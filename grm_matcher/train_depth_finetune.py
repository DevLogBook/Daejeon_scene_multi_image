"""V3: MegaDepth + UAV3DCrop geometry-reference adaptation with exact-H replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader,Dataset
from .config import GRMMatcherConfig
from .matcher import GRMMatcher
from .scratch_transfer import ResizedHomographyDataset,check_homography_splits
from .depth_finetune import DepthCacheDataset,MixedDepthReplay,DepthObjective,FINE_PREFIXES
from .train_scratch_transfer import run_epoch,monitor,rng_state,restore_rng,dataset_signature
from .train_stage25 import _atomic_torch_save,_seed_everything

RECIPE='v3_depth_mix_v1'


class SyntheticTagged(Dataset):
    def __init__(self,data): self.data=data
    def __len__(self): return len(self.data)
    def __getitem__(self,index): return dict(self.data[index],domain='synthetic')


def optimizer_for(model,args):
    groups={}
    for name,param in model.named_parameters():
        param.requires_grad_(True)
        if name.startswith('backbone.'): group,rate='backbone',args.backbone_lr
        elif name.startswith('confidence_head.'): group,rate='confidence',args.confidence_lr
        elif name.startswith(FINE_PREFIXES): group,rate='fine',args.fine_lr
        elif name.startswith(('coarse_transformer.','correlation.','context_bridge.',
                              'fpn.coarse_lateral.','fpn.coarse_output.')): group,rate='coarse',args.coarse_lr
        else: group,rate='mid',args.mid_lr
        groups.setdefault(group,dict(name=group,params=[],lr=rate,peak_lr=rate))['params'].append(param)
    return torch.optim.AdamW(list(groups.values()),weight_decay=args.weight_decay)


def audit_splits(cache):
    rows={s:[json.loads(line) for line in (cache/s/'manifest.jsonl').read_text().splitlines() if line.strip()]
          for s in ('train','val')}
    hashes={s:{r[k] for r in data for k in ('source_sha256_a','source_sha256_b')} for s,data in rows.items()}
    groups={s:{r['group'] for r in data} for s,data in rows.items()}
    if hashes['train']&hashes['val'] or groups['train']&groups['val']:
        raise ValueError('Geometry train/val shares image content or scene/field groups')
    counts={s:{d:sum(r['domain']==d for r in data) for d in ('megadepth','uav')} for s,data in rows.items()}
    if any(n<1 for row in counts.values() for n in row.values()): raise ValueError('Each split needs both real domains')
    return counts


def derived_metrics(metrics):
    metrics['coarse_recall1']=metrics.get('coarse_recall1_count',0)/max(metrics.get('coarse_reference_count',0),1e-8)
    metrics['coarse_recall4']=metrics.get('coarse_recall4_count',0)/max(metrics.get('coarse_reference_count',0),1e-8)
    metrics['initial_24_32_fraction']=metrics.get('initial_24_32_count',0)/max(metrics.get('coarse_reference_count',0),1e-8)
    metrics['reference_coverage_of_outputs']=metrics.get('reference_scored_matches',0)/max(metrics.get('output_matches',0),1e-8)
    metrics['fine_nll_reachable_fraction']=metrics.get('pred_fine_nll_supervised_count',0)/max(metrics.get('pred_local_supervised_count',0),1e-8)
    return metrics


def real_validation_guard(validation,baseline,args):
    # Loss alone can decrease when the learned sampler stops producing matches.
    # Keep collapse from masquerading as an improvement; never filter inference points.
    for domain in ('megadepth','uav'):
        current,initial=validation[domain],baseline[domain]
        if current['output_matches']<max(1.,args.min_output_ratio*initial['output_matches']): return False
        if current['reference_scored_matches']<max(1.,args.min_output_ratio*initial['reference_scored_matches']): return False
        if current['reference_coverage_of_outputs']<initial['reference_coverage_of_outputs']-args.reference_coverage_guard: return False
        if current['coarse_recall4']<initial['coarse_recall4']-args.real_recall_guard: return False
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry-cache',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True,help='Exact-H synthetic replay with isolated train/val manifests')
    p.add_argument('--output-dir',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint',type=Path); source.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int,default=8); p.add_argument('--warmup-epochs',type=int,default=1)
    p.add_argument('--freeze-backbone-epochs',type=int,default=2); p.add_argument('--freeze-fine-epochs',type=int,default=2)
    for name,value in (('backbone',1e-6),('coarse',1e-5),('mid',1e-5),('fine',5e-6),('confidence',5e-6)):
        p.add_argument('--'+name+'-lr',type=float,default=value)
    for name,value in (('anchor-weight',.25),('positive-matchability-weight',.1),('overlap-weight',.1),
                       ('confidence-weight',.1),('cycle-weight',0.),('local-nll-weight',.1),('real-fine-weight',0.)):
        p.add_argument('--'+name,type=float,default=value)
    p.add_argument('--mid-coordinate-weight',type=float,default=.5)
    p.add_argument('--fine-coordinate-weight',type=float,default=1.)
    p.add_argument('--anchor-points',type=int,default=128)
    p.add_argument('--pairs-per-domain',type=int,default=1500)
    p.add_argument('--val-limit',type=int,default=300,help='Per real domain and synthetic; scene-interleaved for real')
    p.add_argument('--size',type=int,default=512); p.add_argument('--accumulation-steps',type=int,default=8)
    p.add_argument('--num-workers',type=int,default=2); p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--min-lr-ratio',type=float,default=.1); p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--gradient-clip',type=float,default=1.); p.add_argument('--device',default='cuda')
    p.add_argument('--log-every',type=int,default=20); p.add_argument('--stop-after-epochs',type=int,default=0)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--synthetic-loss-guard',type=float,default=1.1)
    p.add_argument('--synthetic-pck-guard',type=float,default=.005)
    p.add_argument('--uav-validation-weight',type=float,default=.7)
    p.add_argument('--min-output-ratio',type=float,default=.5)
    p.add_argument('--reference-coverage-guard',type=float,default=.05)
    p.add_argument('--real-recall-guard',type=float,default=.02)
    args=p.parse_args(argv)
    if args.epochs<2 or not 0<=args.warmup_epochs<args.epochs or not 0<=args.freeze_backbone_epochs<args.epochs or not 0<=args.freeze_fine_epochs<args.epochs:
        p.error('Invalid schedule')
    if args.size<64 or args.size%32 or args.accumulation_steps<4 or args.accumulation_steps%4:
        p.error('size must be divisible by32; accumulation must be a positive multiple of4')
    if min(args.anchor_points,args.pairs_per_domain,args.gradient_clip)<=0 or min(args.val_limit,args.num_workers,args.stop_after_epochs,args.log_every,args.weight_decay)<0:
        p.error('Invalid limits')
    if not 0<args.min_lr_ratio<=1 or not 0<=args.uav_validation_weight<=1 or args.synthetic_loss_guard<1 or args.synthetic_pck_guard<0:
        p.error('Invalid ratios or guards')
    if not 0<args.min_output_ratio<=1 or min(args.reference_coverage_guard,args.real_recall_guard)<0:
        p.error('Invalid real-domain collapse guards')
    for key,value in vars(args).items():
        if key.endswith('_lr') and value<=0: p.error('Learning rates must be positive')
        if key.endswith('_weight') and value<0: p.error('Loss weights must be nonnegative')
    _seed_everything(args.seed); torch.set_num_threads(min(4,torch.get_num_threads()))
    raw=torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=False)
    config=GRMMatcherConfig.from_dict(raw['config']); config.use_amp=False
    if config.architecture_version!='v3': raise ValueError('Requires V3; no architecture upgrade is performed')
    if args.resume and raw.get('training_recipe')!=RECIPE: raise ValueError('Use --init-checkpoint for a different training stage')
    counts=audit_splits(args.geometry_cache)
    geometry={(s,d):DepthCacheDataset(args.geometry_cache,s,d,config,args.val_limit if s=='val' else 0)
              for s in ('train','val') for d in ('megadepth','uav')}
    if any(ds.size!=args.size for ds in geometry.values()): raise ValueError('Cache canvas size differs from --size')
    split_info=check_homography_splits(args.data_dir)
    synthetic={s:ResizedHomographyDataset(args.data_dir,s,config,args.size,args.val_limit if s=='val' else 0) for s in ('train','val')}
    mixed=MixedDepthReplay(geometry['train','megadepth'],geometry['train','uav'],synthetic['train'],args.pairs_per_domain,args.seed)
    contract={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
        if k not in ('init_checkpoint','resume','output_dir','device','num_workers','log_every','stop_after_epochs','audit_only')}
    contract.update(cache_signatures={str(k):v.signature for k,v in geometry.items()},synthetic_signature=dataset_signature(args.data_dir,'agri'),
        code_sha256=hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in
            ('train_depth_finetune.py','depth_finetune.py','depth_geometry.py','scratch_transfer.py','stage25.py',
             'real_adaptation.py','train_scratch_transfer.py','architecture_v3.py'))).hexdigest())
    summary=dict(recipe=RECIPE,geometry_counts=counts,synthetic_split=split_info,precision='FP32',
        mix='MegaDepth : UAV : synthetic = 1 : 1 : 2',microbatches_per_epoch=len(mixed),
        optimizer_steps_per_epoch=math.ceil(len(mixed)/args.accumulation_steps),contract=contract,
        caveat='MVS geometry reference, not independent exact GT; loss weights are unablated starting values')
    print(json.dumps(summary,indent=2),flush=True)
    if args.audit_only: return
    if args.resume:
        if raw['contract']!=contract: raise ValueError('Resume data/code/schedule contract changed')
    elif args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError('New stage needs empty output; use --resume only for same stage')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; check GPU allocation and PyTorch driver compatibility')
    model=GRMMatcher(config).to(device); model.load_state_dict(raw['model'],strict=True)
    optimizer=optimizer_for(model,args); objective=DepthObjective(config,args)
    generator=torch.Generator().manual_seed(args.seed)
    args.stage='agri'; args.max_empty_batches=1
    def loader(data): return DataLoader(data,batch_size=1,shuffle=False,num_workers=args.num_workers,
        generator=generator,pin_memory=device.type=='cuda')
    train_loader=loader(mixed)
    validation_loaders={d:loader(geometry['val',d]) for d in ('megadepth','uav')}
    validation_loaders['synthetic']=loader(SyntheticTagged(synthetic['val']))
    steps=math.ceil(len(mixed)/args.accumulation_steps); total=steps*args.epochs; warmup=steps*args.warmup_epochs
    start=step=0; best=1.
    if args.resume:
        optimizer.load_state_dict(raw['optimizer']); restore_rng(raw['rng'],generator)
        start=raw['epoch']+1; step=raw['schedule_step']; best=raw['best_validation_loss']; baseline=raw['baseline']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'run.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    def evaluate(epoch):
        result={}
        for name,val_loader in validation_loaders.items():
            monitor(args,f'[DOMAIN VAL] {name}; PCK uses only known reference positions')
            metrics,_=run_epoch(model,objective,val_loader,optimizer,args,device,epoch,step,total,warmup,train=False)
            result[name]=derived_metrics(metrics)
        return result
    def state(epoch,validation):
        return dict(training_recipe=RECIPE,stage='agri',epoch=epoch,schedule_step=step,global_step=step,
            model=model.state_dict(),optimizer=optimizer.state_dict(),config=config.to_dict(),contract=contract,
            rng=rng_state(generator),best_validation_loss=best,baseline=baseline,validation_metrics=validation,
            confidence_head_trained=config.v3_confidence_head,geometry_heads_trained=False,inlier_head_trained=False,
            provenance=raw.get('provenance') if args.resume else dict(initial_checkpoint=str(args.init_checkpoint),parent=raw.get('provenance')))
    if not args.resume:
        baseline=evaluate(-1)
        (args.output_dir/'baseline.json').write_text(json.dumps(baseline,indent=2))
        _atomic_torch_save(state(-1,baseline),args.output_dir/'best_agri.pt')
    if start>=args.epochs: raise ValueError('Run already complete')
    end=min(args.epochs,start+args.stop_after_epochs) if args.stop_after_epochs else args.epochs
    for epoch in range(start,end):
        mixed.set_epoch(epoch)
        monitor(args,f'[DOMAIN TRAIN] epoch={epoch+1}; frozen_backbone={epoch<args.freeze_backbone_epochs}; '
            f'fine_real_weight={args.real_fine_weight}; negative overlap supervision is synthetic-only')
        train,step=run_epoch(model,objective,train_loader,optimizer,args,device,epoch,step,total,warmup,train=True)
        train=derived_metrics(train)
        for domain in ('megadepth','uav','synthetic'): train[domain+'_loss']/=max(train[domain+'_batch'],1e-8)
        validation=evaluate(epoch)
        score=sum(weight*validation[d]['loss']/max(baseline[d]['loss'],1e-8) for d,weight in
                  (('uav',args.uav_validation_weight),('megadepth',1-args.uav_validation_weight)))
        replay=validation['synthetic']; previous=baseline['synthetic']
        guard=replay['loss']<=previous['loss']*args.synthetic_loss_guard and replay['pck3_predicted']>=previous['pck3_predicted']-args.synthetic_pck_guard
        real_guard=real_validation_guard(validation,baseline,args)
        improved=guard and real_guard and score<best
        if improved: best=score
        checkpoint=state(epoch,validation)
        _atomic_torch_save(checkpoint,args.output_dir/'latest.pt')
        _atomic_torch_save(checkpoint,args.output_dir/f'epoch_{epoch+1:04d}.pt')
        if improved: _atomic_torch_save(checkpoint,args.output_dir/'best_agri.pt')
        record=dict(epoch=epoch+1,train=train,validation=validation,score=score,synthetic_guard_passed=guard,
                    real_guard_passed=real_guard,best_updated=improved,lr={g['name']:g['lr'] for g in optimizer.param_groups})
        with (args.output_dir/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        monitor(args,'[EPOCH END] '+json.dumps(record),record)
    monitor(args,f'[FINISHED] {end}/{args.epochs}; best_score={best:.6f}; 1 means initialization retained')


if __name__=='__main__': main()
