"""V3 UDIS-D adaptation with cache-v2 positives and 1:1 exact-H replay."""
import argparse, hashlib, json, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from .config import GRMMatcherConfig
from .matcher import GRMMatcher
from .scratch_transfer import ScratchTransferObjective, ResizedHomographyDataset, configure_optimizer, check_homography_splits
from .imc_dataset import sparse_bidirectional_labels
from .training_v3 import supervised_v3
from .train_scratch_transfer import run_epoch, monitor, rng_state, restore_rng, dataset_signature
from .train_stage25 import _atomic_torch_save, _seed_everything
from .udis_teacher import UDISCacheDataset


class ReplayDataset(Dataset):
    def __init__(self, real, synthetic, seed):
        self.real=real; self.synthetic=synthetic; self.seed=seed; self.set_epoch(0)
    def set_epoch(self,epoch):
        g=torch.Generator().manual_seed(self.seed+epoch)
        self.real_ids=torch.randperm(len(self.real),generator=g).tolist()
        self.synthetic_ids=[]
        while len(self.synthetic_ids)<len(self.real):
            self.synthetic_ids.extend(torch.randperm(len(self.synthetic),generator=g).tolist())
    def __len__(self): return 2*len(self.real)
    def __getitem__(self,index):
        if index%2==0: return self.real[self.real_ids[index//2]]
        result=dict(self.synthetic[self.synthetic_ids[index//2]]); result['domain']='synthetic'
        return result


class TransferObjective:
    def __init__(self,config): self.base=ScratchTransferObjective(config,epipolar_weight=0.,anchor_weight=1.)
    def agri(self,model,batch,train,epoch):
        if train and not any(p.requires_grad for p in model.backbone.parameters()):
            model.backbone.eval()  # Freeze running statistics as well as weights.
        real=batch.get('domain',['synthetic'])[0]=='real'
        if real:
            a,b=batch['image_a'],batch['image_b']
            reverse = train and bool(torch.rand(()) < .5)
            if reverse: a,b=b,a
            if train:
                # Independent, mild appearance changes. Coordinates stay unchanged.
                def color(x):
                    gain=.9+.2*torch.rand((x.shape[0],3,1,1),device=x.device)
                    bias=(torch.rand((x.shape[0],1,1,1),device=x.device)-.5)*.08
                    return (x*gain+bias).clamp(0,1)
                a,b=color(a),color(b)
            def labels(direction):
                src=batch[direction+'_source']; conf=batch[direction+'_confidence']
                sparse=dict(image_a=a,sparse_a=src,sparse_b=batch[direction+'_target'],
                            sparse_valid=torch.ones_like(conf,dtype=torch.bool),sparse_confidence=conf)
                return sparse_bidirectional_labels(sparse,a.device,model.config,self.base.adaptation).forward
            loss,metrics=supervised_v3(self.base,model,a,b,labels('ba' if reverse else 'ab'),labels('ab' if reverse else 'ba'),None,None)
            if loss is not None:
                metrics['teacher_cycle_mean_px']=torch.cat((batch['ab_cycle'],batch['ba_cycle']),1).mean()
        else:
            loss,metrics=self.base.agri(model,batch,train,epoch)
        if loss is not None:
            metrics.update(real_batch=float(real),synthetic_batch=float(not real),
                           real_loss=float(loss.detach()) if real else 0.,synthetic_loss=float(loss.detach()) if not real else 0.)
        return loss,metrics


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-cache',type=Path,required=True,help='Root containing train/ and val/ cache-v2 files')
    p.add_argument('--data-dir',type=Path,required=True,help='Synthetic replay root with train/val manifests')
    p.add_argument('--output-dir',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint',type=Path); source.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int,default=8); p.add_argument('--warmup-epochs',type=int,default=1)
    p.add_argument('--freeze-backbone-epochs',type=int,default=2)
    p.add_argument('--backbone-lr',type=float,default=2e-6); p.add_argument('--coarse-lr',type=float,default=1e-5)
    p.add_argument('--refinement-lr',type=float,default=1e-5); p.add_argument('--min-lr-ratio',type=float,default=.1)
    p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--gradient-clip',type=float,default=1.)
    p.add_argument('--accumulation-steps',type=int,default=8); p.add_argument('--size',type=int,default=512)
    p.add_argument('--teacher-confidence',type=float,default=.8); p.add_argument('--teacher-cycle-px',type=float,default=2.)
    p.add_argument('--min-teacher-points',type=int,default=32)
    p.add_argument('--num-workers',type=int,default=2); p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--log-every',type=int,default=20); p.add_argument('--limit-train',type=int,default=0)
    p.add_argument('--limit-val',type=int,default=0); p.add_argument('--stop-after-epochs',type=int,default=0)
    p.add_argument('--device',default='cuda'); p.add_argument('--audit-only',action='store_true')
    p.add_argument('--synthetic-loss-guard',type=float,default=1.10)
    p.add_argument('--synthetic-pck-guard',type=float,default=.005,help='Maximum absolute PCK@3 decrease from initialization')
    args=p.parse_args(argv)
    if args.epochs<2 or not 0<=args.warmup_epochs<args.epochs or not 0<=args.freeze_backbone_epochs<args.epochs:
        p.error('Invalid epoch/warmup/freeze schedule')
    if args.size%32 or args.size<64 or args.accumulation_steps<2 or args.accumulation_steps%2:
        p.error('size must be >=64 and divisible by 32; accumulation must be positive and even')
    if not 0<args.teacher_confidence<1 or not 0<args.min_lr_ratio<=1: p.error('Invalid probability/LR ratio')
    if min(args.backbone_lr,args.coarse_lr,args.refinement_lr,args.gradient_clip,args.teacher_cycle_px,args.min_teacher_points)<=0:
        p.error('Rates, clipping and quality thresholds must be positive')
    if min(args.num_workers,args.limit_train,args.limit_val,args.stop_after_epochs,args.log_every,args.weight_decay,args.synthetic_pck_guard)<0 or args.synthetic_loss_guard<1:
        p.error('Invalid limits/workers/guards')
    _seed_everything(args.seed)
    raw=torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=False)
    config=GRMMatcherConfig.from_dict(raw['config'])
    if config.architecture_version!='v3': raise ValueError('This real adaptation stage requires V3; V1/V2 are preserved')
    config.use_amp=False
    if args.resume and raw.get('training_recipe')!='v3_udis_teacher_v1': raise ValueError('Resume requires this same real adaptation recipe')
    real=[UDISCacheDataset(args.teacher_cache/s,config,args.size,args.teacher_confidence,args.teacher_cycle_px,args.min_teacher_points,
                          args.limit_train if s=='train' else args.limit_val) for s in ('train','val')]
    if real[0].source_hashes & real[1].source_hashes: raise ValueError('Teacher train/val image content overlaps, including rejected cache files')
    if real[0].teacher_contracts!=real[1].teacher_contracts: raise ValueError('Teacher train/val used different weights/settings')
    split_info=check_homography_splits(args.data_dir)
    synthetic=[ResizedHomographyDataset(args.data_dir,s,config,args.size,args.limit_train if s=='train' else args.limit_val) for s in ('train','val')]
    mixed=ReplayDataset(real[0],synthetic[0],args.seed)
    contract={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ('init_checkpoint','resume','output_dir','device','audit_only','stop_after_epochs','log_every','num_workers')}
    contract.update(teacher_signatures=[d.signature for d in real],synthetic_signature=dataset_signature(args.data_dir,'agri'),
                    code_signature=hashlib.sha256(b''.join(Path(__file__).with_name(n).read_bytes() for n in
                        ('train_udis_finetune.py','udis_teacher.py','training_v3.py','scratch_transfer.py','real_adaptation.py','train_scratch_transfer.py'))).hexdigest())
    summary={'recipe':'v3_udis_teacher_v1','precision':'FP32','batch_size':1,'replay_ratio':'1 real : 1 synthetic',
             'train_real_accepted':len(real[0]),'val_real_accepted':len(real[1]),'cache_rejected':[d.rejected for d in real],
             'teacher_quality_audit':[d.audit for d in real],
             'microbatches_per_epoch':len(mixed),'epochs':args.epochs,'contract':contract,'synthetic_split':split_info,
             'warning':'Teacher consistency is not ground-truth accuracy; UDIS-D is real-domain, not agriculture-specific'}
    print(json.dumps(summary,indent=2),flush=True)
    if args.audit_only: return
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError('Use an empty output directory, or --resume latest.pt')
    if args.resume and raw['contract']!=contract: raise ValueError('Resume contract changed: preserve data, code, thresholds and schedule')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'run.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; check GPU allocation and PyTorch environment')
    model=GRMMatcher(config).to(device); model.load_state_dict(raw['model'],strict=True)
    optimizer=configure_optimizer(model,'agri',imc_lr=args.coarse_lr,backbone_lr=args.backbone_lr,
        coarse_lr=args.coarse_lr,refinement_lr=args.refinement_lr,geometry_lr=args.refinement_lr,weight_decay=args.weight_decay)
    objective=TransferObjective(config)
    generator=torch.Generator().manual_seed(args.seed)
    args.stage='agri'; args.max_empty_batches=1
    def loader(dataset):
        return DataLoader(dataset,batch_size=1,shuffle=False,num_workers=args.num_workers,generator=generator,pin_memory=device.type=='cuda')
    train_loader=loader(mixed); real_val_loader=loader(real[1]); synthetic_val_loader=loader(synthetic[1])
    per_epoch=math.ceil(len(mixed)/args.accumulation_steps); total=args.epochs*per_epoch; warmup=args.warmup_epochs*per_epoch
    start,step,best=0,0,1.
    if args.resume:
        optimizer.load_state_dict(raw['optimizer']); restore_rng(raw['rng'],generator)
        start=raw['epoch']+1; step=raw['schedule_step']; best=raw['best_validation_loss']; baseline=raw['baseline']
    def evaluate(epoch):
        common=dict(model=model,objective=objective,optimizer=optimizer,args=args,device=device,epoch=epoch,
                    schedule_step=step,total_steps=total,warmup_steps=warmup,train=False)
        r,_=run_epoch(loader=real_val_loader,**common); s,_=run_epoch(loader=synthetic_val_loader,**common)
        # The real metric is explicitly agreement on trusted Teacher anchors.
        r.pop('pck3_predicted',None)
        r['teacher_anchor_pck3']=r.get('anchor_predicted_correct_3px',0)/max(r.get('confidence_supervised_count',0),1)
        return {'real_teacher':r,'synthetic_gt':s}
    def save(epoch,validation):
        return dict(training_recipe='v3_udis_teacher_v1',stage='agri',epoch=epoch,schedule_step=step,global_step=step,
            model=model.state_dict(),optimizer=optimizer.state_dict(),config=config.to_dict(),contract=contract,
            rng=rng_state(generator),best_validation_loss=best,baseline=baseline,validation_metrics=validation,
            geometry_heads_trained=False,inlier_head_trained=False,local_role_positive_supervision=False,
            confidence_head_trained=config.v3_confidence_head,
            provenance=raw.get('provenance') if args.resume else {'initial_checkpoint':str(args.init_checkpoint),'parent':raw.get('provenance')})
    if not args.resume:
        monitor(args,'[BASELINE] evaluating the exact initialization before updates')
        baseline=evaluate(-1)
        _atomic_torch_save(save(-1,baseline),args.output_dir/'best_agri.pt')
        (args.output_dir/'baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    if start>=args.epochs: raise ValueError('Run already completed')
    end=min(args.epochs,start+args.stop_after_epochs) if args.stop_after_epochs else args.epochs
    for epoch in range(start,end):
        frozen=epoch<args.freeze_backbone_epochs
        for param in model.backbone.parameters(): param.requires_grad_(not frozen)
        mixed.set_epoch(epoch)
        monitor(args,f'[DOMAIN] epoch={epoch+1} backbone_frozen={frozen}; coarse and refinement train throughout')
        metrics,step=run_epoch(model,objective,train_loader,optimizer,args,device,epoch,step,total,warmup,train=True)
        for domain in ('real','synthetic'):
            metrics[domain+'_loss']/=max(metrics[domain+'_batch'],1e-12)
        validation=evaluate(epoch)
        r,s=validation['real_teacher'],validation['synthetic_gt']; br,bs=baseline['real_teacher'],baseline['synthetic_gt']
        score=.5*(r['loss']/max(br['loss'],1e-12)+s['loss']/max(bs['loss'],1e-12))
        guard=s['loss']<=bs['loss']*args.synthetic_loss_guard and s['pck3_predicted']>=bs['pck3_predicted']-args.synthetic_pck_guard
        improved=guard and score<best
        if improved: best=score
        state=save(epoch,validation)
        _atomic_torch_save(state,args.output_dir/'latest.pt')
        _atomic_torch_save(state,args.output_dir/f'epoch_{epoch+1:04d}.pt')
        if improved: _atomic_torch_save(state,args.output_dir/'best_agri.pt')
        record=dict(epoch=epoch+1,train=metrics,validation=validation,score=score,synthetic_guard_passed=guard,best_updated=improved,
                    lr={g['name']:g['lr'] for g in optimizer.param_groups})
        with (args.output_dir/'metrics.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(record)+'\n')
        monitor(args,'[EPOCH END] '+json.dumps(record),record)
    monitor(args,f'[FINISHED] epoch={end}/{args.epochs}; best_agri.pt score={best:.6f}; score=1 means initialization retained')

if __name__=='__main__': main()
