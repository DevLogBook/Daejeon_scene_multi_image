"""R1 full MegaDepth traversal; R2 full UAV traversal with audited UDIS Teacher."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader, Subset

from .config import GRMMatcherConfig
from .matcher import GRMMatcher
from .depth_curriculum import TaggedSynthetic, group_indices, interleaved_indices, parameter_group, reference_metrics
from .real_curriculum import (FullCoverageEpoch, RealCurriculumObjective, RealGeometryData,
                              TeacherGeometryData, audit_geometry)
from .scratch_transfer import ResizedHomographyDataset, check_homography_splits, lr_multiplier
from .train_scratch_transfer import monitor, rng_state, restore_rng
from .train_stage25 import _atomic_torch_save, _seed_everything
from .train_depth_curriculum import file_sha256, json_write, synthetic_signature

RECIPE = 'grm_foundation_real_r1_r2_v1'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('R1','R2'), required=True)
    p.add_argument('--geometry-cache', type=Path, required=True)
    p.add_argument('--teacher-cache', type=Path)
    p.add_argument('--megadepth-validation-cache', type=Path, help='Optional R2 MD forgetting monitor; validation only')
    p.add_argument('--synthetic-data', type=Path, required=True, help='Validation ONLY; no synthetic training replay')
    p.add_argument('--output-dir', type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint', type=Path)
    source.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--warmup-epochs', type=float, default=1.)
    p.add_argument('--accumulation-steps', type=int, default=8)
    p.add_argument('--backbone-lr', type=float)
    p.add_argument('--coarse-lr', type=float)
    p.add_argument('--refinement-lr', type=float)
    p.add_argument('--confidence-lr', type=float)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--gradient-clip', type=float, default=1.)
    p.add_argument('--local-freeze-updates', type=int, help='R1 default500, R2 default0; local weights and state frozen')
    p.add_argument('--anchor-points', type=int, default=128)
    p.add_argument('--anchor-jitter', type=float, default=8.)
    p.add_argument('--geometry-fine-weight', type=float, default=.25)
    p.add_argument('--geometry-fine-min-weight', type=float, default=.8)
    p.add_argument('--geometry-fine-cycle-px', type=float, default=.5)
    p.add_argument('--teacher-confidence', type=float, default=.8)
    p.add_argument('--teacher-cycle-px', type=float, default=2.)
    p.add_argument('--teacher-fine-weight', type=float, default=.1)
    p.add_argument('--teacher-fine-confidence', type=float, default=.95)
    p.add_argument('--teacher-fine-cycle-px', type=float, default=.5)
    p.add_argument('--min-teacher-points', type=int, default=32)
    p.add_argument('--udis-fraction', type=float, default=.4)
    p.add_argument('--udis-start-fraction', type=float, default=.2)
    p.add_argument('--udis-warmup-epochs', type=int, default=1)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--val-limit', type=int, default=300, help='Fixed scene-interleaved per domain; 0=all')
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--stop-after-updates', type=int, default=0, help='Pilot length; full epoch/LR plan stays unchanged')
    p.add_argument('--fine-pilot-updates', type=int, default=0,
                   help='R1 only: train Fine/FPN Fine for exactly this many updates, with its own LR schedule')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args(argv)
    if min(args.epochs,args.accumulation_steps,args.anchor_points,args.eval_every,args.save_every,args.log_every)<=0:
        p.error('Epochs, accumulation and intervals must be positive')
    if args.size<64 or args.size%32 or not 0<=args.warmup_epochs<args.epochs:
        p.error('Invalid size/warmup')
    if args.phase=='R2' and args.teacher_cache is None:
        p.error('R2 requires --teacher-cache with complete cache-v2 train/val')
    if args.resume and args.evaluate_only:
        p.error('Evaluate using --init-checkpoint')
    if not 0<args.min_lr_ratio<=1 or not 0<args.udis_fraction<1 or not 0<args.udis_start_fraction<1:
        p.error('Invalid LR/mixture fraction')
    if min(args.num_workers,args.val_limit,args.stop_after_updates,args.fine_pilot_updates,
           args.weight_decay,args.udis_warmup_epochs)<0:
        p.error('Negative limits/decay')
    if args.fine_pilot_updates and (args.phase != 'R1' or args.warmup_epochs != 0
                                   or args.local_freeze_updates not in (None, 0)
                                   or args.geometry_fine_weight <= 0):
        p.error('Fine pilot requires R1, --warmup-epochs 0, no local freeze and positive geometry Fine weight')
    rates = (1e-5,2e-5,5e-6) if args.phase=='R1' else (5e-6,1e-5,2e-6)
    for name, default in zip(('backbone_lr','coarse_lr','refinement_lr'), rates):
        if getattr(args,name) is None: setattr(args,name,default)
    if args.confidence_lr is None: args.confidence_lr=args.refinement_lr
    if args.local_freeze_updates is None:
        args.local_freeze_updates=500 if args.phase=='R1' and not args.fine_pilot_updates else 0
    if args.local_freeze_updates<0 or not 0<=args.anchor_jitter<=8:
        p.error('Invalid local freeze/jitter')
    for name in ('backbone_lr','coarse_lr','refinement_lr','confidence_lr','gradient_clip',
                 'geometry_fine_cycle_px','teacher_cycle_px','teacher_fine_cycle_px','min_teacher_points'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0: p.error('Invalid '+name)
    for name in ('geometry_fine_weight','teacher_fine_weight','geometry_fine_min_weight',
                 'teacher_confidence','teacher_fine_confidence'):
        if not 0<=getattr(args,name)<=1: p.error('Invalid '+name)
    if args.teacher_fine_confidence<args.teacher_confidence or args.teacher_fine_cycle_px>args.teacher_cycle_px:
        p.error('Teacher Fine gate must be at least as strict as its Coarse gate')
    return args


def fraction(args, epoch):
    blend = min(1., epoch/max(1,args.udis_warmup_epochs)) if args.udis_warmup_epochs else 1.
    return args.udis_start_fraction+(args.udis_fraction-args.udis_start_fraction)*blend


def fixed_subset(data, limit, real=True):
    ids = interleaved_indices(group_indices(data.records,real), limit)
    return Subset(data, ids), [data.records[i] for i in ids]


def build_data(args, config):
    focus = 'megadepth' if args.phase=='R1' else 'uav'
    audit = {'geometry':audit_geometry(args.geometry_cache,focus,config,args.size)}
    kw = dict(fine_min_weight=args.geometry_fine_min_weight, fine_cycle_px=args.geometry_fine_cycle_px)
    sources = {focus:RealGeometryData(args.geometry_cache,'train',focus,config,**kw)}
    full_val = {focus:RealGeometryData(args.geometry_cache,'val',focus,config,**kw)}
    if args.phase=='R2':
        for split, target in (('train',sources),('val',full_val)):
            target['udis'] = TeacherGeometryData(args.teacher_cache/split,config,args.size,
                args.teacher_confidence,args.teacher_cycle_px,args.min_teacher_points,
                args.teacher_fine_confidence,args.teacher_fine_cycle_px)
        if sources['udis'].teacher_contracts != full_val['udis'].teacher_contracts:
            raise ValueError('Teacher train/val model settings differ')
        audit['teacher'] = {s:ds.audit for s,ds in (('train',sources['udis']),('val',full_val['udis']))}
        if args.megadepth_validation_cache:
            audit['megadepth_monitor']=audit_geometry(args.megadepth_validation_cache,'megadepth',config,args.size)
            full_val['megadepth']=RealGeometryData(args.megadepth_validation_cache,'val','megadepth',config,**kw)
    # Audit all source content, including rejected Teacher files, before limits.
    train_hashes=set().union(*(ds.source_hashes for ds in sources.values()))
    val_hashes=set().union(*(ds.source_hashes for ds in full_val.values()))
    if train_hashes & val_hashes:
        raise ValueError('Real train/validation image content overlaps across domains')
    audit['synthetic_split'] = check_homography_splits(args.synthetic_data)
    synth = TaggedSynthetic(ResizedHomographyDataset(args.synthetic_data,'val',config,args.size))
    full_val['synthetic'] = synth
    validation, selected = {}, {}
    for d, ds in full_val.items():
        validation[d], selected[d] = fixed_subset(ds,args.val_limit,d!='synthetic')
    signatures={d:ds.signature for d,ds in sources.items()}
    signatures.update({'val/'+d:ds.signature for d,ds in full_val.items() if d!='synthetic'})
    signatures['synthetic'] = synthetic_signature(args.synthetic_data,{'val':synth})
    return sources, validation, dict(audit=audit, signatures=signatures, validation_records=selected)


def objective(args, config, capture=False):
    return RealCurriculumObjective(config, anchor_points=args.anchor_points, anchor_jitter=args.anchor_jitter,
        geometry_fine_weight=args.geometry_fine_weight, teacher_fine_weight=args.teacher_fine_weight,
        local_freeze_updates=args.local_freeze_updates, capture_prediction=capture,
        fine_only=bool(args.fine_pilot_updates))


def loader(data, args, device, seed):
    return DataLoader(data,batch_size=1,shuffle=False,num_workers=args.num_workers,
        pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(seed))


def to_device(batch, device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def evaluate(model, validation, args, device, step):
    obj=objective(args,model.config,True); obj.update=step
    result={}
    with torch.no_grad(), (args.output_dir/f'validation_step_{step:07d}_pairs.jsonl').open('w',encoding='utf-8') as stream:
        for domain, ds in validation.items():
            scenes=defaultdict(lambda:defaultdict(float)); totals=defaultdict(float)
            excluded=0
            for cpu in loader(ds,args,device,args.seed):
                loss, metrics=obj.agri(model,to_device(cpu,device),False)
                if loss is None or not torch.isfinite(loss) or not all(math.isfinite(float(v)) for v in metrics.values()):
                    raise FloatingPointError(f'Invalid validation: {cpu["pair_id"]}')
                scene=cpu['scene_id'][0]
                stream.write(json.dumps(dict(domain=domain,scene=scene,pair_id=cpu['pair_id'][0],**metrics))+'\n')
                if not metrics.get('reference_count',0):
                    excluded+=1; continue
                for k,v in metrics.items(): totals[k]+=float(v);scenes[scene][k]+=float(v)
                totals['pairs']+=1;scenes[scene]['pairs']+=1
            if not scenes: raise ValueError('No scorable validation '+domain)
            by_scene={s:reference_metrics(v) for s,v in scenes.items()}
            macro={k:sum(v[k] for v in by_scene.values())/len(by_scene)
                   for k in next(iter(by_scene.values())) if k!='eligible'}
            result[domain]=dict(pairs=len(ds),excluded=excluded,micro=reference_metrics(totals),macro=macro,
                by_scene=by_scene,loss=totals['loss']/max(1,totals['pairs']),
                reference_kind='Teacher agreement, not GT accuracy' if domain=='udis' else
                'exact synthetic correspondence' if domain=='synthetic' else 'trusted reconstructed geometry')
            monitor(args,f'[VAL] step={step} domain={domain} '+json.dumps(macro))
    json_write(args.output_dir/f'validation_step_{step:07d}.json',result)
    return result


def selection_scores(result, phase, fine_only=False):
    focus=result['megadepth' if phase=='R1' else 'uav']['macro']
    if fine_only:
        return dict(phase=focus['reference_f1_1'], pck1=focus['pck1'],
                    synthetic=result['synthetic']['macro']['pck1'])
    return dict(phase=focus['coarse_recall1'] if phase=='R1' else focus['reference_f1_3'],
                refinement=focus['reference_f1_3'],synthetic=result['synthetic']['macro']['pck1'])


def make_optimizer(model, args):
    groups=defaultdict(list)
    for n,p in model.named_parameters():
        group = parameter_group(n)
        if not args.fine_pilot_updates or group == 'fine':
            groups[group].append(p)
    rates=dict(backbone=args.backbone_lr,coarse=args.coarse_lr,mid=args.refinement_lr,
               fine=args.refinement_lr,bridge=args.refinement_lr,confidence=args.confidence_lr)
    return torch.optim.AdamW([dict(name=k,params=v,lr=rates[k],peak_lr=rates[k]) for k,v in groups.items()],
                             weight_decay=args.weight_decay)


def code_signature():
    paths=sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()


def main(argv=None):
    args=parse_args(argv); _seed_everything(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    raw=torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=False)
    config=GRMMatcherConfig.from_dict(raw['config']);config.use_amp=False
    if config.architecture_version!='v3' or getattr(config,'adaptive_enabled',False):
        raise ValueError('Requires plain V3/F8/F8-HRA with Adaptive disabled')
    if (config.input_height,config.input_width)!=(args.size,args.size):
        raise ValueError('Checkpoint canvas differs from --size')
    if args.resume and (raw.get('training_recipe')!=RECIPE or raw.get('phase')!=args.phase):
        raise ValueError('Resume requires this exact phase; phase changes use --init-checkpoint')
    sources,validation,data=build_data(args,config)
    lengths=[];epoch_plans=[]
    for e in range(args.epochs):
        plan=FullCoverageEpoch(sources,args.phase,args.seed,e,1-fraction(args,e))
        lengths.append(len(plan));epoch_plans.append(dict(epoch=e+1,pairs=len(plan),counts=plan.counts,unique=plan.unique_counts))
    total_steps=sum(math.ceil(n/args.accumulation_steps) for n in lengths)
    warmup_steps=round(args.warmup_epochs*math.ceil(lengths[0]/args.accumulation_steps))
    if args.fine_pilot_updates:
        if args.fine_pilot_updates > total_steps:
            raise ValueError('Fine pilot needs more data or --epochs for the requested update budget')
        total_steps = args.fine_pilot_updates
        warmup_steps = min(100, total_steps // 10)
    if warmup_steps>=total_steps: raise ValueError('Warmup consumes the complete training plan')
    ignored={'init_checkpoint','resume','output_dir','device','audit_only','evaluate_only','stop_after_updates','num_workers','log_every','save_every'}
    contract={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ignored}
    contract.update(config=config.to_dict(),data_signatures=data['signatures'],
        fixed_validation_sha256=hashlib.sha256(json.dumps(data['validation_records'],sort_keys=True).encode()).hexdigest(),
        source_sha256=code_signature(),total_steps=total_steps,warmup_steps=warmup_steps)
    selection = ('MD macro trusted-reference F1@1 (Fine pilot)' if args.fine_pilot_updates else
                 'MD macro coarse Recall@1' if args.phase=='R1' else 'UAV macro trusted-reference F1@3')
    summary=dict(recipe=RECIPE,phase=args.phase,precision='FP32',microbatch=1,accumulation=args.accumulation_steps,
        training_pairs={d:len(ds) for d,ds in sources.items()},validation_pairs={d:len(ds) for d,ds in validation.items()},
        epoch_plans=epoch_plans,total_steps=total_steps,warmup_steps=warmup_steps,contract=contract,
        selection=selection,
        training_scope='fine_only_update_budget' if args.fine_pilot_updates else 'full_epoch_traversal',
        synthetic_use='validation only; not training replay')
    print(json.dumps(summary,indent=2),flush=True)
    if args.audit_only: return summary
    if args.resume:
        if raw['contract']!=contract or Path(raw['output_dir']).resolve()!=args.output_dir.resolve():
            raise ValueError('Exact resume contract/output changed; preserve schedule/data/code/config')
        if not args.output_dir.is_dir(): raise ValueError('Resume directory is missing')
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new empty output directory; interrupted run uses --resume')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in allocated environment')
    model=GRMMatcher(config).to(device);model.load_state_dict(raw['model'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise FloatingPointError('Initialization contains nonfinite weights')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    json_write(args.output_dir/'run.json',summary);json_write(args.output_dir/'data_audit.json',data['audit'])
    json_write(args.output_dir/'validation_pairs.json',data['validation_records'])
    generator=torch.Generator().manual_seed(args.seed)
    opt=make_optimizer(model,args);obj=objective(args,config)
    epoch=offset=step=0;best={};baseline=None;validation_metrics=None;validation_step=-1
    lineage=(raw.get('provenance') if args.resume else dict(initial_checkpoint=str(args.init_checkpoint.resolve()),
        sha256=file_sha256(args.init_checkpoint),parent_recipe=raw.get('training_recipe'),parent=raw.get('provenance')))
    if args.resume:
        opt.load_state_dict(raw['optimizer']);restore_rng(raw['rng'],generator)
        epoch,offset,step=raw['next_epoch'],raw['next_pair_offset'],raw['schedule_step']
        best=raw['best_scores'];baseline=raw['baseline'];validation_metrics=raw['validation_metrics'];validation_step=raw['validation_step']
    interrupted=[]
    def request_stop(signum, frame): interrupted.append(signum)
    signal.signal(signal.SIGTERM,request_stop)
    if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,request_stop)
    def state():
        return dict(training_recipe=RECIPE,stage='agri',phase=args.phase,config=config.to_dict(),model=model.state_dict(),
            optimizer=opt.state_dict(),schedule_step=step,global_step=step,next_epoch=epoch,next_pair_offset=offset,
            output_dir=str(args.output_dir.resolve()),contract=contract,rng=rng_state(generator),best_scores=best.copy(),
            baseline=baseline,validation_metrics=validation_metrics,validation_step=validation_step,provenance=lineage,
            confidence_head_trained=True,geometry_heads_trained=False,inlier_head_trained=False)
    def save(name):_atomic_torch_save(state(),args.output_dir/name)
    def validate_and_select():
        nonlocal baseline,validation_metrics,validation_step,best
        before=rng_state(generator)
        validation_metrics=evaluate(model,validation,args,device,step);validation_step=step
        restore_rng(before,generator)
        if baseline is None:baseline=validation_metrics
        improved=[]
        for key,value in selection_scores(validation_metrics,args.phase,bool(args.fine_pilot_updates)).items():
            if key not in best or value>best[key]:best[key]=value;improved.append(key)
        for key in improved:save('best_'+key+'.pt')
        monitor(args,f'[SELECTION] step={step} improved={improved} best={best}')
    if args.evaluate_only:
        evaluate(model,validation,args,device,0);return
    if step>=total_steps:raise ValueError('Run already complete')
    if not args.resume:validate_and_select();save('latest.pt')
    stop_at=min(total_steps,step+args.stop_after_updates) if args.stop_after_updates else total_steps
    scope = 'Fine only, fixed update budget' if args.fine_pilot_updates else 'full traversal, no scene cap'
    monitor(args,f'[START] {args.phase} update={step}/{total_steps} stop={stop_at}; {scope}')
    while epoch<args.epochs and step<stop_at:
        plan=FullCoverageEpoch(sources,args.phase,args.seed,epoch,1-fraction(args,epoch),offset)
        monitor(args,f'[EPOCH PLAN] epoch={epoch+1} cursor={offset}/{len(plan.plan)} counts={plan.counts} unique={plan.unique_counts}')
        iterator=iter(loader(plan,args,device,args.seed+epoch))
        remaining=len(plan);within=0
        while within<remaining and step<stop_at:
            count=min(args.accumulation_steps,remaining-within)
            opt.zero_grad(set_to_none=True);acc=defaultdict(float);pairs=[];by_domain=defaultdict(lambda:defaultdict(float));started=time.monotonic()
            obj.update=step
            for _ in range(count):
                cpu=next(iterator);domain=cpu['domain'][0];pairs.append(dict(domain=domain,pair_id=cpu['pair_id'][0]))
                loss,metrics=obj.agri(model,to_device(cpu,device),True,epoch)
                if loss is None or not torch.isfinite(loss) or not all(math.isfinite(float(v)) for v in metrics.values()):
                    json_write(args.output_dir/'failure.json',dict(step=step,pairs=pairs,metrics=repr(metrics)))
                    raise FloatingPointError('Invalid real loss/supervision; see failure.json')
                (loss/count).backward()
                for k,v in metrics.items():acc[k]+=float(v);by_domain[domain][k]+=float(v)
                by_domain[domain]['pairs']+=1
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip,error_if_nonfinite=True))
            factor=lr_multiplier(step,total_steps,warmup_steps,args.min_lr_ratio)
            for group in opt.param_groups:group['lr']=group['peak_lr']*factor
            opt.step();step+=1;within+=count;offset+=count
            if offset==lengths[epoch]:epoch+=1;offset=0;epoch_done=True
            else:epoch_done=False
            if step==1 or step%args.log_every==0 or epoch_done:
                means={k:v/count for k,v in acc.items()}
                event=dict(phase=args.phase,step=step,total_steps=total_steps,next_epoch=epoch,next_pair_offset=offset,
                    microbatches=count,metrics=means,reference_metrics=reference_metrics(acc),
                    by_domain={d:dict(pairs=int(v['pairs']),loss=v['loss']/v['pairs'],**reference_metrics(v)) for d,v in by_domain.items()},
                    lr={g['name']:g['lr'] for g in opt.param_groups},gradient_norm=norm,
                    seconds=time.monotonic()-started,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None)
                monitor(args,f'[TRAIN] {args.phase} step={step}/{total_steps} loss={means["loss"]:.4f} coarse={means["coarse"]:.4f} '+
                    f'PCK1={event["reference_metrics"]["pck1"]:.2%} coarse_R1={event["reference_metrics"]["coarse_recall1"]:.2%} grad={norm:.3f}',event)
            # Scheduler termination grace periods can be shorter than validation.
            # Persist the completed update BEFORE any long validation work.
            if interrupted:
                save('latest.pt')
                monitor(args,'[STOP] signal received; checkpoint saved at update boundary without waiting for validation')
                return
            if step%args.eval_every==0 or epoch_done or step==stop_at:validate_and_select()
            if step%args.save_every==0 or epoch_done or step==stop_at:save('latest.pt')
            if epoch_done:save(f'epoch_{epoch:04d}.pt')
            if epoch_done:break
    monitor(args,f'[FINISH] updates={step}/{total_steps}; next_epoch={epoch}; offset={offset}')
    return dict(step=step,total_steps=total_steps,best=best)


if __name__=='__main__':
    main()
