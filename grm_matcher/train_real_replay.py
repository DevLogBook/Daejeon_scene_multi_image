"""R3: four-domain replay repair from the archived R2 step9000 checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import signal
import time

import torch

from .config import GRMMatcherConfig
from .matcher import GRMMatcher
from .depth_curriculum import TaggedSynthetic, parameter_group, reference_metrics
from .real_curriculum import RealGeometryData
from .real_replay import ReplayPlan, ReplayObjective, guarded_scores, set_learning_rates
from .scratch_transfer import ResizedHomographyDataset
from .train_real_curriculum import build_data as build_r2_data, evaluate, loader, to_device
from .train_depth_curriculum import json_write, synthetic_signature
from .train_scratch_transfer import monitor, rng_state, restore_rng
from .train_stage25 import _atomic_torch_save, _seed_everything

RECIPE = 'grm_foundation_real_r3_replay_v1'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('geometry-cache', 'megadepth-cache', 'teacher-cache', 'synthetic-data', 'output-dir'):
        p.add_argument('--'+flag, type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint', type=Path)
    source.add_argument('--resume', type=Path)
    for flag, default in dict(total_updates=8000, warmup_updates=300, local_freeze_updates=1000,
                              local_warmup_updates=200, accumulation_steps=8, anchor_points=128,
                              min_teacher_points=32, size=512, val_limit=300, eval_every=1000,
                              save_every=250, log_every=20, stop_after_updates=0, num_workers=2,
                              seed=2026, guard_patience=2).items():
        p.add_argument('--'+flag.replace('_','-'), type=int, default=default)
    for flag, default in dict(coarse_lr=3e-6, refinement_lr=5e-7, confidence_lr=5e-7,
                              min_lr_ratio=.1, weight_decay=1e-4, gradient_clip=1., anchor_jitter=8.,
                              geometry_fine_weight=.25, geometry_fine_min_weight=.8, geometry_fine_cycle_px=.5,
                              teacher_confidence=.8, teacher_cycle_px=2., teacher_fine_weight=.1,
                              teacher_fine_confidence=.95, teacher_fine_cycle_px=.5,
                              md_guard=.01, uav_guard=.005, synthetic_guard=.005).items():
        p.add_argument('--'+flag.replace('_','-'), type=float, default=default)
    p.add_argument('--device', default='cuda')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args(argv)
    positive = ('total_updates','accumulation_steps','anchor_points','min_teacher_points','eval_every',
                'save_every','log_every','guard_patience','coarse_lr','refinement_lr','confidence_lr',
                'gradient_clip','geometry_fine_cycle_px','teacher_cycle_px','teacher_fine_cycle_px')
    if any(not math.isfinite(getattr(args,n)) or getattr(args,n)<=0 for n in positive):
        p.error('Update counts, rates and intervals must be finite and positive')
    if not 0 <= args.warmup_updates < args.total_updates or not 0 <= args.local_freeze_updates < args.total_updates:
        p.error('Warmup/freeze must fit the complete update plan')
    for n in ('local_warmup_updates','num_workers','val_limit','stop_after_updates','weight_decay',
              'md_guard','uav_guard','synthetic_guard'):
        if not math.isfinite(getattr(args,n)) or getattr(args,n)<0: p.error('Invalid '+n)
    if args.size<64 or args.size%32 or not 0 <= args.anchor_jitter <= 8 or not 0 < args.min_lr_ratio <= 1:
        p.error('Invalid canvas, jitter or minimum LR')
    for n in ('geometry_fine_weight','geometry_fine_min_weight','teacher_confidence',
              'teacher_fine_weight','teacher_fine_confidence'):
        if not 0 <= getattr(args,n) <= 1: p.error('Invalid '+n)
    if args.teacher_fine_confidence<args.teacher_confidence or args.teacher_fine_cycle_px>args.teacher_cycle_px:
        p.error('Fine Teacher gate must be at least as strict as Coarse')
    if args.resume and args.evaluate_only: p.error('Evaluate using --init-checkpoint')
    return args


def build_data(args, config):
    # Reuse the existing audits and fixed validation protocol; add actual MD/SYN training replay.
    legacy = argparse.Namespace(**vars(args), phase='R2', megadepth_validation_cache=args.megadepth_cache)
    sources, validation, data = build_r2_data(legacy, config)
    sources['megadepth'] = RealGeometryData(args.megadepth_cache,'train','megadepth',config,
        fine_min_weight=args.geometry_fine_min_weight,fine_cycle_px=args.geometry_fine_cycle_px)
    sources['synthetic'] = TaggedSynthetic(ResizedHomographyDataset(args.synthetic_data,'train',config,args.size))
    train_hashes = set().union(*(ds.source_hashes for d,ds in sources.items() if d!='synthetic'))
    val_hashes = set().union(*(ds.dataset.source_hashes for d,ds in validation.items() if d!='synthetic'))
    if train_hashes & val_hashes:
        raise ValueError('Cross-domain real train/validation content overlap')
    data['signatures']['megadepth'] = sources['megadepth'].signature
    data['signatures']['synthetic'] = synthetic_signature(args.synthetic_data,
        dict(train=sources['synthetic'], val=validation['synthetic'].dataset))
    return sources, validation, data


def make_optimizer(model, args):
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        group = parameter_group(name)
        if group == 'backbone': p.requires_grad_(False)
        else: groups[group].append(p)
    rates = dict(coarse=args.coarse_lr, mid=args.refinement_lr, fine=args.refinement_lr,
                 bridge=args.refinement_lr, confidence=args.confidence_lr)
    return torch.optim.AdamW([dict(name=k, params=v, lr=rates[k], peak_lr=rates[k]) for k,v in groups.items()],
                             weight_decay=args.weight_decay, betas=(.9,.999), eps=1e-8)


def same_training_contract(saved, current):
    old=dict(saved);new=dict(current)
    for key in ('source_sha256','expected_init_sha256','plan_sha256','fixed_validation_sha256'):
        old.pop(key,None);new.pop(key,None)
    if 'validation_records' not in old:new.pop('validation_records',None)
    a=dict(old.get('data_signatures',{}));b=dict(new.get('data_signatures',{}))
    for domain in ('udis','val/udis'):
        if isinstance(a.get(domain),str) and isinstance(b.get(domain),list):a[domain]=b[domain]
    old['data_signatures']=a;new['data_signatures']=b
    return old==new


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    source = args.resume or args.init_checkpoint
    raw = torch.load(source, map_location='cpu', weights_only=False)
    if args.resume and (raw.get('training_recipe')!=RECIPE or raw.get('phase')!='R3'):
        raise ValueError('Resume requires this exact R3 recipe; phase changes use --init-checkpoint')
    config = GRMMatcherConfig.from_dict(raw['config']); config.use_amp = False
    if config.architecture_version!='v3' or getattr(config,'adaptive_enabled',False) or config.coarse_stride!=8:
        raise ValueError('R3 requires plain V3 F8/F8-HRA, Adaptive disabled')
    if (config.input_height,config.input_width)!=(args.size,args.size):
        raise ValueError('Checkpoint canvas differs from --size')
    if args.resume:
        if Path(raw['output_dir']).resolve()!=args.output_dir.resolve() or not args.output_dir.is_dir():
            raise ValueError('Resume output directory differs or is missing')
    elif not args.audit_only and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new empty output directory; interrupted run uses --resume')
    sources, validation, data = build_data(args,config)
    plan = ReplayPlan(sources,args.seed,args.total_updates,args.accumulation_steps)
    ignored = {'init_checkpoint','resume','output_dir','device','audit_only','evaluate_only',
               'stop_after_updates','num_workers','log_every','save_every'}
    contract = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ignored}
    contract.update(config=config.to_dict(),data_signatures=data['signatures'],validation_records=data['validation_records'])
    if args.resume and not same_training_contract(raw['contract'],contract):
        raise ValueError('Resume contract changed: preserve data/config/schedule')
    summary = dict(recipe=RECIPE,phase='R3',precision='FP32',contract=contract,
        training_pairs={d:len(ds) for d,ds in sources.items()}, planned_visits=plan.statistics(),
        validation_pairs={d:len(ds) for d,ds in validation.items()},backbone='frozen throughout',
        total_steps=args.total_updates,warmup_steps=args.warmup_updates,
        selection='Guarded UAV macro F1@3; separate guarded UAV macro PCK@1')
    print(json.dumps(summary,indent=2),flush=True)
    if args.audit_only: return summary
    device = torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    model = GRMMatcher(config).to(device); model.load_state_dict(raw['model'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise FloatingPointError('Initialization contains nonfinite weights')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    json_write(args.output_dir/'run.json',summary)
    json_write(args.output_dir/'data_audit.json',data['audit'])
    json_write(args.output_dir/'validation_pairs.json',data['validation_records'])
    generator = torch.Generator().manual_seed(args.seed)
    opt = make_optimizer(model,args)
    obj = ReplayObjective(config,anchor_points=args.anchor_points,anchor_jitter=args.anchor_jitter,
        local_freeze_updates=args.local_freeze_updates,geometry_fine_weight=args.geometry_fine_weight,
        teacher_fine_weight=args.teacher_fine_weight)
    step = 0; baseline = validation_metrics = None; validation_step = -1
    best = {}; guard_streak = {}; halted = False; interval = []
    lineage = raw.get('provenance') if args.resume else dict(initial_checkpoint=str(source.resolve()),
        parent_step=raw.get('schedule_step',raw.get('global_step')),
        parent_recipe=raw.get('training_recipe'),parent=raw.get('provenance'))
    if args.resume:
        opt.load_state_dict(raw['optimizer']); restore_rng(raw['rng'],generator)
        step=raw['schedule_step']; baseline=raw['baseline']; best=raw['best_scores']
        validation_metrics=raw['validation_metrics']; validation_step=raw['validation_step']
        guard_streak=raw['guard_streak']; halted=raw['halted']; interval=raw['interval']
    interrupted = []
    def request_stop(signum, frame): interrupted.append(signum)
    signal.signal(signal.SIGTERM,request_stop)
    if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,request_stop)

    def state():
        return dict(training_recipe=RECIPE,phase='R3',stage='agri',config=config.to_dict(),model=model.state_dict(),
            optimizer=opt.state_dict(),schedule_step=step,global_step=step,next_pair_offset=step*args.accumulation_steps,
            output_dir=str(args.output_dir.resolve()),contract=contract,rng=rng_state(generator),
            baseline=baseline,best_scores=best.copy(),validation_metrics=validation_metrics,validation_step=validation_step,
            guard_streak=guard_streak.copy(),halted=halted,interval=interval.copy(),provenance=lineage,
            confidence_head_trained=True,geometry_heads_trained=False,inlier_head_trained=False)

    def save(name): _atomic_torch_save(state(),args.output_dir/name)

    def save_latest():
        latest=args.output_dir/'latest.pt'
        if latest.exists(): latest.replace(args.output_dir/'previous.pt')
        save('latest.pt')

    def validate_and_select():
        nonlocal baseline,validation_metrics,validation_step,halted
        before=rng_state(generator)
        try: validation_metrics=evaluate(model,validation,args,device,step)
        finally: restore_rng(before,generator)
        validation_step=step
        if baseline is None: baseline=validation_metrics
        failures,scores=guarded_scores(validation_metrics,baseline,args)
        for key in ('megadepth/coarse_recall1','uav/pck1','synthetic/pck1'):
            guard_streak[key]=guard_streak.get(key,0)+1 if key in failures else 0
        halted=any(v>=args.guard_patience for v in guard_streak.values())
        improved=[]
        if not failures:
            for key,value in scores.items():
                if key not in best or value>tuple(best[key]): best[key]=value; improved.append(key)
        for key in improved: save('best_'+key+'.pt')
        if step: save(f'validation_{step:07d}.pt')
        json_write(args.output_dir/'selection.json',dict(step=step,best=best,failures=failures,
            guard_streak=guard_streak,halted=halted,actual_visits=plan.statistics(step*args.accumulation_steps)))
        monitor(args,f'[SELECTION] step={step} improved={improved} failures={failures} halted={halted}')

    def flush_interval():
        if not interval: return
        totals=defaultdict(float); domains=defaultdict(lambda:defaultdict(float))
        for record in interval:
            for k,v in record['metrics'].items(): totals[k]+=v
            for d,values in record['by_domain'].items():
                for k,v in values.items(): domains[d][k]+=v
        n=len(interval)*args.accumulation_steps
        seconds=sum(r['seconds'] for r in interval)
        event=dict(phase='R3',step=step,total_steps=args.total_updates,stage='coarse' if step<=args.local_freeze_updates else 'joint',
            interval_start=step-len(interval)+1,updates=len(interval),microbatches=n,
            metrics={k:v/n for k,v in totals.items()},reference_metrics=reference_metrics(totals),
            by_domain={d:dict(pairs=int(v['pairs']),loss=v['loss']/v['pairs'],**reference_metrics(v)) for d,v in domains.items()},
            lr={g['name']:g['lr'] for g in opt.param_groups},backbone_lr=0.,
            gradient_norm_mean=sum(r['gradient_norm'] for r in interval)/len(interval),
            gradient_norm_max=max(r['gradient_norm'] for r in interval),seconds=seconds,
            pairs_per_second=n/max(seconds,1e-9),
            peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None)
        monitor(args,f'[TRAIN] R3 step={step}/{args.total_updates} stage={event["stage"]} '+
            f'interval={event["interval_start"]}-{step} loss={event["metrics"]["loss"]:.4f} '+
            f'coarse={event["metrics"]["coarse"]:.4f} lr={event["lr"]}',event)
        interval.clear()

    if args.evaluate_only: evaluate(model,validation,args,device,0); return
    if halted: raise ValueError('Run stopped by repeated guard failures; inspect selection.json before a new experiment')
    if step>=args.total_updates:
        if validation_step<step:
            validate_and_select(); save_latest()
            return dict(step=step,total_steps=args.total_updates,best=best,halted=halted)
        raise ValueError('Run already complete')
    if not args.resume: validate_and_select(); save_latest()
    elif validation_step<step and (step%args.eval_every==0 or step==args.local_freeze_updates):
        validate_and_select(); save_latest()
    stop_at=min(args.total_updates,step+args.stop_after_updates) if args.stop_after_updates else args.total_updates
    plan.offset=step*args.accumulation_steps
    iterator=iter(loader(plan,args,device,args.seed))
    monitor(args,f'[START] R3 update={step}/{args.total_updates} stop={stop_at}; full 8k LR plan retained')
    while step<stop_at and not halted:
        started=time.monotonic(); opt.zero_grad(set_to_none=True); obj.update=step
        set_learning_rates(opt,args,step+1)
        metrics_sum=defaultdict(float); by_domain=defaultdict(lambda:defaultdict(float)); pairs=[]
        for _ in range(args.accumulation_steps):
            cpu=next(iterator); d=cpu['domain'][0]; pairs.append(dict(domain=d,pair_id=cpu['pair_id'][0]))
            loss,metrics=obj.agri(model,to_device(cpu,device),True)
            if loss is None or not torch.isfinite(loss) or not all(math.isfinite(float(v)) for v in metrics.values()):
                json_write(args.output_dir/'failure.json',dict(completed_step=step,pairs=pairs,metrics=repr(metrics)))
                raise FloatingPointError('Invalid replay supervision/loss; inspect failure.json')
            (loss/args.accumulation_steps).backward()
            for k,v in metrics.items(): metrics_sum[k]+=float(v); by_domain[d][k]+=float(v)
            by_domain[d]['pairs']+=1
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip,error_if_nonfinite=True))
        opt.step(); step+=1
        if device.type=='cuda': torch.cuda.synchronize(device)
        interval.append(dict(metrics=dict(metrics_sum),by_domain={d:dict(v) for d,v in by_domain.items()},
                             gradient_norm=norm,seconds=time.monotonic()-started))
        if step%args.log_every==0 or step==stop_at or step==args.local_freeze_updates: flush_interval()
        if interrupted:
            flush_interval(); save_latest(); monitor(args,'[STOP] signal: saved at completed update boundary'); break
        if step%args.eval_every==0 or step==args.local_freeze_updates or step==args.total_updates:
            # Persist before long validation; resume redoes an unfinished validation.
            save_latest()
            validate_and_select()
            save_latest()
        elif step%args.save_every==0 or step==stop_at or halted: save_latest()
    monitor(args,f'[FINISH] updates={step}/{args.total_updates} halted={halted}; best={best}')
    return dict(step=step,total_steps=args.total_updates,best=best,halted=halted)


if __name__=='__main__': main()
