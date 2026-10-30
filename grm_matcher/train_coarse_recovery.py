"""Independent C -> D/E training; 2000 coarse updates, optional 1000 joint updates."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .coarse_recovery import (PHASES, QUOTA, STRATA, RecoveryObjective, StratifiedMixture,
    gradient_audit, inventory, make_optimizer, scoring_rates, visibility_bucket)
from .config import GRMMatcherConfig
from .matcher import GRMMatcher
from .depth_curriculum import reference_metrics
from .scratch_transfer import lr_multiplier
from .train_depth_curriculum import (build_data, code_signature, data_loader, file_sha256,
    json_write, synthetic_signature, to_device)
from .train_scratch_transfer import monitor, rng_state, restore_rng
from .train_stage25 import _atomic_torch_save, _seed_everything
from agri_eval_common import fit_homography, error_auc, project

RECIPE = 'v3_coarse_recovery_r1'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=tuple(PHASES))
    p.add_argument('--geometry-cache', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint', type=Path)
    source.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, help='Full schedule; defaults D=2000, E=1000')
    p.add_argument('--warmup-updates', type=int)
    p.add_argument('--stop-after-updates', type=int, default=0, help='Interrupt after this many additional updates; schedule unchanged')
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--val-limit', type=int, default=300, help='Same scene-interleaved fixed pairs as old curriculum; 0=all')
    p.add_argument('--anchor-points', type=int, default=128)
    p.add_argument('--anchor-jitter', type=float, default=8.)
    p.add_argument('--real-fine-min-weight', type=float, default=.5)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--gradient-clip', type=float, default=1.)
    p.add_argument('--synthetic-pck-guard', type=float, default=.005)
    p.add_argument('--synthetic-recall-guard', type=float, default=.02)
    p.add_argument('--domain-reach-guard', type=float, default=.02)
    p.add_argument('--joint-coarse-guard', type=float, default=.005)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda')
    p.add_argument('--require-all-strata', action='store_true', help='Fail audit if train/val lacks any real visibility stratum')
    p.add_argument('--audit-only', action='store_true', help='CPU metadata/file audit; no training or output directory')
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args(argv)
    args.steps = PHASES[args.phase]['steps'] if args.steps is None else args.steps
    args.warmup_updates = PHASES[args.phase]['warmup'] if args.warmup_updates is None else args.warmup_updates
    if args.steps <= 0 or not 0 <= args.warmup_updates < args.steps:
        p.error('Require steps>0 and 0<=warmup-updates<steps')
    if min(args.eval_every,args.save_every,args.anchor_points,args.gradient_clip) <= 0:
        p.error('Intervals, anchors and gradient clip must be positive')
    if min(args.stop_after_updates,args.log_every,args.val_limit,args.num_workers,args.weight_decay) < 0:
        p.error('Limits/decay must be nonnegative')
    if not 0 < args.min_lr_ratio <= 1 or not 0 <= args.anchor_jitter <= 8 or not 0 <= args.real_fine_min_weight <= 1:
        p.error('Invalid LR floor, jitter or real Fine quality threshold')
    if any(not math.isfinite(v) or v < 0 for v in (args.synthetic_pck_guard,args.synthetic_recall_guard,
            args.domain_reach_guard,args.joint_coarse_guard,args.weight_decay,args.gradient_clip)):
        p.error('Guards/decay/clip must be finite and nonnegative')
    if args.evaluate_only and args.resume:
        p.error('Use --init-checkpoint for evaluate-only')
    return args


def rates(counts):
    return dict(reference_metrics(counts), **scoring_rates(counts))


def macro(rows):
    return {k: sum(r[k] for r in rows)/len(rows) for k in rows[0] if k != 'eligible'}


def coarse_score(validation):
    return sum(validation[d]['visibility_macro']['raw_reachable32'] for d in ('megadepth','uav'))/2


def selection_score(validation, phase):
    return coarse_score(validation) if phase == 'D' else sum(
        validation[d]['visibility_macro']['reference_f1_3'] for d in ('megadepth','uav'))/2


def guard_failures(validation, baseline, args):
    failures = []
    syn, old = validation['synthetic']['macro'], baseline['synthetic']['macro']
    for key, gap in [('pck3',args.synthetic_pck_guard), ('reference_recall3',args.synthetic_recall_guard)]:
        if syn[key] < old[key]-gap:
            failures.append(f'synthetic/{key} dropped more than {gap}')
    for d in ('megadepth','uav'):
        if validation[d]['visibility_macro']['raw_reachable32'] < baseline[d]['visibility_macro']['raw_reachable32']-args.domain_reach_guard:
            failures.append(d+'/balanced_raw_reachable32 dropped beyond domain guard')
    if args.phase == 'E' and coarse_score(validation) < coarse_score(baseline)-args.joint_coarse_guard:
        failures.append('joint balanced raw reachability dropped beyond coarse guard')
    return failures


def validate(model, datasets, args, device, update):
    objective = RecoveryObjective(model.config,args.phase,anchor_points=args.anchor_points,anchor_jitter=0,
        real_fine_min_weight=args.real_fine_min_weight,capture_prediction=True)
    result = {}; started = time.monotonic()
    path = args.output_dir/f'validation_step_{update:06d}_pairs.jsonl'
    with path.open('w',encoding='utf-8') as stream, torch.inference_mode():
        for domain, subset in datasets.items():
            totals = defaultdict(float); scenes = defaultdict(lambda:defaultdict(float))
            strata = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
            pairs = defaultdict(int); errors = defaultdict(list)
            for index, cpu in enumerate(data_loader(subset,args.num_workers,device,args.seed)):
                loss, metrics = objective.agri(model,to_device(cpu,device),False)
                if loss is None or not torch.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError(f'Invalid validation loss/metrics: {cpu.get("pair_id")}')
                raw_row = subset.dataset.records[subset.indices[index]]
                bucket = visibility_bucket(dict(raw_row,domain=domain))
                scene = cpu['scene_id'][0]; pairs[bucket] += 1
                row = dict(pair_id=cpu['pair_id'][0],domain=domain,scene=scene,visibility_bucket=bucket,
                    counts=metrics,rates=rates(metrics))
                if domain == 'synthetic':
                    pa,pb = objective.last_prediction
                    seed = int(hashlib.sha256(row['pair_id'].encode()).hexdigest()[:7],16)
                    h, inliers, status = fit_homography(pa,pb,seed,3.)
                    corner_error = math.inf
                    if h is not None:
                        size = cpu['image_a'].shape[-1]
                        corners = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]],float)
                        e = np.linalg.norm(project(corners,h)-project(corners,cpu['H_ab'][0].numpy()),axis=1)
                        if np.isfinite(e).all(): corner_error = float(e.mean())
                    errors[scene].append(corner_error)
                    row['homography'] = dict(status=status,inliers=inliers,corner_error_input_px=corner_error if math.isfinite(corner_error) else None)
                if metrics['scoring_reference_count'] <= 0:
                    raise ValueError(f'No reference coarse cells in fixed validation pair {row["pair_id"]}')
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                for k,v in metrics.items():
                    totals[k] += v; scenes[scene][k] += v; strata[bucket][scene][k] += v
                if index == 0 or (index+1)%100 == 0:
                    monitor(args,f'[VAL] {domain} {index+1}/{len(subset)} update={update}')
            by_scene = {k:rates(v) for k,v in scenes.items()}
            by_visibility = {b:dict(pairs=pairs[b],scenes=len(ss),
                macro=macro([rates(v) for v in ss.values()]),
                micro=rates({k:sum(v[k] for v in ss.values()) for k in totals})) for b,ss in strata.items()}
            result[domain] = dict(pairs=len(subset),micro=rates(totals),macro=macro(list(by_scene.values())),
                by_scene=by_scene,by_visibility=by_visibility,
                visibility_macro=macro([r['macro'] for r in by_visibility.values()]))
            if errors:
                for threshold in (5,10):
                    result[domain]['macro'][f'h_auc{threshold}'] = sum(error_auc(e,threshold) for e in errors.values())/len(errors)
            brief = {b:{k:round(row['macro'][k],4) for k in
                ('raw_reachable32','dual_reachable32','weighted_reachable32','reference_recall3','pck3')}
                for b,row in by_visibility.items()}
            monitor(args,f'[VAL SUMMARY] update={update} {domain} strata={brief}')
    result['elapsed_seconds'] = time.monotonic()-started
    json_write(args.output_dir/f'validation_step_{update:06d}.json',result)
    return result


def validate_parent(raw,args):
    if args.resume:
        if raw.get('training_recipe') != RECIPE or raw.get('phase') != args.phase:
            raise ValueError('Resume is only for this exact D/E run; use --init-checkpoint when changing phases')
    elif not args.evaluate_only:
        if args.phase == 'D' and raw.get('phase') not in ('C','D'):
            raise ValueError('D must initialize from curriculum C, or restart a D checkpoint')
        if args.phase == 'E':
            if raw.get('training_recipe') != RECIPE or raw.get('phase') not in ('D','E'):
                raise ValueError('E requires a checkpoint from this recovery D phase; use --init-checkpoint, not --resume')
            if raw.get('phase') == 'D' and not raw.get('selected_improvement',False):
                raise ValueError('D has no validated guarded improvement. Inspect D metrics before starting E; use best_phase.pt')


def main(argv=None):
    args = parse_args(argv); _seed_everything(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    raw = torch.load(args.resume or args.init_checkpoint,map_location='cpu',weights_only=False)
    validate_parent(raw,args)
    config = GRMMatcherConfig.from_dict(raw['config']); config.use_amp = False
    if config.architecture_version != 'v3' or config.v3_revision < 1 or not config.v3_confidence_head:
        raise ValueError('Requires existing V3 revision>=1 with confidence head; no architecture upgrade is performed')
    sources,validation,info = build_data(args,config)
    info['sampling_inventory'] = inventory(sources)
    val_sources = {d:SubsetRecords(ds) for d,ds in validation.items()}
    info['validation_inventory'] = inventory(val_sources)
    for split,inv in [('train',info['sampling_inventory']),('val',info['validation_inventory'])]:
        for d,row in inv.items():
            if row['missing_strata']:
                message = f'{split}/{d} missing visibility strata {row["missing_strata"]}'
                if args.require_all_strata: raise ValueError(message)
                print('[STRATA WARNING]',message,flush=True)
    runtime = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in
        ('init_checkpoint','resume','output_dir','stop_after_updates','num_workers','device','log_every','audit_only','evaluate_only')}
    contract = dict(recipe=RECIPE,runtime=runtime,phase_definition=PHASES[args.phase],quota=QUOTA,
        sampler='A->B support bins <.3/.3-.6/>=.6; cycle available bins; uniform scenes then pairs',
        config=config.to_dict(),code_sha256=code_signature(),
        geometry_signatures={d:ds.signature for d,ds in sources.items() if d!='synthetic'},
        validation_signatures={d:ds.dataset.signature for d,ds in validation.items() if d!='synthetic'},
        synthetic_signature=synthetic_signature(args.data_dir,dict(train=sources['synthetic'],val=validation['synthetic'].dataset)),
        validation_records_sha256=hashlib.sha256(json.dumps(info['validation_records'],sort_keys=True).encode()).hexdigest())
    provenance = raw.get('provenance') if args.resume else dict(initial_checkpoint=str(args.init_checkpoint),
        initial_checkpoint_sha256=file_sha256(args.init_checkpoint),parent_recipe=raw.get('training_recipe'),
        parent_phase=raw.get('phase'),parent_phase_step=raw.get('phase_step'),parent=raw.get('provenance'))
    summary = dict(recipe=RECIPE,contract=contract,provenance=provenance,precision='FP32',
        torch_version=str(torch.__version__),accumulation_steps=10,sampling_inventory=info['sampling_inventory'],
        validation_inventory=info['validation_inventory'],
        checkpoint_selection='D: domain/stratum/scene-balanced raw reach32; E: balanced final reference F1@3; both guarded',
        note='Depth reference is uncertain. No test data or test predictions used in sampler.')
    print(json.dumps(dict(phase=args.phase,steps=args.steps,warmup=args.warmup_updates,quota=QUOTA,
        learning_rates=PHASES[args.phase]['lr'],train_inventory=info['sampling_inventory'],
        validation_inventory=info['validation_inventory']),ensure_ascii=False,indent=2),flush=True)
    if args.audit_only: return
    if args.resume:
        if raw['contract'] != contract:
            raise ValueError('Resume contract changed (data/code/schedule/config); use a new output and --init-checkpoint')
        if Path(raw['output_dir']).resolve() != args.output_dir.resolve() or not args.output_dir.is_dir():
            raise ValueError('Resume must target its existing original output directory')
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('New run requires empty output; resume only interrupted runs of the SAME phase')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check GPU allocation and driver/PyTorch compatibility')
    model = GRMMatcher(config).to(device); model.load_state_dict(raw['model'],strict=True)
    del raw['model']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    json_write(args.output_dir/'data_audit.json',info)
    if args.evaluate_only:
        validate(model,validation,args,device,0); return
    optimizer = make_optimizer(model,args.phase,args.weight_decay)
    objective = RecoveryObjective(config,args.phase,anchor_points=args.anchor_points,anchor_jitter=args.anchor_jitter,
        real_fine_min_weight=args.real_fine_min_weight)
    generator = torch.Generator().manual_seed(args.seed)
    step = 0; baseline = last_val = None; validation_step = 0; best_score = None
    best_step = 0; checked = False; selected_improvement = False; failures = []
    if args.resume:
        optimizer.load_state_dict(raw['optimizer']); restore_rng(raw['rng'],generator)
        step = raw['phase_step']; baseline = raw['baseline']; last_val = raw['validation_metrics']
        validation_step = raw['validation_step']; best_score = raw['best_score']; best_step = raw['best_step']
        checked = raw.get('gradient_checked',False); selected_improvement = raw.get('selected_improvement',False)
        failures = raw.get('guard_failures',[])
    if step >= args.steps: raise ValueError('This phase is complete; start E with --init-checkpoint if D improved')
    json_write(args.output_dir/'run.json',summary)
    stopped = []
    def request_stop(signum,frame): stopped.append(signum)
    signal.signal(signal.SIGTERM,request_stop)
    if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,request_stop)
    def save(name):
        _atomic_torch_save(dict(training_recipe=RECIPE,stage='agri',phase=args.phase,phase_step=step,
            global_step=step,seen_pairs=step*10,config=config.to_dict(),model=model.state_dict(),
            optimizer=optimizer.state_dict(),output_dir=str(args.output_dir.resolve()),contract=contract,
            rng=rng_state(generator),provenance=provenance,baseline=baseline,validation_metrics=last_val,
            validation_step=validation_step,best_score=best_score,best_step=best_step,gradient_checked=checked,
            selected_improvement=selected_improvement,guard_failures=failures,
            confidence_head_trained=bool(raw.get('confidence_head_trained')) or (args.phase=='E' and step>0),
            geometry_heads_trained=False,inlier_head_trained=False),args.output_dir/name)
    if not args.resume:
        baseline = last_val = validate(model,validation,args,device,0)
        best_score = selection_score(baseline,args.phase)
        save('best_phase.pt'); save('latest.pt')
    end = min(args.steps,step+args.stop_after_updates) if args.stop_after_updates else args.steps
    monitor(args,f'[START] {args.phase} next={step+1} stop={end}/{args.steps}; 10 pairs/update; baseline_score={selection_score(baseline,args.phase):.5f}')
    totals = defaultdict(float); interval_pairs = 0; sampled = defaultdict(int); started = time.monotonic()
    while step < end and not stopped:
        boundary = min(end,(step//args.eval_every+1)*args.eval_every)
        loader = iter(data_loader(StratifiedMixture(sources,args.phase,args.seed,step,boundary-step),args.num_workers,device,args.seed+step))
        while step < boundary and not stopped:
            tick = time.monotonic(); objective.update = step; optimizer.zero_grad(set_to_none=True)
            current_pairs = []
            try:
                for _ in range(10):
                    cpu = next(loader); d = cpu['domain'][0]; bucket = cpu['visibility_bucket'][0]
                    current_pairs.append(dict(domain=d,pair_id=cpu['pair_id'][0],bucket=bucket))
                    loss,metrics = objective.agri(model,to_device(cpu,device),True)
                    if loss is None or not torch.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError('Missing/nonfinite loss or diagnostics')
                    (loss/10).backward(); interval_pairs += 1; sampled[d+'/'+bucket] += 1
                    for key,value in metrics.items():
                        totals[key] += value; totals[d+'/'+key] += value
                    totals[d+'/pairs'] += 1
                if not checked:
                    json_write(args.output_dir/'gradient_audit.json',gradient_audit(model,args.phase)); checked = True
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip,error_if_nonfinite=True))
                factor = lr_multiplier(step,args.steps,args.warmup_updates,args.min_lr_ratio)
                for g in optimizer.param_groups: g['lr'] = g['peak_lr']*factor
                optimizer.step()
            except Exception as error:
                _atomic_torch_save(dict(phase=args.phase,completed_updates=step,pairs=current_pairs,
                    error=repr(error),cpu_batch=locals().get('cpu')),args.output_dir/'failed_batch.pt')
                monitor(args,f'[FAILED] update={step+1} pairs={current_pairs} error={error!r}'); raise
            step += 1; selected_improvement = False
            if step==1 or (args.log_every and step%args.log_every==0) or step==end:
                domains = {}
                for d in sources:
                    counts = {k.split('/',1)[1]:v for k,v in totals.items() if k.startswith(d+'/')}
                    domains[d] = dict(rates(counts),loss=counts.get('loss',0)/max(counts.get('pairs',0),1))
                event = dict(phase=args.phase,optimizer_step=step,total_steps=args.steps,seen_pairs=step*10,
                    losses={k:totals[k]/max(interval_pairs,1) for k in ('loss','coarse','anchor','refinement','matchability','overlap','confidence')},
                    by_domain=domains,sampled_domain_strata=dict(sampled),gradient_norm=norm,
                    lr={g['name']:g['lr'] for g in optimizer.param_groups},seconds_per_update=time.monotonic()-tick,
                    elapsed_seconds=time.monotonic()-started,
                    gpu_peak_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None)
                brief={d:dict(loss=round(v['loss'],4),raw32=round(v['raw_reachable32'],4),weighted32=round(v['weighted_reachable32'],4)) for d,v in domains.items()}
                monitor(args,f'[TRAIN] {args.phase} {step}/{args.steps} norm={norm:.3f} lr={event["lr"]} domains={brief}',event)
                totals.clear(); interval_pairs=0; sampled.clear()
            if step%args.save_every==0: save('latest.pt')
        del loader
        if stopped: break
        # Validate partial stops too, so a stopped checkpoint never pretends to have current validation.
        last_val = validate(model,validation,args,device,step); validation_step = step
        failures = guard_failures(last_val,baseline,args); score = selection_score(last_val,args.phase)
        improved = not failures and score > best_score
        if improved:
            best_score=score; best_step=step; selected_improvement=True; save('best_phase.pt')
        record = dict(step=step,phase=args.phase,score=score,coarse_score=coarse_score(last_val),
            guard_passed=not failures,guard_failures=failures,updated_best=improved,best_step=best_step,
            best_score=best_score,validation=last_val)
        with (args.output_dir/'metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        save(f'step_{step:06d}.pt'); save('latest.pt')
        monitor(args,f'[VALIDATION] step={step} score={score:.5f} best={best_score:.5f}@{best_step} updated={improved} guards={failures}',
            dict(event='validation',**{k:v for k,v in record.items() if k!='validation'}))
    save('latest.pt')
    monitor(args,f'[STOPPED] {args.phase} {step}/{args.steps}; best_step={best_step}; validation_step={validation_step}; '
        'best_phase.pt remains initialization if best_step=0. No automatic phase transition.')


class SubsetRecords:
    def __init__(self,subset): self.records=[subset.dataset.records[i] for i in subset.indices]
    def __len__(self): return len(self.records)


if __name__=='__main__': main()
