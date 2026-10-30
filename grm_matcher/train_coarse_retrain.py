"""Independent UAV-focused coarse retraining from an existing V3 checkpoint."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch

from .coarse_recovery import RecoveryObjective, gradient_audit, scoring_rates
from .coarse_retrain import (RECIPE, QUOTA, SUPPORT_CYCLE, RetrainMixture, make_optimizer,
    selection_metrics, guard_failures, require_strata, parameter_snapshot, relative_updates)
from .config import GRMMatcherConfig
from .depth_curriculum import reference_metrics
from .matcher import GRMMatcher
from .scratch_transfer import lr_multiplier
from .train_coarse_recovery import validate
from .train_depth_curriculum import (build_data, code_signature, data_loader, file_sha256,
    json_write, synthetic_signature, to_device)
from .train_scratch_transfer import monitor, rng_state, restore_rng
from .train_stage25 import _atomic_torch_save, _seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry-cache', 'data-dir', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint', type=Path)
    source.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--warmup-updates', type=int, default=500)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--coarse-lr', type=float, default=1e-4)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--gradient-clip', type=float, default=5.)
    p.add_argument('--target-guard', type=float, default=.002)
    p.add_argument('--other-guard', type=float, default=.02)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--val-limit', type=int, default=300)
    p.add_argument('--stop-after-updates', type=int, default=0)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args(argv)
    if args.steps <= 0 or not 0 <= args.warmup_updates < args.steps:
        p.error('Require steps>0 and 0<=warmup-updates<steps')
    if min(args.eval_every, args.save_every) <= 0 or min(args.log_every, args.val_limit,
            args.stop_after_updates, args.num_workers) < 0:
        p.error('Invalid interval or limit')
    for name in ('backbone_lr', 'coarse_lr', 'gradient_clip', 'min_lr_ratio'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(name+' must be finite and positive')
    for name in ('weight_decay', 'target_guard', 'other_guard'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(name+' must be finite and nonnegative')
    if args.min_lr_ratio > 1 or args.target_guard > 1 or args.other_guard > 1:
        p.error('Ratios and rate guards must not exceed 1')
    if args.resume and (args.evaluate_only or args.audit_only):
        p.error('Use --init-checkpoint for audit/evaluate-only')
    # Reuse the existing coarse-only objective and full-model validation exactly.
    args.phase = 'D'
    args.anchor_points = 128
    args.anchor_jitter = 0.
    args.real_fine_min_weight = .5
    return args


def rates(counts):
    return dict(reference_metrics(counts), **scoring_rates(counts))


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    initial = args.resume or args.init_checkpoint
    raw = torch.load(initial, map_location='cpu', weights_only=False)
    if args.resume and raw.get('training_recipe') != RECIPE:
        raise ValueError('Resume requires this retraining recipe; initialize a new run instead')
    config = GRMMatcherConfig.from_dict(raw['config'])
    config.use_amp = False
    if config.architecture_version != 'v3' or config.v3_revision < 1 or not config.v3_confidence_head:
        raise ValueError('Requires existing V3 revision>=1 with confidence head')
    sources, validation, info = build_data(args, config)
    info['support_inventories'] = require_strata(sources, validation)
    runtime = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in
               ('init_checkpoint', 'resume', 'output_dir', 'stop_after_updates', 'num_workers',
                'device', 'log_every', 'audit_only', 'evaluate_only')}
    contract = dict(recipe=RECIPE, runtime=runtime, config=config.to_dict(), quota=QUOTA,
        support_cycle=SUPPORT_CYCLE, code_sha256=code_signature(),
        geometry_signatures={d: ds.signature for d, ds in sources.items() if d != 'synthetic'},
        validation_signatures={d: ds.dataset.signature for d, ds in validation.items() if d != 'synthetic'},
        synthetic_signature=synthetic_signature(args.data_dir, dict(train=sources['synthetic'],
            val=validation['synthetic'].dataset)),
        validation_records_sha256=hashlib.sha256(json.dumps(info['validation_records'], sort_keys=True).encode()).hexdigest())
    provenance = raw['provenance'] if args.resume else dict(initial_checkpoint=str(initial),
        initial_checkpoint_sha256=file_sha256(initial), parent_recipe=raw.get('training_recipe'),
        parent_phase=raw.get('phase'), parent_phase_step=raw.get('phase_step'), parent=raw.get('provenance'))
    summary = dict(recipe=RECIPE, contract=contract, provenance=provenance, precision='FP32',
        torch_version=str(torch.__version__), accumulation_steps=10,
        checkpoint_selection='UAV low-support scene-macro mean(raw nearest-cell recall, raw reach32), guarded',
        objective='Unchanged bidirectional coarse CE + positive real matchability / synthetic overlap',
        note='Warm-start stage retraining, not random initialization. No online hard mining or new geometry augmentation.')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        return
    if args.resume:
        # Tuples serialize as lists in JSON but torch checkpoints retain tuples.
        if raw['contract'] != contract:
            raise ValueError('Resume contract changed: use a new output directory and --init-checkpoint')
        if Path(raw['output_dir']).resolve() != args.output_dir.resolve() or not args.output_dir.is_dir():
            raise ValueError('Resume must target the original output directory')
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('New training requires an empty output directory')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model = GRMMatcher(config).to(device)
    model.load_state_dict(raw.pop('model'), strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_write(args.output_dir/'data_audit.json', info)
    json_write(args.output_dir/'run.json', summary)
    if args.evaluate_only:
        validate(model, validation, args, device, 0)
        return
    optimizer = make_optimizer(model, args)
    objective = RecoveryObjective(config, 'D', anchor_points=args.anchor_points, anchor_jitter=0.)
    generator = torch.Generator().manual_seed(args.seed)
    step = best_step = validation_step = 0
    baseline = last_val = None
    best_score = None
    checked = selected = False
    failures = []
    if args.resume:
        optimizer.load_state_dict(raw['optimizer'])
        restore_rng(raw['rng'], generator)
        step, best_step = raw['phase_step'], raw['best_step']
        baseline, last_val = raw['baseline'], raw['validation_metrics']
        best_score, validation_step = raw['best_score'], raw['validation_step']
        checked, selected = raw['gradient_checked'], raw['selected_improvement']
        failures = raw['guard_failures']
    if step >= args.steps:
        raise ValueError('This training schedule is complete')
    stopped = []
    def stop(signum, frame):
        stopped.append(signum)
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, stop)

    def save(name):
        _atomic_torch_save(dict(training_recipe=RECIPE, stage='agri', phase='R', phase_step=step,
            global_step=step, seen_pairs=step*10, config=config.to_dict(), model=model.state_dict(),
            optimizer=optimizer.state_dict(), output_dir=str(args.output_dir.resolve()), contract=contract,
            rng=rng_state(generator), provenance=provenance, baseline=baseline, validation_metrics=last_val,
            validation_step=validation_step, best_score=best_score, best_step=best_step,
            gradient_checked=checked, selected_improvement=selected, guard_failures=failures,
            confidence_head_trained=bool(raw.get('confidence_head_trained')),
            requires_local_adaptation=step > 0, geometry_heads_trained=False, inlier_head_trained=False),
            args.output_dir/name)

    if not args.resume:
        baseline = last_val = validate(model, validation, args, device, 0)
        best_score = selection_metrics(baseline)['score']
        save('best_phase.pt')
        save('latest.pt')
    end = min(args.steps, step+args.stop_after_updates) if args.stop_after_updates else args.steps
    totals = defaultdict(float)
    sampled = defaultdict(int)
    interval_pairs = 0
    started = time.monotonic()
    monitor(args, f'[START] coarse retrain {step}/{args.steps}; quota={QUOTA}; best={best_score:.6f}')
    while step < end and not stopped:
        boundary = min(end, (step//args.eval_every+1)*args.eval_every)
        loader = iter(data_loader(RetrainMixture(sources, args.seed, step, boundary-step),
                                 args.num_workers, device, args.seed+step))
        while step < boundary and not stopped:
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            log_step = step == 0 or (args.log_every and (step+1) % args.log_every == 0) or step+1 == end
            before = parameter_snapshot(model) if log_step else None
            current_pairs = []
            try:
                for _ in range(10):
                    cpu = next(loader)
                    domain, bucket = cpu['domain'][0], cpu['visibility_bucket'][0]
                    current_pairs.append(dict(domain=domain, bucket=bucket, pair_id=cpu['pair_id'][0]))
                    loss, metrics = objective.agri(model, to_device(cpu, device), True)
                    if loss is None or not torch.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError('Missing/nonfinite training loss or metrics')
                    (loss/10).backward()
                    interval_pairs += 1
                    sampled[domain+'/'+bucket] += 1
                    for k, v in metrics.items():
                        totals[k] += v
                        totals[domain+'/'+k] += v
                    totals[domain+'/pairs'] += 1
                if not checked:
                    json_write(args.output_dir/'gradient_audit.json', gradient_audit(model, 'D'))
                    checked = True
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip,
                                                           error_if_nonfinite=True))
                factor = lr_multiplier(step, args.steps, args.warmup_updates, args.min_lr_ratio)
                for group in optimizer.param_groups:
                    group['lr'] = group['peak_lr']*factor
                optimizer.step()
            except Exception as error:
                _atomic_torch_save(dict(completed_updates=step, pairs=current_pairs, error=repr(error),
                                       cpu_batch=locals().get('cpu')), args.output_dir/'failed_batch.pt')
                raise
            step += 1
            if log_step:
                domains = {}
                for d in sources:
                    counts = {k.split('/', 1)[1]: v for k, v in totals.items() if k.startswith(d+'/')}
                    domains[d] = dict(rates(counts), loss=counts.get('loss', 0)/max(counts.get('pairs', 0), 1))
                event = dict(optimizer_step=step, seen_pairs=step*10, coarse_only_training=True,
                    loss=totals['loss']/max(interval_pairs, 1), by_domain=domains,
                    sampled_domain_strata=dict(sampled), gradient_norm_before_clip=norm,
                    clip_scale=min(1., args.gradient_clip/max(norm, 1e-30)),
                    relative_parameter_update=relative_updates(model, before),
                    lr={g['name']: g['lr'] for g in optimizer.param_groups},
                    seconds_per_update=time.monotonic()-tick, elapsed_seconds=time.monotonic()-started)
                monitor(args, f'[TRAIN] R {step}/{args.steps} loss={event["loss"]:.4f} '
                        f'grad={norm:.3f} update={event["relative_parameter_update"]}', event)
                totals.clear()
                sampled.clear()
                interval_pairs = 0
            del before
            if step % args.save_every == 0:
                save('latest.pt')
        del loader
        if stopped:
            break
        last_val = validate(model, validation, args, device, step)
        validation_step = step
        scores = selection_metrics(last_val)
        failures = guard_failures(last_val, baseline, args)
        improved = not failures and scores['score'] > best_score
        if improved:
            best_score, best_step, selected = scores['score'], step, True
            save('best_phase.pt')
        record = dict(step=step, scores=scores, guard_failures=failures, updated_best=improved,
                      best_step=best_step, best_score=best_score)
        with (args.output_dir/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        save(f'step_{step:06d}.pt')
        save('latest.pt')
        monitor(args, f'[VALIDATION] R {step} scores={scores} best_step={best_step} guards={failures}',
                dict(event='validation', **record))
    save('latest.pt')
    monitor(args, f'[STOPPED] R {step}/{args.steps}; best_step={best_step}; validation_step={validation_step}; '
            'best_step=0 means initialization retained; no automatic local adaptation')


if __name__ == '__main__':
    main()
