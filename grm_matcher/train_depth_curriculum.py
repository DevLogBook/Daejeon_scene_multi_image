"""V3 A/B/C depth curriculum. Separate phase runs, no silent auto-transition."""
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
import numpy as np
from torch.utils.data import DataLoader, Subset

from .config import GRMMatcherConfig
from .matcher import GRMMatcher
from .scratch_transfer import ResizedHomographyDataset, check_homography_splits, lr_multiplier
from .train_stage25 import _atomic_torch_save, _seed_everything
from .train_scratch_transfer import rng_state, restore_rng, monitor, dataset_signature
from .depth_curriculum import (PHASES, PREFIXES, GeometryData, TaggedSynthetic, StepMixture,
    CurriculumObjective, phase_mode, parameter_group, make_optimizer, group_indices,
    interleaved_indices, reference_metrics)
from tools.audit_grm_curriculum_data import audit
from agri_eval_common import fit_homography, error_auc, project

RECIPE = 'v3_depth_curriculum_r2'


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def data_loader(data, workers, device, seed):
    # DataLoader iterator creation must not consume the augmentation RNG.
    return DataLoader(data, batch_size=1, shuffle=False, num_workers=workers,
        pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(seed))


def validate(model, datasets, args, device, update, prefix='validation'):
    """Fixed pairs; scene/date macro is primary, pooled micro is also retained."""
    objective = CurriculumObjective(model.config, 'C', anchor_points=args.anchor_points, anchor_jitter=0,
        real_fine_min_weight=args.real_fine_min_weight, disable_real_fine=args.disable_real_fine,
        capture_prediction=True)
    result = {}
    started = time.monotonic()
    per_pair_path = args.output_dir/f'{prefix}_step_{update:06d}_pairs.jsonl'
    with per_pair_path.open('w', encoding='utf-8') as stream, torch.no_grad():
        for domain, dataset in datasets.items():
            scenes = defaultdict(lambda: defaultdict(float)); totals = defaultdict(float)
            excluded = 0; geometry_errors = defaultdict(list); reference_errors = []
            loader = data_loader(dataset, args.num_workers, device, args.seed)
            for index, cpu in enumerate(loader):
                loss, metrics = objective.agri(model, to_device(cpu, device), False)
                if loss is None or not torch.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
                    raise ValueError(f'Invalid validation supervision/loss: {cpu.get("pair_id")}')
                derived = reference_metrics(metrics)
                scene = cpu['scene_id'][0]
                row = dict(domain=domain, scene=scene, pair_id=cpu['pair_id'][0], **metrics, **derived)
                errors = objective.last_reference_errors
                if errors is not None and len(errors):
                    reference_errors.extend(errors.tolist())
                    row['prediction_error_input_px'] = dict(zip(('median', 'p90', 'p95'), np.quantile(errors, [.5, .9, .95]).tolist()))
                else:
                    row['prediction_error_input_px'] = None
                if domain == 'synthetic':
                    pa, pb = objective.last_prediction
                    pair_seed = int(hashlib.sha256(row['pair_id'].encode()).hexdigest()[:7], 16)
                    estimate, inliers, status = fit_homography(pa, pb, pair_seed, 3.)
                    corner_error = math.inf
                    if estimate is not None:
                        size = cpu['image_a'].shape[-1]
                        corners = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], dtype=float)
                        gt_h = cpu['H_ab'][0].numpy()
                        errors = np.linalg.norm(project(corners, estimate)-project(corners, gt_h), axis=1)
                        if np.isfinite(errors).all():
                            corner_error = float(errors.mean())
                    geometry_errors[scene].append(corner_error)
                    row['geometry'] = dict(status=status, inliers=inliers,
                        corner_mean_input_px=corner_error if math.isfinite(corner_error) else None)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                if not derived['eligible']:
                    excluded += 1
                    continue
                for key, value in metrics.items():
                    totals[key] += value
                    scenes[scene][key] += value
                scenes[scene]['pairs'] += 1
                if index == 0 or (index+1) % 100 == 0:
                    monitor(args, f'[VAL] {domain} {index+1}/{len(dataset)} step={update}')
            if not scenes:
                raise ValueError(f'No scorable validation pairs in {domain}')
            by_scene = {name: dict(pairs=int(counts['pairs']), **reference_metrics(counts)) for name, counts in scenes.items()}
            macro_keys = [k for k, v in next(iter(by_scene.values())).items() if k not in ('eligible', 'pairs')]
            macro = {key: sum(row[key] for row in by_scene.values())/len(by_scene) for key in macro_keys}
            if geometry_errors:
                for name, errors in geometry_errors.items():
                    by_scene[name].update({f'h_auc{threshold}': error_auc(errors, threshold) for threshold in (5, 10)})
                macro.update({f'h_auc{threshold}': sum(r[f'h_auc{threshold}'] for r in by_scene.values())/len(by_scene)
                              for threshold in (5, 10)})
            result[domain] = dict(pairs=len(dataset), excluded_no_reference=excluded, totals=dict(totals),
                                  micro=reference_metrics(totals), macro=macro, by_scene=by_scene)
            result[domain]['prediction_error_input_px'] = (dict(zip(('median', 'p90', 'p95'),
                np.quantile(reference_errors, [.5, .9, .95]).tolist())) if reference_errors else None)
            if geometry_errors:
                all_errors = [e for errors in geometry_errors.values() for e in errors]
                result[domain]['micro'].update({f'h_auc{threshold}': error_auc(all_errors, threshold) for threshold in (5, 10)})
    result['elapsed_seconds'] = time.monotonic()-started
    json_write(args.output_dir/f'{prefix}_step_{update:06d}.json', result)
    return result


def selection_scores(validation, phase):
    md, uav, synthetic = [validation[d]['macro'] for d in ('megadepth', 'uav', 'synthetic')]
    return dict(phase=(md['coarse_recall1']+uav['coarse_recall1'])/2 if phase == 'A' else uav['reference_f1_3'],
                md=md['reference_f1_3'], agri=uav['reference_f1_3'],
                synthetic=synthetic['h_auc5'])


def gradient_audit(model, phase):
    report = {name: dict(parameters=0, with_gradient=0, squared_norm=0.) for name in PREFIXES}
    for name, param in model.named_parameters():
        row = report[parameter_group(name)]
        row['parameters'] += param.numel()
        if param.grad is not None:
            if parameter_group(name) not in PHASES[phase]['lr']:
                raise RuntimeError(f'Frozen parameter received gradient: {name}')
            if not torch.isfinite(param.grad).all():
                raise FloatingPointError(f'Nonfinite gradient: {name}')
            row['with_gradient'] += param.numel()
            row['squared_norm'] += float(param.grad.float().square().sum())
    missing = [name for name in PHASES[phase]['lr'] if report[name]['parameters'] and report[name]['squared_norm'] == 0]
    # Confidence needs real outputs; a collapse should be visible rather than a false pass.
    if missing:
        raise RuntimeError(f'Active parameter groups have zero gradient: {missing}; check outputs/known anchors')
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=tuple(PHASES), required=True)
    p.add_argument('--geometry-cache', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--init-checkpoint', type=Path)
    source.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, help='Full phase schedule, not the pilot length')
    p.add_argument('--stop-after-updates', type=int, default=0, help='Additional updates then checkpoint; 0 completes phase')
    p.add_argument('--warmup-updates', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--val-limit', type=int, default=300, help='Fixed scene-interleaved pairs per domain; 0 = full validation')
    p.add_argument('--anchor-points', type=int, default=128)
    p.add_argument('--anchor-jitter', type=float, default=8., help='Input pixels; training GT anchors only')
    p.add_argument('--real-fine-min-weight', type=float, default=.5, help='Conservative reference weight mask, not proof of GT accuracy')
    p.add_argument('--disable-real-fine', action='store_true')
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--gradient-clip', type=float, default=1.)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda')
    p.add_argument('--audit-only', action='store_true', help='Read data/contracts only; no GPU model or output directory needed')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--compare-checkpoints', type=Path, nargs='+', help='Extra candidates evaluated with exactly the same validation list; implies evaluate-only')
    args = p.parse_args(argv)
    args.steps = PHASES[args.phase]['steps'] if args.steps is None else args.steps
    if args.steps <= 0 or not 0 <= args.warmup_updates < args.steps:
        p.error('Require steps>0 and 0<=warmup-updates<steps')
    if min(args.stop_after_updates, args.num_workers, args.val_limit, args.log_every, args.weight_decay) < 0:
        p.error('Limits and decay must be nonnegative')
    if min(args.eval_every, args.save_every, args.anchor_points, args.gradient_clip) <= 0:
        p.error('Intervals/anchor count/clip must be positive')
    if not 0 < args.min_lr_ratio <= 1 or not 0 <= args.real_fine_min_weight <= 1 or not 0 <= args.anchor_jitter <= 8:
        p.error('Invalid LR ratio, fine weight mask or jitter (maximum8 input pixels)')
    if args.resume and (args.evaluate_only or args.compare_checkpoints):
        p.error('Evaluate candidates using --init-checkpoint, not --resume')
    if args.compare_checkpoints:
        args.evaluate_only = True
    return args


def build_data(args, config):
    audit_result = audit(args.geometry_cache/'preparation.json', args.geometry_cache)
    if audit_result['errors']:
        raise ValueError('Geometry audit failed: '+ '; '.join(audit_result['errors'][:10]))
    if config.coarse_stride != 16:
        raise ValueError('v1 geometry cache uses stride16')
    split_info = check_homography_splits(args.data_dir)
    sources = {domain: GeometryData(args.geometry_cache, 'train', domain, config) for domain in ('megadepth', 'uav')}
    validation = {domain: GeometryData(args.geometry_cache, 'val', domain, config) for domain in ('megadepth', 'uav')}
    if any(ds.size != 512 for ds in list(sources.values())+list(validation.values())):
        raise ValueError('Curriculum R2 expects 512 cache; rebuild for any size change')
    sources['synthetic'] = TaggedSynthetic(ResizedHomographyDataset(args.data_dir, 'train', config, 512))
    validation['synthetic'] = TaggedSynthetic(ResizedHomographyDataset(args.data_dir, 'val', config, 512))
    selected = {}; selected_records = {}
    for domain, dataset in validation.items():
        groups = group_indices(dataset.records, domain != 'synthetic')
        indices = interleaved_indices(groups, args.val_limit)
        represented = {next(k for k, values in groups.items() if i in values) for i in indices}
        if len(represented) != len(groups):
            raise ValueError(f'val-limit too small to include every {domain} scene/source ({len(groups)})')
        selected[domain] = Subset(dataset, indices)
        selected_records[domain] = [dataset.records[i] for i in indices]
    return sources, selected, dict(audit=audit_result, synthetic_split=split_info, validation_records=selected_records)


def code_signature():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    # Includes all transitive matcher/loss/data code; changing it requires a new run.
    for path in sorted(root.glob('*.py'))+[root.parent/'tools/audit_grm_curriculum_data.py', root.parent/'agri_eval_common.py']:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def synthetic_signature(root, datasets):
    digest = hashlib.sha256(dataset_signature(root, 'agri').encode())
    for split, tagged in datasets.items():
        data = tagged.data
        seen = set()
        for row in data.records:
            for key, folder in (('image_a', 'input1'), ('image_b', 'input2'), ('h_gt_a_to_b', 'homography'),
                                ('valid_a', 'masks/valid1'), ('valid_b', 'masks/valid2'), ('overlap_a', 'masks/overlap1')):
                path = data._resolve_record_path(row[key], folder)
                if path not in seen:
                    stat = path.stat(); seen.add(path)
                    digest.update(f'{split}:{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    raw = torch.load(args.resume or args.init_checkpoint, map_location='cpu', weights_only=False)
    config = GRMMatcherConfig.from_dict(raw['config']); config.use_amp = False
    if config.architecture_version != 'v3' or not config.v3_confidence_head:
        raise ValueError('Requires a V3 checkpoint with confidence head; architecture is not upgraded')
    if args.resume and (raw.get('training_recipe') != RECIPE or raw.get('phase') != args.phase):
        raise ValueError('Resume is only for the exact same curriculum phase; use --init-checkpoint for phase changes')
    if not args.resume and not args.evaluate_only and args.phase in ('B', 'C'):
        expected = 'A' if args.phase == 'B' else 'B'
        if raw.get('training_recipe') != RECIPE or raw.get('phase') not in (expected, args.phase):
            raise ValueError(f'Phase {args.phase} must initialize from curriculum phase {expected}, or restart its own phase')
    sources, validation, data_info = build_data(args, config)
    runtime = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k not in ('init_checkpoint', 'resume', 'output_dir', 'stop_after_updates', 'num_workers',
                            'device', 'log_every', 'audit_only', 'evaluate_only', 'compare_checkpoints')}
    # Synthetic manifest identity plus source content checks from the existing loader's source split audit.
    contract = dict(recipe=RECIPE, runtime=runtime, phase_definition=PHASES[args.phase], config=config.to_dict(),
                    geometry_signatures={d: ds.signature for d, ds in sources.items() if d != 'synthetic'},
                    validation_signatures={d: ds.dataset.signature for d, ds in validation.items() if d != 'synthetic'},
                    synthetic_signature=synthetic_signature(args.data_dir, dict(train=sources['synthetic'], val=validation['synthetic'].dataset)),
                    validation_records_sha256=hashlib.sha256(json.dumps(data_info['validation_records'], sort_keys=True).encode()).hexdigest(),
                    code_sha256=code_signature())
    lineage = raw.get('provenance') if args.resume else dict(initial_checkpoint=str(args.init_checkpoint),
        initial_checkpoint_sha256=file_sha256(args.init_checkpoint), parent_recipe=raw.get('training_recipe'),
        parent_phase=raw.get('phase'), parent_phase_step=raw.get('phase_step'), parent=raw.get('provenance'))
    summary = dict(recipe=RECIPE, contract=contract, provenance=lineage, torch_version=str(torch.__version__),
                   precision='FP32', accumulation_steps=10,
                   source_pairs={d: len(ds) for d, ds in sources.items()},
                   validation_pairs={d: len(ds) for d, ds in validation.items()},
                   caveat='Depth references are uncertain; fixed-reference metrics are conditional on trusted geometry')
    print(json.dumps({k: v for k, v in summary.items() if k != 'contract'}, indent=2), flush=True)
    print(f'[CONTRACT] phase={args.phase}; code={contract["code_sha256"]}; '
          f'fixed_val={contract["validation_records_sha256"]}', flush=True)
    if args.audit_only:
        return
    if args.resume:
        if raw['contract'] != contract:
            raise ValueError('Resume contract changed (code/data/config/schedule); start a new output using --init-checkpoint')
        if Path(raw['output_dir']).resolve() != args.output_dir.resolve():
            raise ValueError('Resume must target its original output directory')
        if not args.output_dir.is_dir():
            raise ValueError('Resume output directory is missing')
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('New run requires empty output directory; --resume is only for this exact phase')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check allocated GPU and driver/PyTorch compatibility')
    model = GRMMatcher(config).to(device); model.load_state_dict(raw['model'], strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_write(args.output_dir/'data_audit.json', data_info['audit'])
    json_write(args.output_dir/'validation_pairs.json', data_info['validation_records'])
    if args.evaluate_only:
        candidates = list(dict.fromkeys([args.init_checkpoint]+(args.compare_checkpoints or [])))
        comparison = []
        for index, path in enumerate(candidates):
            candidate = torch.load(path, map_location='cpu', weights_only=False)
            candidate_config = GRMMatcherConfig.from_dict(candidate['config']); candidate_config.use_amp = False
            if candidate_config.to_dict() != config.to_dict():
                raise ValueError(f'Candidate architecture/config differs: {path}')
            model.load_state_dict(candidate['model'], strict=True)
            val = validate(model, validation, args, device, 0, prefix=f'candidate_{index:02d}')
            comparison.append(dict(checkpoint=str(path), scores=selection_scores(val, args.phase), validation=val))
        json_write(args.output_dir/'candidate_comparison.json', comparison)
        monitor(args, '[EVALUATION COMPLETE] '+json.dumps([dict(checkpoint=r['checkpoint'], scores=r['scores']) for r in comparison]))
        return

    optimizer = make_optimizer(model, args.phase, args.weight_decay)
    objective = CurriculumObjective(config, args.phase, anchor_points=args.anchor_points,
        anchor_jitter=args.anchor_jitter, real_fine_min_weight=args.real_fine_min_weight,
        disable_real_fine=args.disable_real_fine)
    generator = torch.Generator().manual_seed(args.seed)
    step = 0; best = {}; validation_metrics = None; validation_step = 0
    checks_without_improvement = 0; gradient_checked = False
    baseline = None
    if args.resume:
        optimizer.load_state_dict(raw['optimizer']); restore_rng(raw['rng'], generator)
        step = raw['phase_step']; best = raw['best_scores']; baseline = raw['baseline']
        validation_metrics = raw['validation_metrics']; validation_step = raw['validation_step']
        checks_without_improvement = raw.get('checks_without_improvement', 0)
        gradient_checked = raw.get('gradient_checked', False)
    if step >= args.steps:
        raise ValueError('Phase is already complete')
    json_write(args.output_dir/'run.json', summary)
    interrupted = []
    def request_stop(signum, frame):
        interrupted.append(signum)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, request_stop)

    def state():
        return dict(training_recipe=RECIPE, stage='agri', phase=args.phase, phase_step=step,
            global_step=step, seen_pairs=step*10, output_dir=str(args.output_dir.resolve()),
            config=config.to_dict(), model=model.state_dict(), optimizer=optimizer.state_dict(),
            contract=contract, rng=rng_state(generator), sampler=dict(seed=args.seed, phase=args.phase, next_update=step),
            baseline=baseline, validation_metrics=validation_metrics, validation_step=validation_step,
            best_scores=best.copy(), gradient_checked=gradient_checked,
            checks_without_improvement=checks_without_improvement,
            confidence_head_trained=bool(raw.get('confidence_head_trained')) or (args.phase != 'A' and step > 0),
            geometry_heads_trained=False, inlier_head_trained=False,
            provenance=lineage)

    def save(name):
        _atomic_torch_save(state(), args.output_dir/name)

    if not args.resume:
        baseline = validation_metrics = validate(model, validation, args, device, 0)
        best = selection_scores(baseline, args.phase)
        for name in ('phase', 'md', 'agri', 'synthetic'):
            save(f'best_{name}.pt')
        save('latest.pt')
    end = min(args.steps, step+args.stop_after_updates) if args.stop_after_updates else args.steps
    monitor(args, f'[START] phase={args.phase} next_update={step+1}, stop={end}/{args.steps}, quotas={PHASES[args.phase]["quota"]}')
    train_totals = defaultdict(float); interval_pairs = 0; start_time = time.monotonic()
    while step < end:
        boundary = min(end, (step//args.eval_every+1)*args.eval_every)
        mixture = StepMixture(sources, args.phase, args.seed, step, boundary-step)
        loader = iter(data_loader(mixture, args.num_workers, device, args.seed+step))
        while step < boundary:
            objective.update = step
            phase_mode(model, args.phase, True)
            optimizer.zero_grad(set_to_none=True)
            pairs = []; local_metrics = defaultdict(float)
            update_started = time.monotonic()
            try:
                for _ in range(10):
                    cpu = next(loader); pairs.append(dict(domain=cpu['domain'][0], pair_id=cpu['pair_id'][0]))
                    batch = to_device(cpu, device)
                    loss, metrics = objective.agri(model, batch, True)
                    if loss is None or not torch.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError('Missing/nonfinite supervision or loss')
                    (loss/10).backward()
                    for key, value in metrics.items():
                        local_metrics[key] += value
                        train_totals[key] += value
                    for key, value in metrics.items():
                        train_totals[cpu['domain'][0]+'/'+key] += value
                    train_totals[cpu['domain'][0]+'/pairs'] += 1
                if not gradient_checked:
                    report = gradient_audit(model, args.phase)
                    json_write(args.output_dir/'gradient_audit.json', report)
                    gradient_checked = True
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True))
                factor = lr_multiplier(step, args.steps, args.warmup_updates, args.min_lr_ratio)
                for group in optimizer.param_groups:
                    group['lr'] = group['peak_lr']*factor
                optimizer.step()
            except Exception as error:
                diagnostic = dict(phase=args.phase, phase_step=step, pairs=pairs, metrics=dict(local_metrics), error=repr(error))
                # Do not overwrite the last known-good checkpoint with invalid parameters.
                _atomic_torch_save(dict(diagnostic=diagnostic, cpu_batch=locals().get('cpu')), args.output_dir/'failed_batch.pt')
                monitor(args, '[FAILED] '+str(diagnostic))
                raise
            step += 1; interval_pairs += 10
            if step == 1 or (args.log_every and step % args.log_every == 0) or step == end:
                averages = {k: v/interval_pairs for k, v in train_totals.items() if '/' not in k}
                domain_keys = ('loss', 'coarse', 'refinement', 'anchor', 'confidence', 'reference_count',
                               'output_matches', 'reference_scored_matches', 'anchor_fine_nll_supervised_count',
                               'pred_fine_nll_supervised_count')
                domains = {}
                for d in sources:
                    domains[d] = {k: train_totals[d+'/'+k]/max(train_totals[d+'/pairs'], 1) for k in domain_keys}
                    domain_counts = {k.split('/', 1)[1]: v for k, v in train_totals.items() if k.startswith(d+'/')}
                    domains[d].update(reference_metrics(domain_counts))
                event = dict(phase=args.phase, optimizer_step=step, total_steps=args.steps, seen_pairs=step*10,
                    averages=averages, by_domain=domains, coarse_only_training=args.phase == 'A',
                    lr={g['name']: g['lr'] for g in optimizer.param_groups},
                    gradient_norm=norm, real_fine_weight=objective.fine_weight(True) if args.phase != 'A' else 0.,
                    seconds_per_update=time.monotonic()-update_started,
                    elapsed_seconds=time.monotonic()-start_time,
                    gpu_peak_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None)
                compact = {d: dict(loss=round(r['loss'], 4), R1=round(r['coarse_recall1'], 4),
                                   fine_labels=round(r['anchor_fine_nll_supervised_count']+r['pred_fine_nll_supervised_count'], 1))
                           for d, r in domains.items()}
                monitor(args, f'[TRAIN] {args.phase} {step}/{args.steps} norm={norm:.3f} '
                    f'fine_real={event["real_fine_weight"]:.3f} lr={event["lr"]} domains={compact}', event)
                train_totals.clear(); interval_pairs = 0
            if step % args.save_every == 0:
                save('latest.pt')
            if interrupted:
                break
        del loader
        if interrupted:
            break
        if step % args.eval_every == 0 or step == args.steps:
            validation_metrics = validate(model, validation, args, device, step)
            validation_step = step
            scores = selection_scores(validation_metrics, args.phase)
            improved = scores['phase'] > best['phase']
            checks_without_improvement = 0 if improved else checks_without_improvement+1
            winners = [key for key, value in scores.items() if value > best[key]]
            best.update({key: scores[key] for key in winners})
            for key in winners:
                save(f'best_{key}.pt')
            save(f'step_{step:06d}.pt'); save('latest.pt')
            record = dict(phase=args.phase, step=step, scores=scores, best=best.copy(), updated=winners,
                          checks_without_improvement=checks_without_improvement,
                          validation=validation_metrics)
            with (args.output_dir/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            monitor(args, '[VALIDATION] '+json.dumps(dict(step=step, scores=scores, updated=winners)))
            if (validation_metrics['synthetic']['macro']['pck3'] < baseline['synthetic']['macro']['pck3']-.005 or
                    validation_metrics['synthetic']['macro']['reference_recall3'] < baseline['synthetic']['macro']['reference_recall3']-.02):
                monitor(args, '[REPLAY WARNING] Synthetic PCK3 dropped >0.5pp or reference recall3 >2pp; '
                              'review the tradeoff. Warning only; no loss-based veto.')
            if checks_without_improvement >= 3:
                monitor(args, '[REVIEW] Three checks without phase-score improvement; inspect tails/support before extending. No automatic phase transition.')
    save('latest.pt')
    monitor(args, f'[STOPPED] phase={args.phase} step={step}/{args.steps}; latest validation step={validation_step}; '
                  f'completed={step >= args.steps}; signal_stop={bool(interrupted)}')


if __name__ == '__main__':
    main()
