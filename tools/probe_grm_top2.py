"""Paired Top-1, second-mode and depth-oracle experiment for fixed-weight GRM."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.evaluate_grm_four_hpc import GRMMatcher, benchmark, load_inventory


ARMS = ('top1', 'top2', 'depth_oracle')


def run_pair(matcher, dataset, pair, frames, patch_args):
    import torch
    from grm_matcher.coordinates import flat_indices_to_image_xy
    from grm_matcher.top2_patch import probe_second_mode

    ag, md, uv = benchmark.ag, benchmark.md, benchmark.uv
    if dataset == 'uav2025':
        a, b = (frames[pair[key]] for key in ('frame_a', 'frame_b'))
        ia, ta, va = uv.prepare_image(a.image, a.w, a.h, 512)
        ib, tb, vb = uv.prepare_image(b.image, b.w, b.h, 512)
        shape_a, shape_b = (a.h, a.w), (b.h, b.w)
    else:
        a, b = (ag.read_image(pair[key]) for key in ('image_a', 'image_b'))
        ia, ta, va = md.preprocess_image(a, 512, 'letterbox')
        ib, tb, vb = md.preprocess_image(b, 512, 'letterbox')
        shape_a, shape_b = a.shape[:2], b.shape[:2]

    model = matcher.native.model
    device = matcher.device
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        matcher.sync()
        start = time.perf_counter()
        output, aux = model(matcher.native._tensor(ia), matcher.native._tensor(ib), return_auxiliary=True,
                            valid_a=torch.from_numpy(va).to(device)[None, None],
                            valid_b=torch.from_numpy(vb).to(device)[None, None])
        matcher.sync()
        base_ms = 1000 * (time.perf_counter() - start)
        start = time.perf_counter()
        second = probe_second_mode(model, output, aux, **patch_args)
        matcher.sync()
        extra_ms = 1000 * (time.perf_counter() - start)
        coarse = flat_indices_to_image_xy(aux.target_indices, aux.features_b['F_coarse'].shape[-1],
                                          model.config.coarse_stride)[0].cpu().numpy()
        source = output.src_xy[0].cpu().numpy()
        base = output.dst_xy[0].cpu().numpy()
        valid = output.valid_mask[0].cpu().numpy().copy()
        alt_slots = np.empty(0, dtype=int)
        if second is not None:
            alt_slots = second.slots.cpu().numpy()
            alt_center, alt_xy = second.center.cpu().numpy(), second.destination.cpu().numpy()
            alt_valid, alt_pick = second.valid.cpu().numpy(), second.replace.cpu().numpy()
        del aux, output, second

    inv_a, inv_b = np.linalg.inv(ta), np.linalg.inv(tb)
    source_original, base_original = ag.project(source, inv_a), ag.project(base, inv_b)
    valid &= ag.in_bounds(source_original, shape_a) & ag.in_bounds(base_original, shape_b)
    valid &= ag.in_bounds(source, ia.shape) & ag.in_bounds(base, ib.shape)
    if dataset == 'uav2025':
        reference = uv.reference(a, b, source_original, 512, .01, 1.)
        gt, supported = reference['target_input'], reference['valid'] & valid
    else:
        gt_original, _, supported, _ = md.depth_errors(source_original, base_original, pair, shape_a, shape_b, .05)
        gt, supported = ag.project(gt_original, tb), supported & valid

    base_error = np.linalg.norm(base - gt, axis=1)
    base_reach = np.max(np.abs(coarse - gt), axis=1) <= 32.
    patched, oracle = base.copy(), base.copy()
    rescued = harmed = recoverable = oracle_wins = routed = switches = 0
    if alt_slots.size:
        alt_valid &= valid[alt_slots] & ag.in_bounds(alt_xy, ib.shape)
        alt_valid &= ag.in_bounds(ag.project(alt_xy, inv_b), shape_b)
        chosen = alt_valid & alt_pick
        alternative_error = np.linalg.norm(alt_xy - gt[alt_slots], axis=1)
        alternative_reach = np.max(np.abs(alt_center - gt[alt_slots]), axis=1) <= 32.
        oracle_choice = alt_valid & supported[alt_slots] & (alternative_error < base_error[alt_slots])
        patched[alt_slots[chosen]] = alt_xy[chosen]
        oracle[alt_slots[oracle_choice]] = alt_xy[oracle_choice]
        correct_before = base_error[alt_slots] <= 3.
        correct_after = alternative_error <= 3.
        rescued = int((supported[alt_slots] & chosen & ~correct_before & correct_after).sum())
        harmed = int((supported[alt_slots] & chosen & correct_before & ~correct_after).sum())
        recoverable = int((supported[alt_slots] & ~base_reach[alt_slots] & alternative_reach).sum())
        oracle_wins = int((oracle_choice & ~correct_before & correct_after).sum())
        routed, switches = len(alt_slots), int(chosen.sum())

    seed = 2026 + pair['index']
    row = dict(pair_id=pair['pair_id'], scene=pair['scene'], overlap=pair.get('geometry_fraction_ab'),
               matches=int(valid.sum()), supported=int(supported.sum()), reach32=int((supported & base_reach).sum()),
               routed=routed, switches=switches, reach_rescue=recoverable, oracle_pck_rescue=oracle_wins,
               rescued=rescued, harmed=harmed, base_ms=base_ms, extra_ms=extra_ms)
    for arm, destination in zip(ARMS, (base, patched, oracle)):
        original = ag.project(destination, inv_b)
        if dataset == 'uav2025':
            epi, rotation, translation = uv.pose_metrics(source_original[valid], original[valid], a, b, 512, seed)
        else:
            epi = md.epi_error(source_original[valid], original[valid], pair['ka'], pair['kb'], pair['rotation'], pair['translation'])
            pose = md.estimate_pose(source[valid], destination[valid], ta @ pair['ka'], tb @ pair['kb'], .5, seed)
            rotation, translation = (md.angular_errors(*pose[:2], pair['rotation'], pair['translation'])
                                     if pose is not None else (np.inf, np.inf))
        error = np.linalg.norm(destination - gt, axis=1)
        row[arm] = dict(pck3=int((supported & (error <= 3.)).sum()),
                        epipolar_correct=int((epi < 1e-4).sum()), pose=max(rotation, translation))
    return row


def summarize(rows):
    if not rows:
        return None
    sums = {key: sum(r[key] for r in rows) for key in ('matches', 'supported', 'reach32', 'routed', 'switches',
            'reach_rescue', 'oracle_pck_rescue', 'rescued', 'harmed')}
    result = dict(pairs=len(rows), **sums, mean_base_ms=float(np.mean([r['base_ms'] for r in rows])),
                  mean_extra_ms=float(np.mean([r['extra_ms'] for r in rows])))
    for arm in ARMS:
        pose = [r[arm]['pose'] if r[arm]['pose'] is not None else np.inf for r in rows]
        result[arm] = dict(pck3_input=sum(r[arm]['pck3'] for r in rows)/max(1, sums['supported']),
                           epi_precision=sum(r[arm]['epipolar_correct'] for r in rows)/max(1, sums['matches']),
                           **{f'auc{t}': benchmark.md.pose_auc(pose, t) for t in (5, 10, 20)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--megadepth-root', type=Path)
    parser.add_argument('--uav-root', type=Path)
    parser.add_argument('--uav-prepared', type=Path, default=Path('benchmarks_uav2025_20261023/prepared'))
    parser.add_argument('--datasets', nargs='+', choices=('megadepth1500', 'uav2025'), default=('megadepth1500', 'uav2025'))
    parser.add_argument('--md-limit', type=int, default=120, help='Evenly spaced MD pairs. Use 0 for all 1500.')
    parser.add_argument('--budget', type=int, default=64)
    args = parser.parse_args()
    if args.md_limit < 0 or args.budget < 1 or len(set(args.datasets)) != len(args.datasets):
        parser.error('Require md-limit >= 0, budget >= 1 and unique datasets')
    data, metadata, frames, _ = load_inventory(args, model='GRM')
    matcher = GRMMatcher(args.checkpoint.resolve())
    patch_args = dict(budget=args.budget, separation_px=24., min_score_ratio=.30,
                      confidence_margin=.02, cycle_limit_px=8.)
    args.output.mkdir(parents=True, exist_ok=False)
    benchmark.write(args.output/'settings.json', dict(model=matcher.metadata, datasets=metadata, patch=patch_args))
    for dataset in args.datasets:
        pairs = data[dataset]
        if dataset == 'megadepth1500' and 0 < args.md_limit < len(pairs):
            pairs = [pairs[i] for i in np.linspace(0, len(pairs)-1, args.md_limit, dtype=int)]
        # Warm up the same inference and diagnostic calls outside the measured rows.
        for _ in range(3):
            run_pair(matcher, dataset, pairs[0], frames, patch_args)
        rows = []
        with (args.output/f'{dataset}.jsonl').open('w', encoding='utf-8') as journal:
            for pair in pairs:
                row = benchmark.uv.clean(run_pair(matcher, dataset, pair, frames, patch_args))
                rows.append(row)
                journal.write(json.dumps(row, allow_nan=False)+'\n')
                journal.flush()
                if len(rows) % 20 == 0 or len(rows) == len(pairs):
                    print(f'{dataset}: {len(rows)}/{len(pairs)}', flush=True)
        result = summarize(rows)
        if dataset == 'uav2025':
            result['low_overlap'] = summarize([r for r in rows if r['overlap'] < .30])
        benchmark.write(args.output/f'{dataset}_summary.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
