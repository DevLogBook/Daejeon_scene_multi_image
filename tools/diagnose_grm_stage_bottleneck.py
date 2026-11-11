"""Audit fixed-weight GRM retrieval, refinement and pose on depth-supported pairs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.evaluate_grm_four_hpc import GRMMatcher, benchmark, load_inventory


SCORES = ('fpn_cosine', 'context_cosine', 'dual', 'deployed')
STAGES = ('coarse', 'mid', 'fine')
POSE_ARMS = ('all_pred', 'supported_pred', 'verified_pred', 'verified_reference', 'supported_reference')
FALLBACK_SOURCES = ('deployed', 'fpn')
LOCAL_STAGES = ('mid', 'fine', 'oracle_center', 'fine_oracle_center')


def fine_center_diagnostic(model, output, aux, supported, gt, ta, tb, pair):
    import torch
    from grm_matcher.coordinates import feature_xy_to_image_xy, image_xy_to_feature_xy

    slots = np.flatnonzero(supported)
    stride = model.config.fine_stride
    source = output.src_xy[:, slots]
    with torch.inference_mode():
        reference = source.new_tensor(gt[slots])[None]
        center = feature_xy_to_image_xy(image_xy_to_feature_xy(reference, stride).round(), stride)
        fa, fb = aux.features_a, aux.features_b
        fine = model.fine_refiner(
            fa['F_fine'], fb['F_fine'], source, center, model.points_valid(source, fa),
            source_valid_map=fa['valid_fine'], target_valid_map=fb['valid_fine'])
        inside = model.points_valid(fine.dst_xy, fb)
        candidates = (
            (aux.mid.dst_xy[:, slots], aux.mid.valid_mask[:, slots]),
            (output.dst_xy[:, slots], output.valid_mask[:, slots]),
            (center, model.points_valid(center, fb)),
            (fine.dst_xy, fine.valid_mask & inside),
        )
        result = dict(points=len(slots))
        md = benchmark.md
        for name, (xy, valid) in zip(LOCAL_STAGES, candidates):
            error = torch.linalg.vector_norm(xy - reference, dim=-1)[0].cpu().numpy()
            valid = valid[0].cpu().numpy()
            xy = xy[0].cpu().numpy()
            pose = md.estimate_pose(source[0].cpu().numpy()[valid], xy[valid],
                                    ta @ pair['ka'], tb @ pair['kb'], .5, 2026 + pair['index'])
            angle = max(md.angular_errors(*pose[:2], pair['rotation'], pair['translation'])) if pose else np.inf
            result[name] = dict(valid=int(valid.sum()), pose=angle,
                                pck05=int((valid & (error <= .5)).sum()),
                                pck1=int((valid & (error <= 1.)).sum()),
                                pck3=int((valid & (error <= 3.)).sum()),
                                median_px=float(np.median(error[valid])) if valid.any() else None,
                                p90_px=float(np.percentile(error[valid], 90)) if valid.any() else None)
    return result


def compare_fallback(ag, second, base, coarse, gt, supported, final_valid, shape_b, input_shape, inv_b, bound, pose):
    patched, oracle = base.copy(), base.copy()
    before = np.linalg.norm(base - gt, axis=-1)
    metrics = dict(refined=0, refined_valid=0, switches=0, rescued=0, harmed=0, oracle_rescues=0,
                   reach_rescues=0, switched_supported=0)
    if second is not None:
        slots = second.slots.cpu().numpy()
        center, alt = second.center.cpu().numpy(), second.destination.cpu().numpy()
        valid = second.valid.cpu().numpy() & final_valid[slots]
        valid &= ag.in_bounds(alt, input_shape) & ag.in_bounds(ag.project(alt, inv_b), shape_b)
        chosen = second.replace.cpu().numpy() & valid
        after = np.linalg.norm(alt - gt[slots], axis=-1)
        eligible = supported[slots] & valid
        rescue = eligible & (before[slots] > 3.) & (after <= 3.)
        oracle_choice = eligible & (after < before[slots])
        patched[slots[chosen]] = alt[chosen]
        oracle[slots[oracle_choice]] = alt[oracle_choice]
        outside = np.max(np.abs(coarse[slots] - gt[slots]), axis=1) > bound
        within = np.max(np.abs(center - gt[slots]), axis=1) <= bound
        metrics.update(refined=len(slots), refined_valid=int(valid.sum()), switches=int(chosen.sum()),
                       switched_supported=int((chosen & eligible).sum()),
                       rescued=int((chosen & rescue).sum()),
                       harmed=int((eligible & chosen & (before[slots] <= 3.) & (after > 3.)).sum()),
                       oracle_rescues=int(rescue.sum()), reach_rescues=int((eligible & outside & within).sum()))
    for name, xy in (('patch', patched), ('oracle', oracle)):
        metrics[name] = dict(pck3=int((supported & (np.linalg.norm(xy - gt, axis=-1) <= 3.)).sum()),
                             pose=pose(xy))
    return metrics


def run_pair(matcher, dataset, pair, frames, *, fallback_budget=0, fine_oracle=False):
    import torch
    from torch.nn.functional import normalize
    from grm_matcher.coordinates import flat_indices_to_image_xy

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
    stride = model.config.coarse_stride
    bound = (model.config.mid_window_size // 2) * model.config.mid_stride + (
        model.config.fine_window_size // 2) * model.config.fine_stride
    fallback = {}
    with torch.inference_mode():
        if fallback_budget:
            from grm_matcher.top2_patch import probe_routed_candidate, route_difficult_queries
            matcher.sync()
            started = time.perf_counter()
        output, aux = model(matcher.native._tensor(ia), matcher.native._tensor(ib), return_auxiliary=True,
                            valid_a=torch.from_numpy(va).to(device)[None, None],
                            valid_b=torch.from_numpy(vb).to(device)[None, None])
        if fallback_budget:
            matcher.sync()
            base_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            routes = route_difficult_queries(model, output, aux, budget=fallback_budget)
            matcher.sync()
            route_ms = (time.perf_counter() - started) * 1000
            for name in FALLBACK_SOURCES:
                started = time.perf_counter()
                second = probe_routed_candidate(model, output, aux, routes, source=name)
                matcher.sync()
                fallback[name] = (second, (time.perf_counter() - started) * 1000)
        source = output.src_xy[0].cpu().numpy()
        fine = output.dst_xy[0].cpu().numpy()
        final_valid = output.valid_mask[0].cpu().numpy().copy()
        query_valid = aux.query_valid_mask[0].cpu().numpy()
        fa, fb = aux.features_a['F_coarse'], aux.features_b['F_coarse']
        width = fb.shape[-1]
        query = aux.query_indices[0]
        corr = aux.correlation
        fpn_a = normalize(fa[0].flatten(1).T[query].float(), dim=-1)
        fpn_b = normalize(fb[0].flatten(1).T.float(), dim=-1)
        dual = corr.dual_probability[0, query]
        deployed = corr.probability[0, query] if model.config.v3_revision >= 1 else dual
        scores = (fpn_a @ fpn_b.T, corr.similarity[0, query], dual, deployed)
        candidate_mask = aux.features_b['valid_coarse'][0].flatten()
        coarse = flat_indices_to_image_xy(aux.target_indices[0], width, stride).cpu().numpy()
        mid = aux.mid.dst_xy[0].cpu().numpy() if aux.mid is not None else fine
        mid_valid = aux.mid.valid_mask[0].cpu().numpy() if aux.mid is not None else final_valid
        ranks = {}
        for name, score in zip(SCORES, scores):
            ids = score.masked_fill(~candidate_mask, -torch.inf).topk(3, dim=-1).indices
            if name == 'deployed':
                # Honor the deployment path's first-index tie break.
                ids[:, 0] = aux.target_indices[0]
                tied_out = (ids[:, 1:] == ids[:, :1]).any(dim=-1)
                if tied_out.any():
                    rerank = score[tied_out].masked_fill(~candidate_mask, -torch.inf)
                    rerank.scatter_(1, ids[tied_out, :1], -torch.inf)
                    ids[tied_out, 1:] = rerank.topk(2, dim=-1).indices
            ranks[name] = (flat_indices_to_image_xy(ids, width, stride).cpu().numpy(),
                           candidate_mask[ids].cpu().numpy())

    source_orig = ag.project(source, np.linalg.inv(ta))
    inv_b = np.linalg.inv(tb)
    fine_orig = ag.project(fine, inv_b)
    final_valid &= ag.in_bounds(source_orig, shape_a) & ag.in_bounds(fine_orig, shape_b)
    final_valid &= ag.in_bounds(source, ia.shape) & ag.in_bounds(fine, ib.shape)
    if dataset == 'uav2025':
        reference = uv.reference(a, b, source_orig, 512, .01, 1.)
        gt, gt_orig, depth_valid = (reference[k] for k in ('target_input', 'target_original', 'valid'))
    else:
        gt_orig, _, depth_valid, _ = md.depth_errors(source_orig, fine_orig, pair, shape_a, shape_b, .05)
        gt = ag.project(gt_orig, tb)
    supported = depth_valid & query_valid & ag.in_bounds(gt, ib.shape)
    selected = supported & final_valid
    gt_cell = np.rint((gt[supported] + .5) / stride - .5)
    row = dict(pair_id=pair['pair_id'], scene=pair['scene'], overlap=pair.get('geometry_fraction_ab'),
               queries=int(query_valid.sum()), matches=int(final_valid.sum()), supported=int(supported.sum()),
               supported_final=int(selected.sum()), bound=bound)
    top1_reach = np.max(np.abs(coarse - gt), axis=1) <= bound
    for name, (xy, valid) in ranks.items():
        cell = (xy[supported] + .5) / stride - .5
        exact = (cell == gt_cell[:, None]).all(axis=-1) & valid[supported]
        reach = (np.max(np.abs(xy - gt[:, None]), axis=-1) <= bound) & valid
        row[name] = dict(cell1=int(exact[:, 0].sum()), cell3=int(exact.any(axis=-1).sum()),
                         reach1=int((supported & reach[:, 0]).sum()), reach3=int((supported & reach.any(axis=-1)).sum()),
                         rescue3=int((supported & ~top1_reach & reach.any(axis=-1)).sum()))
    for name, xy, valid in zip(STAGES, (coarse, mid, fine), (query_valid, mid_valid, final_valid)):
        correct = (np.linalg.norm(xy - gt, axis=-1) <= 3.) & valid
        row[name + '_pck3'] = int((supported & correct).sum())
    row['fine_miss_outside'] = int((supported & ~correct & ~top1_reach).sum())
    row['fine_hit_inside'] = int((supported & correct & top1_reach).sum())
    verified = selected & correct
    seed = 2026 + pair['index']
    if fine_oracle and dataset == 'megadepth1500':
        row['fine_center_diagnostic'] = (fine_center_diagnostic(model, output, aux, supported & top1_reach,
                                         gt, ta, tb, pair) if (supported & top1_reach).any() else None)
    if fallback:
        def patched_pose(xy):
            if final_valid.sum() < 5:
                return np.inf
            dest = ag.project(xy[final_valid], inv_b)
            _, rotation, translation = uv.pose_metrics(source_orig[final_valid], dest, a, b, 512, seed)
            return max(rotation, translation)

        row['fallback'] = dict(base_ms=base_ms, route_ms=route_ms, routed=int(routes.slots.numel()),
                               routed_supported=int(supported[routes.slots.cpu().numpy()].sum()),
                               routed_outside=int((supported & ~top1_reach)[routes.slots.cpu().numpy()].sum()))
        for name, (second, extra_ms) in fallback.items():
            row['fallback'][name] = compare_fallback(ag, second, fine, coarse, gt, supported & final_valid,
                                                    final_valid, shape_b, ib.shape, inv_b, bound, patched_pose)
            row['fallback'][name]['extra_ms'] = extra_ms + route_ms
    for name, mask in zip(POSE_ARMS, (final_valid, selected, verified, verified, selected)):
        points = int(mask.sum())
        if points < 5:
            row[name] = dict(points=points, pose=np.inf)
            continue
        reference_coordinates = name.endswith('_reference')
        destination = gt if reference_coordinates else fine
        original = gt_orig if reference_coordinates else fine_orig
        if dataset == 'uav2025':
            _, rotation, translation = uv.pose_metrics(source_orig[mask], original[mask], a, b, 512, seed)
        else:
            pose = md.estimate_pose(source[mask], destination[mask], ta @ pair['ka'], tb @ pair['kb'], .5, seed)
            rotation, translation = (md.angular_errors(*pose[:2], pair['rotation'], pair['translation'])
                                     if pose is not None else (np.inf, np.inf))
        row[name] = dict(points=points, pose=max(rotation, translation))
    return row


def summarize(rows):
    totals = {k: sum(r[k] for r in rows) for k in (
        'queries', 'matches', 'supported', 'supported_final', 'coarse_pck3', 'mid_pck3', 'fine_pck3',
        'fine_miss_outside', 'fine_hit_inside')}
    result = dict(pairs=len(rows), bound=rows[0]['bound'], **totals)
    for name in SCORES:
        result[name] = {k: sum(r[name][k] for r in rows) for k in rows[0][name]}
    for name in POSE_ARMS:
        errors = [r[name]['pose'] if r[name]['pose'] is not None else np.inf for r in rows]
        result[name] = dict(mean_points=float(np.mean([r[name]['points'] for r in rows])),
                            finite_poses=int(np.isfinite(errors).sum()),
                            **{f'auc{t}': benchmark.md.pose_auc(errors, t) for t in (5, 10, 20)})
    if 'fine_center_diagnostic' in rows[0]:
        local = [r['fine_center_diagnostic'] for r in rows]
        point_count = sum(r['points'] for r in local if r is not None)
        result['fine_center_diagnostic'] = dict(points=point_count)
        for name in LOCAL_STAGES:
            stage = [r[name] if r is not None else None for r in local]
            errors = [r['pose'] if r is not None and r['pose'] is not None else np.inf for r in stage]
            measured = [r for r in stage if r is not None and r['median_px'] is not None]
            result['fine_center_diagnostic'][name] = dict(
                **{f'pck{t}': sum(r[f'pck{t}'] for r in stage if r is not None) / max(1, point_count)
                   for t in ('05', '1', '3')},
                median_pair_median_px=float(np.median([r['median_px'] for r in measured])) if measured else None,
                median_pair_p90_px=float(np.median([r['p90_px'] for r in measured])) if measured else None,
                **{f'auc{t}': benchmark.md.pose_auc(errors, t) for t in (5, 10, 20)})
    if 'fallback' in rows[0]:
        routed = [r['fallback'] for r in rows]
        result['fallback'] = dict(
            **{k: sum(r[k] for r in routed) for k in ('routed', 'routed_supported', 'routed_outside')},
            **{k: float(np.mean([r[k] for r in routed])) for k in ('base_ms', 'route_ms')})
        for name in FALLBACK_SOURCES:
            arm = [r[name] for r in routed]
            metrics = {k: sum(r[k] for r in arm) for k in ('refined', 'refined_valid', 'switches', 'rescued',
                       'harmed', 'oracle_rescues', 'reach_rescues', 'switched_supported')}
            metrics['extra_ms'] = float(np.mean([r['extra_ms'] for r in arm]))
            for variant in ('patch', 'oracle'):
                errors = [r[variant]['pose'] if r[variant]['pose'] is not None else np.inf for r in arm]
                metrics[variant] = dict(pck3=sum(r[variant]['pck3'] for r in arm) / max(1, totals['supported']),
                                        **{f'auc{t}': benchmark.md.pose_auc(errors, t) for t in (5, 10, 20)})
            result['fallback'][name] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--megadepth-root', type=Path)
    parser.add_argument('--uav-root', type=Path)
    parser.add_argument('--uav-prepared', type=Path, default=Path('benchmarks_uav2025_20261023/prepared'))
    parser.add_argument('--datasets', nargs='+', choices=('megadepth1500', 'uav2025'), default=('megadepth1500', 'uav2025'))
    parser.add_argument('--md-limit', type=int, default=120, help='Evenly spaced sample size. Zero uses all 1500.')
    parser.add_argument('--uav-fallback-budget', type=int, default=0, help='Maximum fallback queries per UAV pair.')
    parser.add_argument('--md-fine-oracle', action='store_true', help='Probe Fine with a quantized GT center on MD.')
    args = parser.parse_args()
    if min(args.md_limit, args.uav_fallback_budget) < 0 or len(set(args.datasets)) != len(args.datasets):
        parser.error('Use nonnegative limits and unique datasets.')
    data, metadata, frames, _ = load_inventory(args, model='GRM')
    matcher = GRMMatcher(args.checkpoint.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    benchmark.write(args.output / 'settings.json', dict(model=matcher.metadata, datasets=metadata,
                    md_limit=args.md_limit, pck_input_px=3, uav_low_overlap_max=.30,
                    uav_low_overlap_field='geometry_fraction_ab', uav_fallback_budget=args.uav_fallback_budget,
                    md_fine_oracle=args.md_fine_oracle))
    for dataset in args.datasets:
        pairs = data[dataset]
        if dataset == 'megadepth1500' and 0 < args.md_limit < len(pairs):
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, args.md_limit, dtype=int)]
        options = dict(fallback_budget=args.uav_fallback_budget if dataset == 'uav2025' else 0,
                       fine_oracle=args.md_fine_oracle)
        if options['fallback_budget']:
            for _ in range(3):
                run_pair(matcher, dataset, pairs[0], frames, **options)
        rows = []
        with (args.output / f'{dataset}.jsonl').open('w', encoding='utf-8') as journal:
            for pair in pairs:
                row = benchmark.uv.clean(run_pair(matcher, dataset, pair, frames, **options))
                rows.append(row)
                journal.write(json.dumps(row, allow_nan=False) + '\n')
                if len(rows) % 20 == 0 or len(rows) == len(pairs):
                    journal.flush()
                    print(f'{dataset}: {len(rows)}/{len(pairs)}', flush=True)
        summary = summarize(rows)
        low = [r for r in rows if r['overlap'] is not None and r['overlap'] < .30]
        if low:
            summary['low_overlap'] = summarize(low)
            summary['other_overlap'] = summarize([r for r in rows if r['overlap'] >= .30])
        benchmark.write(args.output / f'{dataset}_summary.json', summary)


if __name__ == '__main__':
    main()
