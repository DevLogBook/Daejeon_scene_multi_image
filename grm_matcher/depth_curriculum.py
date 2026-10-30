"""Independent V3 depth curriculum: scopes, balanced sampling and masked losses.

The legacy depth trainer/objective are not modified. Cache geometry is a
reconstruction reference, not exact subpixel ground truth.
"""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .architecture_v3 import matching_probability
from .coordinates import flat_indices_to_image_xy, image_xy_to_normalized_grid
from .depth_finetune import DepthCacheDataset, DepthObjective, masked_mean
from .imc_dataset import sparse_bidirectional_labels
from .stage25 import GRMStage25CoarseOutput, _local_probability_nll, transform_points


PREFIXES = {
    'backbone': ('backbone.',),
    'coarse': ('fpn.coarse_lateral.', 'fpn.coarse_output.', 'coarse_transformer.', 'correlation.'),
    'mid': ('fpn.mid_lateral.', 'fpn.coarse_to_mid.', 'fpn.mid_output.', 'mid_refiner.'),
    'fine': ('fpn._fine_alpha_logit', 'fpn.fine_lateral.', 'fpn.mid_to_fine.', 'fpn.fine_output.', 'fine_refiner.'),
    'bridge': ('context_bridge.',),
    'confidence': ('confidence_head.',),
}
PHASES = {
    'A': dict(steps=10000, quota=dict(megadepth=6, uav=2, synthetic=2),
              lr=dict(backbone=1e-5, coarse=1e-4)),
    'B': dict(steps=5000, quota=dict(megadepth=4, uav=4, synthetic=2),
              lr=dict(mid=2e-5, fine=2e-5, bridge=1e-5, confidence=1e-5)),
    'C': dict(steps=5000, quota=dict(megadepth=2, uav=6, synthetic=2),
              lr=dict(backbone=2e-6, coarse=2e-5, mid=1e-5, fine=1e-5, bridge=1e-5, confidence=1e-5)),
}


def parameter_group(name):
    matches = [key for key, prefixes in PREFIXES.items() if name.startswith(prefixes)]
    if len(matches) != 1:
        raise ValueError(f'Unassigned or ambiguous V3 parameter: {name}')
    return matches[0]


def phase_mode(model, phase, train):
    """Freeze parameters and stateful layers, preserving differentiable local paths."""
    active = PHASES[phase]['lr']
    model.train(train)
    for name, param in model.named_parameters():
        param.requires_grad_(parameter_group(name) in active)
    if train:
        for module in model.modules():
            params = list(module.parameters())
            if params and not any(p.requires_grad for p in params):
                module.eval()


def make_optimizer(model, phase, weight_decay):
    groups = defaultdict(list)
    for name, param in model.named_parameters():
        groups[parameter_group(name)].append(param)
    rates = PHASES[phase]['lr']
    return torch.optim.AdamW([dict(name=k, params=groups[k], lr=lr, peak_lr=lr)
                             for k, lr in rates.items() if groups[k]], weight_decay=weight_decay)


def group_indices(records, real=True):
    groups = defaultdict(list)
    for index, row in enumerate(records):
        key = row['scene'] if real else str(row['source_image']).replace('\\', '/').casefold()
        groups[key].append(index)
    return dict(sorted(groups.items()))


def interleaved_indices(groups, limit=0):
    result = []
    for offset in range(max(map(len, groups.values()))):
        for indices in groups.values():
            if offset < len(indices):
                result.append(indices[offset])
                if limit and len(result) == limit:
                    return result
    return result


class GeometryData(DepthCacheDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        row['scene_id'] = self.records[index]['scene']
        for direction in ('ab', 'ba'):
            target, valid, weight = [row[direction+'_'+key] for key in ('target', 'valid', 'weight')]
            if not torch.isfinite(target[valid]).all() or not torch.isfinite(weight).all():
                raise ValueError(f'Nonfinite geometry label: {row["pair_id"]}/{direction}')
            if (weight[valid] <= 0).any() or (weight > 1).any() or (weight < 0).any():
                raise ValueError(f'Invalid geometry weight: {row["pair_id"]}/{direction}')
            if ((target[valid] < 0) | (target[valid] > self.size-1)).any():
                raise ValueError(f'Out-of-canvas trusted target: {row["pair_id"]}/{direction}')
            # Unknown cells may not propagate NaN through masked losses.
            row[direction+'_target'] = torch.where(valid[:, None], target, torch.zeros_like(target))
            row[direction+'_weight'] = torch.where(valid, weight, torch.zeros_like(weight))
        return row


class TaggedSynthetic(Dataset):
    def __init__(self, data):
        self.data = data
        self.records = data.records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(self.data[index], domain='synthetic',
                    scene_id=str(self.records[index]['source_image']).replace('\\', '/').casefold())


class StepMixture(Dataset):
    """Index plan is a pure function of seed/phase/update, immune to prefetch."""
    def __init__(self, sources, phase, seed, start_update, updates):
        self.sources = sources
        self.plan = []
        groups = {d: list(group_indices(ds.records, d != 'synthetic').values()) for d, ds in sources.items()}
        for update in range(start_update, start_update+updates):
            rng = torch.Generator().manual_seed(seed + (ord(phase)-64)*1_000_000_000 + update)
            block = []
            for domain, count in PHASES[phase]['quota'].items():
                for _ in range(count):
                    scene = groups[domain][torch.randint(len(groups[domain]), (), generator=rng).item()]
                    index = scene[torch.randint(len(scene), (), generator=rng).item()]
                    block.append((domain, index))
            self.plan.extend(block[i] for i in torch.randperm(len(block), generator=rng).tolist())

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        domain, row = self.plan[index]
        return self.sources[domain][row]


def local_terms(mid, fine, target, initial, known, weight, config, fine_weight, fine_known):
    """Mid 24px + Fine 8px; strict real Fine quality only affects Fine terms."""
    mr = (config.mid_window_size//2)*config.mid_stride
    fr = (config.fine_window_size//2)*config.fine_stride
    mask = known & ((target-initial).abs() <= mr+fr).all(-1) & mid.valid_mask & fine.valid_mask
    mt = initial + (target-initial).clamp(-mr, mr)
    mid_coord = masked_mean(F.smooth_l1_loss(mid.dst_xy, mt, reduction='none').mean(-1), mask, weight)
    mid_nll = _local_probability_nll(mid.local_probability, initial, mt, config.mid_stride, mask, weight, 1e-8)
    fmask = mask & fine_known & (fine_weight > 0)
    fine_coord = masked_mean(F.smooth_l1_loss(fine.dst_xy, target, reduction='none').mean(-1), fmask, weight)
    nmask = fmask & ((target-mid.dst_xy.detach()).abs() <= fr).all(-1)
    fine_nll = _local_probability_nll(fine.local_probability, mid.dst_xy.detach(), target,
                                    config.fine_stride, nmask, weight, 1e-8)
    total = .5*mid_coord + .1*mid_nll + fine_weight*(fine_coord+.1*fine_nll)
    return total, dict(mid_coordinate=float(mid_coord.detach()), fine_coordinate=float(fine_coord.detach()),
                       mid_nll=float(mid_nll.detach()), fine_nll=float(fine_nll.detach()),
                       local_supervised_count=float(mask.sum()), fine_coordinate_supervised_count=float(fmask.sum()),
                       fine_nll_supervised_count=float(nmask.sum()))


class CurriculumObjective(DepthObjective):
    def __init__(self, config, phase, *, anchor_points=128, anchor_jitter=8., real_fine_min_weight=.5,
                 disable_real_fine=False, capture_prediction=False):
        super().__init__(config, SimpleNamespace(anchor_weight=.25))
        self.phase = phase
        self.anchor_points = anchor_points
        self.anchor_jitter = anchor_jitter
        self.real_fine_min_weight = real_fine_min_weight
        self.disable_real_fine = disable_real_fine
        self.capture_prediction = capture_prediction
        self.last_prediction = None
        self.last_reference_errors = None
        self.update = 0

    def fine_weight(self, real):
        if not real:
            return 1.
        if self.disable_real_fine:
            return 0.
        return .1+.15*min(self.update/1000, 1.) if self.phase == 'B' else .25

    def agri(self, model, batch, train, epoch=0):
        self.last_prediction = None
        self.last_reference_errors = None
        phase_mode(model, self.phase, train)
        real = batch['domain'][0] != 'synthetic'
        a, b = batch['image_a'], batch['image_b']
        size = a.shape[-1]; h = size//self.config.coarse_stride
        if a.shape[0] != 1 or a.shape[-2] != size:
            raise ValueError('Curriculum requires square canvas and microbatch=1')
        grid = flat_indices_to_image_xy(torch.arange(h*h, device=a.device)[None], h, self.config.coarse_stride)
        directions = {}
        for direction, side in (('ab', 'a'), ('ba', 'b')):
            if real:
                target = batch[direction+'_target'].float()
                valid = batch[direction+'_valid'].bool()
                weight = batch[direction+'_weight'].float()
            else:
                sampling = image_xy_to_normalized_grid(grid, size, size).unsqueeze(2)
                field = batch.get('gt_dense_mapping' if direction == 'ab' else 'gt_dense_mapping_ba')
                target = (F.grid_sample(field.float(), sampling, align_corners=False)[:, :, :, 0].transpose(1, 2)
                          if field is not None else transform_points(grid, torch.linalg.inv(batch['H_ab'])))
                valid = F.grid_sample(batch['gt_overlap_'+side].float(), sampling, align_corners=False)[:, 0, :, 0] > .999
                valid &= torch.isfinite(target).all(-1) & (target >= 0).all(-1) & (target <= size-1).all(-1)
                weight = valid.float()
            target = torch.where(valid[..., None], target, torch.zeros_like(target))
            directions[direction] = target, valid, weight

        def labels(direction):
            target, valid, weight = directions[direction]
            return sparse_bidirectional_labels(dict(image_a=a, sparse_a=grid, sparse_b=target,
                sparse_valid=valid, sparse_confidence=weight), a.device, self.config, self.base.adaptation).forward

        forward, backward = labels('ab'), labels('ba')
        if not forward.valid_mask.any() or not backward.valid_mask.any():
            return None, dict(valid_count=0.)
        if train and self.config.input_range == 'zero_one':
            def color(image):
                return (image*(.9+.2*torch.rand((1, 3, 1, 1), device=image.device))
                        + (torch.rand((1, 1, 1, 1), device=image.device)-.5)*.06).clamp(0, 1)
            a, b = color(a), color(b)

        coarse_only = train and self.phase == 'A'
        if coarse_only:
            fa, fb, ta, tb = model.encode_pair(a, b, batch.get('valid_a'), batch.get('valid_b'))
            corr = model.correlation(ta, tb, fa['valid_coarse'].flatten(1), fb['valid_coarse'].flatten(1))
            output = aux = None
        else:
            output, aux = model(a, b, return_auxiliary=True, valid_a=batch.get('valid_a'), valid_b=batch.get('valid_b'))
            fa, fb, corr = aux.features_a, aux.features_b, aux.correlation
        coarse_output = GRMStage25CoarseOutput(corr.similarity, corr.matchability_a, corr.matchability_b, h, h)
        zero = corr.matchability_a.sum()*0
        coarse = self.base.coarse(coarse_output, forward, backward)['total'] if self.phase != 'B' else zero
        anchor = predicted = confidence = positive = overlap = zero
        ametrics = {}; pmetrics = {}
        gt, gt_valid, gt_weight = directions['ab']
        fw = self.fine_weight(real)

        if self.phase != 'A':
            ids = torch.where(forward.valid_mask[0])[0]
            if len(ids) > self.anchor_points:
                ids = ids[torch.linspace(0, len(ids)-1, self.anchor_points, device=a.device).round().long()]
            initial = forward.coarse_dst_xy[:, ids]
            if train and self.anchor_jitter:
                initial = (initial+(torch.rand_like(initial)*2-1)*self.anchor_jitter).clamp(0, size-1)
            am, af = model.refine_points(fa, fb, grid[:, ids], initial, forward.valid_mask[:, ids])
            fine_known = (forward.confidence[:, ids] >= self.real_fine_min_weight) if real else forward.valid_mask[:, ids]
            anchor, ametrics = local_terms(am, af, forward.fine_dst_xy[:, ids], initial,
                forward.valid_mask[:, ids], forward.confidence[:, ids], self.config, fw, fine_known)

        metrics = dict(reference_count=float(gt_valid.sum()), output_matches=0., reference_scored_matches=0.,
                       correct_at_1px=0., correct_at_3px=0., correct_at_5px=0., mid_correct_at_3px=0.,
                       confidence_sum=0., confidence_brier_sum=0., fine_weight=fw if self.phase != 'A' else 0.)
        if output is not None:
            metrics['output_matches'] = float(output.valid_mask.sum())
            if output.src_xy.shape[1]:
                q = aux.query_indices
                target = gt.gather(1, q[..., None].expand(-1, -1, 2))
                known = gt_valid.gather(1, q) & output.valid_mask
                weight = gt_weight.gather(1, q)
                error = (output.dst_xy-target).norm(dim=-1)
                if self.capture_prediction and not train:
                    self.last_reference_errors = error[known].detach().cpu().numpy()
                initial = flat_indices_to_image_xy(aux.target_indices, h, self.config.coarse_stride)
                if self.phase != 'A':
                    fknown = (weight >= self.real_fine_min_weight) if real else known
                    predicted, pmetrics = local_terms(aux.mid, aux.fine, target, initial, known, weight,
                                                      self.config, fw, fknown)
                    if aux.confidence_logits is not None:
                        confidence = masked_mean(F.binary_cross_entropy_with_logits(aux.confidence_logits,
                            (error.detach() <= 3).float(), reduction='none'), known, weight)
                metrics['reference_scored_matches'] = float(known.sum())
                for threshold in (1, 3, 5):
                    metrics[f'correct_at_{threshold}px'] = float(((error <= threshold) & known).sum())
                metrics['mid_correct_at_3px'] = float((((aux.mid.dst_xy-target).norm(dim=-1) <= 3) & known).sum())
                metrics['confidence_sum'] = float(output.p_match[output.valid_mask].detach().sum())
                metrics['confidence_brier_sum'] = float(((output.p_match.detach()-(error <= 3).float()).square()*known).sum())

        if self.phase != 'B':
            if real:
                positive = self.base.coarse._positive_matchability_loss(coarse_output, forward, backward)
            else:
                values = []
                for side, pred, feat in (('a', corr.matchability_a, fa), ('b', corr.matchability_b, fb)):
                    truth = F.interpolate(batch['gt_overlap_'+side].float(), (h, h), mode='nearest-exact').flatten(1)
                    mask = feat['valid_coarse'].flatten(1)
                    if mask.any():
                        values.append(F.binary_cross_entropy(pred[mask].clamp(1e-6, 1-1e-6), truth[mask]))
                if values:
                    overlap = torch.stack(values).mean()
        total = coarse + .25*anchor + predicted + .1*confidence + .1*positive + .1*overlap
        probability = matching_probability(corr, self.config)
        ranking = probability.topk(min(4, h*h), dim=-1).indices
        cells = torch.round((gt+.5)/self.config.coarse_stride-.5).long()
        known = gt_valid & (cells >= 0).all(-1) & (cells < h).all(-1)
        target_cells = cells[..., 1].clamp(0, h-1)*h+cells[..., 0].clamp(0, h-1)
        initial_all = flat_indices_to_image_xy(ranking[..., 0], h, self.config.coarse_stride)
        metrics.update(loss=float(total.detach()), coarse=float(coarse.detach()), anchor=float(anchor.detach()),
                       refinement=float(predicted.detach()), confidence=float(confidence.detach()),
                       matchability=float(positive.detach()), overlap=float(overlap.detach()),
                       coarse_reference_count=float(known.sum()),
                       coarse_recall1_count=float(((ranking[..., 0] == target_cells) & known).sum()),
                       coarse_recall4_count=float(((ranking == target_cells[..., None]).any(-1) & known).sum()),
                       coarse_error_sum=float((initial_all-gt).norm(dim=-1)[known].detach().sum()))
        metrics.update({'anchor_'+k: v for k, v in ametrics.items()})
        metrics.update({'pred_'+k: v for k, v in pmetrics.items()})
        if self.capture_prediction and not train and output is not None:
            keep = output.valid_mask[0]
            self.last_prediction = (output.src_xy[0, keep].detach().cpu().numpy(),
                                    output.dst_xy[0, keep].detach().cpu().numpy())
        return total, metrics


def reference_metrics(totals):
    """Conditional precision and fixed-reference recall; unknown is unscored."""
    n = totals.get('reference_count', 0.)
    scored = totals.get('reference_scored_matches', 0.)
    outputs = totals.get('output_matches', 0.)
    result = dict(eligible=n > 0, scorable_fraction=scored/max(outputs, 1),
                  mean_confidence=totals.get('confidence_sum', 0.)/max(outputs, 1))
    for px in (1, 3, 5):
        correct = totals.get(f'correct_at_{px}px', 0.)
        precision, recall = correct/max(scored, 1), correct/max(n, 1)
        result.update({f'pck{px}': precision, f'reference_recall{px}': recall,
                       f'reference_f1_{px}': 2*precision*recall/max(precision+recall, 1e-12)})
    for k in (1, 4):
        result[f'coarse_recall{k}'] = totals.get(f'coarse_recall{k}_count', 0.)/max(totals.get('coarse_reference_count', 0.), 1)
    result['mid_pck3'] = totals.get('mid_correct_at_3px', 0.)/max(scored, 1)
    result['confidence_brier'] = totals.get('confidence_brier_sum', 0.)/max(scored, 1)
    return result
