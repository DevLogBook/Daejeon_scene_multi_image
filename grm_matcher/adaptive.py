"""Risk-driven Mid search. Public coordinates and covariance use image pixels.

The only grouped loop is over two window sizes, never over individual queries.
Oracle entry points are deliberately separate from the image-only matcher API.
"""
from dataclasses import dataclass, fields
import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from .coordinates import image_xy_to_feature_xy, image_xy_to_normalized_grid
from .local_correlation import GRMRefinementOutput


class SearchRiskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if features.shape[-1] != 8:
            raise ValueError("Risk features must end in dimension 8")
        return self.net(features.float()).squeeze(-1)


def risk_features(corr, query, target, temperature):
    # Gather only selected rows; reuse the existing all-pairs result.
    p = corr.dual_probability.gather(1, query[..., None].expand(-1, -1, corr.dual_probability.shape[-1]))
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
    top = p.topk(min(2, p.shape[-1]), dim=-1).values
    p1 = top[..., 0]
    p2 = top[..., 1] if top.shape[-1] > 1 else torch.zeros_like(p1)
    entropy = -(p * p.clamp_min(1e-8).log()).sum(-1) / math.log(max(2, p.shape[-1]))
    reverse = corr.probability.argmax(1).gather(1, target)
    similarity = corr.similarity.gather(1, query[..., None].expand(-1, -1, p.shape[-1]))
    similarity = similarity.gather(2, target[..., None]).squeeze(-1) * temperature
    return torch.stack((p1, p2, p1-p2, entropy, corr.matchability_a.gather(1, query),
                        corr.matchability_b.gather(1, target), (reverse == query).float(),
                        similarity.clamp(-1, 1)), -1)


@dataclass
class SearchLabels:
    expand: torch.Tensor
    supervised: torch.Tensor
    category: torch.Tensor  # 0 small, 1 expanded, 2 unrecoverable, 3 invisible


@torch.no_grad()
def search_labels(initial, truth, valid, stride=8):
    # L-infinity: these are SQUARE windows, with feature-cell radii 2 and 4.
    error = (image_xy_to_feature_xy(truth, stride) - image_xy_to_feature_xy(initial, stride)).abs().amax(-1)
    visible = valid & torch.isfinite(truth).all(-1) & torch.isfinite(initial).all(-1)
    category = torch.full_like(valid, 3, dtype=torch.long)
    category = torch.where(visible, torch.full_like(category, 2), category)
    category = torch.where(visible & (error <= 4), torch.ones_like(category), category)
    category = torch.where(visible & (error <= 2), torch.zeros_like(category), category)
    return SearchLabels(category == 1, category < 2, category)


def risk_loss(logits, labels, gamma=0.):
    mask = labels.supervised
    if not mask.any():
        return logits.sum() * 0
    x, y = logits[mask].float(), labels.expand[mask].float()
    # Equal total contribution per observed class; one-class batches remain useful.
    positives, negatives = y.sum(), (1-y).sum()
    weight = torch.where(y.bool(), .5 / positives.clamp_min(1), .5 / negatives.clamp_min(1))
    weight = weight / weight.sum()
    loss = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    if gamma:
        pt = torch.where(y.bool(), x.sigmoid(), (-x).sigmoid())
        loss = loss * (1-pt).pow(gamma)
    return (loss * weight).sum()


@torch.no_grad()
def budget_route(scores, valid, config):
    if scores.shape[1] == 0:
        return valid.clone()
    count = valid.sum(1)
    ordered = scores.masked_fill(~valid, -torch.inf).sort(1, descending=True).values
    fraction = scores.new_full((scores.shape[0],), config.adaptive_budget)
    if config.adaptive_dynamic_budget:
        q = (count.float() * config.adaptive_topq).ceil().long().clamp_min(1)
        top = torch.arange(scores.shape[1], device=scores.device)[None] < q[:, None]
        top &= torch.isfinite(ordered)
        difficulty = ordered.masked_fill(~top, 0).sum(1) / top.sum(1).clamp_min(1)
        fraction = config.adaptive_min_budget + difficulty * (config.adaptive_max_budget-config.adaptive_min_budget)
        # Five-percent bins bound the number of budget shapes in deployment.
        fraction = (fraction / .05).round() * .05
        fraction = fraction.clamp(config.adaptive_min_budget, config.adaptive_max_budget)
    k = torch.minimum((count * fraction).floor().long(), count)
    order = scores.masked_fill(~valid, -torch.inf).argsort(dim=1, descending=True, stable=True)
    selected = torch.arange(scores.shape[1], device=scores.device)[None] < k[:, None]
    return torch.zeros_like(valid).scatter(1, order, selected) & valid


def grouped_mid(refiner, fa, fb, source, initial, valid, expand):
    """Gather/pad each size across B, run shared CNN, scatter to query order.

    Probabilities are center-padded to 9x9 AFTER computation; padded mass is zero.
    Batch=1 does exactly N_small*25 + N_large*81 cost cells.
    """
    b, n = valid.shape
    merged = {}
    projected = None
    for size, route in ((5, valid & ~expand), (9, valid & expand)):
        count = int(route.sum(1).max().item()) if n else 0
        if count == 0:
            continue
        index = route.long().argsort(dim=1, descending=True, stable=True)[:, :count]
        active = route.gather(1, index)
        gather = lambda x: x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))
        if projected is None:
            projected = (refiner.correlation.descriptor_projection(fa['F_mid']),
                         refiner.correlation.descriptor_projection(fb['F_mid']))
        part = refiner(fa['F_mid'], fb['F_mid'], gather(source), gather(initial), active,
            source_valid_map=fa.get('valid_mid'), target_valid_map=fb.get('valid_mid'), target_window_size=size,
            projected_features=projected)
        for field in fields(GRMRefinementOutput):
            value = getattr(part, field.name)
            if field.name == 'local_probability' and size == 5:
                value = F.pad(value, (2, 2, 2, 2))
            shape = (b, n, *value.shape[2:])
            mask = active.reshape(b, count, *([1] * (value.ndim-2)))
            value = value * mask
            expanded_index = index.reshape(b, count, *([1] * (value.ndim-2))).expand_as(value)
            # bool scatter-add is not supported on every backend.
            if value.dtype == torch.bool:
                value = value.long()
            scattered = value.new_zeros(shape).scatter_add(1, expanded_index, value)
            merged[field.name] = merged.get(field.name, 0) + scattered
    if not merged:
        zero = source[..., 0] * 0
        vector = source * 0
        cov = source.new_zeros(b, n, 2, 2)
        descriptor = source.new_zeros(b, n, refiner.correlation.channels)
        return GRMRefinementOutput(source, initial, vector, source.new_zeros(b, n, 9, 9), zero,
                                   cov, cov.clone(), zero, descriptor, descriptor.clone(), valid & False)
    merged['src_xy'] = source
    merged['valid_mask'] = merged['valid_mask'].bool()
    merged['dst_xy'] = torch.where(valid[..., None], merged['dst_xy'], initial)
    return GRMRefinementOutput(**merged)


class VarianceCalibrator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.floor = config.adaptive_sigma_min**2
        self.alpha_max = config.adaptive_alpha_max
        self.net = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log((1-.05)/(self.alpha_max-1)))

    def forward(self, fine):
        base = fine.covariance_image.float().detach()
        features = torch.stack((fine.entropy, fine.peak_probability,
            torch.log1p(base[..., 0, 0]), torch.log1p(base[..., 1, 1]),
            base[..., 0, 1] / base.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-6),
            (fine.source_descriptor-fine.target_descriptor).square().mean(-1)), -1).detach()
        alpha = .05 + (self.alpha_max-.05) * self.net(features).squeeze(-1).sigmoid()
        return alpha[..., None, None] * (.5*(base+base.transpose(-1, -2))) + self.floor * torch.eye(2, device=base.device)


def variance_nll(covariance, error, valid):
    if not valid.any():
        return covariance.sum() * 0
    cov, e = covariance[valid].float(), error[valid].float().detach()
    chol = torch.linalg.cholesky(cov)
    mahal = (e[..., None] * torch.cholesky_solve(e[..., None], chol)).sum((-2, -1))
    return .5 * (mahal + 2*chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)).mean()


@torch.no_grad()
def ground_truth_at(batch, points, side='a'):
    """Exact augmented dense fields take precedence over H, in both directions."""
    image = batch['image_'+side]
    h, w = image.shape[-2:]
    grid = image_xy_to_normalized_grid(points.float(), h, w)[:, :, None]
    key = 'gt_dense_mapping' if side == 'a' else 'gt_dense_mapping_ba'
    if key in batch:
        truth = F.grid_sample(batch[key].float(), grid, align_corners=False)[:, :, :, 0].transpose(1, 2)
    else:
        # Never invert a homography to approximate the inverse of a dense flow.
        if side == 'b' and 'gt_dense_mapping' in batch:
            pure_h = ('local_flow' in batch and bool((batch['local_flow'] == 0).all()))
            pure_h |= ('flow_active' in batch and not bool(batch['flow_active'].any()))
            if not pure_h:
                raise ValueError('Reverse Oracle needs the true gt_dense_mapping_ba for local flow')
        from .stage25 import transform_points
        matrix = batch['H_ab'].float()
        truth = transform_points(points.float(), matrix if side == 'a' else torch.linalg.inv(matrix))
    visible = F.grid_sample(batch['gt_overlap_'+side].float(), grid, align_corners=False)[:, 0, :, 0] > .999
    valid = visible & torch.isfinite(truth).all(-1)
    valid &= ((points >= 0) & (points <= points.new_tensor([w-1, h-1]))).all(-1)
    valid &= ((truth >= 0) & (truth <= truth.new_tensor([w-1, h-1]))).all(-1)
    for suffix, xy in ((side, points), ('b' if side == 'a' else 'a', truth)):
        mask = batch.get('valid_'+suffix)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None]
            g = image_xy_to_normalized_grid(xy.float(), h, w)[:, :, None]
            valid &= F.grid_sample(mask.float(), g, align_corners=False)[:, 0, :, 0] > .999
    return truth, valid


def route_points(model, corr, query, target, initial, valid, truth=None, truth_valid=None, *, compute_risk=False):
    config = model.config
    logits = None
    if config.adaptive_mode == 'learned_5_9' or model.training or compute_risk:
        features = risk_features(corr, query, target, config.temperature)
        logits = model.search_risk_head(features.detach() if config.adaptive_detach_risk else features)
    if config.adaptive_mode == 'oracle_5_9':
        if truth is None or truth_valid is None:
            raise ValueError('Oracle requires explicit GT via oracle_forward; ordinary inference accepts images only')
        expand = search_labels(initial, truth, valid & truth_valid, config.mid_stride).expand
    elif config.adaptive_mode == 'learned_5_9':
        expand = budget_route(logits.sigmoid(), valid, config)
    else:
        expand = torch.zeros_like(valid)
    return logits, expand


def oracle_forward(model, batch, return_auxiliary=False):
    if not model.config.adaptive_enabled or model.config.adaptive_mode not in ('oracle_5_9','oracle_5_7_9'):
        raise ValueError('oracle_forward requires oracle_5_9 config')
    from .architecture_v3 import forward_v3
    return forward_v3(model, batch['image_a'], batch['image_b'], return_auxiliary,
                      batch.get('valid_a'), batch.get('valid_b'), oracle_batch=batch)


def load_adaptive_state(model, state, *, upgrade=False):
    """Strict restore, or an explicit V3 migration allowing ONLY new head keys."""
    if not upgrade:
        return model.load_state_dict(state, strict=True)
    own = model.state_dict()
    missing, extra = set(own)-set(state), set(state)-set(own)
    allowed = ('search_risk_head.', 'search_level_head.', 'variance_calibrator.')
    if extra or any(not name.startswith(allowed) for name in missing):
        raise RuntimeError(f'Incompatible V3 checkpoint: missing={sorted(missing)}, unexpected={sorted(extra)}')
    if missing:
        warnings.warn('Explicit V3 migration initializes: '+', '.join(sorted(missing)), UserWarning)
    return model.load_state_dict({**own, **state}, strict=True)


def configure_stage(model, stage):
    if model.config.adaptive_risk_kind=='ordinal_local':
        raise ValueError('Use train_search_levels for ordinal local-evidence supervision')
    prefixes = {'A': ('search_risk_head.',), 'B': ('mid_refiner.', 'search_risk_head.'),
                'C': ('mid_refiner.', 'search_risk_head.'), 'D': ('variance_calibrator.',), 'E': ('',)}
    if stage not in prefixes:
        raise ValueError('Stage must be A/B/C/D/E')
    if not model.config.adaptive_enabled:
        raise ValueError('Stage training requires adaptive_enabled')
    if stage == 'D' and not model.config.adaptive_variance:
        raise ValueError('Stage D requires adaptive_variance')
    model.config.adaptive_mode = {'A': 'fixed_7', 'B': 'oracle_5_9'}.get(stage, 'learned_5_9')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes[stage]))
    return [p for p in model.parameters() if p.requires_grad]
