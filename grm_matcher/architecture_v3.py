"""Opt-in Top-1 matcher: local cost processing, bounded coverage, score-only confidence."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .backbone import build_grm_backbone
from .feature_pyramid import GRMResidualFPN
from .coarse_transformer import GRMCoarseContextTransformer
from .correlation import GRMExplicitCorrelation
from .context_bridge import GRMContextBridge
from .coordinates import flat_indices_to_image_xy
from .local_correlation import (GRMLocalCorrelation, GRMLocalRefinerCore,
    compute_spatial_expectation, compute_spatial_covariance, sample_feature_windows)
from .outputs import GRMMatcherOutput


def peak_moments(probability):
    """Locate within the winning 3x3 mode; retain full-distribution covariance."""
    h, w = probability.shape[-2:]
    index = probability.flatten(-2).argmax(-1)
    y, x = torch.meshgrid(torch.arange(h, device=probability.device),
                          torch.arange(w, device=probability.device), indexing="ij")
    neighborhood = ((x-index[..., None, None] % w).abs() <= 1) & (
        (y-index[..., None, None] // w).abs() <= 1)
    local = probability * neighborhood
    local = local / local.sum((-2, -1), keepdim=True).clamp_min(1e-8)
    return compute_spatial_expectation(local), compute_spatial_covariance(probability)


class CostResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class CostVolumeCorrelation(GRMLocalCorrelation):
    def __init__(self, channels, temperature, eps, hidden, patch_size=0):
        super().__init__(channels, temperature, eps)
        self.peak_localization = True
        self.patch_size = patch_size
        self.input_channels = 4 + 2*patch_size**2
        self.cost_net = nn.Sequential(nn.Conv2d(self.input_channels, hidden, 3, padding=1), nn.GELU(),
            CostResidual(hidden), CostResidual(hidden), nn.Conv2d(hidden, 1, 1))
        nn.init.normal_(self.cost_net[-1].weight, std=1e-3)
        nn.init.zeros_(self.cost_net[-1].bias)

    def patch_cost(self, source, target, source_xy, target_xy, window, source_map, target_map):
        """One plane per source offset, compared to the same offset at every target hypothesis."""
        size = self.patch_size
        src, src_valid = sample_feature_windows(self.descriptor_projection(source), source_xy, size)
        dst, dst_valid = sample_feature_windows(self.descriptor_projection(target), target_xy, window+size-1)
        if source_map is not None:
            sampled, _ = sample_feature_windows(source_map[:, None].float(), source_xy, size)
            src_valid = src_valid & (sampled[:, :, 0] > .999)
        if target_map is not None:
            sampled, _ = sample_feature_windows(target_map[:, None].float(), target_xy, window+size-1)
            dst_valid = dst_valid & (sampled[:, :, 0] > .999)
        src, dst = F.normalize(src.float(), dim=2, eps=self.eps), F.normalize(dst.float(), dim=2, eps=self.eps)
        costs, masks = [], []
        for y in range(size):
            for x in range(size):
                mask = src_valid[..., y, x, None, None] & dst_valid[..., y:y+window, x:x+window]
                plane = (src[..., y, x, None, None] * dst[..., y:y+window, x:x+window]).sum(2)
                costs.append(plane*mask)
                masks.append(mask.float())
        return torch.stack(costs, 2), torch.stack(masks, 2)

    def forward(self, feature_source, feature_target, source_centers_xy, target_centers_xy,
                source_patch_size, target_window_size, valid_mask=None, source_valid_map=None, target_valid_map=None):
        result = super().forward(feature_source, feature_target, source_centers_xy, target_centers_xy,
            source_patch_size, target_window_size, valid_mask, source_valid_map, target_valid_map)
        b, n, h, w = result.logits.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=result.logits.device),
                              torch.linspace(-1, 1, w, device=result.logits.device), indexing="ij")
        valid = result.window_valid_mask
        cost = torch.stack((result.logits * self.temperature * valid, valid.float(),
                            x.expand(b, n, h, w), y.expand(b, n, h, w)), dim=2)
        if self.patch_size:
            if source_patch_size != self.patch_size:
                raise ValueError("Configured source patch does not match cost-network inputs")
            planes, masks = self.patch_cost(feature_source, feature_target, source_centers_xy,
                target_centers_xy, target_window_size, source_valid_map, target_valid_map)
            cost = torch.cat((cost, planes*valid[:, :, None], masks*valid[:, :, None]), 2)
        logits = result.logits + self.cost_net(cost.reshape(b*n, self.input_channels, h, w)).reshape(b, n, h, w)
        prob = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).flatten(-2).softmax(-1).reshape_as(logits)
        prob = prob * valid
        prob = prob / prob.sum((-2, -1), keepdim=True).clamp_min(self.eps)
        result.logits, result.probability = logits, prob
        result.delta_xy, result.covariance = peak_moments(prob)
        result.entropy = -(prob * prob.clamp_min(self.eps).log()).sum((-2, -1))
        result.peak_probability = prob.flatten(-2).amax(-1)
        # The inherited descriptors are not used for V3 confidence or geometry.
        return result


def initialize_v3(model, config):
    model.backbone = build_grm_backbone(config)
    model.fpn = GRMResidualFPN(config)
    model.coarse_transformer = GRMCoarseContextTransformer(config)
    model.correlation = GRMExplicitCorrelation(config)
    model.context_bridge = GRMContextBridge(config)
    for name, channels, stride, patch, window, hidden in (
        ("mid_refiner", config.mid_dim, config.mid_stride, config.mid_source_patch_size,
         config.mid_window_size, config.v3_mid_hidden),
        ("fine_refiner", config.fine_dim, config.fine_stride, config.fine_source_patch_size,
         config.fine_window_size, config.v3_fine_hidden)):
        refiner = GRMLocalRefinerCore(channels, stride, patch, window, config.local_temperature, config.probability_eps)
        refiner.correlation = CostVolumeCorrelation(channels, config.local_temperature, config.probability_eps,
            hidden, patch_size=patch if config.v3_revision >= 1 else 0)
        setattr(model, name, refiner)
    model.confidence_head = (nn.Sequential(nn.Linear(10, 32), nn.GELU(), nn.Linear(32, 1))
                             if config.v3_confidence_head else None)


def row_statistics(correlation):
    p = correlation.dual_probability
    conditional = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
    entropy = -(conditional * conditional.clamp_min(1e-8).log()).sum(-1) / math.log(max(2, p.shape[-1]))
    values = conditional.topk(min(2, p.shape[-1]), dim=-1).values
    margin = values[..., 0] - (values[..., 1] if values.shape[-1] > 1 else 0)
    return entropy, margin


def matching_probability(correlation, config):
    """Use one pair score for inference, supervised initialization and reverse matching."""
    if config.v3_revision == 0:
        return correlation.dual_probability
    return (correlation.dual_probability * correlation.matchability_a[..., None]
            * correlation.matchability_b[:, None, :])


def target_support(correlation, source_valid, target_valid, config):
    probability = matching_probability(correlation, config)
    best, targets = probability.masked_fill(~target_valid[:, None], -1).max(-1)
    valid = source_valid & target_valid.gather(1, targets)
    if config.v3_revision == 0:
        best = best.clamp_min(0) * correlation.matchability_a * correlation.matchability_b.gather(1, targets)
    else:
        valid &= (correlation.matchability_a > config.v3_min_matchability)
        valid &= (correlation.matchability_b.gather(1, targets) > config.v3_min_matchability)
        maximum = best.masked_fill(~valid, 0).amax(-1, keepdim=True)
        valid &= (best > config.v3_relative_reliability*maximum) & (best > 0)
    return best.clamp_min(0), targets, valid


@torch.no_grad()
def select_queries(reliability, valid, width, count, bonus=.2, sigma=2., underfill=True):
    """Greedy bounded coverage bonus; no quotas, confidence thresholds, or duplicates."""
    b, total = reliability.shape
    count = min(count, total)
    if underfill:
        # Batch=1 has no padded work. Larger batches pad only to their largest support.
        count = min(count, int(valid.sum(-1).max().item()))
    if count == 0:
        return (torch.empty(b, 0, dtype=torch.long, device=valid.device),
                torch.empty(b, 0, dtype=torch.bool, device=valid.device))
    ids = torch.arange(total, device=reliability.device)
    xy = torch.stack((ids % width, ids // width), -1).float()
    if reliability.is_cuda and reliability.dtype == torch.float32 and total <= 1024 and count >= 4:
        # At 512px there are only 32x32 coarse cells. Hundreds of sequential
        # CUDA launches cost more than this small greedy decision on the CPU.
        # Compute the SAME Gaussian distances on the input device first, then
        # transfer once; do not substitute CPU exp, cdist, or approximate NMS.
        # The limit bounds the quadratic table to 4 MiB (FP32), independent of B.
        # For <4 queries, avoid table setup/copies and keep the short device loop.
        distance = (xy[:, None] - xy[None]).square().sum(-1)
        separation = (1-torch.exp(-distance/(2*sigma*sigma))).cpu().numpy()
        quality = reliability.cpu().numpy()
        remaining_cpu = valid.cpu().numpy().copy()
        diversity_cpu = np.ones_like(quality)
        chosen_cpu = np.empty((b, count), dtype=np.int64)
        masks_cpu = np.empty((b, count), dtype=np.bool_)
        rows = np.arange(b)
        for slot in range(count):
            scores = quality * (1 + bonus * diversity_cpu)
            scores[~remaining_cpu] = -1
            index = scores.argmax(-1)
            chosen_cpu[:, slot] = index
            masks_cpu[:, slot] = remaining_cpu.any(-1)
            np.minimum(diversity_cpu, separation[index], out=diversity_cpu)
            remaining_cpu[rows, index] = False
        return (torch.from_numpy(chosen_cpu).to(valid.device),
                torch.from_numpy(masks_cpu).to(valid.device))
    remaining = valid.clone()
    diversity = torch.ones_like(reliability)
    chosen, masks = [], []
    for _ in range(count):
        scores = (reliability * (1 + bonus * diversity)).masked_fill(~remaining, -1)
        index = scores.argmax(-1)
        active = remaining.any(-1)
        chosen.append(index)
        masks.append(active)
        distance = (xy[None] - xy[index, None]).square().sum(-1)
        diversity = torch.minimum(diversity, 1-torch.exp(-distance/(2*sigma*sigma)))
        remaining.scatter_(1, index[:, None], False)
    return torch.stack(chosen, 1), torch.stack(masks, 1)


def confidence_logits(model, correlation, query, target, mid, fine, cycle):
    entropy, margin = row_statistics(correlation)
    gather = lambda x: x.gather(1, query)
    appearance = correlation.dual_probability.gather(1, query[..., None].expand(-1, -1, correlation.dual_probability.shape[-1])).gather(2, target[..., None]).squeeze(-1)
    trace = fine.covariance_image.diagonal(dim1=-2, dim2=-1).sum(-1)
    features = torch.stack((appearance, gather(correlation.matchability_a),
        correlation.matchability_b.gather(1, target), gather(entropy), gather(margin),
        mid.peak_probability, fine.peak_probability,
        fine.entropy / math.log(max(2, fine.local_probability.shape[-1]**2)),
        torch.log1p(trace.clamp_min(0)) / 10, cycle.cycle_score), -1)
    # Calibration must not teach the matcher to make its own correctness label easier.
    return model.confidence_head(features.detach()).squeeze(-1) if model.confidence_head is not None else None


@dataclass
class V3Auxiliary:
    features_a: dict
    features_b: dict
    correlation: object
    query_indices: torch.Tensor
    target_indices: torch.Tensor
    mid: object
    fine: object
    confidence_logits: object


def forward_v3(model, image_a, image_b, return_auxiliary=False, valid_a=None, valid_b=None):
    fa, fb, ta, tb = model.encode_pair(image_a, image_b, valid_a, valid_b)
    ma, mb = fa['valid_coarse'].flatten(1), fb['valid_coarse'].flatten(1)
    corr = model.correlation(ta, tb, ma, mb)
    h, w = fa['F_coarse'].shape[-2:]
    # Matchability enters reliability exactly once, not via an already weighted probability.
    reliability, targets, support = target_support(corr, ma, mb, model.config)
    queries, valid = select_queries(reliability, support, w,
        model.config.max_source_queries, model.config.v3_spatial_bonus, model.config.v3_spatial_sigma_cells,
        underfill=model.config.v3_revision >= 1)
    targets = targets.gather(1, queries)
    source = flat_indices_to_image_xy(queries, w, model.config.coarse_stride)
    initial = flat_indices_to_image_xy(targets, w, model.config.coarse_stride)
    if queries.shape[1] == 0:
        # Do not invoke local CNN/grid_sample on phantom padded queries.
        zero = source[..., 0]
        output = GRMMatcherOutput(source, initial, zero, zero, zero, zero, zero, zero,
            source.new_empty(source.shape[0], 0, 2, 2),
            corr.matchability_a.reshape(-1, 1, h, w), corr.matchability_b.reshape(-1, 1, h, w),
            targets[..., None], zero[..., None], valid)
        auxiliary = V3Auxiliary(fa, fb, corr, queries, targets, None, None, None)
        return (output, auxiliary) if return_auxiliary else output
    mid, fine = model.refine_points(fa, fb, source, initial, valid)
    valid = valid & fine.valid_mask
    cycle = model.direct_cycle(fa, fb, matching_probability(corr, model.config), source, fine.dst_xy, valid)
    logits = confidence_logits(model, corr, queries, targets, mid, fine, cycle)
    score = logits.sigmoid() if logits is not None else reliability.gather(1, queries)
    entropy, _ = row_statistics(corr)
    output = GRMMatcherOutput(src_xy=source, dst_xy=fine.dst_xy, p_match=score*valid,
        p_global=valid.float(), p_local=torch.zeros_like(score), p_false=(~valid).float(),
        candidate_entropy=entropy.gather(1, queries), cycle_error=cycle.cycle_error,
        fine_uncertainty=fine.covariance_image, overlap_src=corr.matchability_a.reshape(-1, 1, h, w),
        overlap_dst=corr.matchability_b.reshape(-1, 1, h, w), candidate_indices=targets[..., None],
        candidate_probs=valid[..., None].float(), valid_mask=valid)
    auxiliary = V3Auxiliary(fa, fb, corr, queries, targets, mid, fine, logits)
    return (output, auxiliary) if return_auxiliary else output
