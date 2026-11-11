"""Opt-in, fixed-weight second-mode probe for GRM V3 inference."""

from dataclasses import dataclass

import torch
from torch.nn.functional import normalize

from .architecture_v3 import confidence_logits, matching_probability
from .coordinates import flat_indices_to_image_xy


@dataclass
class SecondMode:
    slots: torch.Tensor
    center: torch.Tensor
    destination: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    replace: torch.Tensor


@dataclass
class CandidateRoutes:
    slots: torch.Tensor
    deployed_targets: torch.Tensor


@torch.inference_mode()
def route_difficult_queries(model, output, aux, *, budget=32, separation_px=24.,
                            min_score_ratio=.30, weak_cycle_px=4.):
    """Share difficult source slots between the deployed-score and FPN arms."""
    config = model.config
    if (config.architecture_version != 'v3' or config.v3_revision < 1
            or config.adaptive_enabled or output.src_xy.shape[0] != 1):
        raise ValueError('Candidate fallback requires batch 1 and fixed-window V3 revision 1+')
    slots = output.valid_mask[0].nonzero(as_tuple=True)[0]
    if not slots.numel() or budget <= 0:
        return CandidateRoutes(slots[:0], slots[:0])
    queries, first = aux.query_indices[0, slots], aux.target_indices[0, slots]
    scores = aux.correlation.probability[0, queries]
    width = aux.features_b['F_coarse'].shape[-1]
    ids = torch.arange(scores.shape[1], device=slots.device)
    separated = _separated(ids[None], first, width, config.coarse_stride, separation_px)
    valid_targets = aux.features_b['valid_coarse'][0].flatten()
    second_score, second = scores.masked_fill(~(separated & valid_targets), 0.).max(-1)
    first_score = scores.gather(1, first[:, None]).squeeze(1)
    ratio = second_score / first_score.clamp_min(torch.finfo(scores.dtype).tiny)
    weak_cycle = output.cycle_error[0, slots] > weak_cycle_px
    eligible = ((second_score > 0) & (weak_cycle | (ratio >= min_score_ratio))).nonzero(as_tuple=True)[0]
    # Prioritize weak cycles, then the existence of a competitive distant mode.
    priority = weak_cycle.float() + ratio.clamp(0., 1.)
    chosen = eligible[torch.argsort(priority[eligible], descending=True, stable=True)[:budget]]
    return CandidateRoutes(slots[chosen], second[chosen])


def _separated(targets, first, width, stride, separation_px):
    dx = (targets % width - (first % width)[:, None]).abs()
    dy = (targets // width - (first // width)[:, None]).abs()
    return torch.maximum(dx, dy) * stride >= separation_px


@torch.inference_mode()
def probe_routed_candidate(model, output, aux, routes, *, source='fpn',
                           separation_px=24., cycle_limit_px=8.,
                           cycle_margin_px=2., confidence_margin=.02):
    """Compare one candidate from either score on the shared routed queries."""
    slots = routes.slots
    if not slots.numel():
        return None
    config = model.config
    queries = aux.query_indices[0, slots]
    width = aux.features_b['F_coarse'].shape[-1]
    target = routes.deployed_targets
    if source == 'fpn':
        fa = normalize(aux.features_a['F_coarse'][0].flatten(1).T[queries].float(), dim=-1)
        fb = normalize(aux.features_b['F_coarse'][0].flatten(1).T.float(), dim=-1)
        valid_targets = aux.features_b['valid_coarse'][0].flatten()
        scores = (fa @ fb.T).masked_fill(~valid_targets, -torch.inf)
        top = scores.topk(min(3, scores.shape[1]), dim=-1).indices
        first = aux.target_indices[0, slots]
        eligible = _separated(top, first, width, config.coarse_stride, separation_px) & valid_targets[top]
        # Among the top three, keep the highest-ranked distinct spatial mode.
        target = top.gather(1, eligible.long().argmax(1, keepdim=True)).squeeze(1)
        keep = eligible.any(dim=1)
        slots, queries, target = slots[keep], queries[keep], target[keep]
        if not slots.numel():
            return None
    elif source != 'deployed':
        raise ValueError(f'Unknown candidate source: {source}')
    correlation = aux.correlation
    source_xy = output.src_xy[:, slots]
    center = flat_indices_to_image_xy(target[None], width, config.coarse_stride)
    mid, fine = model.refine_points(aux.features_a, aux.features_b, source_xy, center,
                                    torch.ones_like(target[None], dtype=torch.bool))
    cycle = model.direct_cycle(aux.features_a, aux.features_b, correlation.probability,
                               source_xy, fine.dst_xy, fine.valid_mask, correlation=correlation)
    logits = confidence_logits(model, correlation, queries[None], target[None], mid, fine, cycle)
    if logits is None:
        raise ValueError('Candidate fallback requires the V3 confidence head')
    confidence = logits.sigmoid()[0]
    valid = fine.valid_mask[0]
    cycle_delta = output.cycle_error[0, slots] - cycle.cycle_error[0]
    confidence_delta = confidence - output.p_match[0, slots]
    improvement = ((cycle_delta >= cycle_margin_px) & (confidence_delta >= -confidence_margin))
    improvement |= (cycle_delta >= -.5) & (confidence_delta > confidence_margin)
    replace = valid & cycle.valid_mask[0] & (cycle.cycle_error[0] <= cycle_limit_px) & improvement
    return SecondMode(slots, center[0], fine.dst_xy[0], confidence, valid, replace)


@torch.inference_mode()
def probe_second_mode(model, output, aux, *, budget=64, separation_px=24.,
                      min_score_ratio=.30, confidence_margin=.02, cycle_limit_px=8.):
    """Refine one separated alternative per ambiguous source, without using GT."""
    config = model.config
    if output.src_xy.shape[0] != 1 or config.v3_revision < 1 or getattr(config, 'adaptive_enabled', False):
        raise ValueError('Second-mode probe requires batch 1 and fixed-window GRM V3 revision 1+')
    slots = output.valid_mask[0].nonzero(as_tuple=True)[0]
    if not slots.numel() or not budget:
        return None

    correlation = aux.correlation
    queries = aux.query_indices[0, slots]
    first = aux.target_indices[0, slots]
    height, width = aux.features_b['F_coarse'].shape[-2:]
    stride = config.coarse_stride
    target_ids = torch.arange(height * width, device=slots.device)
    dx = (target_ids % width - (first % width)[:, None]).abs()
    dy = (target_ids // width - (first // width)[:, None]).abs()
    separated = torch.maximum(dx, dy) * stride >= separation_px
    target_ok = aux.features_b['valid_coarse'].flatten(1)[0].bool()
    target_ok = target_ok & (correlation.matchability_b[0] > config.v3_min_matchability)
    scores = (correlation.dual_probability[0, queries]
              * correlation.matchability_a[0, queries, None]
              * correlation.matchability_b[0, None, :])
    first_scores = scores.gather(1, first[:, None]).squeeze(1)
    alt_scores, alt = scores.masked_fill(~(separated & target_ok), 0.).max(dim=1)
    ratios = alt_scores / first_scores.clamp_min(torch.finfo(scores.dtype).tiny)
    # Stable ties preserve the original source query order.
    candidates = ((alt_scores > 0) & (ratios >= min_score_ratio)).nonzero(as_tuple=True)[0]
    if not candidates.numel():
        return None
    chosen = candidates[torch.argsort(ratios[candidates], descending=True, stable=True)[:budget]]
    slots, queries, alt = slots[chosen], queries[chosen][None], alt[chosen][None]
    source = output.src_xy[:, slots]
    center = flat_indices_to_image_xy(alt, width, stride)
    valid = torch.ones_like(alt, dtype=torch.bool)
    mid, fine = model.refine_points(aux.features_a, aux.features_b, source, center, valid)
    valid = fine.valid_mask
    cycle = model.direct_cycle(aux.features_a, aux.features_b, matching_probability(correlation, config),
                               source, fine.dst_xy, valid, correlation=correlation)
    logits = confidence_logits(model, correlation, queries, alt, mid, fine, cycle)
    if logits is None:
        raise ValueError('Second-mode probe requires the learned V3 confidence head')
    confidence = logits.sigmoid()
    replace = (valid & (cycle.cycle_error <= cycle_limit_px)
               & (confidence > output.p_match[:, slots] + confidence_margin))
    return SecondMode(slots, center[0], fine.dst_xy[0], confidence[0], valid[0], replace[0])
