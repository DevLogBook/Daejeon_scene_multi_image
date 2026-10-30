"""Losses and pseudo-label utilities for real-domain Stages 2.5-B and 3.5.

Stage 2.5-A remains in :mod:`grm_matcher.stage25`: it trains only the local
mid/fine path.  This file deliberately keeps coarse adaptation and geometry
adaptation explicit so neither can be enabled accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import GRMMatcherConfig
from .coordinates import image_xy_to_feature_xy
from .matcher import GRMMatcher
from .outputs import GRMMatcherAuxiliary, GRMMatcherOutput
from .stage25 import (
    GRMStage25CoarseOutput,
    GRMStage25Config,
    GRMStage25ConsistencyLoss,
    GRMStage25PseudoLabels,
    GRMStage25StudentOutput,
    transform_pseudo_labels,
)


@dataclass
class GRMStage35PseudoLabels:
    """Geometry pseudo labels: 0=Global, 1=Local, 2=False, -1=ignore."""

    matches: GRMStage25PseudoLabels
    role_labels: Tensor
    homography_valid: Tensor
    homography_residual: Tensor
    local_flow_consistent: Tensor


def forward_student_coarse(
    student: GRMMatcher,
    image_a: Tensor,
    image_b: Tensor,
    valid_a: Tensor | None = None,
    valid_b: Tensor | None = None,
) -> GRMStage25CoarseOutput:
    """Execute the Student backbone/FPN/Transformer/correlation path."""

    features_a, features_b, tokens_a, tokens_b = student.encode_pair(image_a,image_b,valid_a,valid_b)
    coarse_a = features_a["F_coarse"]
    correlation = student.correlation(tokens_a,tokens_b,features_a["valid_coarse"].flatten(1),features_b["valid_coarse"].flatten(1))
    height, width = coarse_a.shape[-2:]
    return GRMStage25CoarseOutput(
        similarity=correlation.similarity,
        matchability_a=correlation.matchability_a,
        matchability_b=correlation.matchability_b,
        coarse_height=int(height),
        coarse_width=int(width),
    )


class GRMStage25CoarsePseudoLoss(nn.Module):
    """Bidirectional coarse CE plus conservative positive matchability loss."""

    def __init__(
        self,
        matcher_config: GRMMatcherConfig,
        adaptation_config: GRMStage25Config,
    ) -> None:
        super().__init__()
        self.matcher_config = matcher_config
        self.config = adaptation_config

    def forward(
        self,
        student: GRMStage25CoarseOutput,
        forward_labels: GRMStage25PseudoLabels,
        backward_labels: GRMStage25PseudoLabels,
    ) -> Dict[str, Tensor]:
        forward = self._direction_loss(student, forward_labels, transpose=False)
        backward = self._direction_loss(student, backward_labels, transpose=True)
        matchability = self._positive_matchability_loss(
            student, forward_labels, backward_labels
        )
        total = (
            self.config.coarse_pseudo_ce_weight * forward
            + self.config.coarse_bidirectional_weight * backward
            + self.config.matchability_distillation_weight * matchability
        )
        return {
            "total": total,
            "coarse_forward": forward,
            "coarse_backward": backward,
            "matchability": matchability,
            "valid_count": (
                self._supervision_indices(student, forward_labels, False)[2].sum()
                + self._supervision_indices(student, backward_labels, True)[2].sum()
            ).to(dtype=total.dtype).detach(),
        }

    def _supervision_indices(self, student, labels, transpose):
        source, target, inside = _coarse_indices(
            labels, self.matcher_config.coarse_stride,
            student.coarse_height, student.coarse_width)
        logits = student.similarity.transpose(1, 2) if transpose else student.similarity
        batch = torch.arange(logits.shape[0], device=logits.device)[:, None]
        target_logit = logits[batch, source, target]
        # Pixel-valid labels may round onto an invalid coarse cell at a padding
        # or augmentation boundary. Correlation masks those pairs with finfo.min.
        # Do not teach CE (or positive matchability) to select a masked cell.
        # Keep NaNs/+inf visible to the existing nonfinite failure diagnostics.
        unmasked = (target_logit != torch.finfo(logits.dtype).min) & ~torch.isneginf(target_logit)
        valid = labels.valid_mask & inside & unmasked
        mask = _strict_coarse_mask(replace(labels, valid_mask=valid), self.config, student)
        return source, target, mask

    def _direction_loss(
        self,
        student: GRMStage25CoarseOutput,
        labels: GRMStage25PseudoLabels,
        *,
        transpose: bool,
    ) -> Tensor:
        source, target, mask = self._supervision_indices(student, labels, transpose)
        logits = student.similarity.transpose(1, 2) if transpose else student.similarity
        if not mask.any():
            # Masked logits contain finfo.min; summing even finite entries can
            # overflow to -inf, and -inf * 0 is NaN. Empty sum keeps autograd.
            return logits.reshape(-1)[:0].sum()
        batch = torch.arange(logits.shape[0], device=logits.device)[:, None]
        selected_logits = logits[batch.expand_as(source)[mask], source[mask]]
        per_item = F.cross_entropy(selected_logits.float(), target[mask], reduction="none")
        weights = labels.confidence[mask].float().clamp_min(self.config.probability_eps)
        weights = weights / weights.mean().clamp_min(self.config.probability_eps)
        return (per_item * weights).sum() / weights.sum().clamp_min(1.0)

    def _positive_matchability_loss(
        self,
        student: GRMStage25CoarseOutput,
        forward: GRMStage25PseudoLabels,
        backward: GRMStage25PseudoLabels,
    ) -> Tensor:
        source_a, _, mask_a = self._supervision_indices(student, forward, False)
        source_b, _, mask_b = self._supervision_indices(student, backward, True)
        losses = []
        for probability, indices, mask in (
            (student.matchability_a, source_a, mask_a),
            (student.matchability_b, source_b, mask_b),
        ):
            if mask.any():
                batch = torch.arange(probability.shape[0], device=probability.device)[:, None]
                positive = probability[batch.expand_as(indices)[mask], indices[mask]]
                # -log(p) is AMP-safe and does not incorrectly label unknown cells negative.
                losses.append(-torch.log(positive.float().clamp_min(self.config.probability_eps)).mean())
        return torch.stack(losses).mean() if losses else student.similarity.reshape(-1)[:0].sum()


class GRMStage25BCoarseAndRefinementLoss(nn.Module):
    """Stage 2.5-B objective with low-LR coarse and continued local adaptation."""

    def __init__(
        self,
        matcher_config: GRMMatcherConfig,
        adaptation_config: GRMStage25Config,
    ) -> None:
        super().__init__()
        self.coarse = GRMStage25CoarsePseudoLoss(matcher_config, adaptation_config)
        self.refinement = GRMStage25ConsistencyLoss(matcher_config, adaptation_config)

    def forward(
        self,
        coarse: GRMStage25CoarseOutput,
        refinement: GRMStage25StudentOutput,
        forward_labels: GRMStage25PseudoLabels,
        backward_labels: GRMStage25PseudoLabels,
    ) -> Dict[str, Tensor]:
        coarse_losses = self.coarse(coarse, forward_labels, backward_labels)
        refinement_losses = self.refinement(refinement, forward_labels)
        total = coarse_losses["total"] + refinement_losses["total"]
        result = {f"coarse_{key}": value for key, value in coarse_losses.items()}
        result.update({f"refine_{key}": value for key, value in refinement_losses.items()})
        result["total"] = total
        result["valid_count"] = coarse_losses["valid_count"]
        return result


@torch.no_grad()
def build_geometry_role_pseudo_labels(
    teacher: GRMMatcher,
    labels: GRMStage25PseudoLabels,
    config: GRMStage25Config,
) -> GRMStage35PseudoLabels:
    """Create conservative role labels without consulting Teacher role logits."""

    shape = labels.valid_mask.shape
    entropy = _optional(labels.normalized_entropy, labels.cycle_error.new_zeros(shape))
    mutual = _optional(labels.mutual_top1, torch.ones_like(labels.valid_mask))
    base_valid = _optional(labels.base_valid_mask, labels.valid_mask)
    seed_valid_matches = labels.valid_mask & mutual
    seed = teacher.seed_geometry(
        labels.src_xy,
        labels.fine_dst_xy,
        labels.confidence.clamp(0.0, 1.0),
        entropy,
        labels.cycle_error,
        seed_valid_matches,
        mutual,
    )
    residual = seed.forward_residual
    flow_consistent = _local_flow_consistency(
        labels.src_xy,
        labels.fine_dst_xy,
        seed_valid_matches,
        config.local_neighbor_count,
        config.local_flow_consistency_threshold,
    )
    homography_available = seed.homography_valid[:, None]
    reliable = labels.valid_mask & mutual
    global_mask = (
        reliable
        & homography_available
        & (residual <= config.global_residual_threshold)
    )
    local_mask = (
        reliable
        & homography_available
        & (residual > config.global_residual_threshold)
        & (residual <= config.local_residual_threshold)
        & flow_consistent
    )
    peak = _optional(labels.fine_peak_probability, labels.fine_std.new_ones(shape))
    evidence_count = (
        (labels.cycle_error >= config.false_cycle_error).to(torch.int32)
        + (entropy >= config.false_normalized_entropy).to(torch.int32)
        + (labels.fine_std >= config.false_fine_std).to(torch.int32)
        + (peak <= config.false_fine_peak_probability).to(torch.int32)
        + (~mutual).to(torch.int32)
    )
    false_mask = base_valid & ~reliable & (evidence_count >= 2)
    role = torch.full(shape, -1, device=labels.src_xy.device, dtype=torch.long)
    role[global_mask] = 0
    role[local_mask] = 1
    role[false_mask] = 2
    return GRMStage35PseudoLabels(
        matches=labels,
        role_labels=role,
        homography_valid=seed.homography_valid,
        homography_residual=residual,
        local_flow_consistent=flow_consistent,
    )


class GRMStage35GeometryLoss(nn.Module):
    """Align Teacher source points to Student queries and train Stage-3 heads."""

    def __init__(self, config: GRMStage25Config) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        output: GRMMatcherOutput,
        auxiliary: GRMMatcherAuxiliary,
        pseudo: GRMStage35PseudoLabels,
    ) -> Dict[str, Tensor]:
        student_src = auxiliary.candidate_src_xy[:, :, 0].float()
        teacher_src = pseudo.matches.src_xy.float()
        teacher_valid = pseudo.role_labels >= 0
        distance = torch.cdist(student_src, teacher_src)
        distance = distance.masked_fill(~teacher_valid[:, None, :], torch.finfo(distance.dtype).max)
        nearest_distance, nearest_index = distance.min(dim=-1)
        aligned = nearest_distance <= self.config.geometry_source_alignment_radius
        batch = torch.arange(student_src.shape[0], device=student_src.device)[:, None]
        teacher_role = pseudo.role_labels[batch, nearest_index]
        teacher_target = pseudo.matches.fine_dst_xy[batch, nearest_index]
        candidate_error = torch.linalg.vector_norm(
            auxiliary.fine_dst_xy.float() - teacher_target[:, :, None], dim=-1
        )
        candidate_error = candidate_error.masked_fill(
            ~auxiliary.candidate_valid_mask, torch.finfo(candidate_error.dtype).max
        )
        target_error, target_k = candidate_error.min(dim=-1)
        true_role = (teacher_role == 0) | (teacher_role == 1)
        rerank_mask = (
            aligned
            & true_role
            & (target_error <= self.config.geometry_rerank_target_radius)
        )
        rerank = _masked_cross_entropy(auxiliary.rerank_logits, target_k, rerank_mask)

        selected_correct = auxiliary.selected_indices == target_k
        role_mask = aligned & ((teacher_role == 2) | (true_role & rerank_mask & selected_correct))
        role = _masked_cross_entropy(auxiliary.role_logits, teacher_role.clamp_min(0), role_mask)

        calibration_target = torch.where(
            teacher_role == 2,
            torch.zeros_like(output.p_match),
            (rerank_mask & selected_correct).float(),
        )
        calibration_mask = aligned & (teacher_role >= 0)
        calibration = _masked_mean(
            (output.p_match.float() - calibration_target).square(), calibration_mask
        )
        total = (
            self.config.rerank_pseudo_weight * rerank
            + self.config.role_pseudo_weight * role
            + self.config.calibration_pseudo_weight * calibration
        )
        return {
            "total": total,
            "rerank": rerank,
            "role": role,
            "calibration": calibration,
            "rerank_count": rerank_mask.sum().to(total.dtype).detach(),
            "role_count": role_mask.sum().to(total.dtype).detach(),
            "global_count": (role_mask & (teacher_role == 0)).sum().to(total.dtype).detach(),
            "local_count": (role_mask & (teacher_role == 1)).sum().to(total.dtype).detach(),
            "false_count": (role_mask & (teacher_role == 2)).sum().to(total.dtype).detach(),
        }


def transform_geometry_role_pseudo_labels(
    pseudo: GRMStage35PseudoLabels,
    transform_a: Tensor,
    transform_b: Tensor,
    height: int,
    width: int,
    matcher_config: GRMMatcherConfig,
    config: GRMStage25Config,
) -> GRMStage35PseudoLabels:
    """Move geometry pseudo coordinates to the Student view and keep ignore safe."""

    transformed = transform_pseudo_labels(
        pseudo.matches,
        transform_a,
        transform_b,
        height,
        width,
        matcher_config,
        config,
    )
    inside = (
        (transformed.src_xy[..., 0] >= 0.0)
        & (transformed.src_xy[..., 0] <= width - 1)
        & (transformed.src_xy[..., 1] >= 0.0)
        & (transformed.src_xy[..., 1] <= height - 1)
        & (transformed.fine_dst_xy[..., 0] >= 0.0)
        & (transformed.fine_dst_xy[..., 0] <= width - 1)
        & (transformed.fine_dst_xy[..., 1] >= 0.0)
        & (transformed.fine_dst_xy[..., 1] <= height - 1)
    )
    roles = pseudo.role_labels.clone()
    roles[~inside] = -1
    return GRMStage35PseudoLabels(
        matches=transformed,
        role_labels=roles,
        homography_valid=pseudo.homography_valid,
        homography_residual=pseudo.homography_residual,
        local_flow_consistent=pseudo.local_flow_consistent,
    )


def _coarse_indices(
    labels: GRMStage25PseudoLabels,
    stride: int,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    source_xy = image_xy_to_feature_xy(labels.src_xy.float(), stride).round().long()
    target_xy = image_xy_to_feature_xy(labels.fine_dst_xy.float(), stride).round().long()
    inside = (
        (source_xy[..., 0] >= 0)
        & (source_xy[..., 0] < width)
        & (source_xy[..., 1] >= 0)
        & (source_xy[..., 1] < height)
        & (target_xy[..., 0] >= 0)
        & (target_xy[..., 0] < width)
        & (target_xy[..., 1] >= 0)
        & (target_xy[..., 1] < height)
    )
    source = source_xy[..., 1].clamp(0, height - 1) * width + source_xy[..., 0].clamp(0, width - 1)
    target = target_xy[..., 1].clamp(0, height - 1) * width + target_xy[..., 0].clamp(0, width - 1)
    return source, target, inside


def _strict_coarse_mask(
    labels: GRMStage25PseudoLabels,
    config: GRMStage25Config,
    student: GRMStage25CoarseOutput,
) -> Tensor:
    mask = labels.valid_mask.clone()
    if config.coarse_require_mutual_top1 and labels.mutual_top1 is not None:
        mask &= labels.mutual_top1
    # Teacher sampler region IDs already encode the dynamic coarse grid and are
    # preserved through Student augmentation. No image-size or stride constant
    # is reintroduced here.
    for batch_index in range(mask.shape[0]):
        indices = torch.nonzero(mask[batch_index], as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            continue
        if labels.region_ids is None:
            region_id = indices
        else:
            region_id = labels.region_ids[batch_index, indices]
        enough = (
            indices.numel() >= config.coarse_min_pseudo_matches
            and torch.unique(region_id).numel() >= config.coarse_min_covered_regions
        )
        if not enough:
            mask[batch_index] = False
    return mask


def _local_flow_consistency(
    src_xy: Tensor,
    dst_xy: Tensor,
    valid: Tensor,
    neighbors: int,
    threshold: float,
) -> Tensor:
    result = torch.zeros_like(valid)
    displacement = dst_xy.float() - src_xy.float()
    for batch_index in range(src_xy.shape[0]):
        indices = torch.nonzero(valid[batch_index], as_tuple=False).squeeze(-1)
        if indices.numel() < 2:
            continue
        source = src_xy[batch_index, indices].float()
        distance = torch.cdist(source, source)
        distance.fill_diagonal_(torch.finfo(distance.dtype).max)
        count = min(neighbors, int(indices.numel()) - 1)
        nearest = torch.topk(distance, count, largest=False).indices
        flows = displacement[batch_index, indices]
        neighbor_flows = flows[nearest]
        difference = torch.linalg.vector_norm(neighbor_flows - flows[:, None], dim=-1)
        result[batch_index, indices] = difference.median(dim=-1).values <= threshold
    return result


def _optional(value: Tensor | None, fallback: Tensor) -> Tensor:
    return fallback if value is None else value


def _masked_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits.float()[mask], target[mask])


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return values.sum() * 0.0
    return values[mask].mean()
