"""Interpretable staged losses for standalone GRM-Matcher training."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import GRMMatcherConfig
from .coordinates import (
    feature_xy_to_image_xy,
    image_xy_to_feature_xy,
    image_xy_to_normalized_grid,
)
from .outputs import GRMMatcherAuxiliary, GRMMatcherOutput
from .role_targets import build_geometry_role_targets


class GRMMatcherLoss(nn.Module):
    """Coarse, refinement, rerank, role, matchability, and Brier losses."""

    valid_stages = {"stage1", "stage2", "stage3", "joint"}

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        self.config = config
        self.eps = config.probability_eps

    def forward(
        self,
        output: GRMMatcherOutput,
        targets: Mapping[str, Tensor],
        auxiliary: GRMMatcherAuxiliary,
        *,
        stage: str = "joint",
        joint_source: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        if stage not in self.valid_stages:
            raise ValueError(f"stage must be one of {sorted(self.valid_stages)}, got {stage!r}.")
        self._validate_targets(targets, output.batch_size)

        coarse_loss = self._coarse_loss(targets, auxiliary)
        mid_loss = self._local_refinement_loss(
            targets,
            auxiliary.candidate_src_xy,
            auxiliary.candidate_initial_dst_xy,
            auxiliary.mid_dst_xy,
            auxiliary.mid_probability,
            auxiliary.candidate_valid_mask,
            stride=self.config.mid_stride,
        )
        fine_loss = self._local_refinement_loss(
            targets,
            auxiliary.candidate_src_xy,
            auxiliary.fine_initial_dst_xy,
            auxiliary.fine_dst_xy,
            auxiliary.fine_probability,
            auxiliary.candidate_valid_mask,
            stride=self.config.fine_stride,
        )
        rerank_loss = self._rerank_loss(targets, auxiliary)
        binary = self.config.classification_mode == "inlier"
        if binary:
            if auxiliary.inlier_logits is None:
                raise ValueError("Binary training requires inlier_logits")
            labels = self._inlier_targets(output, targets)
            role_loss = _masked_mean(F.binary_cross_entropy_with_logits(
                auxiliary.inlier_logits.float(), labels, reduction="none"), output.valid_mask)
        elif self.config.classification_mode == "role":
            role_loss = self._role_loss(output, targets, auxiliary.role_logits)
        else:
            role_loss = rerank_loss * 0.0
        classifier_weight = self.config.inlier_loss_weight if binary else self.config.role_loss_weight
        matchability_loss = self._matchability_loss(targets, auxiliary)
        calibration_loss = (rerank_loss * 0.0 if self.config.classification_mode == "none"
                            else self._calibration_loss(output, targets))

        zero = coarse_loss * 0.0
        if stage == "stage1":
            total = (
                self.config.coarse_loss_weight * coarse_loss
                + self.config.matchability_loss_weight * matchability_loss
            )
        elif stage == "stage2":
            total = (
                self.config.mid_loss_weight * mid_loss
                + self.config.fine_loss_weight * fine_loss
            )
        elif stage == "stage3":
            total = (
                self.config.rerank_loss_weight * rerank_loss
                + classifier_weight * role_loss
                + self.config.calibration_loss_weight * calibration_loss
            )
        elif joint_source in {"homography", "role_augmented"}:
            prefix = (
                "joint_homography"
                if joint_source == "homography"
                else "joint_role"
            )
            total = (
                getattr(self.config, f"{prefix}_coarse_loss_weight") * coarse_loss
                + getattr(self.config, f"{prefix}_mid_loss_weight") * mid_loss
                + getattr(self.config, f"{prefix}_fine_loss_weight") * fine_loss
                + getattr(self.config, f"{prefix}_rerank_loss_weight") * rerank_loss
                + (self.config.inlier_loss_weight if binary else getattr(self.config, f"{prefix}_role_loss_weight")) * role_loss
                + getattr(self.config, f"{prefix}_matchability_loss_weight")
                * matchability_loss
                + getattr(self.config, f"{prefix}_calibration_loss_weight")
                * calibration_loss
            )
        else:
            # Backward-compatible generic Joint behavior for callers that do
            # not annotate the batch source.
            total = (
                self.config.coarse_loss_weight * coarse_loss
                + self.config.mid_loss_weight * mid_loss
                + self.config.fine_loss_weight * fine_loss
                + self.config.rerank_loss_weight * rerank_loss
                + classifier_weight * role_loss
                + self.config.matchability_loss_weight * matchability_loss
                + self.config.calibration_loss_weight * calibration_loss
            )
        result = {
            "total": total,
            "coarse": coarse_loss,
            "mid": mid_loss,
            "fine": fine_loss,
            "rerank": rerank_loss,
            ("inlier" if binary else "role"): role_loss,
            "matchability": matchability_loss,
            "calibration": calibration_loss,
            "zero": zero,
        }
        if self.config.classification_mode == "none":
            result.pop("role")
            result.pop("calibration")
        return result

    def _coarse_loss(
        self, targets: Mapping[str, Tensor], auxiliary: GRMMatcherAuxiliary
    ) -> Tensor:
        batch = auxiliary.coarse_similarity.shape[0]
        device = auxiliary.coarse_similarity.device
        flat = torch.arange(
            auxiliary.coarse_height * auxiliary.coarse_width,
            device=device,
            dtype=torch.long,
        )[None].expand(batch, -1)
        source_xy = feature_xy_to_image_xy(
            torch.stack(
                (
                    torch.remainder(flat, auxiliary.coarse_width),
                    torch.div(flat, auxiliary.coarse_width, rounding_mode="floor"),
                ),
                dim=-1,
            ).float(),
            self.config.coarse_stride,
        )
        gt_dst = _sample_field(targets["gt_dense_mapping"], source_xy)
        overlap = _sample_field(targets["gt_overlap_a"].float(), source_xy)[..., 0] > 0.5
        target_feature_xy = image_xy_to_feature_xy(gt_dst, self.config.coarse_stride)
        target_cell_xy = target_feature_xy.round().long()
        inside = (
            (target_cell_xy[..., 0] >= 0)
            & (target_cell_xy[..., 0] < auxiliary.coarse_width)
            & (target_cell_xy[..., 1] >= 0)
            & (target_cell_xy[..., 1] < auxiliary.coarse_height)
        )
        target_indices = (
            target_cell_xy[..., 1].clamp(0, auxiliary.coarse_height - 1)
            * auxiliary.coarse_width
            + target_cell_xy[..., 0].clamp(0, auxiliary.coarse_width - 1)
        )
        logits = auxiliary.coarse_similarity
        target_logit = torch.gather(logits, -1, target_indices[..., None]).squeeze(-1)
        # Pixel overlap does not guarantee that the rounded coarse target cell
        # survives the feature validity mask. Never supervise a masked class.
        unmasked = (target_logit != torch.finfo(logits.dtype).min) & ~torch.isneginf(target_logit)
        valid = overlap & inside & torch.isfinite(gt_dst).all(-1) & unmasked
        if not valid.any():
            return logits.reshape(-1)[:0].sum()
        # Select before CE: invalid rows must not produce losses or gradients.
        return F.cross_entropy(logits[valid].float(), target_indices[valid])

    def _local_refinement_loss(
        self,
        targets: Mapping[str, Tensor],
        src_xy: Tensor,
        initial_dst_xy: Tensor,
        refined_dst_xy: Tensor,
        local_probability: Tensor,
        candidate_valid: Tensor,
        *,
        stride: int,
    ) -> Tensor:
        batch, queries, topk, _ = src_xy.shape
        flat_source = src_xy.reshape(batch, queries * topk, 2)
        gt_dst = _sample_field(targets["gt_dense_mapping"], flat_source).reshape_as(src_xy)
        overlap = _sample_field(
            targets["gt_overlap_a"].float(), flat_source
        )[..., 0].reshape(batch, queries, topk) > 0.5
        window_size = local_probability.shape[-1]
        radius = window_size // 2
        target_offset = (gt_dst - initial_dst_xy) / float(stride)
        nearest_offset = target_offset.round().long()
        reachable = (
            (nearest_offset[..., 0].abs() <= radius)
            & (nearest_offset[..., 1].abs() <= radius)
        )
        local_x = (nearest_offset[..., 0] + radius).clamp(0, window_size - 1)
        local_y = (nearest_offset[..., 1] + radius).clamp(0, window_size - 1)
        flat_index = local_y * window_size + local_x
        selected_probability = torch.gather(
            local_probability.flatten(-2), -1, flat_index[..., None]
        ).squeeze(-1)
        nll = -torch.log(selected_probability.clamp_min(self.eps))
        coordinate = F.smooth_l1_loss(
            refined_dst_xy,
            gt_dst,
            beta=self.config.coordinate_loss_beta,
            reduction="none",
        ).mean(dim=-1)
        valid = candidate_valid & overlap & reachable
        return _masked_mean(coordinate + nll, valid)

    def _rerank_loss(
        self, targets: Mapping[str, Tensor], auxiliary: GRMMatcherAuxiliary
    ) -> Tensor:
        batch, queries, topk, _ = auxiliary.candidate_src_xy.shape
        source = auxiliary.candidate_src_xy[:, :, 0]
        gt_dst = _sample_field(targets["gt_dense_mapping"], source)
        overlap = _sample_field(targets["gt_overlap_a"].float(), source)[..., 0] > 0.5
        error = torch.linalg.vector_norm(
            auxiliary.fine_dst_xy - gt_dst[:, :, None], dim=-1
        )
        error = error.masked_fill(
            ~auxiliary.candidate_valid_mask, torch.finfo(error.dtype).max
        )
        best_error, best_candidate = error.min(dim=-1)
        valid = overlap & (
            best_error <= self.config.correspondence_correct_threshold
        )
        losses = F.cross_entropy(
            auxiliary.rerank_logits.reshape(batch * queries, topk),
            best_candidate.reshape(-1),
            reduction="none",
        ).reshape(batch, queries)
        return _masked_mean(losses, valid)

    def _role_loss(
        self,
        output: GRMMatcherOutput,
        targets: Mapping[str, Tensor],
        role_logits: Tensor,
    ) -> Tensor:
        role_target = build_geometry_role_targets(
            output.src_xy,
            output.dst_xy,
            targets,
            correspondence_correct_threshold=(
                self.config.correspondence_correct_threshold
            ),
        ).labels
        valid = output.valid_mask
        if not valid.any():
            return role_logits.sum() * 0.0
        selected_target = role_target[valid]
        counts = torch.bincount(selected_target, minlength=3).float() + 1.0
        class_weights = counts.sum() / (3.0 * counts)
        return F.cross_entropy(role_logits[valid], selected_target, weight=class_weights)

    def _matchability_loss(
        self, targets: Mapping[str, Tensor], auxiliary: GRMMatcherAuxiliary
    ) -> Tensor:
        batch = auxiliary.matchability_a.shape[0]
        device = auxiliary.matchability_a.device
        flat = torch.arange(
            auxiliary.coarse_height * auxiliary.coarse_width,
            device=device,
            dtype=torch.long,
        )[None].expand(batch, -1)
        points = feature_xy_to_image_xy(
            torch.stack(
                (
                    torch.remainder(flat, auxiliary.coarse_width),
                    torch.div(flat, auxiliary.coarse_width, rounding_mode="floor"),
                ),
                dim=-1,
            ).float(),
            self.config.coarse_stride,
        )
        target_a = _sample_field(targets["gt_overlap_a"].float(), points)[..., 0]
        target_b = _sample_field(targets["gt_overlap_b"].float(), points)[..., 0]
        # The correlation module intentionally exposes sigmoid probabilities,
        # not logits. CUDA autocast rejects probability-space BCE, so keep this
        # small loss calculation in FP32 without changing the public semantics.
        with torch.autocast(device_type=device.type, enabled=False):
            loss_a = F.binary_cross_entropy(
                auxiliary.matchability_a.float().clamp(self.eps, 1.0 - self.eps),
                target_a.float(),
            )
            loss_b = F.binary_cross_entropy(
                auxiliary.matchability_b.float().clamp(self.eps, 1.0 - self.eps),
                target_b.float(),
            )
        return 0.5 * (loss_a + loss_b)

    def _calibration_loss(
        self, output: GRMMatcherOutput, targets: Mapping[str, Tensor]
    ) -> Tensor:
        if self.config.classification_mode == "inlier":
            correctness = self._inlier_targets(output, targets)
            return _masked_mean((output.p_match - correctness).square(), output.valid_mask)
        role_targets = build_geometry_role_targets(
            output.src_xy,
            output.dst_xy,
            targets,
            correspondence_correct_threshold=(
                self.config.correspondence_correct_threshold
            ),
        )
        correctness = role_targets.correct_mask.float()
        brier = (output.p_match - correctness).square()
        return _masked_mean(brier, output.valid_mask)

    @torch.no_grad()
    def _inlier_targets(self, output, targets):
        """GT correspondence correctness; no seed H or Global/Local annotations."""
        source, destination = output.src_xy.detach(), output.dst_xy.detach()
        mapping = targets["gt_dense_mapping"]
        gt = _sample_field(mapping, source)
        overlap = _sample_field(targets["gt_overlap_a"].float(), source)[..., 0] > .5
        height, width = mapping.shape[-2:]
        inside = ((source >= 0) & (source <= source.new_tensor([width-1, height-1]))).all(-1)
        finite = torch.isfinite(gt).all(-1) & torch.isfinite(destination).all(-1)
        correct = torch.linalg.vector_norm(destination-gt, dim=-1) <= self.config.correspondence_correct_threshold
        return (overlap & inside & finite & correct).float()

    def _validate_targets(self, targets: Mapping[str, Tensor], batch: int) -> None:
        required = {
            "gt_dense_mapping",
            "gt_overlap_a",
            "gt_overlap_b",
        }
        if self.config.classification_mode == "role":
            required.update(("gt_global_mask", "gt_local_mask"))
        missing = sorted(required - set(targets))
        if missing:
            raise ValueError(f"Training targets are missing keys: {missing}.")
        if targets["gt_dense_mapping"].ndim != 4:
            raise ValueError("gt_dense_mapping must be [B, 2, H, W].")
        if targets["gt_dense_mapping"].shape[0] != batch:
            raise ValueError("Target and output batch sizes must match.")


def _sample_field(field: Tensor, points_xy: Tensor) -> Tensor:
    """Bilinearly sample ``[B,C,H,W]`` at image ``[B,N,2]`` points."""

    if field.ndim != 4 or points_xy.ndim != 3 or points_xy.shape[-1] != 2:
        raise ValueError("field must be [B,C,H,W] and points_xy [B,N,2].")
    grid = image_xy_to_normalized_grid(
        points_xy.float(), field.shape[-2], field.shape[-1]
    ).unsqueeze(2)
    sampled = F.grid_sample(
        field.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return sampled.squeeze(-1).transpose(1, 2)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    # Multiplying by zero does not remove NaN/Inf. Select supervised entries;
    # empty sum preserves the gradient connection without reducing sentinels.
    selected = values.masked_select(mask.bool())
    return selected.sum() / max(selected.numel(), 1)
