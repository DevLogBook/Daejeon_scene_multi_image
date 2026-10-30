"""Stage-2 refinement metrics independent of geometry, reranking, and role heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import GRMMatcherConfig
from .coordinates import image_xy_to_normalized_grid
from .outputs import GRMMatcherAuxiliary


@dataclass
class GRMStage2Predictions:
    """Candidate coordinates needed to evaluate coarse -> mid -> fine refinement."""

    src_xy: Tensor  # [B, M, K, 2]
    coarse_dst_xy: Tensor  # [B, M, K, 2]
    mid_dst_xy: Tensor  # [B, M, K, 2]
    fine_dst_xy: Tensor  # [B, M, K, 2]
    candidate_valid_mask: Tensor  # [B, M, K]

    @classmethod
    def from_auxiliary(cls, auxiliary: GRMMatcherAuxiliary) -> "GRMStage2Predictions":
        return cls(
            src_xy=auxiliary.candidate_src_xy,
            coarse_dst_xy=auxiliary.candidate_initial_dst_xy,
            mid_dst_xy=auxiliary.mid_dst_xy,
            fine_dst_xy=auxiliary.fine_dst_xy,
            candidate_valid_mask=auxiliary.candidate_valid_mask,
        )


@dataclass
class GRMStage2MetricComponents:
    """Numerators and denominators supporting exact aggregation across batches."""

    numerators: Dict[str, Tensor]
    denominators: Dict[str, Tensor]
    counts: Dict[str, Tensor]

    def metrics(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name, numerator in self.numerators.items():
            denominator = self.denominators[name]
            result[name] = (
                float((numerator / denominator.clamp_min(1.0)).detach().cpu())
                if float(denominator.detach().cpu()) > 0.0
                else 0.0
            )
        result.update(
            {name: float(value.detach().cpu()) for name, value in self.counts.items()}
        )
        return result


class GRMStage2MetricAccumulator:
    """CPU accumulator used by the independent Stage-2 validator."""

    def __init__(self) -> None:
        self.numerators: Dict[str, float] = {}
        self.denominators: Dict[str, float] = {}
        self.counts: Dict[str, float] = {}

    def update(self, components: GRMStage2MetricComponents) -> None:
        for name, value in components.numerators.items():
            self.numerators[name] = self.numerators.get(name, 0.0) + float(
                value.detach().double().cpu()
            )
        for name, value in components.denominators.items():
            self.denominators[name] = self.denominators.get(name, 0.0) + float(
                value.detach().double().cpu()
            )
        for name, value in components.counts.items():
            self.counts[name] = self.counts.get(name, 0.0) + float(
                value.detach().double().cpu()
            )

    def metrics(self) -> Dict[str, float]:
        result = {
            name: self.numerators[name] / denominator if denominator > 0.0 else 0.0
            for name, denominator in self.denominators.items()
        }
        result.update(self.counts)
        return result


def compute_stage2_metric_components(
    predictions: GRMStage2Predictions,
    targets: Mapping[str, Tensor],
    config: GRMMatcherConfig,
) -> GRMStage2MetricComponents:
    """Measure refinement without executing or consulting Stage-3 modules.

    ``top1`` always means candidate index zero from the Stage-1 probability
    ranking. ``oracle`` chooses one candidate per query using the smallest
    *coarse* GT error, then follows that same candidate through mid and fine.
    This avoids artificial candidate switching between refinement levels.
    """

    _validate_predictions(predictions)
    if "gt_dense_mapping" not in targets or "gt_overlap_a" not in targets:
        raise KeyError("Stage-2 metrics require gt_dense_mapping and gt_overlap_a.")

    source = predictions.src_xy[:, :, 0]
    gt_dst = _sample_dense_field(targets["gt_dense_mapping"], source)
    overlap = _sample_dense_field(targets["gt_overlap_a"].float(), source)[..., 0] > 0.5
    valid = predictions.candidate_valid_mask & overlap[..., None]

    coarse_error = torch.linalg.vector_norm(
        predictions.coarse_dst_xy - gt_dst[..., None, :], dim=-1
    )
    mid_error = torch.linalg.vector_norm(
        predictions.mid_dst_xy - gt_dst[..., None, :], dim=-1
    )
    fine_error = torch.linalg.vector_norm(
        predictions.fine_dst_xy - gt_dst[..., None, :], dim=-1
    )

    top1_valid = valid[..., 0]
    oracle_valid = valid.any(dim=-1)
    masked_coarse = coarse_error.masked_fill(~valid, torch.finfo(coarse_error.dtype).max)
    oracle_index = masked_coarse.argmin(dim=-1)

    coarse_oracle = _gather_candidate(coarse_error, oracle_index)
    mid_oracle = _gather_candidate(mid_error, oracle_index)
    fine_oracle = _gather_candidate(fine_error, oracle_index)

    mid_radius_px = (config.mid_window_size // 2) * config.mid_stride
    fine_radius_px = (config.fine_window_size // 2) * config.fine_stride
    mid_offset = (gt_dst[..., None, :] - predictions.coarse_dst_xy).abs()
    fine_offset = (gt_dst[..., None, :] - predictions.mid_dst_xy).abs()
    mid_reachable = valid & (mid_offset <= float(mid_radius_px)).all(dim=-1)
    fine_reachable = valid & (fine_offset <= float(fine_radius_px)).all(dim=-1)
    mid_oracle_reachable = _gather_candidate(mid_reachable, oracle_index)
    fine_oracle_reachable = _gather_candidate(fine_reachable, oracle_index)

    numerators: Dict[str, Tensor] = {}
    denominators: Dict[str, Tensor] = {}

    def add_mean(name: str, values: Tensor, mask: Tensor) -> None:
        numerators[name] = values.masked_select(mask).sum()
        denominators[name] = mask.sum().to(dtype=values.dtype)

    add_mean("coarse_top1_pixel_error", coarse_error[..., 0], top1_valid)
    add_mean("mid_top1_pixel_error", mid_error[..., 0], top1_valid)
    add_mean("fine_top1_pixel_error", fine_error[..., 0], top1_valid)
    add_mean("coarse_oracle_pixel_error", coarse_oracle, oracle_valid)
    add_mean("mid_oracle_pixel_error", mid_oracle, oracle_valid)
    add_mean("fine_oracle_pixel_error", fine_oracle, oracle_valid)

    add_mean(
        "mid_top1_reachable_ratio",
        mid_reachable[..., 0].to(dtype=coarse_error.dtype),
        top1_valid,
    )
    add_mean(
        "fine_top1_reachable_ratio",
        fine_reachable[..., 0].to(dtype=coarse_error.dtype),
        top1_valid,
    )
    add_mean(
        "mid_oracle_reachable_ratio",
        mid_oracle_reachable.to(dtype=coarse_error.dtype),
        oracle_valid,
    )
    add_mean(
        "fine_oracle_reachable_ratio",
        fine_oracle_reachable.to(dtype=coarse_error.dtype),
        oracle_valid,
    )
    add_mean(
        "mid_candidate_reachable_ratio",
        mid_reachable.to(dtype=coarse_error.dtype),
        valid,
    )
    add_mean(
        "fine_candidate_reachable_ratio",
        fine_reachable.to(dtype=coarse_error.dtype),
        valid,
    )
    top1_mid_reachable = mid_reachable[..., 0] & top1_valid
    numerators["fine_top1_reachable_given_mid_ratio"] = (
        fine_reachable[..., 0] & top1_mid_reachable
    ).sum().to(dtype=coarse_error.dtype)
    denominators["fine_top1_reachable_given_mid_ratio"] = top1_mid_reachable.sum().to(
        dtype=coarse_error.dtype
    )

    stage_errors = {
        "coarse": (coarse_error[..., 0], coarse_oracle),
        "mid": (mid_error[..., 0], mid_oracle),
        "fine": (fine_error[..., 0], fine_oracle),
    }
    for stage_name, (top1_error, oracle_error) in stage_errors.items():
        for threshold in (1.0, 3.0, 5.0):
            suffix = str(int(threshold))
            add_mean(
                f"{stage_name}_top1_pck_{suffix}",
                (top1_error <= threshold).to(dtype=top1_error.dtype),
                top1_valid,
            )
            add_mean(
                f"{stage_name}_oracle_pck_{suffix}",
                (oracle_error <= threshold).to(dtype=oracle_error.dtype),
                oracle_valid,
            )

    counts = {
        "stage2_top1_query_count": top1_valid.sum().to(dtype=coarse_error.dtype),
        "stage2_oracle_query_count": oracle_valid.sum().to(dtype=coarse_error.dtype),
        "stage2_valid_candidate_count": valid.sum().to(dtype=coarse_error.dtype),
    }
    return GRMStage2MetricComponents(numerators, denominators, counts)


def _sample_dense_field(field: Tensor, points_xy: Tensor) -> Tensor:
    if field.ndim != 4 or points_xy.ndim != 3 or points_xy.shape[-1] != 2:
        raise ValueError("field must be [B,C,H,W] and points_xy [B,N,2].")
    grid = image_xy_to_normalized_grid(
        points_xy.float(), field.shape[-2], field.shape[-1]
    ).unsqueeze(2)
    return F.grid_sample(
        field.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(-1).transpose(1, 2)


def _gather_candidate(values: Tensor, indices: Tensor) -> Tensor:
    return torch.gather(values, 2, indices[..., None]).squeeze(-1)


def _validate_predictions(predictions: GRMStage2Predictions) -> None:
    expected = predictions.src_xy.shape
    if len(expected) != 4 or expected[-1] != 2:
        raise ValueError("Stage-2 coordinate tensors must be [B,M,K,2].")
    for name in ("coarse_dst_xy", "mid_dst_xy", "fine_dst_xy"):
        if getattr(predictions, name).shape != expected:
            raise ValueError(f"{name} must match src_xy shape {expected}.")
    if predictions.candidate_valid_mask.shape != expected[:-1]:
        raise ValueError("candidate_valid_mask must be [B,M,K].")
    if predictions.candidate_valid_mask.dtype != torch.bool:
        raise TypeError("candidate_valid_mask must be bool.")
