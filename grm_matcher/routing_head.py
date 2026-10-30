"""Independent soft geometry routing head and offline target construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class GRMGeometryRoutingConfig:
    """Architecture and soft-label thresholds for the optional routing branch."""

    hidden_dim: int = 128
    sigma_h_px: float = 3.0
    global_residual_threshold_px: float = 3.0
    max_deform_threshold_px: float = 24.0
    ambiguous_margin_px: float = 1.0
    local_neighbors: int = 8
    local_consistency_threshold_px: float = 4.0
    minimum_match_probability: float = 0.2
    maximum_cycle_error_px: Optional[float] = None

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.local_neighbors <= 0:
            raise ValueError("hidden_dim and local_neighbors must be positive.")
        if min(
            self.sigma_h_px,
            self.global_residual_threshold_px,
            self.max_deform_threshold_px,
            self.ambiguous_margin_px,
            self.local_consistency_threshold_px,
        ) <= 0.0:
            raise ValueError("Routing geometry thresholds must be positive.")
        if self.max_deform_threshold_px <= self.global_residual_threshold_px:
            raise ValueError("max_deform_threshold_px must exceed the global threshold.")
        if not 0.0 <= self.minimum_match_probability <= 1.0:
            raise ValueError("minimum_match_probability must lie in [0,1].")


@dataclass
class GRMGeometryRoutingOutput:
    """Two non-exclusive per-match geometry support predictions."""

    h_support_logits: Tensor
    deform_support_logits: Tensor
    w_h: Tensor
    w_deform: Tensor
    valid_mask: Tensor

    def validate(self) -> None:
        expected = self.valid_mask.shape
        for name in ("h_support_logits", "deform_support_logits", "w_h", "w_deform"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must match valid_mask shape {expected}.")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean.")


@dataclass
class GRMGeometryRoutingTargets:
    """Soft labels and independent ignore masks for routing supervision."""

    target_h: Tensor
    target_deform: Tensor
    h_mask: Tensor
    deform_mask: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


class GRMGeometryRoutingHead(nn.Module):
    """Predict independent H support and static-deformation support."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.head_h = nn.Linear(self.hidden_dim, 1)
        self.head_deform = nn.Linear(self.hidden_dim, 1)

    def forward(self, selected_graph_tokens: Tensor, valid_mask: Tensor) -> GRMGeometryRoutingOutput:
        if (
            selected_graph_tokens.ndim != 3
            or selected_graph_tokens.shape[-1] != self.hidden_dim
            or valid_mask.shape != selected_graph_tokens.shape[:2]
        ):
            raise ValueError(
                "selected_graph_tokens must be [B,N,C] and valid_mask [B,N]."
            )
        hidden = self.trunk(selected_graph_tokens)
        h_logits = self.head_h(hidden).squeeze(-1)
        deform_logits = self.head_deform(hidden).squeeze(-1)
        w_h = torch.sigmoid(h_logits.float())
        w_deform = torch.sigmoid(deform_logits.float())
        w_h = torch.where(valid_mask, w_h, torch.zeros_like(w_h))
        w_deform = torch.where(valid_mask, w_deform, torch.zeros_like(w_deform))
        output = GRMGeometryRoutingOutput(
            h_support_logits=h_logits,
            deform_support_logits=deform_logits,
            w_h=w_h,
            w_deform=w_deform,
            valid_mask=valid_mask,
        )
        output.validate()
        return output


def fallback_geometry_routing(
    valid_mask: Tensor,
    reliability: Tensor,
    *,
    legacy_p_local: Optional[Tensor] = None,
    prefer_legacy_local: bool = False,
) -> GRMGeometryRoutingOutput:
    """Explicit no-checkpoint fallback: ``w_H=1`` and deform=reliability/legacy."""

    if valid_mask.shape != reliability.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask and reliability must have matching [B,N] shapes.")
    deform = (
        legacy_p_local
        if prefer_legacy_local and legacy_p_local is not None
        else reliability
    )
    if deform.shape != valid_mask.shape:
        raise ValueError("legacy_p_local must match valid_mask.")
    w_h = valid_mask.float()
    w_deform = torch.where(valid_mask, deform.float().clamp(0.0, 1.0), 0.0)
    eps = torch.finfo(w_deform.dtype).eps
    return GRMGeometryRoutingOutput(
        h_support_logits=torch.where(valid_mask, torch.full_like(w_h, 20.0), -20.0),
        deform_support_logits=torch.logit(w_deform.clamp(eps, 1.0 - eps)),
        w_h=w_h,
        w_deform=w_deform,
        valid_mask=valid_mask,
    )


def _apply_h(points_xy: Tensor, h_ab: Tensor) -> Tensor:
    if points_xy.ndim != 2 or points_xy.shape[-1] != 2 or h_ab.shape != (3, 3):
        raise ValueError("points_xy must be [N,2] and h_ab [3,3].")
    homogeneous = torch.cat((points_xy.float(), torch.ones_like(points_xy[:, :1])), dim=-1)
    projected = homogeneous @ h_ab.float().T
    denominator = projected[:, 2:3]
    sign = torch.where(denominator >= 0.0, 1.0, -1.0)
    denominator = torch.where(denominator.abs() > 1e-8, denominator, sign * 1e-8)
    return projected[:, :2] / denominator


@torch.no_grad()
def build_geometry_routing_targets(
    src_xy: Tensor,
    dst_xy: Tensor,
    h_ab: Tensor,
    p_match: Tensor,
    cycle_error: Tensor,
    uncertainty_score: Tensor,
    valid_mask: Tensor,
    *,
    config: Optional[GRMGeometryRoutingConfig] = None,
) -> GRMGeometryRoutingTargets:
    """Construct mask-aware offline soft targets; ambiguous matches are ignored."""

    cfg = config or GRMGeometryRoutingConfig()
    count = src_xy.shape[0]
    if src_xy.shape != (count, 2) or dst_xy.shape != (count, 2):
        raise ValueError("src_xy and dst_xy must be [N,2].")
    for name, value in (
        ("p_match", p_match),
        ("cycle_error", cycle_error),
        ("uncertainty_score", uncertainty_score),
        ("valid_mask", valid_mask),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must be [N].")
    residual_xy = dst_xy.float() - _apply_h(src_xy, h_ab)
    residual = torch.linalg.vector_norm(residual_xy, dim=-1)
    target_h = torch.exp(-residual.square() / (2.0 * cfg.sigma_h_px**2))
    reliable = (
        valid_mask
        & torch.isfinite(src_xy).all(dim=-1)
        & torch.isfinite(dst_xy).all(dim=-1)
        & (p_match >= cfg.minimum_match_probability)
        & torch.isfinite(cycle_error)
        & torch.isfinite(uncertainty_score)
        & (uncertainty_score > 0.0)
    )
    if cfg.maximum_cycle_error_px is not None:
        reliable &= cycle_error <= cfg.maximum_cycle_error_px

    if count > 1:
        distance = torch.cdist(src_xy.float()[None], src_xy.float()[None])[0]
        distance.fill_diagonal_(float("inf"))
        neighbors = min(cfg.local_neighbors, count - 1)
        indices = torch.topk(distance, neighbors, largest=False).indices
        neighbor_residual = residual_xy[indices]
        local_center = neighbor_residual.median(dim=1).values
        local_error = torch.linalg.vector_norm(residual_xy - local_center, dim=-1)
    else:
        local_error = torch.full_like(residual, float("inf"))
    spatially_consistent = local_error <= cfg.local_consistency_threshold_px

    low = cfg.global_residual_threshold_px
    high = cfg.max_deform_threshold_px
    margin = cfg.ambiguous_margin_px
    positive = reliable & (residual >= low + margin) & (residual <= high - margin) & spatially_consistent
    clear_negative = reliable & (
        (residual <= low - margin)
        | (residual >= high + margin)
        | (local_error >= cfg.local_consistency_threshold_px + margin)
    )
    deform_mask = positive | clear_negative
    target_deform = positive.float()
    h_mask = reliable & ~(
        (residual > low - margin) & (residual < low + margin)
    )
    return GRMGeometryRoutingTargets(
        target_h=target_h,
        target_deform=target_deform,
        h_mask=h_mask,
        deform_mask=deform_mask,
        diagnostics={
            "reliable_count": int(reliable.sum()),
            "h_supervised_count": int(h_mask.sum()),
            "deform_positive_count": int(positive.sum()),
            "deform_supervised_count": int(deform_mask.sum()),
            "residual_to_h_px": residual,
            "local_residual_consistency_px": local_error,
        },
    )


def geometry_routing_loss(
    output: GRMGeometryRoutingOutput,
    targets: GRMGeometryRoutingTargets,
) -> dict[str, Tensor]:
    """Return independent mask-aware BCE terms and their sum."""

    output.validate()

    def masked_bce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if logits.shape != target.shape or mask.shape != target.shape:
            raise ValueError("Routing logits, targets, and masks must have equal shapes.")
        loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
        weight = mask.float()
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    h_loss = masked_bce(output.h_support_logits, targets.target_h, targets.h_mask)
    deform_loss = masked_bce(
        output.deform_support_logits, targets.target_deform, targets.deform_mask
    )
    return {
        "routing_h_bce": h_loss,
        "routing_deform_bce": deform_loss,
        "routing_total": h_loss + deform_loss,
    }
