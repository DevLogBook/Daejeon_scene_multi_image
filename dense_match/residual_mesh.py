"""Non-learning robust residual mesh baseline for GRM correspondences.

The mesh stores B-pixel residuals relative to an A-to-B base homography:
``dst_xy - H_ab(src_xy)``.  It is never responsible for seam ownership or
blending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from dense_match.geometry_utils import apply_homography_xy, image_pixel_grid


ResidualMeshMode = Literal[
    "h_only",
    "all_reliable_mesh",
    "legacy_global_only_h",
    "legacy_local_only_mesh",
]


@dataclass(frozen=True)
class ResidualMeshConfig:
    """Robust mesh optimization and safety limits."""

    mesh_rows: int = 12
    mesh_cols: int = 12
    minimum_residual_matches: int = 6
    aggregation: Literal["huber_irls", "trimmed_weighted_mean"] = "huber_irls"
    irls_iterations: int = 5
    huber_delta_px: float = 3.0
    trim_fraction: float = 0.15
    data_weight: float = 1.0
    smooth_weight: float = 2.0
    bend_weight: float = 0.5
    anchor_weight: float = 0.02
    boundary_weight: float = 2.0
    boundary_fade_width_px: int = 16
    support_scale: float = 0.5
    max_residual_px: float = 24.0
    minimum_jacobian: float = 0.05
    maximum_fold_ratio: float = 0.0
    fold_backoff_steps: int = 6
    legacy_local_threshold: float = 0.5
    linear_solve_regularization: float = 1e-5

    def __post_init__(self) -> None:
        if self.mesh_rows < 2 or self.mesh_cols < 2:
            raise ValueError("mesh_rows and mesh_cols must be >= 2.")
        if self.minimum_residual_matches < 1 or self.irls_iterations < 1:
            raise ValueError("match and iteration counts must be positive.")
        if self.aggregation not in {"huber_irls", "trimmed_weighted_mean"}:
            raise ValueError(f"Unsupported aggregation={self.aggregation!r}.")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must lie in [0,0.5).")
        nonnegative = (
            self.data_weight,
            self.smooth_weight,
            self.bend_weight,
            self.anchor_weight,
            self.boundary_weight,
            self.support_scale,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("Mesh energy weights must be nonnegative.")
        if self.huber_delta_px <= 0.0 or self.max_residual_px <= 0.0:
            raise ValueError("Residual thresholds must be positive.")
        if not 0.0 <= self.maximum_fold_ratio <= 1.0:
            raise ValueError("maximum_fold_ratio must lie in [0,1].")
        if not 0.0 <= self.legacy_local_threshold <= 1.0:
            raise ValueError("legacy_local_threshold must lie in [0,1].")


@dataclass
class ResidualMeshResult:
    """Optimized mesh and dense B-pixel residual field."""

    mesh_offset: Tensor
    dense_residual_grid: Tensor
    support_weight: Tensor
    residual_variance: Tensor
    deformation_confidence: Tensor
    fold_mask: Tensor
    base_mapping_xy: Tensor
    final_mapping_xy: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _mesh_interpolation_matrix(
    src_xy: Tensor,
    image_hw: Tuple[int, int],
    mesh_hw: Tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Return bilinear interpolation matrix ``[N,V]`` and vertex indices."""

    height, width = image_hw
    rows, cols = mesh_hw
    if src_xy.ndim != 2 or src_xy.shape[-1] != 2:
        raise ValueError("src_xy must be [N,2].")
    gx = src_xy[:, 0] * (cols - 1) / max(width - 1, 1)
    gy = src_xy[:, 1] * (rows - 1) / max(height - 1, 1)
    x0 = gx.floor().long().clamp(0, cols - 1)
    y0 = gy.floor().long().clamp(0, rows - 1)
    x1 = (x0 + 1).clamp(0, cols - 1)
    y1 = (y0 + 1).clamp(0, rows - 1)
    fx = (gx - x0.to(gx.dtype)).clamp(0.0, 1.0)
    fy = (gy - y0.to(gy.dtype)).clamp(0.0, 1.0)
    indices = torch.stack(
        (y0 * cols + x0, y0 * cols + x1, y1 * cols + x0, y1 * cols + x1),
        dim=-1,
    )
    coefficients = torch.stack(
        ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy),
        dim=-1,
    )
    matrix = src_xy.new_zeros((src_xy.shape[0], rows * cols), dtype=torch.float64)
    matrix.scatter_add_(1, indices, coefficients.double())
    return matrix, indices


def _regularization_rows(
    mesh_hw: Tuple[int, int],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build first-order, second-order, and outer-boundary linear operators."""

    rows, cols = mesh_hw
    vertices = rows * cols
    first: list[Tensor] = []
    second: list[Tensor] = []
    for y in range(rows):
        for x in range(cols - 1):
            row = torch.zeros(vertices, device=device, dtype=torch.float64)
            row[y * cols + x] = -1.0
            row[y * cols + x + 1] = 1.0
            first.append(row)
    for y in range(rows - 1):
        for x in range(cols):
            row = torch.zeros(vertices, device=device, dtype=torch.float64)
            row[y * cols + x] = -1.0
            row[(y + 1) * cols + x] = 1.0
            first.append(row)
    for y in range(rows):
        for x in range(1, cols - 1):
            row = torch.zeros(vertices, device=device, dtype=torch.float64)
            row[y * cols + x - 1] = 1.0
            row[y * cols + x] = -2.0
            row[y * cols + x + 1] = 1.0
            second.append(row)
    for y in range(1, rows - 1):
        for x in range(cols):
            row = torch.zeros(vertices, device=device, dtype=torch.float64)
            row[(y - 1) * cols + x] = 1.0
            row[y * cols + x] = -2.0
            row[(y + 1) * cols + x] = 1.0
            second.append(row)
    boundary = torch.zeros(vertices, device=device, dtype=torch.float64)
    for y in range(rows):
        for x in range(cols):
            if y in {0, rows - 1} or x in {0, cols - 1}:
                boundary[y * cols + x] = 1.0
    first_matrix = torch.stack(first) if first else torch.zeros(0, vertices, device=device, dtype=torch.float64)
    second_matrix = torch.stack(second) if second else torch.zeros(0, vertices, device=device, dtype=torch.float64)
    return first_matrix, second_matrix, boundary


def _fade_inside_overlap(overlap_mask: Optional[Tensor], image_hw: Tuple[int, int], width: int) -> Tensor:
    """Return a 0→1 erosion-distance ramp inside overlap."""

    height, image_width = image_hw
    if overlap_mask is None:
        return torch.ones(height, image_width, dtype=torch.float32)
    mask = overlap_mask
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim == 4:
        mask = mask[0, 0] if mask.shape[1] == 1 else mask[0, ..., 0]
    if mask.ndim != 2:
        raise ValueError("overlap_mask must resolve to [H,W].")
    mask = mask.float()[None, None]
    if mask.shape[-2:] != image_hw:
        mask = F.interpolate(mask, size=image_hw, mode="nearest")
    mask = torch.nan_to_num(mask, nan=0.0).clamp(0.0, 1.0)
    if width <= 0:
        return mask[0, 0]
    active = mask
    ramp = torch.zeros_like(active)
    for _ in range(width):
        active = 1.0 - F.max_pool2d(1.0 - active, kernel_size=3, stride=1, padding=1)
        ramp = ramp + active
    return (ramp / float(width)).clamp(0.0, 1.0)[0, 0]


def _jacobian_fold_mask(mapping_xy: Tensor, minimum_jacobian: float) -> tuple[Tensor, Tensor]:
    if mapping_xy.ndim != 3 or mapping_xy.shape[-1] != 2:
        raise ValueError("mapping_xy must be [H,W,2].")
    if mapping_xy.shape[0] < 2 or mapping_xy.shape[1] < 2:
        empty = torch.zeros(
            max(mapping_xy.shape[0] - 1, 0),
            max(mapping_xy.shape[1] - 1, 0),
            device=mapping_xy.device,
            dtype=torch.bool,
        )
        return empty, mapping_xy.new_zeros(empty.shape)
    dx = mapping_xy[:-1, 1:] - mapping_xy[:-1, :-1]
    dy = mapping_xy[1:, :-1] - mapping_xy[:-1, :-1]
    determinant = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    return determinant <= minimum_jacobian, determinant


def _empty_result(
    h_ab: Tensor,
    image_hw: Tuple[int, int],
    mesh_hw: Tuple[int, int],
    reason: str,
    diagnostics: Optional[dict[str, Any]] = None,
) -> ResidualMeshResult:
    height, width = image_hw
    rows, cols = mesh_hw
    pixel_grid = image_pixel_grid(image_hw, device=h_ab.device, dtype=h_ab.dtype)
    base = apply_homography_xy(pixel_grid.reshape(-1, 2), h_ab).reshape(height, width, 2)
    fold, determinant = _jacobian_fold_mask(base, minimum_jacobian=0.0)
    info = dict(diagnostics or {})
    info.update(
        {
            "fallback_reason": reason,
            "h_only": True,
            "fold_ratio": float(fold.float().mean()) if fold.numel() else 0.0,
            "min_jacobian": float(determinant.min()) if determinant.numel() else 0.0,
        }
    )
    return ResidualMeshResult(
        mesh_offset=torch.zeros(rows, cols, 2, device=h_ab.device, dtype=h_ab.dtype),
        dense_residual_grid=torch.zeros(height, width, 2, device=h_ab.device, dtype=h_ab.dtype),
        support_weight=torch.zeros(rows, cols, device=h_ab.device, dtype=h_ab.dtype),
        residual_variance=torch.zeros(rows, cols, device=h_ab.device, dtype=h_ab.dtype),
        deformation_confidence=torch.zeros(height, width, device=h_ab.device, dtype=h_ab.dtype),
        fold_mask=fold,
        base_mapping_xy=base,
        final_mapping_xy=base,
        diagnostics=info,
    )


def estimate_residual_mesh(
    src_xy: Tensor,
    dst_xy: Tensor,
    h_ab: Tensor,
    image_a_hw: Tuple[int, int],
    weights: Optional[Tensor] = None,
    *,
    overlap_mask: Optional[Tensor] = None,
    w_deform: Optional[Tensor] = None,
    legacy_p_local: Optional[Tensor] = None,
    mode: ResidualMeshMode = "all_reliable_mesh",
    config: Optional[ResidualMeshConfig] = None,
) -> ResidualMeshResult:
    """Fit a safe local residual mesh without learning or ROLE hard routing."""

    cfg = config or ResidualMeshConfig()
    mesh_hw = (cfg.mesh_rows, cfg.mesh_cols)
    if src_xy.ndim != 2 or src_xy.shape[-1] != 2 or dst_xy.shape != src_xy.shape:
        raise ValueError("src_xy and dst_xy must both be [N,2].")
    if h_ab.shape != (3, 3):
        raise ValueError("h_ab must be [3,3].")
    count = src_xy.shape[0]
    if weights is None:
        weights = torch.ones(count, device=src_xy.device, dtype=src_xy.dtype)
    if weights.shape != (count,):
        raise ValueError("weights must be [N].")
    if w_deform is not None and w_deform.shape != (count,):
        raise ValueError("w_deform must be [N].")
    if legacy_p_local is not None and legacy_p_local.shape != (count,):
        raise ValueError("legacy_p_local must be [N].")
    if mode not in {
        "h_only",
        "all_reliable_mesh",
        "legacy_global_only_h",
        "legacy_local_only_mesh",
    }:
        raise ValueError(f"Unsupported residual mesh mode={mode!r}.")
    if mode in {"h_only", "legacy_global_only_h"}:
        return _empty_result(h_ab, image_a_hw, mesh_hw, f"mode:{mode}", {"mode": mode})

    finite = (
        torch.isfinite(src_xy).all(dim=-1)
        & torch.isfinite(dst_xy).all(dim=-1)
        & torch.isfinite(weights)
        & (weights > 0.0)
    )
    if mode == "legacy_local_only_mesh":
        if legacy_p_local is None:
            return _empty_result(
                h_ab, image_a_hw, mesh_hw, "legacy_local_probabilities_missing", {"mode": mode}
            )
        finite &= legacy_p_local >= cfg.legacy_local_threshold
    src = src_xy[finite]
    dst = dst_xy[finite]
    base_weights = weights[finite].float().clamp_min(0.0)
    if w_deform is not None:
        base_weights = base_weights * w_deform[finite].float().clamp(0.0, 1.0)
    if src.shape[0] < cfg.minimum_residual_matches or float(base_weights.sum()) <= 0.0:
        return _empty_result(
            h_ab,
            image_a_hw,
            mesh_hw,
            f"insufficient_residual_support:{int(src.shape[0])}",
            {"mode": mode, "selected_match_count": int(src.shape[0])},
        )

    predicted = apply_homography_xy(src, h_ab)
    residual = (dst - predicted).double()
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    if cfg.aggregation == "trimmed_weighted_mean" and residual.shape[0] >= 8:
        keep_count = max(
            cfg.minimum_residual_matches,
            int(round(residual.shape[0] * (1.0 - cfg.trim_fraction))),
        )
        center = residual.median(dim=0).values
        ordering = torch.argsort(torch.linalg.vector_norm(residual - center, dim=-1))
        keep = ordering[:keep_count]
        src, dst = src[keep], dst[keep]
        base_weights, residual, residual_norm = (
            base_weights[keep],
            residual[keep],
            residual_norm[keep],
        )

    interpolation, vertex_indices = _mesh_interpolation_matrix(src, image_a_hw, mesh_hw)
    vertices = cfg.mesh_rows * cfg.mesh_cols
    first, second, outer_boundary = _regularization_rows(mesh_hw, device=src.device)
    identity = torch.eye(vertices, device=src.device, dtype=torch.float64)
    regularizer = cfg.linear_solve_regularization * identity
    if first.numel():
        regularizer = regularizer + cfg.smooth_weight * (first.T @ first)
    if second.numel():
        regularizer = regularizer + cfg.bend_weight * (second.T @ second)
    regularizer = regularizer + cfg.anchor_weight * identity
    regularizer = regularizer + cfg.boundary_weight * torch.diag(outer_boundary)

    mesh_flat = torch.zeros(vertices, 2, device=src.device, dtype=torch.float64)
    robust_weight = base_weights.double()
    solve_failure = ""
    for _ in range(cfg.irls_iterations):
        predicted_residual = interpolation @ mesh_flat
        error_norm = torch.linalg.vector_norm(predicted_residual - residual, dim=-1)
        if cfg.aggregation == "huber_irls":
            robust = torch.where(
                error_norm <= cfg.huber_delta_px,
                torch.ones_like(error_norm),
                cfg.huber_delta_px / error_norm.clamp_min(1e-8),
            )
        else:
            robust = torch.ones_like(error_norm)
        robust_weight = cfg.data_weight * base_weights.double() * robust
        weighted_matrix = interpolation * robust_weight[:, None]
        system = interpolation.T @ weighted_matrix + regularizer
        rhs = interpolation.T @ (robust_weight[:, None] * residual)
        try:
            mesh_flat = torch.linalg.solve(system, rhs)
        except RuntimeError as error:
            solve_failure = f"linear_solve_failed:{error}"
            break
        if not torch.isfinite(mesh_flat).all():
            solve_failure = "linear_solve_non_finite"
            break
    if solve_failure:
        return _empty_result(
            h_ab,
            image_a_hw,
            mesh_hw,
            solve_failure,
            {"mode": mode, "selected_match_count": int(src.shape[0])},
        )

    magnitude = torch.linalg.vector_norm(mesh_flat, dim=-1, keepdim=True)
    mesh_flat = mesh_flat * torch.clamp(cfg.max_residual_px / magnitude.clamp_min(1e-8), max=1.0)
    mesh = mesh_flat.reshape(cfg.mesh_rows, cfg.mesh_cols, 2)

    # Weighted bilinear support and residual variance at mesh vertices.
    support_flat = interpolation.T @ base_weights.double()
    fitted_error_sq = torch.linalg.vector_norm(interpolation @ mesh_flat - residual, dim=-1).square()
    variance_flat = interpolation.T @ (base_weights.double() * fitted_error_sq)
    variance_flat = variance_flat / support_flat.clamp_min(1e-8)
    support = support_flat.reshape(cfg.mesh_rows, cfg.mesh_cols)
    variance = variance_flat.reshape(cfg.mesh_rows, cfg.mesh_cols)

    dense = F.interpolate(
        mesh.permute(2, 0, 1).unsqueeze(0),
        size=image_a_hw,
        mode="bilinear",
        align_corners=True,
    )[0].permute(1, 2, 0)
    support_dense = F.interpolate(
        support[None, None],
        size=image_a_hw,
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    confidence = support_dense / (support_dense + cfg.support_scale)
    overlap_fade = _fade_inside_overlap(
        overlap_mask, image_a_hw, cfg.boundary_fade_width_px
    ).to(device=src.device, dtype=confidence.dtype)
    confidence = (confidence * overlap_fade).clamp(0.0, 1.0)
    dense = dense * confidence[..., None]

    height, width = image_a_hw
    source_grid = image_pixel_grid(image_a_hw, device=src.device, dtype=h_ab.dtype)
    base_mapping = apply_homography_xy(source_grid.reshape(-1, 2), h_ab).reshape(height, width, 2)
    residual_scale = 1.0
    final_mapping = base_mapping + dense.to(base_mapping.dtype)
    fold, determinant = _jacobian_fold_mask(final_mapping, cfg.minimum_jacobian)
    induced_fold = fold
    base_fold, _ = _jacobian_fold_mask(base_mapping, cfg.minimum_jacobian)
    if base_fold.shape == fold.shape:
        induced_fold = fold & ~base_fold
    fold_ratio = float(induced_fold.float().mean()) if induced_fold.numel() else 0.0
    for _ in range(cfg.fold_backoff_steps):
        if fold_ratio <= cfg.maximum_fold_ratio:
            break
        residual_scale *= 0.5
        final_mapping = base_mapping + residual_scale * dense.to(base_mapping.dtype)
        fold, determinant = _jacobian_fold_mask(final_mapping, cfg.minimum_jacobian)
        induced_fold = fold & ~base_fold if base_fold.shape == fold.shape else fold
        fold_ratio = float(induced_fold.float().mean()) if induced_fold.numel() else 0.0
    dense = dense * residual_scale
    mesh = mesh * residual_scale
    confidence = confidence * residual_scale
    final_mapping = base_mapping + dense.to(base_mapping.dtype)

    diagnostics: dict[str, Any] = {
        "mode": mode,
        "fallback_reason": "",
        "h_only": False,
        "input_match_count": int(count),
        "selected_match_count": int(src.shape[0]),
        "residual_weight_sum": float(base_weights.sum()),
        "residual_px_median": float(residual_norm.median()),
        "residual_px_p90": float(torch.quantile(residual_norm, 0.9)),
        "mesh_max_offset_px": float(torch.linalg.vector_norm(mesh, dim=-1).max()),
        "support_vertex_count": int((support > 1e-8).sum()),
        "fold_backoff_scale": residual_scale,
        "fold_ratio": fold_ratio,
        "min_jacobian": float(determinant.min()) if determinant.numel() else 0.0,
        "residual_vectors": residual.to(dtype=src_xy.dtype),
        "selected_src_xy": src,
        "selected_dst_xy": dst,
        "vertex_indices": vertex_indices,
        "robust_weights": robust_weight.to(dtype=src_xy.dtype),
    }
    return ResidualMeshResult(
        mesh_offset=mesh.to(dtype=src_xy.dtype),
        dense_residual_grid=dense.to(dtype=src_xy.dtype),
        support_weight=support.to(dtype=src_xy.dtype),
        residual_variance=variance.to(dtype=src_xy.dtype),
        deformation_confidence=confidence.to(dtype=src_xy.dtype),
        fold_mask=induced_fold,
        base_mapping_xy=base_mapping.to(dtype=src_xy.dtype),
        final_mapping_xy=final_mapping.to(dtype=src_xy.dtype),
        diagnostics=diagnostics,
    )
