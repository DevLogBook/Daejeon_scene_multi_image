"""Robust all-reliable MAGSAC A-to-B homography estimation with diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from dense_match.geometry_utils import apply_homography_xy

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class GlobalHomographyConfig:
    """Validation, refinement, and explicit fallback configuration."""

    ransac_threshold_px: float = 3.0
    confidence: float = 0.999
    max_iterations: int = 10_000
    min_determinant: float = 1e-10
    max_condition_number: float = 1e10
    max_perspective_per_px: float = 0.02
    collinear_ratio_threshold: float = 1e-3
    minimum_valid_warp_ratio: float = 0.01
    refine_weighted_dlt: bool = True
    allow_affine_fallback: bool = True
    allow_translation_fallback: bool = True
    allow_identity_fallback: bool = True

    def __post_init__(self) -> None:
        if self.ransac_threshold_px <= 0.0:
            raise ValueError("ransac_threshold_px must be positive.")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must lie in (0,1].")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if min(
            self.min_determinant,
            self.max_condition_number,
            self.max_perspective_per_px,
            self.collinear_ratio_threshold,
        ) <= 0.0:
            raise ValueError("Homography validation thresholds must be positive.")
        if not 0.0 <= self.minimum_valid_warp_ratio <= 1.0:
            raise ValueError("minimum_valid_warp_ratio must lie in [0,1].")


@dataclass
class GlobalHomographyResult:
    """Result of robust global geometry estimation for one image pair."""

    H_ab: Tensor
    valid: bool
    input_match_count: int
    inlier_mask: Tensor
    inlier_count: int
    reprojection_error: Tensor
    failure_reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _spatial_rank(points_xy: Tensor, ratio_threshold: float) -> bool:
    if points_xy.shape[0] < 3:
        return False
    centered = points_xy.double() - points_xy.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    return bool(singular[-1] / singular[0].clamp_min(1e-12) >= ratio_threshold)


def _weighted_dlt(src_xy: Tensor, dst_xy: Tensor, weights: Tensor) -> Optional[Tensor]:
    """Hartley-normalized weighted DLT for an already robust inlier set."""

    if src_xy.shape[0] < 4:
        return None

    def normalize(points: Tensor) -> tuple[Tensor, Tensor]:
        centroid = points.mean(dim=0)
        centered = points - centroid
        mean_distance = torch.linalg.vector_norm(centered, dim=-1).mean()
        if not torch.isfinite(mean_distance) or mean_distance < 1e-8:
            raise ValueError("degenerate point normalization")
        scale = np.sqrt(2.0) / mean_distance
        transform = points.new_tensor(
            [[scale, 0.0, -scale * centroid[0]],
             [0.0, scale, -scale * centroid[1]],
             [0.0, 0.0, 1.0]]
        )
        normalized = centered * scale
        return normalized, transform

    try:
        src = src_xy.double()
        dst = dst_xy.double()
        src_n, t_src = normalize(src)
        dst_n, t_dst = normalize(dst)
        x, y = src_n.unbind(-1)
        u, v = dst_n.unbind(-1)
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        row_u = torch.stack((-x, -y, -one, zero, zero, zero, u * x, u * y, u), -1)
        row_v = torch.stack((zero, zero, zero, -x, -y, -one, v * x, v * y, v), -1)
        matrix = torch.stack((row_u, row_v), dim=1)
        sqrt_weight = weights.double().clamp_min(1e-8).sqrt()[:, None, None]
        matrix = (matrix * sqrt_weight).reshape(-1, 9)
        _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        h_normalized = vh[-1].reshape(3, 3)
        h = torch.linalg.inv(t_dst) @ h_normalized @ t_src
        if h[2, 2].abs() < 1e-12:
            return None
        return h / h[2, 2]
    except (RuntimeError, ValueError):
        return None


def _warp_valid_ratio(
    h: Tensor,
    image_a_hw: Optional[Tuple[int, int]],
    image_b_hw: Optional[Tuple[int, int]],
) -> float:
    if image_a_hw is None or image_b_hw is None:
        return 1.0
    ha, wa = image_a_hw
    hb, wb = image_b_hw
    ys = torch.linspace(0.0, max(ha - 1, 0), 9, device=h.device, dtype=h.dtype)
    xs = torch.linspace(0.0, max(wa - 1, 0), 9, device=h.device, dtype=h.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    projected = apply_homography_xy(points, h)
    valid = (
        torch.isfinite(projected).all(dim=-1)
        & (projected[:, 0] >= -0.5)
        & (projected[:, 0] <= wb - 0.5)
        & (projected[:, 1] >= -0.5)
        & (projected[:, 1] <= hb - 0.5)
    )
    return float(valid.float().mean())


def _validate_h(
    h: Tensor,
    config: GlobalHomographyConfig,
    image_a_hw: Optional[Tuple[int, int]],
    image_b_hw: Optional[Tuple[int, int]],
) -> tuple[bool, str, dict[str, float]]:
    if h.shape != (3, 3) or not torch.isfinite(h).all():
        return False, "homography_non_finite", {}
    if h[2, 2].abs() < 1e-12:
        return False, "homography_zero_scale", {}
    normalized = h.double() / h[2, 2].double()
    try:
        determinant = float(torch.linalg.det(normalized))
        condition = float(torch.linalg.cond(normalized))
    except RuntimeError:
        return False, "homography_linear_algebra_failure", {}
    perspective = float(normalized[2, :2].abs().max())
    valid_ratio = _warp_valid_ratio(normalized, image_a_hw, image_b_hw)
    metrics = {
        "determinant": determinant,
        "condition_number": condition,
        "perspective_max_abs": perspective,
        "valid_warp_ratio": valid_ratio,
    }
    if abs(determinant) < config.min_determinant:
        return False, "homography_determinant_too_small", metrics
    if not np.isfinite(condition) or condition > config.max_condition_number:
        return False, "homography_condition_number_too_large", metrics
    if perspective > config.max_perspective_per_px:
        return False, "homography_perspective_too_large", metrics
    if valid_ratio < config.minimum_valid_warp_ratio:
        return False, "homography_valid_warp_area_too_small", metrics
    return True, "", metrics


def _translation_fallback(src: Tensor, dst: Tensor, weights: Tensor) -> Optional[Tensor]:
    if src.shape[0] < 1:
        return None
    delta = dst.double() - src.double()
    weight = weights.double().clamp_min(0.0)
    if float(weight.sum()) <= 0.0:
        weight = torch.ones_like(weight)
    translation = (delta * weight[:, None]).sum(dim=0) / weight.sum()
    h = torch.eye(3, dtype=torch.float64, device=src.device)
    h[:2, 2] = translation
    return h


def estimate_global_homography(
    src_xy: Tensor,
    dst_xy: Tensor,
    weights: Optional[Tensor] = None,
    *,
    image_a_hw: Optional[Tuple[int, int]] = None,
    image_b_hw: Optional[Tuple[int, int]] = None,
    previous_h_ab: Optional[Tensor] = None,
    config: Optional[GlobalHomographyConfig] = None,
) -> GlobalHomographyResult:
    """Estimate A→B H from every finite reliable match using USAC_MAGSAC."""

    cfg = config or GlobalHomographyConfig()
    if src_xy.ndim != 2 or src_xy.shape[-1] != 2 or dst_xy.shape != src_xy.shape:
        raise ValueError("src_xy and dst_xy must both be [N,2].")
    count = int(src_xy.shape[0])
    device = src_xy.device
    output_dtype = src_xy.dtype if src_xy.is_floating_point() else torch.float32
    if weights is None:
        weights = torch.ones(count, device=device, dtype=torch.float32)
    if weights.shape != (count,):
        raise ValueError("weights must be [N].")
    finite = (
        torch.isfinite(src_xy).all(dim=-1)
        & torch.isfinite(dst_xy).all(dim=-1)
        & torch.isfinite(weights)
        & (weights >= 0.0)
    )
    src = src_xy[finite].double()
    dst = dst_xy[finite].double()
    weight = weights[finite].double()
    finite_indices = torch.nonzero(finite, as_tuple=False).squeeze(-1)
    inlier_full = torch.zeros(count, device=device, dtype=torch.bool)
    failure: list[str] = []
    diagnostics: dict[str, Any] = {
        "finite_match_count": int(src.shape[0]),
        "backend": "opencv_usac_magsac" if cv2 is not None else "unavailable",
        "fallback": "",
    }
    h: Optional[Tensor] = None
    local_inliers = torch.zeros(src.shape[0], dtype=torch.bool, device=device)

    if src.shape[0] < 4:
        failure.append(f"fewer_than_four_points:{int(src.shape[0])}")
    elif not _spatial_rank(src, cfg.collinear_ratio_threshold):
        failure.append("source_points_nearly_collinear")
    elif not _spatial_rank(dst, cfg.collinear_ratio_threshold):
        failure.append("destination_points_nearly_collinear")
    elif cv2 is None:
        failure.append("opencv_unavailable")
    else:
        try:
            matrix, mask = cv2.findHomography(
                src.cpu().numpy(),
                dst.cpu().numpy(),
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=cfg.ransac_threshold_px,
                maxIters=cfg.max_iterations,
                confidence=cfg.confidence,
            )
        except cv2.error as error:
            matrix, mask = None, None
            failure.append(f"opencv_magsac_error:{error}")
        if matrix is None or mask is None:
            if not failure:
                failure.append("opencv_magsac_no_model")
        else:
            h = torch.from_numpy(np.asarray(matrix)).to(device=device, dtype=torch.float64)
            local_inliers = torch.from_numpy(mask.reshape(-1).astype(bool)).to(device=device)
            if int(local_inliers.sum()) < 4:
                failure.append("magsac_fewer_than_four_inliers")
                h = None
            elif cfg.refine_weighted_dlt:
                refined = _weighted_dlt(
                    src[local_inliers],
                    dst[local_inliers],
                    weight[local_inliers],
                )
                if refined is not None:
                    h = refined
                    diagnostics["weighted_dlt_refined"] = True
                else:
                    diagnostics["weighted_dlt_refined"] = False
            if h is not None:
                accepted, reason, metrics = _validate_h(
                    h, cfg, image_a_hw, image_b_hw
                )
                diagnostics.update(metrics)
                if not accepted:
                    failure.append(reason)
                    h = None

    primary_success = h is not None
    if h is None and previous_h_ab is not None:
        candidate = previous_h_ab.to(device=device, dtype=torch.float64)
        accepted, reason, metrics = _validate_h(candidate, cfg, image_a_hw, image_b_hw)
        if accepted:
            h = candidate / candidate[2, 2]
            diagnostics.update(metrics)
            diagnostics["fallback"] = "previous_h"
        else:
            failure.append(f"previous_h_rejected:{reason}")

    if h is None and cfg.allow_affine_fallback and src.shape[0] >= 3 and cv2 is not None:
        try:
            affine, affine_mask = cv2.estimateAffine2D(
                src.cpu().numpy(),
                dst.cpu().numpy(),
                method=cv2.RANSAC,
                ransacReprojThreshold=cfg.ransac_threshold_px,
                maxIters=cfg.max_iterations,
                confidence=cfg.confidence,
                refineIters=10,
            )
        except cv2.error as error:
            affine, affine_mask = None, None
            failure.append(f"affine_error:{error}")
        if affine is not None:
            candidate = torch.eye(3, device=device, dtype=torch.float64)
            candidate[:2] = torch.from_numpy(affine).to(device=device, dtype=torch.float64)
            accepted, reason, metrics = _validate_h(candidate, cfg, image_a_hw, image_b_hw)
            if accepted:
                h = candidate
                diagnostics.update(metrics)
                diagnostics["fallback"] = "affine"
                if affine_mask is not None:
                    local_inliers = torch.from_numpy(
                        affine_mask.reshape(-1).astype(bool)
                    ).to(device=device)
            else:
                failure.append(f"affine_rejected:{reason}")

    if h is None and cfg.allow_translation_fallback and src.shape[0] >= 1:
        candidate = _translation_fallback(src, dst, weight)
        assert candidate is not None
        accepted, reason, metrics = _validate_h(candidate, cfg, image_a_hw, image_b_hw)
        if accepted:
            h = candidate
            diagnostics.update(metrics)
            diagnostics["fallback"] = "translation"
            local_inliers = torch.ones(src.shape[0], device=device, dtype=torch.bool)
        else:
            failure.append(f"translation_rejected:{reason}")

    if h is None and cfg.allow_identity_fallback:
        candidate = torch.eye(3, device=device, dtype=torch.float64)
        accepted, reason, metrics = _validate_h(candidate, cfg, image_a_hw, image_b_hw)
        if accepted:
            h = candidate
            diagnostics.update(metrics)
            diagnostics["fallback"] = "identity"
        else:
            failure.append(f"identity_rejected:{reason}")

    usable = h is not None
    if h is None:
        h = torch.eye(3, device=device, dtype=torch.float64)
        diagnostics["fallback"] = "explicit_failure"
    h = h / h[2, 2].clamp_min(1e-12)
    if local_inliers.numel() == finite_indices.numel():
        inlier_full[finite_indices[local_inliers]] = True
    projected = apply_homography_xy(src_xy.to(dtype=torch.float64), h)
    error = torch.linalg.vector_norm(projected - dst_xy.to(dtype=torch.float64), dim=-1)
    error = torch.where(torch.isfinite(error), error, torch.full_like(error, float("inf")))
    diagnostics["primary_magsac_success"] = primary_success
    diagnostics["model"] = "homography" if primary_success else diagnostics["fallback"]
    return GlobalHomographyResult(
        H_ab=h.to(dtype=output_dtype),
        valid=usable,
        input_match_count=count,
        inlier_mask=inlier_full,
        inlier_count=int(inlier_full.sum()),
        reprojection_error=error.to(dtype=output_dtype),
        failure_reason=";".join(failure),
        diagnostics=diagnostics,
    )
