"""Instability-aware post-warp source selection with explicit backend fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class InstabilitySeamConfig:
    color_weight: float = 1.0
    gradient_weight: float = 0.5
    feature_weight: float = 0.5
    uncertainty_weight: float = 1.0
    strain_weight: float = 1.0
    instability_weight: float = 2.0
    smooth_weight: float = 0.2
    unstable_threshold: float = 0.6
    minimum_component_area: int = 16
    backend: Literal["auto", "opencv_graphcut", "dynamic_programming"] = "auto"


@dataclass
class InstabilitySeamResult:
    """Binary source ownership after final geometry warp."""

    source_selection_mask: np.ndarray
    seam_boundary: np.ndarray
    seam_cost: np.ndarray
    unstable_map: np.ndarray
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _normalize(value: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        array = np.linalg.norm(array, axis=-1)
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-6:
        return np.clip(array, 0.0, 1.0)
    return ((array - low) / (high - low)).astype(np.float32)


def _cost_and_instability(
    image_a: np.ndarray,
    image_b: np.ndarray,
    overlap: np.ndarray,
    config: InstabilitySeamConfig,
    *,
    feature_difference: Optional[np.ndarray],
    deformation_uncertainty: Optional[np.ndarray],
    mesh_strain: Optional[np.ndarray],
    forward_backward_inconsistency: Optional[np.ndarray],
    sparse_instability: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    shape = overlap.shape
    lab_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    color = _normalize(np.linalg.norm(lab_a - lab_b, axis=-1), shape)
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient = _normalize(
        np.abs(cv2.Laplacian(gray_a, cv2.CV_32F) - cv2.Laplacian(gray_b, cv2.CV_32F)),
        shape,
    )
    feature = _normalize(feature_difference, shape)
    uncertainty = _normalize(deformation_uncertainty, shape)
    strain = _normalize(mesh_strain, shape)
    fb = _normalize(forward_backward_inconsistency, shape)
    sparse = _normalize(sparse_instability, shape)
    unstable = np.clip(
        0.30 * color + 0.15 * gradient + 0.20 * uncertainty
        + 0.15 * strain + 0.10 * fb + 0.10 * sparse,
        0.0,
        1.0,
    )
    cost = (
        config.color_weight * color
        + config.gradient_weight * gradient
        + config.feature_weight * feature
        + config.uncertainty_weight * uncertainty
        + config.strain_weight * strain
        + config.instability_weight * np.maximum.reduce((unstable, fb, sparse))
    )
    cost = np.where(overlap, cost, 1e6).astype(np.float32)
    return cost, unstable.astype(np.float32)


def _dp_path(cost: np.ndarray, overlap: np.ndarray, smooth_weight: float) -> tuple[np.ndarray, str]:
    ys, xs = np.where(overlap)
    if xs.size == 0:
        return np.zeros(cost.shape[0], dtype=np.int32), "vertical"
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = cost[y0:y1, x0:x1]
    cropped_overlap = overlap[y0:y1, x0:x1]
    axis = "vertical"
    if cropped.shape[1] < cropped.shape[0]:
        cropped = cropped.T
        cropped_overlap = cropped_overlap.T
        axis = "horizontal"
    invalid = ~cropped_overlap
    work = cropped.copy()
    work[invalid] = 1e6
    height, width = work.shape
    dp = work.copy()
    parent = np.zeros((height, width), dtype=np.int32)
    for row in range(1, height):
        previous = dp[row - 1]
        for column in range(width):
            left, right = max(0, column - 2), min(width, column + 3)
            candidates = previous[left:right] + smooth_weight * np.abs(
                np.arange(left, right) - column
            )
            choice = int(np.argmin(candidates))
            parent[row, column] = left + choice
            dp[row, column] += candidates[choice]
    path = np.zeros(height, dtype=np.int32)
    path[-1] = int(np.argmin(dp[-1]))
    for row in range(height - 2, -1, -1):
        path[row] = parent[row + 1, path[row + 1]]
    path += x0 if axis == "vertical" else y0
    return path, axis


def _dp_ownership(
    cost: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    smooth_weight: float,
) -> tuple[np.ndarray, str]:
    overlap = valid_a & valid_b
    path, axis = _dp_path(cost, overlap, smooth_weight)
    ownership = np.full(valid_a.shape, -1, dtype=np.int8)
    ownership[valid_a & ~valid_b] = 0
    ownership[valid_b & ~valid_a] = 1
    yy, xx = np.indices(valid_a.shape)
    if axis == "vertical":
        row_path = np.interp(
            np.arange(valid_a.shape[0]),
            np.linspace(0, valid_a.shape[0] - 1, len(path)),
            path,
        )
        ownership[overlap & (xx <= row_path[:, None])] = 0
        ownership[overlap & (xx > row_path[:, None])] = 1
    else:
        col_path = np.interp(
            np.arange(valid_a.shape[1]),
            np.linspace(0, valid_a.shape[1] - 1, len(path)),
            path,
        )
        ownership[overlap & (yy <= col_path[None, :])] = 0
        ownership[overlap & (yy > col_path[None, :])] = 1
    return ownership, axis


def _opencv_ownership(
    image_a: np.ndarray,
    image_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
) -> np.ndarray:
    factory = getattr(cv2, "detail_GraphCutSeamFinder", None)
    if factory is None:
        raise RuntimeError("opencv_graphcut_unavailable")
    masks = [
        np.ascontiguousarray(valid_a.astype(np.uint8) * 255),
        np.ascontiguousarray(valid_b.astype(np.uint8) * 255),
    ]
    finder = factory("COST_COLOR_GRAD")
    finder.find(
        [np.ascontiguousarray(image_a.astype(np.float32)),
         np.ascontiguousarray(image_b.astype(np.float32))],
        [(0, 0), (0, 0)],
        masks,
    )
    ownership = np.full(valid_a.shape, -1, dtype=np.int8)
    ownership[valid_a & ~valid_b] = 0
    ownership[valid_b & ~valid_a] = 1
    overlap = valid_a & valid_b
    ownership[overlap & (masks[0] > 0)] = 0
    ownership[overlap & (masks[1] > 0) & (masks[0] == 0)] = 1
    undecided = overlap & (ownership < 0)
    ownership[undecided] = 0
    return ownership


def _stabilize_components(
    ownership: np.ndarray,
    unstable: np.ndarray,
    overlap: np.ndarray,
    threshold: float,
    minimum_area: int,
) -> int:
    binary = ((unstable >= threshold) & overlap).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    changed = 0
    for label in range(1, count):
        component = labels == label
        if int(stats[label, cv2.CC_STAT_AREA]) < minimum_area:
            continue
        values = ownership[component]
        values = values[values >= 0]
        if values.size == 0:
            continue
        chosen = int(np.bincount(values, minlength=2).argmax())
        changed += int(np.count_nonzero(ownership[component] != chosen))
        ownership[component] = chosen
    return changed


def estimate_instability_seam(
    image_a: np.ndarray,
    image_b_final_warped: np.ndarray,
    valid_mask_a: np.ndarray,
    valid_mask_b: np.ndarray,
    *,
    feature_difference: Optional[np.ndarray] = None,
    deformation_uncertainty: Optional[np.ndarray] = None,
    mesh_strain: Optional[np.ndarray] = None,
    forward_backward_inconsistency: Optional[np.ndarray] = None,
    sparse_instability: Optional[np.ndarray] = None,
    config: Optional[InstabilitySeamConfig] = None,
) -> InstabilitySeamResult:
    """Select A/B only after final warp; never average unstable components."""

    cfg = config or InstabilitySeamConfig()
    if image_a.shape != image_b_final_warped.shape or image_a.ndim != 3:
        raise ValueError("Final-warp images must share [H,W,3].")
    shape = image_a.shape[:2]
    valid_a = np.asarray(valid_mask_a) > 0
    valid_b = np.asarray(valid_mask_b) > 0
    if valid_a.shape != shape or valid_b.shape != shape:
        raise ValueError("Valid masks must match image spatial shape.")
    overlap = valid_a & valid_b
    cost, unstable = _cost_and_instability(
        image_a,
        image_b_final_warped,
        overlap,
        cfg,
        feature_difference=feature_difference,
        deformation_uncertainty=deformation_uncertainty,
        mesh_strain=mesh_strain,
        forward_backward_inconsistency=forward_backward_inconsistency,
        sparse_instability=sparse_instability,
    )
    fallback_reason = ""
    axis = ""
    use_opencv = cfg.backend in {"auto", "opencv_graphcut"}
    if use_opencv:
        try:
            ownership = _opencv_ownership(image_a, image_b_final_warped, valid_a, valid_b)
            backend = "opencv_graphcut+instability_components"
        except Exception as error:
            fallback_reason = f"opencv_graphcut_failed:{error}"
            ownership, axis = _dp_ownership(cost, valid_a, valid_b, cfg.smooth_weight)
            backend = "dynamic_programming"
    else:
        ownership, axis = _dp_ownership(cost, valid_a, valid_b, cfg.smooth_weight)
        backend = "dynamic_programming"
    ownership[valid_a & ~valid_b] = 0
    ownership[valid_b & ~valid_a] = 1
    component_changes = _stabilize_components(
        ownership,
        unstable,
        overlap,
        cfg.unstable_threshold,
        cfg.minimum_component_area,
    )
    selection_b = ownership == 1
    boundary = cv2.morphologyEx(
        selection_b.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    boundary &= overlap
    return InstabilitySeamResult(
        source_selection_mask=selection_b,
        seam_boundary=boundary,
        seam_cost=cost,
        unstable_map=unstable,
        backend=backend,
        diagnostics={
            "fallback_reason": fallback_reason,
            "overlap_ratio": float(overlap.mean()),
            "seam_pixel_count": int(boundary.sum()),
            "unstable_component_changed_pixels": component_changes,
            "dp_axis": axis,
        },
    )
