from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class SeamOptions:
    mode: str = "none"
    graphcut_cost_type: str = "COST_COLOR"
    color_weight: float = 1.0
    grad_weight: float = 0.5
    geom_weight: float = 2.0
    stability_weight: float = 2.0
    residual_weight: float = 1.0
    row_weight: float = 1.0
    smooth_weight: float = 0.1
    min_component_area: int = 64
    debug: bool = False


@dataclass
class SeamEvidence:
    confidence: np.ndarray | None = None
    residual_mag: np.ndarray | None = None
    stability: np.ndarray | None = None
    geometry_risk: np.ndarray | None = None
    row_dir: np.ndarray | None = None
    row_conf: np.ndarray | None = None
    traversability: np.ndarray | None = None


@dataclass
class SeamResult:
    ownership: np.ndarray
    masks: list[np.ndarray]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)


def normalize_map(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(shape, dtype=np.float32)
    arr = np.where(finite, arr, 0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-6:
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def build_pair_cost_map(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    options: SeamOptions,
    evidence: SeamEvidence | None = None,
) -> np.ndarray:
    h, w = mask_a.shape[:2]
    evidence = evidence or SeamEvidence()
    overlap = ((mask_a > 0) & (mask_b > 0)).astype(np.float32)
    lab_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    color = np.linalg.norm(lab_a - lab_b, axis=2) / 255.0
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_a = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
    grad_b = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)
    grad = normalize_map(np.abs(grad_a - grad_b), (h, w))

    cost = options.color_weight * normalize_map(color, (h, w))
    cost += options.grad_weight * grad
    cost += options.geom_weight * normalize_map(evidence.geometry_risk, (h, w))
    cost += options.residual_weight * normalize_map(evidence.residual_mag, (h, w))
    cost += options.stability_weight * (1.0 - normalize_map(evidence.stability, (h, w)))
    cost += (1.0 - overlap) * 1e6
    return cost.astype(np.float32)


def seam_boundary_energy(soft_mask: np.ndarray, cost_map: np.ndarray) -> float:
    mask = np.asarray(soft_mask, dtype=np.float32)
    cost = np.asarray(cost_map, dtype=np.float32)
    if cost.shape != mask.shape:
        cost = cv2.resize(cost, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    cost = np.where(np.isfinite(cost), cost, 0.0)
    cost = np.maximum(cost, 0.0)
    gx = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
    boundary = np.sqrt(gx * gx + gy * gy)
    return float((boundary * cost).mean())


def compute_row_crossing_penalty(
    seam_x_by_row: np.ndarray,
    row_dir: np.ndarray,
    row_conf: np.ndarray | None = None,
) -> float:
    seam = np.asarray(seam_x_by_row, dtype=np.float32)
    if seam.size < 2:
        return 0.0
    h = seam.shape[0]
    rows = np.arange(h, dtype=np.int32)
    cols = np.clip(np.rint(seam).astype(np.int32), 0, row_dir.shape[1] - 1)
    dirs = np.asarray(row_dir, dtype=np.float32)[rows, cols]
    conf = np.ones(h, dtype=np.float32) if row_conf is None else np.asarray(row_conf, dtype=np.float32)[rows, cols]
    dx = np.gradient(seam)
    seam_angle = np.arctan2(np.ones_like(dx), dx + 1e-6)
    crossing = np.sin(seam_angle - dirs) ** 2
    return float(np.mean(crossing * np.clip(conf, 0.0, 1.0)))


def _smooth_path(cost: np.ndarray, smooth_weight: float) -> np.ndarray:
    h, w = cost.shape
    dp = cost.copy()
    parent = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h):
        prev = dp[y - 1]
        for x in range(w):
            left = max(0, x - 1)
            right = min(w, x + 2)
            candidates = prev[left:right] + smooth_weight * np.abs(np.arange(left, right) - x)
            offset = int(np.argmin(candidates))
            parent[y, x] = left + offset
            dp[y, x] += candidates[offset]
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for y in range(h - 2, -1, -1):
        seam[y] = parent[y + 1, seam[y + 1]]
    return seam


def estimate_pair_seam_masks(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    options: SeamOptions | None = None,
    evidence: SeamEvidence | None = None,
) -> SeamResult:
    options = options or SeamOptions(mode="agri_pair")
    h, w = mask_a.shape[:2]
    valid_a = mask_a > 0
    valid_b = mask_b > 0
    overlap = valid_a & valid_b
    ownership = np.full((h, w), -1, dtype=np.int32)
    ownership[valid_a & ~valid_b] = 0
    ownership[valid_b & ~valid_a] = 1
    if not overlap.any():
        masks = [(ownership == 0).astype(np.uint8) * 255, (ownership == 1).astype(np.uint8) * 255]
        return SeamResult(ownership=ownership, masks=masks, diagnostics={"overlap": 0.0})

    cost = build_pair_cost_map(img_a, img_b, mask_a, mask_b, options, evidence)
    seam_x = _smooth_path(cost, options.smooth_weight)
    xs = np.arange(w)[None, :]
    ownership[overlap & (xs <= seam_x[:, None])] = 0
    ownership[overlap & (xs > seam_x[:, None])] = 1
    masks = [(ownership == 0).astype(np.uint8) * 255, (ownership == 1).astype(np.uint8) * 255]
    overlap_cost = cost[overlap]
    max_cost = max(float(overlap_cost.max()), 1e-6)
    return SeamResult(
        ownership=ownership,
        masks=masks,
        diagnostics={
            "overlap": float(overlap.mean()),
            "seam_x_by_row": seam_x.tolist(),
            "mean_cost": float(overlap_cost.mean()),
        },
        debug_images={"cost": np.clip(cost / max_cost * 255.0, 0, 255).astype(np.uint8)},
    )


def ownership_from_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=np.int32)
    stack = np.stack([m > 0 for m in masks], axis=0)
    valid = stack.any(axis=0)
    ownership = np.argmax(stack, axis=0).astype(np.int32)
    ownership[~valid] = -1
    return ownership


def _mask_debug_images(before_masks: list[np.ndarray], after_masks: list[np.ndarray]) -> dict[str, np.ndarray]:
    debug_images = {}
    for idx, (before, after) in enumerate(zip(before_masks, after_masks)):
        debug_images[f"mask_before_{idx:04d}"] = before.copy()
        debug_images[f"mask_after_{idx:04d}"] = after.copy()
        debug_images[f"mask_diff_{idx:04d}"] = cv2.absdiff(before, after)
    return debug_images


def _component_stats(masks: list[np.ndarray], min_component_area: int) -> tuple[int, int]:
    fragment_count = 0
    small_component_count = 0
    min_area = max(1, int(min_component_area))
    for mask in masks:
        binary = (mask > 0).astype(np.uint8)
        if not binary.any():
            continue
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        fragment_count += max(0, int(num_labels) - 1)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            small_component_count += int(np.count_nonzero(areas < min_area))
    return fragment_count, small_component_count


def _coverage_diagnostics(
    before_stack: np.ndarray,
    after_stack: np.ndarray,
    *,
    min_component_area: int = 64,
) -> dict[str, Any]:
    height, width = before_stack.shape[1:]
    canvas_pixels = max(1, int(height * width))
    before_union = before_stack.any(axis=0)
    after_union = after_stack.any(axis=0)
    before_coverage = before_stack.sum(axis=0)
    after_coverage = after_stack.sum(axis=0)
    overlap = before_coverage > 1
    uncovered = before_union & ~after_union
    overcovered = after_coverage > 1
    if overlap.any():
        changed_per_source = before_stack[:, overlap] != after_stack[:, overlap]
        overlap_changed_ratio = float(changed_per_source.any(axis=0).mean())
    else:
        overlap_changed_ratio = 0.0

    after_masks = [(mask.astype(np.uint8) * 255) for mask in after_stack]
    fragment_count, small_component_count = _component_stats(after_masks, min_component_area)
    after_union_area = max(1, int(after_union.sum()))
    area_after = [int(mask.sum()) for mask in after_stack]
    uncovered_ratio = float(uncovered.sum() / canvas_pixels)
    diagnostics: dict[str, Any] = {
        "coverage_before_ratio": float(before_union.sum() / canvas_pixels),
        "coverage_after_ratio": float(after_union.sum() / canvas_pixels),
        "uncovered_ratio": uncovered_ratio,
        "overlap_ratio": float(overlap.sum() / canvas_pixels),
        "overlap_changed_ratio": overlap_changed_ratio,
        "overlap_ratio_after": float(overcovered.sum() / canvas_pixels),
        "mask_area_before": [int(mask.sum()) for mask in before_stack],
        "mask_area_after": area_after,
        "fragment_count": int(fragment_count),
        "small_component_count": int(small_component_count),
        "source_area_ratio_after": [float(area / after_union_area) for area in area_after],
        "coverage_error": bool(uncovered_ratio > 0.0),
    }
    return diagnostics


def coverage_safe_masks_from_probs(
    before_masks: list[np.ndarray],
    ownership_probs: np.ndarray,
    *,
    min_component_area: int = 64,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Convert network source preferences into mutually exclusive valid masks.

    The probabilities are treated only as preferences. Pixels covered by zero
    input masks stay uncovered, pixels covered by one input mask keep that
    source, and overlap pixels choose the highest-probability valid source.
    """
    if not before_masks:
        return np.zeros((0, 0), dtype=np.int32), [], {
            "coverage_before_ratio": 0.0,
            "coverage_after_ratio": 0.0,
            "uncovered_ratio": 0.0,
            "overlap_ratio": 0.0,
            "overlap_changed_ratio": 0.0,
            "overlap_ratio_after": 0.0,
            "mask_area_before": [],
            "mask_area_after": [],
            "fragment_count": 0,
            "small_component_count": 0,
            "source_area_ratio_after": [],
            "coverage_error": False,
        }
    shape = before_masks[0].shape[:2]
    if any(mask.shape[:2] != shape for mask in before_masks):
        raise ValueError("All before_masks must have the same spatial shape")
    before_stack = np.stack([mask > 0 for mask in before_masks], axis=0)
    probs = np.asarray(ownership_probs, dtype=np.float32)
    if probs.ndim == 4 and probs.shape[1] == 1:
        probs = probs[:, 0]
    if probs.ndim != 3:
        raise ValueError("ownership_probs must have shape [N,H,W] or [N,1,H,W]")
    if probs.shape[0] != len(before_masks):
        raise ValueError("ownership_probs source count must match before_masks")
    if probs.shape[1:] != shape:
        resized = []
        for prob in probs:
            resized.append(cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR))
        probs = np.stack(resized, axis=0).astype(np.float32)
    probs = np.where(np.isfinite(probs), probs, -1e30).astype(np.float32)

    coverage_count = before_stack.sum(axis=0)
    masked_probs = np.where(before_stack, probs, -np.inf)
    ownership = np.full(shape, -1, dtype=np.int32)
    valid_any = coverage_count > 0
    if valid_any.any():
        ownership[valid_any] = np.argmax(masked_probs[:, valid_any], axis=0).astype(np.int32)

    after_stack = np.zeros_like(before_stack, dtype=bool)
    for idx in range(len(before_masks)):
        after_stack[idx] = (ownership == idx) & before_stack[idx]
    masks = [(mask.astype(np.uint8) * 255) for mask in after_stack]
    diagnostics = _coverage_diagnostics(before_stack, after_stack, min_component_area=min_component_area)
    return ownership, masks, diagnostics


def _add_coverage_diagnostics(
    diagnostics: dict[str, Any],
    before_masks: list[np.ndarray],
    after_masks: list[np.ndarray],
    *,
    min_component_area: int = 64,
) -> dict[str, Any]:
    if not before_masks:
        diagnostics.update(_coverage_diagnostics(
            np.zeros((0, 0, 0), dtype=bool),
            np.zeros((0, 0, 0), dtype=bool),
            min_component_area=min_component_area,
        ))
        return diagnostics
    before_stack = np.stack([mask > 0 for mask in before_masks], axis=0)
    after_stack = np.stack([mask > 0 for mask in after_masks], axis=0)
    diagnostics.update(_coverage_diagnostics(before_stack, after_stack, min_component_area=min_component_area))
    return diagnostics


def _preserve_single_source_regions(before_masks: list[np.ndarray], after_masks: list[np.ndarray]) -> list[np.ndarray]:
    probs = np.stack([m > 0 for m in after_masks], axis=0).astype(np.float32)
    _, fixed_masks, _ = coverage_safe_masks_from_probs(before_masks, probs)
    return fixed_masks


def _fallback_pair_result(
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
    options: SeamOptions,
    reason: str,
) -> SeamResult:
    if len(warped_images) < 2:
        masks = [m.copy() for m in warped_masks]
        diagnostics = {"mode": options.mode, "opencv_graphcut_used": False, "fallback_reason": reason}
        _add_coverage_diagnostics(diagnostics, warped_masks, masks, min_component_area=options.min_component_area)
        return SeamResult(
            ownership=ownership_from_masks(masks),
            masks=masks,
            diagnostics=diagnostics,
        )
    result = estimate_pair_seam_masks(
        warped_images[0],
        warped_images[1],
        warped_masks[0],
        warped_masks[1],
        SeamOptions(
            mode="agri_pair",
            color_weight=options.color_weight,
            grad_weight=options.grad_weight,
            geom_weight=options.geom_weight,
            stability_weight=options.stability_weight,
            residual_weight=options.residual_weight,
            row_weight=options.row_weight,
            smooth_weight=options.smooth_weight,
            min_component_area=options.min_component_area,
            debug=options.debug,
        ),
    )
    result.masks = result.masks + [m.copy() for m in warped_masks[2:]]
    result.diagnostics["mode"] = options.mode
    result.diagnostics["opencv_graphcut_used"] = False
    result.diagnostics["fallback_reason"] = reason
    _add_coverage_diagnostics(result.diagnostics, warped_masks, result.masks, min_component_area=options.min_component_area)
    return result


def estimate_opencv_graphcut_masks(
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
    options: SeamOptions | None = None,
) -> SeamResult:
    options = options or SeamOptions(mode="opencv_graphcut")
    if len(warped_images) != len(warped_masks):
        raise ValueError("warped_images and warped_masks must have the same length.")
    if len(warped_images) < 2:
        return _fallback_pair_result(warped_images, warped_masks, options, "need_at_least_two_images")

    finder_factory = getattr(cv2, "detail_GraphCutSeamFinder", None)
    if finder_factory is None:
        return _fallback_pair_result(warped_images, warped_masks, options, "opencv_graphcut_unavailable")

    before_masks = [np.ascontiguousarray(np.where(mask > 0, 255, 0).astype(np.uint8)) for mask in warped_masks]
    graphcut_masks = [mask.copy() for mask in before_masks]
    graphcut_images = [np.ascontiguousarray(img.astype(np.float32)) for img in warped_images]
    corners = [(0, 0) for _ in warped_images]

    try:
        finder = finder_factory(options.graphcut_cost_type)
        finder.find(graphcut_images, corners, graphcut_masks)
    except Exception as exc:
        return _fallback_pair_result(warped_images, warped_masks, options, f"opencv_graphcut_failed:{exc}")

    after_masks = _preserve_single_source_regions(before_masks, graphcut_masks)
    changed_pixels = [int(np.count_nonzero(before != after)) for before, after in zip(before_masks, after_masks)]
    diagnostics = {
        "mode": "opencv_graphcut",
        "opencv_graphcut_used": True,
        "graphcut_cost_type": options.graphcut_cost_type,
        "num_images": len(warped_images),
        "mask_changed_pixels": changed_pixels,
        "total_mask_changed_pixels": int(sum(changed_pixels)),
    }
    _add_coverage_diagnostics(diagnostics, before_masks, after_masks, min_component_area=options.min_component_area)
    return SeamResult(
        ownership=ownership_from_masks(after_masks),
        masks=after_masks,
        diagnostics=diagnostics,
        debug_images=_mask_debug_images(before_masks, after_masks),
    )


def _estimate_agri_pair_stack(
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
    options: SeamOptions,
) -> SeamResult:
    result = estimate_pair_seam_masks(
        warped_images[0],
        warped_images[1],
        warped_masks[0],
        warped_masks[1],
        options,
    )
    result.masks = result.masks + [m.copy() for m in warped_masks[2:]]
    result.diagnostics["mode"] = options.mode
    _add_coverage_diagnostics(result.diagnostics, warped_masks, result.masks, min_component_area=options.min_component_area)
    return result


class BaseSeamModel:
    def __init__(self, options: SeamOptions):
        self.options = options

    def estimate(self, warped_images: list[np.ndarray], warped_masks: list[np.ndarray]) -> SeamResult:
        raise NotImplementedError


class AgriPairSeamModel(BaseSeamModel):
    def estimate(self, warped_images: list[np.ndarray], warped_masks: list[np.ndarray]) -> SeamResult:
        return _estimate_agri_pair_stack(warped_images, warped_masks, self.options)


class OpenCVGraphCutSeamModel(BaseSeamModel):
    def estimate(self, warped_images: list[np.ndarray], warped_masks: list[np.ndarray]) -> SeamResult:
        return estimate_opencv_graphcut_masks(warped_images, warped_masks, self.options)


def build_seam_model(options: SeamOptions) -> BaseSeamModel:
    if options.mode == "agri_pair":
        return AgriPairSeamModel(options)
    if options.mode == "opencv_graphcut":
        return OpenCVGraphCutSeamModel(options)
    raise ValueError(f"Unsupported seam mode: {options.mode}")


def apply_seam_masks_to_stack(
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
    options: SeamOptions | None = None,
) -> SeamResult:
    options = options or SeamOptions(mode="none")
    if options.mode == "none" or len(warped_images) < 2:
        masks = [m.copy() for m in warped_masks]
        diagnostics = {"mode": "none", "num_images": len(warped_images)}
        _add_coverage_diagnostics(diagnostics, warped_masks, masks, min_component_area=options.min_component_area)
        return SeamResult(ownership=ownership_from_masks(masks), masks=masks, diagnostics=diagnostics)
    if options.mode not in {"agri_pair", "opencv_graphcut"}:
        raise ValueError(f"Unsupported seam mode: {options.mode}")
    return build_seam_model(options).estimate(warped_images, warped_masks)


def write_seam_debug(out_dir: Path, result: SeamResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ownership = result.ownership.astype(np.int32)
    if ownership.size:
        vis = np.zeros((*ownership.shape, 3), dtype=np.uint8)
        vis[ownership == 0] = (0, 160, 255)
        vis[ownership == 1] = (255, 80, 0)
        cv2.imwrite(str(out_dir / "ownership.png"), vis)
    for name, image in result.debug_images.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), image)
    (out_dir / "seam_report.json").write_text(json.dumps(result.diagnostics, indent=2), encoding="utf-8")
