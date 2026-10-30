"""Standalone matching, role, geometry, coverage, and calibration metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .coordinates import image_xy_to_normalized_grid
from .outputs import GRMMatcherOutput
from .role_targets import build_geometry_role_targets
from .seed_geometry import apply_homography

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def candidate_recall_at_k(
    probability: Tensor,
    gt_target_indices: Tensor,
    valid_source_mask: Tensor,
    ks: Sequence[int] = (1, 4, 8),
) -> Dict[str, float]:
    """Recall that the supervised target cell appears in full-matrix top-k."""

    if probability.ndim != 3:
        raise ValueError("probability must be [B, S, T].")
    if gt_target_indices.shape != probability.shape[:2]:
        raise ValueError("gt_target_indices must be [B, S].")
    if valid_source_mask.shape != probability.shape[:2]:
        raise ValueError("valid_source_mask must be [B, S].")
    result: Dict[str, float] = {}
    for requested_k in ks:
        if requested_k <= 0:
            raise ValueError("Recall k values must be positive.")
        effective_k = min(requested_k, probability.shape[-1])
        candidates = torch.topk(probability, effective_k, dim=-1).indices
        recalled = (candidates == gt_target_indices[..., None]).any(dim=-1)
        result[f"recall_at_{requested_k}"] = _safe_ratio(
            (recalled & valid_source_mask).sum(), valid_source_mask.sum()
        )
    return result


def threshold_matching_metrics(
    predicted_dst_xy: Tensor,
    gt_dst_xy: Tensor,
    predicted_valid_mask: Tensor,
    gt_valid_mask: Tensor,
    thresholds: Sequence[float] = (1.0, 3.0, 5.0, 10.0),
) -> Dict[str, float]:
    """Precision/recall/F1 under several original-pixel error thresholds."""

    error = torch.linalg.vector_norm(predicted_dst_xy.float() - gt_dst_xy.float(), dim=-1)
    metrics: Dict[str, float] = {}
    for threshold in thresholds:
        correct = predicted_valid_mask & gt_valid_mask & (error <= threshold)
        precision = _safe_ratio(correct.sum(), predicted_valid_mask.sum())
        recall = _safe_ratio(correct.sum(), gt_valid_mask.sum())
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        suffix = _format_threshold(threshold)
        metrics[f"precision_at_{suffix}px"] = precision
        metrics[f"recall_at_{suffix}px"] = recall
        metrics[f"f1_at_{suffix}px"] = f1
    return metrics


def localization_error_metrics(
    predicted_dst_xy: Tensor,
    gt_dst_xy: Tensor,
    valid_mask: Tensor,
) -> Dict[str, float]:
    error = torch.linalg.vector_norm(predicted_dst_xy.float() - gt_dst_xy.float(), dim=-1)
    selected = error[valid_mask]
    if selected.numel() == 0:
        return {"mean_pixel_error": float("nan"), "median_pixel_error": float("nan"), "p90_pixel_error": float("nan")}
    return {
        "mean_pixel_error": float(selected.mean().cpu()),
        "median_pixel_error": float(selected.median().cpu()),
        "p90_pixel_error": float(torch.quantile(selected, 0.9).cpu()),
    }


def spatial_coverage(
    src_xy: Tensor,
    reliable_mask: Tensor,
    image_height: int,
    image_width: int,
    grid_rows: int = 8,
    grid_cols: int = 8,
) -> float:
    """Fraction of source-image grid regions containing a reliable match."""

    if src_xy.ndim != 3 or reliable_mask.shape != src_xy.shape[:2]:
        raise ValueError("src_xy must be [B,N,2] and reliable_mask [B,N].")
    x = (src_xy[..., 0] * grid_cols / max(image_width, 1)).floor().long()
    y = (src_xy[..., 1] * grid_rows / max(image_height, 1)).floor().long()
    region = y.clamp(0, grid_rows - 1) * grid_cols + x.clamp(0, grid_cols - 1)
    covered = torch.zeros(
        src_xy.shape[0], grid_rows * grid_cols, device=src_xy.device, dtype=torch.bool
    )
    for batch_index in range(src_xy.shape[0]):
        covered[batch_index, region[batch_index, reliable_mask[batch_index]]] = True
    return float(covered.float().mean().cpu())


def role_classification_metrics(
    predicted_roles: Tensor,
    target_roles: Tensor,
    valid_mask: Tensor,
) -> Dict[str, float]:
    """Per-class precision/recall/F1 for Global, Local, and False."""

    names = ("global", "local", "false")
    result: Dict[str, float] = {}
    for class_index, name in enumerate(names):
        predicted = (predicted_roles == class_index) & valid_mask
        target = (target_roles == class_index) & valid_mask
        true_positive = (predicted & target).sum()
        precision = _safe_ratio(true_positive, predicted.sum())
        recall = _safe_ratio(true_positive, target.sum())
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        result[f"{name}_precision"] = precision
        result[f"{name}_recall"] = recall
        result[f"{name}_f1"] = f1
    return result


def calibration_metrics(
    predicted_probability: Tensor,
    target_correct: Tensor,
    valid_mask: Tensor,
    bins: int = 15,
) -> Dict[str, float]:
    """Expected Calibration Error and Brier score."""

    probability = predicted_probability[valid_mask].float()
    target = target_correct[valid_mask].float()
    if probability.numel() == 0:
        return {"expected_calibration_error": float("nan"), "brier_score": float("nan")}
    brier = (probability - target).square().mean()
    ece = probability.new_zeros(())
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probability.device)
    for index in range(bins):
        if index == bins - 1:
            in_bin = (probability >= boundaries[index]) & (probability <= boundaries[index + 1])
        else:
            in_bin = (probability >= boundaries[index]) & (probability < boundaries[index + 1])
        if in_bin.any():
            ece = ece + in_bin.float().mean() * (
                probability[in_bin].mean() - target[in_bin].mean()
            ).abs()
    return {
        "expected_calibration_error": float(ece.cpu()),
        "brier_score": float(brier.cpu()),
    }


def homography_quality_metrics(
    src_xy: Tensor,
    dst_xy: Tensor,
    p_global: Tensor,
    valid_mask: Tensor,
    gt_h_ab: Tensor,
    image_height: int,
    image_width: int,
    *,
    probability_threshold: float = 0.5,
    success_corner_threshold: float = 5.0,
) -> Dict[str, float]:
    """Estimate H from selected Global matches and compare with ground truth."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required for homography quality metrics.")
    corner_errors: list[float] = []
    symmetric_errors: list[float] = []
    successes = 0
    for batch_index in range(src_xy.shape[0]):
        selected = valid_mask[batch_index] & (p_global[batch_index] >= probability_threshold)
        if int(selected.sum()) < 4:
            continue
        source = src_xy[batch_index, selected].detach().float().cpu().numpy()
        target = dst_xy[batch_index, selected].detach().float().cpu().numpy()
        matrix, _ = cv2.findHomography(
            source, target, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0
        )
        if matrix is None:
            continue
        predicted_h = torch.from_numpy(matrix).float().unsqueeze(0)
        corners = torch.tensor(
            [[[0.0, 0.0], [image_width - 1.0, 0.0],
              [image_width - 1.0, image_height - 1.0], [0.0, image_height - 1.0]]]
        )
        predicted_corners = apply_homography(corners, predicted_h)
        gt_corners = apply_homography(corners, gt_h_ab[batch_index : batch_index + 1].cpu())
        corner_error = torch.linalg.vector_norm(
            predicted_corners - gt_corners, dim=-1
        ).mean().item()
        corner_errors.append(corner_error)
        predicted_points = apply_homography(
            src_xy[batch_index : batch_index + 1].float().cpu(), predicted_h
        )
        forward = torch.linalg.vector_norm(
            predicted_points - dst_xy[batch_index : batch_index + 1].float().cpu(), dim=-1
        )
        inverse_h = torch.linalg.inv(predicted_h)
        backward_points = apply_homography(
            dst_xy[batch_index : batch_index + 1].float().cpu(), inverse_h
        )
        backward = torch.linalg.vector_norm(
            backward_points - src_xy[batch_index : batch_index + 1].float().cpu(), dim=-1
        )
        symmetric_errors.append(float(torch.sqrt(forward.square() + backward.square()).mean()))
        successes += int(corner_error <= success_corner_threshold)
    attempted = len(corner_errors)
    return {
        "homography_corner_error": sum(corner_errors) / attempted if attempted else float("nan"),
        "homography_symmetric_transfer_error": sum(symmetric_errors) / attempted if attempted else float("nan"),
        "homography_success_rate": successes / max(src_xy.shape[0], 1),
        "homography_estimation_rate": attempted / max(src_xy.shape[0], 1),
    }


def evaluate_output_against_dense_targets(
    output: GRMMatcherOutput,
    targets: Mapping[str, Tensor],
    image_height: int,
    image_width: int,
    correspondence_correct_threshold: float = 5.0,
) -> Dict[str, float]:
    """Convenience evaluation of public output against synthetic dense labels."""

    gt_dst = sample_dense_field(targets["gt_dense_mapping"], output.src_xy)
    overlap = sample_dense_field(targets["gt_overlap_a"].float(), output.src_xy)[..., 0] > 0.5
    valid = output.valid_mask
    metrics = threshold_matching_metrics(output.dst_xy, gt_dst, valid, overlap)
    metrics.update(localization_error_metrics(output.dst_xy, gt_dst, valid & overlap))
    metrics["spatial_coverage"] = spatial_coverage(
        output.src_xy,
        valid & (output.p_match >= 0.5),
        image_height,
        image_width,
    )
    role_targets = build_geometry_role_targets(
        output.src_xy,
        output.dst_xy,
        targets,
        correspondence_correct_threshold=correspondence_correct_threshold,
    )
    predicted_roles = torch.stack(
        (output.p_global, output.p_local, output.p_false), dim=-1
    ).argmax(dim=-1)
    metrics.update(
        role_classification_metrics(predicted_roles, role_targets.labels, valid)
    )
    metrics.update(
        calibration_metrics(output.p_match, role_targets.correct_mask, valid)
    )
    return metrics


def sample_dense_field(field: Tensor, points_xy: Tensor) -> Tensor:
    grid = image_xy_to_normalized_grid(
        points_xy.float(), field.shape[-2], field.shape[-1]
    ).unsqueeze(2)
    return F.grid_sample(
        field.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(-1).transpose(1, 2)


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> float:
    return float((numerator.float() / denominator.float().clamp_min(1.0)).cpu())


def _format_threshold(threshold: float) -> str:
    return str(int(threshold)) if float(threshold).is_integer() else str(threshold)

