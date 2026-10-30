"""Required debug visualizations for the GRM stitching pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from dense_match.geometry_utils import apply_homography_xy
from dense_match.grm_stitch_pipeline import GRMStitchingResult


def _colorize(value: np.ndarray, *, fixed_unit: bool = False) -> np.ndarray:
    array = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if fixed_unit:
        normalized = np.clip(array, 0.0, 1.0)
    else:
        high = max(float(np.percentile(array, 95)), 1e-6)
        normalized = np.clip(array / high, 0.0, 1.0)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _match_canvas(
    image_a: np.ndarray,
    image_b: np.ndarray,
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    colors: np.ndarray,
    *,
    vectors_from: np.ndarray | None = None,
) -> np.ndarray:
    height = max(image_a.shape[0], image_b.shape[0])
    width_a = image_a.shape[1]
    canvas = np.zeros((height, width_a + image_b.shape[1], 3), dtype=np.uint8)
    canvas[:image_a.shape[0], :width_a] = image_a
    canvas[:image_b.shape[0], width_a:] = image_b
    for index, (src, dst) in enumerate(zip(src_xy, dst_xy)):
        color = tuple(int(channel) for channel in colors[index])
        a = tuple(np.rint(src).astype(int))
        b = tuple((np.rint(dst) + np.array([width_a, 0])).astype(int))
        cv2.circle(canvas, a, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        if vectors_from is not None:
            start = tuple((np.rint(vectors_from[index]) + np.array([width_a, 0])).astype(int))
            cv2.arrowedLine(canvas, start, b, color, 1, cv2.LINE_AA, tipLength=0.2)
    return canvas


def _probability_points(image: np.ndarray, xy: np.ndarray, probability: np.ndarray) -> np.ndarray:
    output = image.copy()
    colors = cv2.applyColorMap(
        (np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8)[:, None],
        cv2.COLORMAP_TURBO,
    )[:, 0]
    for point, color in zip(xy, colors):
        cv2.circle(
            output,
            tuple(np.rint(point).astype(int)),
            4,
            tuple(int(channel) for channel in color),
            -1,
            cv2.LINE_AA,
        )
    return output


def write_grm_stitch_debug(
    output_dir: str | Path,
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    result: GRMStitchingResult,
) -> None:
    """Write all required geometry, decoder, seam, and blend diagnostics."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    matches = result.grm_correspondences
    src = matches.src_xy.detach().cpu().numpy()
    dst = matches.dst_xy.detach().cpu().numpy()
    reliability = matches.reliability.detach().cpu().numpy()
    reliability_colors = cv2.applyColorMap(
        (np.clip(reliability, 0.0, 1.0) * 255).astype(np.uint8)[:, None],
        cv2.COLORMAP_TURBO,
    )[:, 0]
    cv2.imwrite(
        str(path / "01_reliable_grm_matches.png"),
        _match_canvas(image_a_bgr, image_b_bgr, src, dst, reliability_colors),
    )

    inlier = result.geometry.H_base.inlier_mask.detach().cpu().numpy()
    if inlier.shape[0] != src.shape[0]:
        inlier = np.zeros(src.shape[0], dtype=bool)
    inlier_colors = np.where(
        inlier[:, None], np.array([[40, 220, 40]]), np.array([[20, 20, 230]])
    ).astype(np.uint8)
    cv2.imwrite(
        str(path / "02_magsac_inlier_outlier.png"),
        _match_canvas(image_a_bgr, image_b_bgr, src, dst, inlier_colors),
    )

    h_projected = apply_homography_xy(
        matches.src_xy, result.geometry.H_base.H_ab
    ).detach().cpu().numpy()
    cv2.imwrite(
        str(path / "03_h_residual_vectors.png"),
        _match_canvas(
            image_a_bgr, image_b_bgr, src, dst, reliability_colors,
            vectors_from=h_projected,
        ),
    )
    mesh = result.geometry.initial_residual_mesh
    cv2.imwrite(
        str(path / "04_mesh_support_count.png"),
        _colorize(mesh.support_weight.detach().cpu().numpy()),
    )
    mesh_vis = image_a_bgr.copy()
    rows, cols = mesh.mesh_offset.shape[:2]
    offsets = mesh.mesh_offset.detach().cpu().numpy()
    for y in range(rows):
        for x in range(cols):
            point = np.array([
                x * (image_a_bgr.shape[1] - 1) / max(cols - 1, 1),
                y * (image_a_bgr.shape[0] - 1) / max(rows - 1, 1),
            ])
            end = point + offsets[y, x]
            cv2.arrowedLine(
                mesh_vis,
                tuple(np.rint(point).astype(int)),
                tuple(np.rint(end).astype(int)),
                (20, 220, 255),
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )
    cv2.imwrite(str(path / "05_mesh_residual_vectors.png"), mesh_vis)
    cv2.imwrite(
        str(path / "06_w_h_distribution.png"),
        _probability_points(image_a_bgr, src, result.routing.w_h[0].detach().cpu().numpy()),
    )
    cv2.imwrite(
        str(path / "07_w_deform_distribution.png"),
        _probability_points(image_a_bgr, src, result.routing.w_deform[0].detach().cpu().numpy()),
    )
    confidence = result.deformation_confidence.detach().cpu().numpy()
    uncertainty = result.deformation_uncertainty.detach().cpu().numpy()
    cv2.imwrite(str(path / "08_deformation_confidence.png"), _colorize(confidence, fixed_unit=True))
    cv2.imwrite(str(path / "09_deformation_uncertainty.png"), _colorize(uncertainty))
    strain = result.debug_data["strain_canvas"]
    fold = mesh.fold_mask.detach().cpu().numpy().astype(np.float32)
    fold_resized = cv2.resize(fold, (strain.shape[1], strain.shape[0]), interpolation=cv2.INTER_NEAREST)
    strain_fold = _colorize(strain)
    strain_fold[fold_resized > 0] = (0, 0, 255)
    cv2.imwrite(str(path / "10_mesh_strain_fold.png"), strain_fold)

    canvas = result.debug_data["final_canvas"]
    difference = cv2.absdiff(canvas.image_a, canvas.image_b)
    difference[(canvas.valid_a == 0) | (canvas.valid_b == 0)] = 0
    cv2.imwrite(str(path / "11_warped_ab_difference.png"), difference)
    cv2.imwrite(str(path / "12_instability_map.png"), _colorize(result.instability_map, fixed_unit=True))
    cv2.imwrite(
        str(path / "13_binary_seam.png"),
        result.binary_seam_mask.astype(np.uint8) * 255,
    )
    seam_overlay = result.final_stitched_image.copy()
    seam_overlay[result.seam.seam_boundary] = (0, 0, 255)
    cv2.imwrite(str(path / "14_seam_overlay.png"), seam_overlay)
    cv2.imwrite(str(path / "15_h_only_stitch.png"), result.h_only_stitched_image)
    cv2.imwrite(str(path / "16_residual_mesh_stitch.png"), result.residual_mesh_stitched_image)
    cv2.imwrite(str(path / "17_learned_mesh_stitch.png"), result.learned_mesh_stitched_image)
    cv2.imwrite(str(path / "18_final_seam_blend_stitch.png"), result.final_stitched_image)

    report = dict(result.diagnostics)
    report["debug_files"] = sorted(item.name for item in path.glob("*.png"))
    report["fallback_reason"] = result.fallback_reason
    (path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
