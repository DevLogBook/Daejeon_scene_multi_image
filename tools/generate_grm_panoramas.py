#!/usr/bin/env python
"""Generate one panorama for every continuous image sequence in pair_list.json.

The standalone GRM matcher estimates adjacent A->B correspondences.  This tool
turns them into robust pairwise homographies, composes the sequence around its
middle frame, and renders a bounded panorama with either feathering or an
exposure-compensated GraphCut/multiband seam.

This is intentionally a GRM-only evaluation.  If the forward A->B prediction is
not solvable, the same pair is evaluated by GRM in reverse and the transform is
inverted.  No classical feature matcher, phase correlation, or copied transform
is allowed to hide a GRM failure.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
for search_path in (ROOT, TOOLS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_grm_landmarks import (  # noqa: E402
    Pair,
    fit_grm_homography,
    infer,
    load_model,
)
from dense_match.seam import SeamOptions, apply_seam_masks_to_stack  # noqa: E402


@dataclass
class Sequence:
    video_id: str
    images: list[Path]
    pair_ids: list[str]


@dataclass
class EdgeResult:
    pair_id: str
    image_a: str
    image_b: str
    method: str
    status: str
    selected_matches: int
    inlier_count: int
    inlier_ratio: float
    geometry: dict[str, float]
    warnings: list[str]
    homography_a_to_b: list[list[float]] | None
    elapsed_sec: float
    source_index: int = 0
    target_index: int = 1
    temporal_span: int = 1

    def matrix(self) -> np.ndarray | None:
        if self.homography_a_to_b is None:
            return None
        return np.asarray(self.homography_a_to_b, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequence panoramas with GRM pairwise matching."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pair-list",
        type=Path,
        help="Existing pair-list JSON. Requires --image-root.",
    )
    input_group.add_argument(
        "--sequence-dir",
        type=Path,
        help=(
            "Image directory scanned in natural filename order; consecutive "
            "images are matched as a temporal sequence."
        ),
    )
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--sequence-id", type=str, default=None)
    parser.add_argument(
        "--sequence-group-size",
        type=int,
        default=0,
        help=(
            "Split each temporal sequence into independent groups of this many "
            "images before matching; 0 keeps the full sequence."
        ),
    )
    parser.add_argument(
        "--sequence-group-overlap",
        type=int,
        default=0,
        help="Number of images shared by consecutive groups.",
    )
    parser.add_argument(
        "--image-extensions",
        type=str,
        default=".jpg,.jpeg,.png,.tif,.tiff",
        help="Comma-separated image suffixes used with --sequence-dir.",
    )
    parser.add_argument(
        "--sequence-image-regex",
        type=str,
        default=None,
        help=(
            "Optional regular expression matched against each filename stem. "
            "Generated files containing 'panorama' are always excluded."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--groups", type=str, default=None)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument(
        "--ransac-threshold-net-px",
        type=float,
        default=None,
        help=(
            "Legacy MAGSAC threshold in resized matcher pixels. When set, it "
            "overrides --ransac-threshold-original-px."
        ),
    )
    parser.add_argument(
        "--ransac-threshold-original-px",
        type=float,
        default=7.5,
        help=(
            "MAGSAC threshold in original-image pixels. This is the default "
            "resolution-independent coordinate convention."
        ),
    )
    parser.add_argument(
        "--matcher-input-width",
        type=int,
        default=None,
        help=(
            "Optional runtime matcher width. Use together with "
            "--matcher-input-height; both must be divisible by 16."
        ),
    )
    parser.add_argument(
        "--matcher-input-height",
        type=int,
        default=None,
        help="Optional runtime matcher height; does not change checkpoint weights.",
    )
    parser.add_argument("--max-dimension", type=int, default=6000)
    parser.add_argument("--max-megapixels", type=float, default=24.0)
    parser.add_argument("--feather-radius", type=float, default=96.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--blend-mode",
        choices=("feather", "graphcut_multiband"),
        default="graphcut_multiband",
    )
    parser.add_argument("--seam-max-dimension", type=int, default=1200)
    parser.add_argument("--seam-graphcut-cost-type", default="COST_COLOR_GRAD")
    parser.add_argument(
        "--seam-mask-dilation",
        type=int,
        default=None,
        help=(
            "Legacy override: dilation radius in seam-resolution pixels. "
            "When omitted, --seam-transition-width-px is used."
        ),
    )
    parser.add_argument(
        "--seam-transition-width-px",
        type=int,
        default=64,
        help=(
            "Approximate full-resolution width of the shared transition band. "
            "This is resolution independent and replaces the old seam-scale dilation default."
        ),
    )
    parser.add_argument(
        "--seam-balance-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Give every geometrically valid frame an interior ownership core before GraphCut. "
            "Prevents one frame from swallowing the overlap and leaving rectangular strips."
        ),
    )
    parser.add_argument(
        "--seam-search-radius",
        type=int,
        default=48,
        help=(
            "GraphCut search radius around balanced ownership boundaries, in seam-resolution pixels."
        ),
    )
    parser.add_argument("--blend-bands", type=int, default=7)
    parser.add_argument(
        "--exposure-compensator",
        choices=("none", "gain", "gain_blocks", "channels", "channels_blocks"),
        default="channels_blocks",
        help=(
            "Exposure compensation before seam estimation. 'channels_blocks' "
            "uses smooth per-channel spatial gains and is recommended when "
            "temporal UAV frames differ in both brightness and color."
        ),
    )
    parser.add_argument("--exposure-block-size", type=int, default=48)
    parser.add_argument("--exposure-gain-min", type=float, default=0.75)
    parser.add_argument("--exposure-gain-max", type=float, default=1.33)
    parser.add_argument(
        "--exposure-gain-filtering-iterations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--composition-model",
        choices=("pose_graph", "homography", "auto", "similarity", "rigid"),
        default="auto",
        help=(
            "Global transform model. 'pose_graph' jointly optimizes image "
            "similarities. 'homography' keeps the exact adjacent A->B projective "
            "mapping and is useful for short groups. 'similarity' and 'rigid' "
            "preserve the legacy per-edge projection modes."
        ),
    )
    parser.add_argument(
        "--max-edge-scale-change",
        type=float,
        default=0.02,
        help="Maximum |scale-1| per edge in similarity mode.",
    )
    parser.add_argument(
        "--pose-graph-max-span",
        type=int,
        default=2,
        help=(
            "Maximum temporal edge span. 2 estimates i<->i+1 and i<->i+2 "
            "constraints before joint optimization."
        ),
    )
    parser.add_argument("--pose-graph-min-inlier-ratio", type=float, default=0.12)
    parser.add_argument("--pose-graph-min-inliers", type=int, default=16)
    parser.add_argument("--pose-graph-max-p90-px", type=float, default=8.0)
    parser.add_argument("--pose-graph-scale-prior-weight", type=float, default=0.20)
    parser.add_argument("--pose-graph-smoothness-weight", type=float, default=0.35)
    parser.add_argument(
        "--pose-graph-max-global-scale-change",
        type=float,
        default=0.20,
        help=(
            "Maximum scale deviation of any optimized image pose from the "
            "center reference. Unlike legacy max-edge-scale-change, this is "
            "a sequence-level safety bound and does not accumulate per edge."
        ),
    )
    parser.add_argument("--pose-graph-huber-px", type=float, default=3.0)
    parser.add_argument("--pose-graph-max-iterations", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_sequences(pair_list: Path, image_root: Path) -> list[Sequence]:
    records = json.loads(pair_list.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("pairs", records.get("data", []))
    if not isinstance(records, list):
        raise ValueError("pair list must be a list or contain pairs/data")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["video_id"])].append(record)

    sequences: list[Sequence] = []
    for video_id in sorted(grouped):
        group = grouped[video_id]
        outgoing: dict[str, dict[str, Any]] = {}
        incoming: set[str] = set()
        for record in group:
            image_a = str(record["image_a"]).replace("\\", "/")
            image_b = str(record["image_b"]).replace("\\", "/")
            if image_a in outgoing:
                raise ValueError(f"{video_id}: branching sequence at {image_a}")
            outgoing[image_a] = record
            incoming.add(image_b)
        starts = [name for name in outgoing if name not in incoming]
        if len(starts) != 1:
            raise ValueError(f"{video_id}: expected one chain start, got {starts}")

        image_names = [starts[0]]
        pair_ids: list[str] = []
        current = starts[0]
        visited: set[str] = set()
        while current in outgoing:
            record = outgoing[current]
            pair_id = str(record["pair_id"])
            if pair_id in visited:
                raise ValueError(f"{video_id}: cycle at {pair_id}")
            visited.add(pair_id)
            pair_ids.append(pair_id)
            current = str(record["image_b"]).replace("\\", "/")
            image_names.append(current)
        if len(visited) != len(group):
            raise ValueError(
                f"{video_id}: chain covered {len(visited)}/{len(group)} pairs"
            )

        images = [(image_root / name).resolve() for name in image_names]
        missing = [str(path) for path in images if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{video_id}: missing images: {missing[:3]}")
        sequences.append(Sequence(video_id, images, pair_ids))
    return sequences


def natural_filename_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    """Sort frame/time numbers numerically while retaining textual context."""
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def load_temporal_sequence(
    sequence_dir: Path,
    sequence_id: str | None,
    image_extensions: str,
    image_name_regex: str | None = None,
) -> Sequence:
    """Build strict adjacent pairs from naturally sorted image filenames."""
    directory = sequence_dir.resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    suffixes = {
        suffix.strip().casefold()
        if suffix.strip().startswith(".")
        else f".{suffix.strip().casefold()}"
        for suffix in image_extensions.split(",")
        if suffix.strip()
    }
    if not suffixes:
        raise ValueError("--image-extensions must contain at least one suffix")
    name_pattern = re.compile(image_name_regex) if image_name_regex else None
    images = sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.casefold() in suffixes
            and "panorama" not in path.stem.casefold()
            and (name_pattern is None or name_pattern.fullmatch(path.stem))
        ),
        key=natural_filename_key,
    )
    if len(images) < 2:
        raise RuntimeError(
            f"{directory}: expected at least two temporal images, got {len(images)}"
        )
    video_id = sequence_id or directory.name
    pair_ids = [
        f"{video_id}_pair_{left.stem}_{right.stem}"
        for left, right in zip(images[:-1], images[1:])
    ]
    return Sequence(video_id=video_id, images=images, pair_ids=pair_ids)


def split_sequence_into_groups(
    sequence: Sequence,
    group_size: int,
    overlap: int = 0,
) -> list[Sequence]:
    """Split a temporal sequence while preserving local adjacent pair IDs."""
    if group_size <= 0 or group_size >= len(sequence.images):
        return [sequence]
    if overlap < 0 or overlap >= group_size:
        raise ValueError("sequence group overlap must satisfy 0 <= overlap < size")
    step = group_size - overlap
    groups: list[Sequence] = []
    start = 0
    group_number = 1
    while start < len(sequence.images):
        end = min(start + group_size, len(sequence.images))
        images = sequence.images[start:end]
        if len(images) < 2:
            break
        pair_ids = sequence.pair_ids[start : end - 1]
        groups.append(
            Sequence(
                video_id=f"{sequence.video_id}_group{group_number:02d}",
                images=images,
                pair_ids=pair_ids,
            )
        )
        if end == len(sequence.images):
            break
        start += step
        group_number += 1
    return groups


def read_cv_image(path: Path, flags: int) -> np.ndarray:
    """OpenCV image reader that supports non-ASCII Windows paths."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise FileNotFoundError(path) from error
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise RuntimeError(f"failed to decode image: {path}")
    return image


def image_shape(path: Path) -> tuple[int, int]:
    image = read_cv_image(path, cv2.IMREAD_GRAYSCALE)
    return image.shape[:2]


def transform_xy(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.astype(np.float32)[None], H.astype(np.float64)
    )[0].astype(np.float64)


def network_to_original_xy(
    points: np.ndarray,
    scale: tuple[float, float],
) -> np.ndarray:
    """Map resized-image pixel centers back to original-image pixel centers."""
    if scale[0] <= 0.0 or scale[1] <= 0.0:
        raise ValueError(f"invalid resize scale: {scale}")
    converted = points.astype(np.float64).copy()
    converted[:, 0] = (converted[:, 0] + 0.5) / float(scale[0]) - 0.5
    converted[:, 1] = (converted[:, 1] + 0.5) / float(scale[1]) - 0.5
    return converted


def fit_original_coordinate_homography(
    src_net: np.ndarray,
    dst_net: np.ndarray,
    selection: np.ndarray,
    scale_a: tuple[float, float],
    scale_b: tuple[float, float],
    threshold_original_px: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit A->B directly in original pixels to avoid threshold ambiguity."""
    if int(selection.sum()) < 4:
        return None, None
    src_original = network_to_original_xy(src_net[selection], scale_a)
    dst_original = network_to_original_xy(dst_net[selection], scale_b)
    cv2.setRNGSeed(0)
    H, inliers = cv2.findHomography(
        src_original.astype(np.float32),
        dst_original.astype(np.float32),
        cv2.USAC_MAGSAC,
        float(threshold_original_px),
        maxIters=10_000,
        confidence=0.999,
    )
    if H is None or not np.isfinite(H).all():
        return None, inliers
    if abs(H[2, 2]) > 1.0e-12:
        H /= H[2, 2]
    return H, inliers


def validate_homography(
    H: np.ndarray | None,
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
) -> tuple[bool, dict[str, float], list[str]]:
    warnings: list[str] = []
    empty = {
        "area_ratio": math.nan,
        "center_shift_diagonals": math.nan,
        "union_area_ratio": math.nan,
        "bbox_overlap_ratio": math.nan,
        "edge_ratio": math.nan,
    }
    if H is None or H.shape != (3, 3) or not np.isfinite(H).all():
        return False, empty, ["missing_or_nonfinite_homography"]

    H = H.astype(np.float64)
    if abs(H[2, 2]) > 1.0e-12:
        H = H / H[2, 2]
    h_a, w_a = shape_a
    h_b, w_b = shape_b
    corners_a = np.asarray(
        [[0, 0], [w_a - 1, 0], [w_a - 1, h_a - 1], [0, h_a - 1]],
        dtype=np.float64,
    )
    homogeneous = np.c_[corners_a, np.ones(4)] @ H.T
    denominators = homogeneous[:, 2]
    if (
        np.any(np.abs(denominators) < 1.0e-8)
        or np.min(denominators) < 0 < np.max(denominators)
    ):
        return False, empty, ["projective_horizon_crosses_image"]
    warped = homogeneous[:, :2] / denominators[:, None]
    if not np.isfinite(warped).all():
        return False, empty, ["nonfinite_warped_corners"]

    area = float(abs(cv2.contourArea(warped.astype(np.float32))))
    area_ratio = area / max(float(w_a * h_a), 1.0)
    edges = np.linalg.norm(np.roll(warped, -1, axis=0) - warped, axis=1)
    edge_ratio = float(edges.max() / max(edges.min(), 1.0e-6))
    center = transform_xy(
        np.asarray([[(w_a - 1) / 2, (h_a - 1) / 2]], dtype=np.float64), H
    )[0]
    center_b = np.asarray([(w_b - 1) / 2, (h_b - 1) / 2], dtype=np.float64)
    diagonal_b = math.hypot(w_b, h_b)
    center_shift = float(np.linalg.norm(center - center_b) / max(diagonal_b, 1.0))

    min_xy = np.minimum(warped.min(axis=0), np.asarray([0.0, 0.0]))
    max_xy = np.maximum(warped.max(axis=0), np.asarray([w_b - 1.0, h_b - 1.0]))
    union_ratio = float(
        np.prod(np.maximum(max_xy - min_xy, 0.0)) / max(float(w_b * h_b), 1.0)
    )
    warped_min = warped.min(axis=0)
    warped_max = warped.max(axis=0)
    overlap_min = np.maximum(warped_min, np.asarray([0.0, 0.0]))
    overlap_max = np.minimum(warped_max, np.asarray([w_b - 1.0, h_b - 1.0]))
    overlap_area = float(np.prod(np.maximum(overlap_max - overlap_min, 0.0)))
    bbox_area = float(np.prod(np.maximum(warped_max - warped_min, 0.0)))
    overlap_ratio = overlap_area / max(min(bbox_area, float(w_b * h_b)), 1.0)

    geometry = {
        "area_ratio": area_ratio,
        "center_shift_diagonals": center_shift,
        "union_area_ratio": union_ratio,
        "bbox_overlap_ratio": overlap_ratio,
        "edge_ratio": edge_ratio,
    }
    if not cv2.isContourConvex(warped.astype(np.float32)):
        warnings.append("warped_quad_not_convex")
    if not 0.12 <= area_ratio <= 8.0:
        warnings.append("implausible_area_ratio")
    if center_shift > 4.0:
        warnings.append("implausible_center_shift")
    if union_ratio > 12.0:
        warnings.append("implausible_union_area")
    if overlap_ratio < 0.005:
        warnings.append("negligible_overlap")
    if edge_ratio > 30.0:
        warnings.append("implausible_edge_ratio")
    return not warnings, geometry, warnings


def make_pair(
    sequence: Sequence,
    source_index: int,
    target_index: int | None = None,
) -> Pair:
    """Create a temporal pair without assuming that its span is exactly one."""
    if target_index is None:
        target_index = source_index + 1
    if not 0 <= source_index < target_index < len(sequence.images):
        raise IndexError(
            f"invalid temporal edge {source_index}->{target_index} for "
            f"{len(sequence.images)} images"
        )
    if target_index == source_index + 1:
        pair_id = sequence.pair_ids[source_index]
    else:
        pair_id = (
            f"{sequence.video_id}_posegraph_"
            f"{sequence.images[source_index].stem}_"
            f"{sequence.images[target_index].stem}"
        )
    return Pair(
        pair_id=pair_id,
        video_id=sequence.video_id,
        scenario="sequence_panorama",
        quality="unknown",
        status="unlabeled",
        image_a=sequence.images[source_index],
        image_b=sequence.images[target_index],
        points_a=np.empty((0, 2), dtype=np.float64),
        points_b=np.empty((0, 2), dtype=np.float64),
    )


def estimate_grm_edge(
    model: Any,
    config: Any,
    pair: Pair,
    device: torch.device,
    threshold_net_px: float | None,
    threshold_original_px: float,
) -> tuple[np.ndarray | None, int, int, dict[str, float]]:
    # The landmark inference helper exposes candidate reranking explicitly.
    # Sequence panoramas use the role-free reliability selector so this tool
    # remains compatible with checkpoints whose stored mode is unset.
    src, dst, values, scale_a, scale_b, _ = infer(
        model, config, pair, device, False, "reliability"
    )
    valid = values["valid"]
    if threshold_net_px is not None:
        H, inliers = fit_grm_homography(
            src, dst, valid, scale_a, scale_b, threshold_net_px
        )
        threshold_coordinate_system = 0.0
        effective_threshold = float(threshold_net_px)
    else:
        H, inliers = fit_original_coordinate_homography(
            src,
            dst,
            valid,
            scale_a,
            scale_b,
            threshold_original_px,
        )
        threshold_coordinate_system = 1.0
        effective_threshold = float(threshold_original_px)
    diagnostics: dict[str, float] = {}
    selected_count = int(valid.sum())
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if H is not None and selected_count > 0:
        src_original = network_to_original_xy(src[valid], scale_a)
        dst_original = network_to_original_xy(dst[valid], scale_b)
        projected = transform_xy(src_original, H)
        errors = np.linalg.norm(projected - dst_original, axis=1)
        finite = np.isfinite(errors)
        if inliers is not None:
            inlier_mask = np.asarray(inliers).reshape(-1).astype(bool)
            finite &= inlier_mask[: len(finite)]
        reliable_errors = errors[finite]
        if reliable_errors.size:
            diagnostics = {
                "inlier_reprojection_median_px": float(
                    np.median(reliable_errors)
                ),
                "inlier_reprojection_mean_px": float(
                    np.mean(reliable_errors)
                ),
                "inlier_reprojection_p90_px": float(
                    np.percentile(reliable_errors, 90)
                ),
                "inlier_reprojection_max_px": float(
                    np.max(reliable_errors)
                ),
            }
    diagnostics["ransac_threshold_value"] = effective_threshold
    # 1 means original-image pixels; 0 preserves the legacy network-pixel mode.
    diagnostics["ransac_threshold_is_original_px"] = threshold_coordinate_system
    return H, selected_count, inlier_count, diagnostics


def estimate_sequence_edges(
    sequence: Sequence,
    model: Any,
    config: Any,
    device: torch.device,
    threshold_net_px: float | None,
    threshold_original_px: float,
    max_span: int = 1,
) -> list[EdgeResult]:
    if max_span < 1:
        raise ValueError("max_span must be at least one")
    edge_specs = [
        (source_index, source_index + span)
        for span in range(1, min(max_span, len(sequence.images) - 1) + 1)
        for source_index in range(len(sequence.images) - span)
    ]
    results: list[EdgeResult] = []
    for edge_number, (source_index, target_index) in enumerate(edge_specs, start=1):
        start = time.perf_counter()
        pair = make_pair(sequence, source_index, target_index)
        shape_a = image_shape(pair.image_a)
        shape_b = image_shape(pair.image_b)
        H, selected, inliers, reprojection = estimate_grm_edge(
            model,
            config,
            pair,
            device,
            threshold_net_px,
            threshold_original_px,
        )
        valid, geometry, warnings = validate_homography(H, shape_a, shape_b)
        geometry.update(reprojection)
        method = "grm_magsac"

        if not valid:
            reverse_pair = Pair(
                pair_id=pair.pair_id,
                video_id=pair.video_id,
                scenario=pair.scenario,
                quality=pair.quality,
                status=pair.status,
                image_a=pair.image_b,
                image_b=pair.image_a,
                points_a=pair.points_b,
                points_b=pair.points_a,
            )
            (
                H_reverse,
                selected_reverse,
                inliers_reverse,
                reprojection_reverse,
            ) = estimate_grm_edge(
                model,
                config,
                reverse_pair,
                device,
                threshold_net_px,
                threshold_original_px,
            )
            H_from_reverse = None
            if H_reverse is not None:
                try:
                    H_from_reverse = np.linalg.inv(H_reverse)
                    H_from_reverse /= H_from_reverse[2, 2]
                except np.linalg.LinAlgError:
                    H_from_reverse = None
            valid_reverse, geometry_reverse, warnings_reverse = validate_homography(
                H_from_reverse, shape_a, shape_b
            )
            if valid_reverse:
                H = H_from_reverse
                selected = selected_reverse
                inliers = inliers_reverse
                geometry = geometry_reverse
                geometry.update(reprojection_reverse)
                warnings = ["forward_grm_rejected"] + warnings
                method = "grm_reverse_magsac"
                valid = True
            else:
                H = None
                selected = max(selected, selected_reverse)
                inliers = max(inliers, inliers_reverse)
                geometry = geometry_reverse
                warnings = (
                    ["forward_grm_rejected"]
                    + warnings
                    + ["reverse_grm_rejected"]
                    + warnings_reverse
                )
                method = "unresolved"

        results.append(
            EdgeResult(
                pair_id=pair.pair_id,
                image_a=pair.image_a.name,
                image_b=pair.image_b.name,
                method=method,
                status="ok" if valid and H is not None else "unresolved",
                selected_matches=int(selected),
                inlier_count=int(inliers),
                inlier_ratio=float(inliers / selected) if selected else 0.0,
                geometry={key: float(value) for key, value in geometry.items()},
                warnings=warnings,
                homography_a_to_b=H.tolist() if H is not None else None,
                elapsed_sec=float(time.perf_counter() - start),
                source_index=source_index,
                target_index=target_index,
                temporal_span=target_index - source_index,
            )
        )
        print(
            f"  edge {edge_number:02d}/{len(edge_specs):02d} "
            f"[{source_index}->{target_index}] "
            f"{method} matches={selected} inliers={inliers}"
            + (
                " median="
                f"{geometry['inlier_reprojection_median_px']:.2f}px"
                " p90="
                f"{geometry['inlier_reprojection_p90_px']:.2f}px"
                if "inlier_reprojection_median_px" in geometry
                else ""
            ),
            flush=True,
        )
    return results


def compose_transforms(
    edges: list[EdgeResult], image_count: int, reference_index: int
) -> list[np.ndarray]:
    transforms: list[np.ndarray | None] = [None] * image_count
    transforms[reference_index] = np.eye(3, dtype=np.float64)
    for index in range(reference_index - 1, -1, -1):
        H = edges[index].matrix()
        if H is None:
            raise RuntimeError(f"edge {index} remains unresolved")
        transforms[index] = transforms[index + 1] @ H
        transforms[index] /= transforms[index][2, 2]
    for index in range(reference_index + 1, image_count):
        H = edges[index - 1].matrix()
        if H is None:
            raise RuntimeError(f"edge {index - 1} remains unresolved")
        transforms[index] = transforms[index - 1] @ np.linalg.inv(H)
        transforms[index] /= transforms[index][2, 2]
    return [transform for transform in transforms if transform is not None]


def affine_approximation(
    H: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    h, w = shape
    xs = np.linspace(0, w - 1, 4)
    ys = np.linspace(0, h - 1, 4)
    src = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)
    dst = transform_xy(src, H).astype(np.float32)
    affine, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
    if affine is None:
        return H
    return np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float64)


def constrained_similarity_approximation(
    H: np.ndarray,
    shape: tuple[int, int],
    *,
    preserve_scale: bool,
    max_scale_change: float,
) -> np.ndarray:
    """Project a pairwise transform to a centre-preserving similarity.

    The closest proper rotation is obtained through the polar/SVD projection.
    Translation is recomputed so that the image centre retains its mapping.
    This prevents small affine shear and anisotropic-scale errors from being
    amplified by long sequence composition.
    """
    affine = affine_approximation(H, shape)
    linear = np.asarray(affine[:2, :2], dtype=np.float64)
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    if preserve_scale:
        raw_scale = math.sqrt(max(abs(float(np.linalg.det(linear))), 1.0e-12))
        scale = float(
            np.clip(
                raw_scale,
                1.0 - max_scale_change,
                1.0 + max_scale_change,
            )
        )
    else:
        scale = 1.0
    h, w = shape
    centre = np.asarray([(w - 1.0) * 0.5, (h - 1.0) * 0.5], dtype=np.float64)
    mapped_centre = transform_xy(centre[None].astype(np.float32), H)[0]
    linear_similarity = scale * rotation
    translation = mapped_centre - linear_similarity @ centre
    result = np.eye(3, dtype=np.float64)
    result[:2, :2] = linear_similarity
    result[:2, 2] = translation
    return result


@dataclass
class PoseGraphConstraint:
    """One robust temporal constraint in centred, diagonal-normalized coordinates."""

    source_index: int
    target_index: int
    source_points: np.ndarray
    target_points: np.ndarray
    similarity_source_to_target: np.ndarray
    weight: float
    pair_id: str


def coordinate_normalization(shape: tuple[int, int]) -> np.ndarray:
    """Return a pixel->centred-normalized transform for an image shape."""
    height, width = shape
    diagonal = max(math.hypot(width - 1.0, height - 1.0), 1.0)
    return np.asarray(
        [
            [1.0 / diagonal, 0.0, -(width - 1.0) * 0.5 / diagonal],
            [0.0, 1.0 / diagonal, -(height - 1.0) * 0.5 / diagonal],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares proper similarity mapping source points to target points."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target must both have shape [N, 2]")
    if len(source) < 2:
        raise ValueError("at least two points are required for a similarity fit")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    scale = float(np.dot(singular_values, sign) / max(source_variance, 1.0e-12))
    translation = target_mean - scale * rotation @ source_mean
    result = np.eye(3, dtype=np.float64)
    result[:2, :2] = scale * rotation
    result[:2, 2] = translation
    return result


def pose_matrix(parameters: np.ndarray) -> np.ndarray:
    log_scale, angle, tx, ty = [float(value) for value in parameters]
    scale = math.exp(log_scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        [
            [scale * cosine, -scale * sine, tx],
            [scale * sine, scale * cosine, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def similarity_parameters(matrix_value: np.ndarray) -> np.ndarray:
    linear = np.asarray(matrix_value[:2, :2], dtype=np.float64)
    scale = math.sqrt(max(abs(float(np.linalg.det(linear))), 1.0e-12))
    angle = math.atan2(linear[1, 0], linear[0, 0])
    return np.asarray(
        [math.log(scale), angle, matrix_value[0, 2], matrix_value[1, 2]],
        dtype=np.float64,
    )


def build_pose_graph_constraints(
    sequence: Sequence,
    edges: list[EdgeResult],
    *,
    min_inlier_ratio: float,
    min_inliers: int,
    max_p90_px: float,
    huber_px: float,
) -> tuple[list[PoseGraphConstraint], dict[str, Any]]:
    """Convert reliable GRM edges to normalized multi-point constraints."""
    shapes = [image_shape(path) for path in sequence.images]
    normalizations = [coordinate_normalization(shape) for shape in shapes]
    constraints: list[PoseGraphConstraint] = []
    rejected: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for edge in edges:
        H = edge.matrix()
        reasons: list[str] = []
        if H is None:
            reasons.append("missing_transform")
        if edge.inlier_count < min_inliers:
            reasons.append("too_few_inliers")
        if edge.inlier_ratio < min_inlier_ratio:
            reasons.append("low_inlier_ratio")
        p90 = float(edge.geometry.get("inlier_reprojection_p90_px", math.inf))
        if not math.isfinite(p90) or p90 > max_p90_px:
            reasons.append("high_or_missing_p90")
        if H is None or reasons:
            edge.warnings.extend(
                reason for reason in reasons if reason not in edge.warnings
            )
            rejected.append({"pair_id": edge.pair_id, "reasons": reasons})
            continue

        source_index = edge.source_index
        target_index = edge.target_index
        source_height, source_width = shapes[source_index]
        xs = np.linspace(0.1 * (source_width - 1), 0.9 * (source_width - 1), 4)
        ys = np.linspace(0.1 * (source_height - 1), 0.9 * (source_height - 1), 4)
        source_pixels = np.asarray(
            [(x, y) for y in ys for x in xs], dtype=np.float64
        )
        target_pixels = transform_xy(source_pixels, H)
        finite = np.isfinite(target_pixels).all(axis=1)
        source_pixels = source_pixels[finite]
        target_pixels = target_pixels[finite]
        if len(source_pixels) < 4:
            edge.warnings.append("insufficient_finite_pose_graph_samples")
            rejected.append(
                {"pair_id": edge.pair_id, "reasons": ["nonfinite_projection"]}
            )
            continue

        source_points = transform_xy(
            source_pixels, normalizations[source_index]
        )
        target_points = transform_xy(
            target_pixels, normalizations[target_index]
        )
        pair_similarity = fit_similarity_transform(source_points, target_points)
        raw_scale = math.sqrt(
            max(abs(float(np.linalg.det(pair_similarity[:2, :2]))), 1.0e-12)
        )
        if not 0.5 <= raw_scale <= 2.0:
            edge.warnings.append("implausible_pose_graph_scale")
            rejected.append(
                {"pair_id": edge.pair_id, "reasons": ["implausible_scale"]}
            )
            continue

        median_error = float(
            edge.geometry.get("inlier_reprojection_median_px", huber_px)
        )
        count_factor = min(edge.inlier_count / 64.0, 1.0)
        error_factor = math.exp(-median_error / max(huber_px, 1.0e-6))
        span_factor = 1.0 / math.sqrt(max(edge.temporal_span, 1))
        weight = max(
            1.0e-4,
            edge.inlier_ratio * count_factor * error_factor * span_factor,
        )
        edge.geometry["pose_graph_weight"] = float(weight)
        edge.geometry["pose_graph_raw_similarity_scale"] = float(raw_scale)
        constraints.append(
            PoseGraphConstraint(
                source_index=source_index,
                target_index=target_index,
                source_points=source_points,
                target_points=target_points,
                similarity_source_to_target=pair_similarity,
                weight=weight,
                pair_id=edge.pair_id,
            )
        )
        accepted_rows.append(
            {
                "pair_id": edge.pair_id,
                "source_index": source_index,
                "target_index": target_index,
                "temporal_span": edge.temporal_span,
                "weight": float(weight),
                "raw_similarity_scale": float(raw_scale),
                "inlier_ratio": float(edge.inlier_ratio),
                "inlier_count": int(edge.inlier_count),
                "p90_px": float(p90),
            }
        )
    return constraints, {"accepted_edges": accepted_rows, "rejected_edges": rejected}


def initialize_pose_graph(
    image_count: int,
    reference_index: int,
    constraints: list[PoseGraphConstraint],
) -> list[np.ndarray]:
    """Initialize poses through the maximum-reliability temporal spanning tree."""
    adjacency: list[list[tuple[int, np.ndarray, float]]] = [
        [] for _ in range(image_count)
    ]
    for constraint in constraints:
        source_to_target = constraint.similarity_source_to_target
        adjacency[constraint.source_index].append(
            (constraint.target_index, source_to_target, constraint.weight)
        )
        adjacency[constraint.target_index].append(
            (constraint.source_index, np.linalg.inv(source_to_target), constraint.weight)
        )

    poses: list[np.ndarray | None] = [None] * image_count
    costs = [math.inf] * image_count
    costs[reference_index] = 0.0
    poses[reference_index] = np.eye(3, dtype=np.float64)
    queue: list[tuple[float, int]] = [(0.0, reference_index)]
    while queue:
        cost, current = heapq.heappop(queue)
        if cost > costs[current] + 1.0e-12:
            continue
        assert poses[current] is not None
        for neighbor, current_to_neighbor, weight in adjacency[current]:
            candidate_cost = cost - math.log(max(weight, 1.0e-8))
            if candidate_cost >= costs[neighbor]:
                continue
            # G_current(x) = G_neighbor(T_current_to_neighbor(x)).
            candidate_pose = poses[current] @ np.linalg.inv(current_to_neighbor)
            candidate_pose /= candidate_pose[2, 2]
            poses[neighbor] = candidate_pose
            costs[neighbor] = candidate_cost
            heapq.heappush(queue, (candidate_cost, neighbor))
    missing = [index for index, pose in enumerate(poses) if pose is None]
    if missing:
        raise RuntimeError(
            "pose graph is disconnected; unreachable image indices: "
            + ", ".join(map(str, missing))
        )
    return [pose for pose in poses if pose is not None]


def optimize_similarity_pose_graph(
    sequence: Sequence,
    edges: list[EdgeResult],
    reference_index: int,
    *,
    min_inlier_ratio: float,
    min_inliers: int,
    max_p90_px: float,
    scale_prior_weight: float,
    smoothness_weight: float,
    max_global_scale_change: float,
    huber_px: float,
    max_iterations: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Jointly solve all image-to-reference similarities with robust soft priors."""
    constraints, diagnostics = build_pose_graph_constraints(
        sequence,
        edges,
        min_inlier_ratio=min_inlier_ratio,
        min_inliers=min_inliers,
        max_p90_px=max_p90_px,
        huber_px=huber_px,
    )
    image_count = len(sequence.images)
    initial_poses = initialize_pose_graph(image_count, reference_index, constraints)
    variable_indices = [index for index in range(image_count) if index != reference_index]
    index_to_offset = {index: offset for offset, index in enumerate(variable_indices)}
    initial_vector = np.concatenate(
        [similarity_parameters(initial_poses[index]) for index in variable_indices]
    )

    def unpack(vector: np.ndarray) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for image_index in range(image_count):
            if image_index == reference_index:
                result.append(np.eye(3, dtype=np.float64))
            else:
                offset = index_to_offset[image_index] * 4
                result.append(pose_matrix(vector[offset : offset + 4]))
        return result

    def residuals(vector: np.ndarray, include_priors: bool = True) -> np.ndarray:
        poses = unpack(vector)
        rows: list[np.ndarray] = []
        for constraint in constraints:
            source_global = transform_xy(
                constraint.source_points, poses[constraint.source_index]
            )
            target_global = transform_xy(
                constraint.target_points, poses[constraint.target_index]
            )
            rows.append(
                math.sqrt(constraint.weight)
                * (source_global - target_global).reshape(-1)
            )
        if include_priors and scale_prior_weight > 0.0:
            scale_rows = np.asarray(
                [
                    similarity_parameters(poses[index])[0]
                    for index in variable_indices
                ],
                dtype=np.float64,
            )
            rows.append(math.sqrt(scale_prior_weight) * scale_rows)
        if include_priors and smoothness_weight > 0.0 and image_count >= 3:
            parameters = np.stack(
                [similarity_parameters(pose) for pose in poses], axis=0
            )
            second_difference = (
                parameters[:-2] - 2.0 * parameters[1:-1] + parameters[2:]
            )
            rows.append(math.sqrt(smoothness_weight) * second_difference.reshape(-1))
        if not rows:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(rows)

    reference_diagonal = max(
        math.hypot(
            image_shape(sequence.images[reference_index])[1] - 1.0,
            image_shape(sequence.images[reference_index])[0] - 1.0,
        ),
        1.0,
    )
    if not 0.0 < max_global_scale_change < 1.0:
        raise ValueError("max_global_scale_change must be in (0, 1)")
    minimum_scale = 1.0 - max_global_scale_change
    maximum_scale = 1.0 + max_global_scale_change
    lower = np.tile(
        np.asarray([math.log(minimum_scale), -math.pi, -10.0, -10.0]),
        len(variable_indices),
    )
    upper = np.tile(
        np.asarray([math.log(maximum_scale), math.pi, 10.0, 10.0]),
        len(variable_indices),
    )
    initial_vector = np.clip(initial_vector, lower + 1.0e-8, upper - 1.0e-8)
    initial_data_residual = residuals(initial_vector, include_priors=False)
    solution = least_squares(
        residuals,
        initial_vector,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=max(huber_px / reference_diagonal, 1.0e-6),
        max_nfev=max_iterations,
        x_scale="jac",
    )
    optimized_normalized = unpack(solution.x)
    final_data_residual = residuals(solution.x, include_priors=False)

    normalizations = [
        coordinate_normalization(image_shape(path)) for path in sequence.images
    ]
    reference_inverse = np.linalg.inv(normalizations[reference_index])
    pixel_transforms = [
        reference_inverse @ pose @ normalization
        for pose, normalization in zip(optimized_normalized, normalizations)
    ]
    pixel_transforms = [matrix_value / matrix_value[2, 2] for matrix_value in pixel_transforms]
    optimized_parameters = np.stack(
        [similarity_parameters(pose) for pose in optimized_normalized], axis=0
    )

    def rms_px(values: np.ndarray) -> float:
        if values.size == 0:
            return math.nan
        return float(math.sqrt(np.mean(values**2)) * reference_diagonal)

    diagnostics.update(
        {
            "optimizer_success": bool(solution.success),
            "optimizer_status": int(solution.status),
            "optimizer_message": str(solution.message),
            "optimizer_nfev": int(solution.nfev),
            "initial_weighted_rms_px": rms_px(initial_data_residual),
            "final_weighted_rms_px": rms_px(final_data_residual),
            "pose_scales": np.exp(optimized_parameters[:, 0]).tolist(),
            "pose_rotations_degrees": np.degrees(optimized_parameters[:, 1]).tolist(),
            "reference_index": int(reference_index),
            "scale_prior_weight": float(scale_prior_weight),
            "smoothness_weight": float(smoothness_weight),
            "global_scale_bounds": [float(minimum_scale), float(maximum_scale)],
        }
    )
    return pixel_transforms, diagnostics


def transformed_polygons(
    sequence: Sequence, transforms: list[np.ndarray]
) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for path, H in zip(sequence.images, transforms):
        h, w = image_shape(path)
        corners = np.asarray(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float64,
        )
        polygons.append(transform_xy(corners, H))
    return polygons


def sequence_geometry_is_sane(
    sequence: Sequence, transforms: list[np.ndarray]
) -> tuple[bool, dict[str, float]]:
    polygons = transformed_polygons(sequence, transforms)
    all_points = np.concatenate(polygons, axis=0)
    if not np.isfinite(all_points).all():
        return False, {"raw_width": math.inf, "raw_height": math.inf}
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    width, height = max_xy - min_xy
    base_shapes = [image_shape(path) for path in sequence.images]
    max_width = max(shape[1] for shape in base_shapes)
    max_height = max(shape[0] for shape in base_shapes)
    image_areas = [shape[0] * shape[1] for shape in base_shapes]
    projected_ratios = [
        abs(cv2.contourArea(poly.astype(np.float32))) / max(area, 1)
        for poly, area in zip(polygons, image_areas)
    ]
    sane = (
        width > 1
        and height > 1
        and width < max_width * (len(sequence.images) + 2) * 2.0
        and height < max_height * (len(sequence.images) + 2) * 2.0
        and min(projected_ratios) > 0.005
        and max(projected_ratios) < 100.0
    )
    return sane, {
        "raw_width": float(width),
        "raw_height": float(height),
        "min_projected_area_ratio": float(min(projected_ratios)),
        "max_projected_area_ratio": float(max(projected_ratios)),
    }


def stabilize_if_needed(
    sequence: Sequence,
    edges: list[EdgeResult],
    reference_index: int,
    composition_model: str = "auto",
    max_edge_scale_change: float = 0.02,
    *,
    pose_graph_edges: list[EdgeResult] | None = None,
    pose_graph_min_inlier_ratio: float = 0.12,
    pose_graph_min_inliers: int = 16,
    pose_graph_max_p90_px: float = 8.0,
    pose_graph_scale_prior_weight: float = 0.20,
    pose_graph_smoothness_weight: float = 0.35,
    pose_graph_max_global_scale_change: float = 0.20,
    pose_graph_huber_px: float = 3.0,
    pose_graph_max_iterations: int = 200,
) -> tuple[list[np.ndarray], str, dict[str, Any]]:
    if composition_model == "homography":
        transforms = compose_transforms(edges, len(sequence.images), reference_index)
        sane, geometry = sequence_geometry_is_sane(sequence, transforms)
        if not sane:
            raise RuntimeError(
                f"{sequence.video_id}: raw homography composition is unstable"
            )
        return transforms, "homography", geometry

    if composition_model == "auto" and len(sequence.images) <= 5:
        transforms = compose_transforms(edges, len(sequence.images), reference_index)
        sane, geometry = sequence_geometry_is_sane(sequence, transforms)
        if sane:
            geometry["auto_selection_reason"] = "short_sequence_preserves_pair_geometry"
            return transforms, "homography_short_group", geometry

    if composition_model in {"pose_graph", "auto"}:
        graph_edges = pose_graph_edges if pose_graph_edges is not None else edges
        try:
            transforms, pose_graph_diagnostics = optimize_similarity_pose_graph(
                sequence,
                graph_edges,
                reference_index,
                min_inlier_ratio=pose_graph_min_inlier_ratio,
                min_inliers=pose_graph_min_inliers,
                max_p90_px=pose_graph_max_p90_px,
                scale_prior_weight=pose_graph_scale_prior_weight,
                smoothness_weight=pose_graph_smoothness_weight,
                max_global_scale_change=pose_graph_max_global_scale_change,
                huber_px=pose_graph_huber_px,
                max_iterations=pose_graph_max_iterations,
            )
            sane, geometry = sequence_geometry_is_sane(sequence, transforms)
            geometry["pose_graph"] = pose_graph_diagnostics
            if not sane:
                raise RuntimeError("jointly optimized pose graph has unstable geometry")
            return transforms, "similarity_pose_graph", geometry
        except Exception:
            if composition_model == "pose_graph":
                raise
            # Auto mode retains the original homography/affine fallback.  The
            # explicit pose_graph mode is strict so failures are never hidden.

    if composition_model in {"similarity", "rigid"}:
        for index, edge in enumerate(edges):
            H = edge.matrix()
            if H is None:
                continue
            edge.homography_a_to_b = constrained_similarity_approximation(
                H,
                image_shape(sequence.images[index]),
                preserve_scale=composition_model == "similarity",
                max_scale_change=max_edge_scale_change,
            ).tolist()
            edge.warnings.append(f"projected_to_{composition_model}_for_sequence_stability")
        transforms = compose_transforms(edges, len(sequence.images), reference_index)
        sane, geometry = sequence_geometry_is_sane(sequence, transforms)
        if not sane:
            raise RuntimeError(
                f"{sequence.video_id}: sequence geometry remains unstable after "
                f"{composition_model} projection"
            )
        return transforms, f"{composition_model}_stabilized", geometry

    transforms = compose_transforms(edges, len(sequence.images), reference_index)
    sane, geometry = sequence_geometry_is_sane(sequence, transforms)
    if sane:
        return transforms, "homography", geometry

    for index, edge in enumerate(edges):
        H = edge.matrix()
        if H is None:
            continue
        edge.homography_a_to_b = affine_approximation(
            H, image_shape(sequence.images[index])
        ).tolist()
        edge.warnings.append("projected_to_affine_for_sequence_stability")
    transforms = compose_transforms(edges, len(sequence.images), reference_index)
    sane_affine, geometry_affine = sequence_geometry_is_sane(sequence, transforms)
    if not sane_affine:
        raise RuntimeError(
            f"{sequence.video_id}: sequence geometry remains unstable after affine projection"
        )
    return transforms, "affine_stabilized", geometry_affine


def output_transform_and_size(
    sequence: Sequence,
    transforms: list[np.ndarray],
    max_dimension: int,
    max_megapixels: float,
) -> tuple[np.ndarray, tuple[int, int], float, dict[str, float]]:
    polygons = transformed_polygons(sequence, transforms)
    all_points = np.concatenate(polygons, axis=0)
    min_xy = np.floor(all_points.min(axis=0))
    max_xy = np.ceil(all_points.max(axis=0))
    raw_width = max(float(max_xy[0] - min_xy[0] + 1), 1.0)
    raw_height = max(float(max_xy[1] - min_xy[1] + 1), 1.0)
    scale_limits = [1.0]
    if max_dimension > 0:
        scale_limits.append(max_dimension / max(raw_width, raw_height))
    if max_megapixels > 0.0:
        max_pixels = max_megapixels * 1_000_000.0
        scale_limits.append(
            math.sqrt(max_pixels / max(raw_width * raw_height, 1.0))
        )
    scale = min(scale_limits)
    width = max(1, int(math.ceil(raw_width * scale)))
    height = max(1, int(math.ceil(raw_height * scale)))
    canvas_transform = np.asarray(
        [
            [scale, 0.0, -min_xy[0] * scale],
            [0.0, scale, -min_xy[1] * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (
        canvas_transform,
        (width, height),
        scale,
        {
            "raw_width": raw_width,
            "raw_height": raw_height,
            "output_width": float(width),
            "output_height": float(height),
            "output_scale": float(scale),
        },
    )


def feather_blend(
    sequence: Sequence,
    transforms: list[np.ndarray],
    output_path: Path,
    max_dimension: int,
    max_megapixels: float,
    feather_radius: float,
    jpeg_quality: int,
) -> dict[str, float]:
    canvas_transform, (width, height), scale, canvas_info = (
        output_transform_and_size(
            sequence, transforms, max_dimension, max_megapixels
        )
    )
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    effective_feather = max(8.0, feather_radius * min(1.0, scale))

    for path, global_transform in zip(sequence.images, transforms):
        image = read_cv_image(path, cv2.IMREAD_COLOR)
        H_output = canvas_transform @ global_transform
        warped = cv2.warpPerspective(
            image,
            H_output,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        mask = cv2.warpPerspective(
            source_mask,
            H_output,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        weights = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        weights = np.minimum(weights, effective_feather)
        weights[mask == 0] = 0.0
        accumulator += warped.astype(np.float32) * weights[..., None]
        weight_sum += weights

    valid = weight_sum > 0
    if not np.any(valid):
        raise RuntimeError(f"{sequence.video_id}: empty panorama")
    output = np.zeros_like(accumulator, dtype=np.uint8)
    output[valid] = np.clip(
        accumulator[valid] / weight_sum[valid, None], 0, 255
    ).astype(np.uint8)

    ys, xs = np.nonzero(valid)
    output = output[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        output,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    ):
        raise RuntimeError(f"failed to write {output_path}")
    canvas_info["final_width"] = float(output.shape[1])
    canvas_info["final_height"] = float(output.shape[0])
    return canvas_info


def balanced_graphcut_candidate_masks(
    valid_masks: list[np.ndarray],
    search_radius: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Restrict GraphCut to a band around balanced, interior-first ownership.

    Vanilla multi-image GraphCut is free to give nearly the entire common
    overlap to the first image.  In a high-overlap temporal scan that leaves
    later frames as thin rectangular edge strips.  We first assign every
    covered pixel to the valid image where it lies deepest inside that image,
    then dilate those ownership cores to create a local GraphCut search band.
    The core masks cover the complete valid union, so this restriction cannot
    create holes.
    """
    if not valid_masks:
        return [], {
            "enabled": True,
            "search_radius": int(max(0, search_radius)),
            "core_source_area_ratio": [],
            "candidate_overlap_ratio": 0.0,
        }
    shape = valid_masks[0].shape[:2]
    if any(mask.shape[:2] != shape for mask in valid_masks):
        raise ValueError("All seam valid masks must have the same spatial shape")

    valid_stack = np.stack([mask > 0 for mask in valid_masks], axis=0)
    valid_union = valid_stack.any(axis=0)
    distances = []
    for valid in valid_stack:
        # A one-pixel zero border makes distance-to-boundary meaningful even
        # when a warped mask happens to cover the complete seam canvas.
        padded = np.pad(valid.astype(np.uint8), 1, mode="constant")
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        distances.append(np.where(valid, distance, -1.0))
    distance_stack = np.stack(distances, axis=0)
    ownership = np.argmax(distance_stack, axis=0)

    core_masks = [
        np.where(valid_union & (ownership == index), 255, 0).astype(np.uint8)
        for index in range(len(valid_masks))
    ]
    radius = max(0, int(search_radius))
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        candidate_masks = [
            np.where(
                valid,
                cv2.dilate(core, kernel),
                0,
            ).astype(np.uint8)
            for core, valid in zip(core_masks, valid_stack)
        ]
    else:
        candidate_masks = core_masks

    union_area = max(1, int(valid_union.sum()))
    candidate_support = np.zeros(shape, dtype=np.uint16)
    for mask in candidate_masks:
        candidate_support += (mask > 0).astype(np.uint16)
    diagnostics = {
        "enabled": True,
        "search_radius": radius,
        "core_source_area_ratio": [
            float(np.count_nonzero(mask) / union_area) for mask in core_masks
        ],
        "candidate_source_area_ratio": [
            float(np.count_nonzero(mask) / union_area) for mask in candidate_masks
        ],
        "candidate_overlap_ratio": float(np.mean(candidate_support > 1)),
        "core_uncovered_pixels": int(
            np.count_nonzero(valid_union & ~(np.stack(core_masks, axis=0) > 0).any(axis=0))
        ),
    }
    return candidate_masks, diagnostics


def graphcut_multiband_blend(
    sequence: Sequence,
    transforms: list[np.ndarray],
    output_path: Path,
    max_dimension: int,
    max_megapixels: float,
    seam_max_dimension: int,
    graphcut_cost_type: str,
    seam_mask_dilation: int | None,
    seam_transition_width_px: int,
    seam_balance_sources: bool,
    seam_search_radius: int,
    blend_bands: int,
    exposure_compensator_mode: str,
    exposure_block_size: int,
    exposure_gain_min: float,
    exposure_gain_max: float,
    exposure_gain_filtering_iterations: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    """Estimate a low-resolution multi-image seam and blend at full resolution."""
    canvas_transform, (width, height), scale, canvas_info = (
        output_transform_and_size(
            sequence, transforms, max_dimension, max_megapixels
        )
    )
    seam_scale = min(1.0, float(seam_max_dimension) / max(width, height))
    seam_width = max(2, int(round(width * seam_scale)))
    seam_height = max(2, int(round(height * seam_scale)))
    seam_resize = np.asarray(
        [[seam_scale, 0.0, 0.0], [0.0, seam_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    seam_images: list[np.ndarray] = []
    seam_valid_masks: list[np.ndarray] = []
    full_valid_masks: list[np.ndarray] = []
    source_images: list[np.ndarray] = []

    for path, global_transform in zip(sequence.images, transforms):
        image = read_cv_image(path, cv2.IMREAD_COLOR)
        source_images.append(image)
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        output_h = canvas_transform @ global_transform
        seam_h = seam_resize @ output_h
        seam_images.append(
            cv2.warpPerspective(
                image,
                seam_h,
                (seam_width, seam_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
        seam_valid_masks.append(
            cv2.warpPerspective(
                source_mask,
                seam_h,
                (seam_width, seam_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
        full_valid_masks.append(
            cv2.warpPerspective(
                source_mask,
                output_h,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )

    if exposure_gain_min <= 0.0 or exposure_gain_max < exposure_gain_min:
        raise ValueError(
            "exposure gain limits must satisfy 0 < min <= max"
        )

    def apply_gain_field(
        image: np.ndarray,
        gain_field: np.ndarray,
    ) -> np.ndarray:
        """Apply scalar or per-channel low-frequency exposure gains."""
        field = np.asarray(gain_field, dtype=np.float32)
        if field.ndim not in {2, 3}:
            raise ValueError(
                f"expected a 2-D/3-D exposure gain field, got {field.shape}"
            )
        if exposure_compensator_mode == "channels":
            channel_gains = field.reshape(-1)
            if channel_gains.size != image.shape[2]:
                raise ValueError(
                    "channel exposure compensator returned "
                    f"{field.shape}, expected one gain per image channel"
                )
            gain_map = np.broadcast_to(
                channel_gains.reshape(1, 1, -1),
                image.shape,
            )
        else:
            gain_map = cv2.resize(
                field,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            if gain_map.ndim == 2:
                gain_map = gain_map[..., None]
        gain_map = np.clip(
            gain_map,
            exposure_gain_min,
            exposure_gain_max,
        )
        corrected = image.copy()
        for channel in range(corrected.shape[2]):
            values = corrected[..., channel].astype(np.float32)
            gain_channel = gain_map[..., min(channel, gain_map.shape[2] - 1)]
            values *= gain_channel
            corrected[..., channel] = np.clip(values, 0.0, 255.0).astype(
                np.uint8
            )
        return corrected

    exposure_gain_fields: list[np.ndarray] = [
        np.ones((1, 1), dtype=np.float32) for _ in seam_images
    ]
    compensated_seam_images = seam_images
    if exposure_compensator_mode != "none":
        if exposure_compensator_mode == "gain_blocks":
            exposure_compensator = cv2.detail_BlocksGainCompensator()
            block_size = max(8, int(exposure_block_size))
            exposure_compensator.setBlockSize(block_size, block_size)
            exposure_compensator.setNrGainsFilteringIterations(
                max(0, int(exposure_gain_filtering_iterations))
            )
        elif exposure_compensator_mode == "channels_blocks":
            exposure_compensator = cv2.detail_BlocksChannelsCompensator()
            block_size = max(8, int(exposure_block_size))
            exposure_compensator.setBlockSize(block_size, block_size)
            exposure_compensator.setNrGainsFilteringIterations(
                max(0, int(exposure_gain_filtering_iterations))
            )
        elif exposure_compensator_mode == "channels":
            exposure_compensator = cv2.detail_ChannelsCompensator()
        else:
            exposure_compensator = (
                cv2.detail.ExposureCompensator_createDefault(
                    cv2.detail.EXPOSURE_COMPENSATOR_GAIN
                )
            )
        seam_corners = [(0, 0)] * len(seam_images)
        exposure_compensator.feed(
            seam_corners,
            seam_images,
            seam_valid_masks,
        )
        exposure_gain_fields = [
            np.asarray(gain, dtype=np.float32)
            for gain in exposure_compensator.getMatGains()
        ]
        compensated_seam_images = [
            apply_gain_field(image, gain_field)
            for image, gain_field in zip(
                seam_images,
                exposure_gain_fields,
            )
        ]

    graphcut_input_masks = seam_valid_masks
    balance_diagnostics: dict[str, Any] = {"enabled": False}
    if seam_balance_sources:
        graphcut_input_masks, balance_diagnostics = balanced_graphcut_candidate_masks(
            seam_valid_masks,
            seam_search_radius,
        )

    seam_result = apply_seam_masks_to_stack(
        compensated_seam_images,
        graphcut_input_masks,
        SeamOptions(
            mode="opencv_graphcut",
            graphcut_cost_type=graphcut_cost_type,
            min_component_area=16,
        ),
    )
    if not bool(seam_result.diagnostics.get("opencv_graphcut_used", False)):
        raise RuntimeError(
            "OpenCV GraphCut seam was not applied: "
            + str(seam_result.diagnostics.get("fallback_reason", "unknown reason"))
        )

    seam_masks: list[np.ndarray] = []
    coverage_before = np.zeros((height, width), dtype=np.uint16)
    first_valid_owner = np.full((height, width), -1, dtype=np.int16)
    if seam_mask_dilation is None:
        # Dilation happens at seam resolution. Half the requested transition
        # width on each side produces the requested full-resolution band.
        dilation_radius = max(
            0,
            int(math.ceil(0.5 * max(0, seam_transition_width_px) * seam_scale)),
        )
    else:
        dilation_radius = max(0, int(seam_mask_dilation))
    if dilation_radius > 0:
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_radius + 1, 2 * dilation_radius + 1),
        )
    else:
        dilation_kernel = None
    for index, valid_mask in enumerate(full_valid_masks):
        valid = valid_mask > 0
        coverage_before += valid.astype(np.uint16)
        first_valid_owner[(first_valid_owner < 0) & valid] = index
        low_resolution_mask = seam_result.masks[index]
        if dilation_kernel is not None:
            low_resolution_mask = cv2.dilate(
                low_resolution_mask,
                dilation_kernel,
            )
        upsampled = cv2.resize(
            low_resolution_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        seam_masks.append(
            np.where(valid, upsampled, 0).astype(np.uint8)
        )

    assigned = np.zeros((height, width), dtype=bool)
    for mask in seam_masks:
        assigned |= mask > 0
    valid_union = coverage_before > 0
    uncovered = valid_union & ~assigned
    for index, mask in enumerate(seam_masks):
        mask[uncovered & (first_valid_owner == index)] = 255

    blend_support_count = np.zeros((height, width), dtype=np.uint16)
    for mask in seam_masks:
        blend_support_count += (mask > 0).astype(np.uint16)

    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(max(1, int(blend_bands)))
    blender.prepare((0, 0, width, height))
    for index, (image, global_transform, seam_mask) in enumerate(zip(
        source_images, transforms, seam_masks
    )):
        warped = cv2.warpPerspective(
            image,
            canvas_transform @ global_transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if exposure_compensator_mode != "none":
            warped = apply_gain_field(
                warped,
                exposure_gain_fields[index],
            )
        blender.feed(warped.astype(np.int16), seam_mask, (0, 0))
    destination = np.zeros((height, width, 3), dtype=np.int16)
    destination_mask = np.zeros((height, width), dtype=np.uint8)
    blended, blended_mask = blender.blend(destination, destination_mask)
    output = np.clip(blended, 0, 255).astype(np.uint8)
    valid_output = blended_mask > 0
    if not np.any(valid_output):
        raise RuntimeError(f"{sequence.video_id}: empty graphcut panorama")
    ys, xs = np.nonzero(valid_output)
    output = output[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        output,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    ):
        raise RuntimeError(f"failed to write {output_path}")

    canvas_info.update(
        {
            "final_width": float(output.shape[1]),
            "final_height": float(output.shape[0]),
            "blend_mode": "graphcut_multiband",
            "blend_bands": int(blend_bands),
            "seam_scale": float(seam_scale),
            "seam_input_width": int(seam_width),
            "seam_input_height": int(seam_height),
            "graphcut_cost_type": graphcut_cost_type,
            "seam_mask_dilation": int(dilation_radius),
            "seam_transition_width_px": int(seam_transition_width_px),
            "seam_balance_sources": bool(seam_balance_sources),
            "seam_balance_diagnostics": balance_diagnostics,
            "seam_blend_overlap_pixel_ratio": float(
                np.mean(blend_support_count > 1)
            ),
            "exposure_compensator": exposure_compensator_mode,
            "exposure_block_size": int(exposure_block_size),
            "exposure_gain_limits": [
                float(exposure_gain_min),
                float(exposure_gain_max),
            ],
            "exposure_gain_ranges": [
                [
                    float(np.nanmin(gain_field)),
                    float(np.nanmax(gain_field)),
                ]
                for gain_field in exposure_gain_fields
            ],
            "overlap_pixel_ratio_before": float(np.mean(coverage_before > 1)),
            "uncovered_pixels_repaired": int(np.count_nonzero(uncovered)),
            "seam_diagnostics": seam_result.diagnostics,
        }
    )
    return canvas_info


def render_sequence_panorama(
    sequence: Sequence,
    transforms: list[np.ndarray],
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.blend_mode == "graphcut_multiband":
        return graphcut_multiband_blend(
            sequence,
            transforms,
            output_path,
            args.max_dimension,
            args.max_megapixels,
            args.seam_max_dimension,
            args.seam_graphcut_cost_type,
            args.seam_mask_dilation,
            args.seam_transition_width_px,
            args.seam_balance_sources,
            args.seam_search_radius,
            args.blend_bands,
            args.exposure_compensator,
            args.exposure_block_size,
            args.exposure_gain_min,
            args.exposure_gain_max,
            args.exposure_gain_filtering_iterations,
            args.jpeg_quality,
        )
    return feather_blend(
        sequence,
        transforms,
        output_path,
        args.max_dimension,
        args.max_megapixels,
        args.feather_radius,
        args.jpeg_quality,
    )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(float(value)) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.ransac_threshold_net_px is not None and args.ransac_threshold_net_px <= 0:
        raise ValueError("--ransac-threshold-net-px must be positive")
    if args.ransac_threshold_original_px <= 0:
        raise ValueError("--ransac-threshold-original-px must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pair_list is not None:
        if args.image_root is None:
            raise ValueError("--image-root is required with --pair-list")
        sequences = load_sequences(args.pair_list, args.image_root)
        sequence_source_mode = "pair_list"
    else:
        if args.image_root is not None:
            raise ValueError("--image-root is only valid with --pair-list")
        sequences = [
            load_temporal_sequence(
                args.sequence_dir,
                args.sequence_id,
                args.image_extensions,
                args.sequence_image_regex,
            )
        ]
        sequence_source_mode = "natural_temporal_directory"
    if args.groups:
        requested = {value.strip() for value in args.groups.split(",") if value.strip()}
        sequences = [sequence for sequence in sequences if sequence.video_id in requested]
    if args.sequence_group_size < 0:
        raise ValueError("--sequence-group-size must be non-negative")
    if args.sequence_group_size > 0:
        sequences = [
            group
            for sequence in sequences
            for group in split_sequence_into_groups(
                sequence,
                args.sequence_group_size,
                args.sequence_group_overlap,
            )
        ]
    if args.limit_groups is not None:
        sequences = sequences[: args.limit_groups]
    if not sequences:
        raise RuntimeError("no sequences selected")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, config, checkpoint_metadata = load_model(args.checkpoint, device)
    checkpoint_input_size = [int(config.input_width), int(config.input_height)]
    override_width = args.matcher_input_width
    override_height = args.matcher_input_height
    if (override_width is None) != (override_height is None):
        raise ValueError(
            "--matcher-input-width and --matcher-input-height must be set together"
        )
    if override_width is not None and override_height is not None:
        if override_width % config.coarse_stride != 0:
            raise ValueError(
                "matcher input width must be divisible by coarse_stride "
                f"({config.coarse_stride})"
            )
        if override_height % config.coarse_stride != 0:
            raise ValueError(
                "matcher input height must be divisible by coarse_stride "
                f"({config.coarse_stride})"
            )
        config.input_width = int(override_width)
        config.input_height = int(override_height)
        config.validate()
    summary_path = args.output_dir / "panorama_summary.json"
    summary: dict[str, Any] = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence_source_mode": sequence_source_mode,
        "pair_list": str(args.pair_list) if args.pair_list else None,
        "image_root": str(args.image_root) if args.image_root else None,
        "sequence_dir": str(args.sequence_dir) if args.sequence_dir else None,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "precision": "fp32",
        "checkpoint_matcher_input_size": checkpoint_input_size,
        "runtime_matcher_input_size": [
            int(config.input_width),
            int(config.input_height),
        ],
        "ransac_threshold_net_px": args.ransac_threshold_net_px,
        "ransac_threshold_original_px": args.ransac_threshold_original_px,
        "ransac_coordinate_system": (
            "network_resized_pixels"
            if args.ransac_threshold_net_px is not None
            else "original_image_pixels"
        ),
        "checkpoint_metadata": checkpoint_metadata,
        "sequence_count": len(sequences),
        "sequence_group_size": int(args.sequence_group_size),
        "sequence_group_overlap": int(args.sequence_group_overlap),
        "sequences": [],
    }
    overall_start = time.perf_counter()
    failed = 0

    for sequence_index, sequence in enumerate(sequences, start=1):
        missing_images = [
            str(path) for path in sequence.images if not path.is_file()
        ]
        if missing_images:
            raise FileNotFoundError(
                "sequence input is incomplete before inference: "
                + ", ".join(missing_images)
            )
        output_path = args.output_dir / f"{sequence.video_id}_panorama.jpg"
        diagnostics_path = args.output_dir / f"{sequence.video_id}_diagnostics.json"
        if diagnostics_path.is_file() and not args.force:
            try:
                previous = json.loads(
                    diagnostics_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                previous = {}
            previous_status = previous.get("status")
            complete_output = previous_status == "ok" and output_path.is_file()
            partial_outputs = previous.get("segments", [])
            complete_partial = (
                previous_status == "partial"
                and bool(partial_outputs)
                and all(
                    Path(segment["output_path"]).is_file()
                    for segment in partial_outputs
                    if segment.get("status") == "ok"
                )
            )
            if complete_output or complete_partial:
                print(
                    f"[{sequence_index:02d}/{len(sequences):02d}] "
                    f"{sequence.video_id} skipped (already exists)",
                    flush=True,
                )
                summary["sequences"].append(previous)
                write_summary(summary_path, summary)
                continue

        sequence_start = time.perf_counter()
        print(
            f"[{sequence_index:02d}/{len(sequences):02d}] {sequence.video_id} "
            f"images={len(sequence.images)}",
            flush=True,
        )
        diagnostics: dict[str, Any] = {
            "video_id": sequence.video_id,
            "image_count": len(sequence.images),
            "pair_count": len(sequence.pair_ids),
            "sequence_source_mode": sequence_source_mode,
            "temporal_image_order": [path.name for path in sequence.images],
            "output_path": str(output_path),
        }
        try:
            all_edges = estimate_sequence_edges(
                sequence,
                model,
                config,
                device,
                args.ransac_threshold_net_px,
                args.ransac_threshold_original_px,
                max_span=(
                    args.pose_graph_max_span
                    if args.composition_model == "pose_graph"
                    or (
                        args.composition_model == "auto"
                        and len(sequence.images) > 5
                    )
                    else 1
                ),
            )
            edges = [edge for edge in all_edges if edge.temporal_span == 1]
            skip_edges = [edge for edge in all_edges if edge.temporal_span > 1]
            edge_method_counts = {
                method: sum(edge.method == method for edge in all_edges)
                for method in sorted({edge.method for edge in all_edges})
            }
            diagnostics.update(
                {
                    "edge_method_counts": edge_method_counts,
                    "edges": [asdict(edge) for edge in edges],
                    "pose_graph_skip_edges": [asdict(edge) for edge in skip_edges],
                }
            )
            unresolved = [
                edge.pair_id for edge in edges if edge.matrix() is None
            ]
            if unresolved:
                unresolved_indices = {
                    index for index, edge in enumerate(edges) if edge.matrix() is None
                }
                ranges: list[tuple[int, int]] = []
                range_start = 0
                for edge_index in sorted(unresolved_indices):
                    ranges.append((range_start, edge_index))
                    range_start = edge_index + 1
                ranges.append((range_start, len(sequence.images) - 1))

                segment_rows: list[dict[str, Any]] = []
                excluded_images: list[str] = []
                rendered_segment_count = 0
                for start_index, end_index in ranges:
                    segment_images = sequence.images[start_index : end_index + 1]
                    if len(segment_images) < 2:
                        excluded_images.extend(path.name for path in segment_images)
                        continue
                    segment_edges = [
                        replace(
                            edge,
                            source_index=edge.source_index - start_index,
                            target_index=edge.target_index - start_index,
                        )
                        for edge in all_edges
                        if start_index <= edge.source_index
                        and edge.target_index <= end_index
                    ]
                    segment_sequence = Sequence(
                        video_id=sequence.video_id,
                        images=segment_images,
                        pair_ids=sequence.pair_ids[start_index:end_index],
                    )
                    rendered_segment_count += 1
                    segment_output = (
                        args.output_dir
                        / f"{sequence.video_id}_segment{rendered_segment_count:02d}_panorama.jpg"
                    )
                    segment_reference = len(segment_images) // 2
                    (
                        segment_transforms,
                        segment_model,
                        segment_geometry,
                    ) = stabilize_if_needed(
                        segment_sequence,
                        [
                            edge
                            for edge in segment_edges
                            if edge.temporal_span == 1
                        ],
                        segment_reference,
                        args.composition_model,
                        args.max_edge_scale_change,
                        pose_graph_edges=segment_edges,
                        pose_graph_min_inlier_ratio=args.pose_graph_min_inlier_ratio,
                        pose_graph_min_inliers=args.pose_graph_min_inliers,
                        pose_graph_max_p90_px=args.pose_graph_max_p90_px,
                        pose_graph_scale_prior_weight=args.pose_graph_scale_prior_weight,
                        pose_graph_smoothness_weight=args.pose_graph_smoothness_weight,
                        pose_graph_max_global_scale_change=(
                            args.pose_graph_max_global_scale_change
                        ),
                        pose_graph_huber_px=args.pose_graph_huber_px,
                        pose_graph_max_iterations=args.pose_graph_max_iterations,
                    )
                    segment_canvas = render_sequence_panorama(
                        segment_sequence,
                        segment_transforms,
                        segment_output,
                        args,
                    )
                    segment_rows.append(
                        {
                            "status": "ok",
                            "start_image_index": start_index,
                            "end_image_index": end_index,
                            "image_count": len(segment_images),
                            "images": [path.name for path in segment_images],
                            "output_path": str(segment_output),
                            "reference_image": segment_images[
                                segment_reference
                            ].name,
                            "composition_model": segment_model,
                            "sequence_geometry": segment_geometry,
                            "canvas": segment_canvas,
                        }
                    )
                if not segment_rows:
                    raise RuntimeError(
                        "GRM could not form any multi-image connected segment: "
                        + ", ".join(unresolved)
                    )
                diagnostics.update(
                    {
                        "status": "partial",
                        "unresolved_pairs": unresolved,
                        "excluded_single_images": excluded_images,
                        "segments": segment_rows,
                        "elapsed_sec": float(time.perf_counter() - sequence_start),
                    }
                )
                print(
                    f"  PARTIAL: segments={len(segment_rows)} "
                    f"unresolved_edges={len(unresolved)}",
                    flush=True,
                )
            else:
                reference_index = len(sequence.images) // 2
                transforms, composition_model, sequence_geometry = stabilize_if_needed(
                    sequence,
                    edges,
                    reference_index,
                    args.composition_model,
                    args.max_edge_scale_change,
                    pose_graph_edges=all_edges,
                    pose_graph_min_inlier_ratio=args.pose_graph_min_inlier_ratio,
                    pose_graph_min_inliers=args.pose_graph_min_inliers,
                    pose_graph_max_p90_px=args.pose_graph_max_p90_px,
                    pose_graph_scale_prior_weight=args.pose_graph_scale_prior_weight,
                    pose_graph_smoothness_weight=args.pose_graph_smoothness_weight,
                    pose_graph_max_global_scale_change=(
                        args.pose_graph_max_global_scale_change
                    ),
                    pose_graph_huber_px=args.pose_graph_huber_px,
                    pose_graph_max_iterations=args.pose_graph_max_iterations,
                )
                canvas = render_sequence_panorama(
                    sequence,
                    transforms,
                    output_path,
                    args,
                )
                diagnostics.update(
                    {
                        "status": "ok",
                        "reference_index": reference_index,
                        "reference_image": sequence.images[reference_index].name,
                        "composition_model": composition_model,
                        "unresolved_pairs": [],
                        "sequence_geometry": sequence_geometry,
                        "canvas": canvas,
                        "elapsed_sec": float(time.perf_counter() - sequence_start),
                    }
                )
            diagnostics["edges"] = [asdict(edge) for edge in edges]
            diagnostics["pose_graph_skip_edges"] = [
                asdict(edge) for edge in skip_edges
            ]
        except Exception as exc:
            failed += 1
            diagnostics.update(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "elapsed_sec": float(time.perf_counter() - sequence_start),
                }
            )
            print(f"  FAILED: {exc}", flush=True)
        write_summary(diagnostics_path, diagnostics)
        summary["sequences"].append(diagnostics)
        summary["elapsed_sec"] = float(time.perf_counter() - overall_start)
        summary["failed_sequence_count"] = failed
        write_summary(summary_path, summary)

    summary["elapsed_sec"] = float(time.perf_counter() - overall_start)
    summary["failed_sequence_count"] = failed
    summary["successful_sequence_count"] = sum(
        item.get("status") == "ok" for item in summary["sequences"]
    )
    summary["partial_sequence_count"] = sum(
        item.get("status") == "partial" for item in summary["sequences"]
    )
    write_summary(summary_path, summary)
    print(
        f"[done] successful={summary['successful_sequence_count']} "
        f"partial={summary['partial_sequence_count']} "
        f"failed={failed} elapsed={summary['elapsed_sec']:.1f}s",
        flush=True,
    )
    return 0 if failed == 0 and summary["partial_sequence_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
