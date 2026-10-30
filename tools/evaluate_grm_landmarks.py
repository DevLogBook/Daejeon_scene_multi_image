#!/usr/bin/env python
"""Evaluate a standalone GRM-Matcher checkpoint on annotated image pairs.

GRM-Matcher emits sparse A->B correspondences but intentionally does not own a
global solver or stitcher.  This tool adds an external USAC/MAGSAC homography,
measures the resulting annotation reprojection error in original-image pixels,
and saves a small representative set of stitched diagnostic images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grm_matcher import GRMMatcher, GRMMatcherConfig
from grm_matcher.config import GRM_BUILTIN_CANDIDATE_RERANK_MODES
from grm_matcher.outputs import GRMMatcherOutput


PRIMARY_METHOD = "grm_all_magsac"
METHODS = ("identity", PRIMARY_METHOD, "grm_global_magsac", "annotation_planar_fit")


@dataclass
class Pair:
    pair_id: str
    video_id: str
    scenario: str
    quality: str
    status: str
    image_a: Path
    image_b: Path
    points_a: np.ndarray
    points_b: np.ndarray


@dataclass
class PairArtifact:
    pair: Pair
    homography_a_to_b: np.ndarray
    pair_metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standalone GRM correspondences against manual landmarks."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pair-list", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stitched-dir", type=Path, required=True)
    parser.add_argument("--statuses", type=str, default="labeled")
    parser.add_argument("--min-points", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="AMP is disabled by default because low GRM probabilities can underflow in FP16.",
    )
    parser.add_argument("--ransac-threshold-net-px", type=float, default=3.0)
    parser.add_argument("--global-role-threshold", type=float, default=0.5)
    parser.add_argument(
        "--candidate-rerank-mode",
        choices=GRM_BUILTIN_CANDIDATE_RERANK_MODES,
        default=None,
        help=(
            "Override candidate selection for checkpoint-compatible ablations. "
            "Omitting this option preserves the checkpoint configuration."
        ),
    )
    parser.add_argument("--visuals-per-group", type=int, default=3)
    parser.add_argument(
        "--save-all-visuals",
        action="store_true",
        help="Save every pair with a successfully estimated primary homography.",
    )
    parser.add_argument("--visual-max-dimension", type=int, default=2400)
    return parser.parse_args()


def load_pairs(args: argparse.Namespace) -> tuple[list[Pair], dict[str, Any]]:
    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("pairs", records.get("annotations", records.get("data", [])))
    if not isinstance(records, list):
        raise ValueError("annotations must be a list or contain pairs/annotations/data")

    pair_list_ids: set[str] | None = None
    pair_list_count = None
    if args.pair_list is not None:
        pair_records = json.loads(args.pair_list.read_text(encoding="utf-8"))
        if isinstance(pair_records, dict):
            pair_records = pair_records.get("pairs", pair_records.get("data", []))
        if not isinstance(pair_records, list):
            raise ValueError("pair-list must be a list or contain pairs/data")
        pair_list_ids = {str(record["pair_id"]) for record in pair_records}
        pair_list_count = len(pair_records)

    statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    pairs: list[Pair] = []
    status_counts: dict[str, int] = defaultdict(int)
    quality_counts: dict[str, int] = defaultdict(int)
    with_points = 0
    missing_from_pair_list: list[str] = []
    for record in records:
        status = str(record.get("status", "")).strip()
        quality = str(record.get("quality", "")).strip() or "unknown"
        status_counts[status or "unknown"] += 1
        quality_counts[quality] += 1
        points_a = np.asarray(record.get("points_a", []), dtype=np.float64).reshape(-1, 2)
        points_b = np.asarray(record.get("points_b", []), dtype=np.float64).reshape(-1, 2)
        if points_a.shape != points_b.shape:
            raise ValueError(f"{record.get('pair_id')}: points_a/points_b shape mismatch")
        if len(points_a):
            with_points += 1
        pair_id = str(record.get("pair_id", f"pair_{len(pairs):04d}"))
        if pair_list_ids is not None and pair_id not in pair_list_ids:
            missing_from_pair_list.append(pair_id)
            continue
        if statuses and status not in statuses:
            continue
        if len(points_a) < args.min_points:
            continue
        pairs.append(
            Pair(
                pair_id=pair_id,
                video_id=str(record.get("video_id", "")),
                scenario=str(record.get("scenario", "")),
                quality=quality,
                status=status,
                image_a=args.image_root / str(record["image_a"]),
                image_b=args.image_root / str(record["image_b"]),
                points_a=points_a,
                points_b=points_b,
            )
        )
        if args.limit is not None and len(pairs) >= args.limit:
            break

    inventory = {
        "annotation_record_count": len(records),
        "annotation_pairs_with_points": with_points,
        "selected_pair_count": len(pairs),
        "selected_point_count": int(sum(len(pair.points_a) for pair in pairs)),
        "selected_video_count": len({pair.video_id for pair in pairs}),
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "annotation_quality_counts": dict(sorted(quality_counts.items())),
        "pair_list_count": pair_list_count,
        "annotations_missing_from_pair_list": missing_from_pair_list,
    }
    return pairs, inventory


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[GRMMatcher, GRMMatcherConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise KeyError("GRM checkpoint must contain config and model")
    config = GRMMatcherConfig.from_dict(checkpoint["config"])
    model = GRMMatcher(config)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device).eval()
    metadata = {
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best": checkpoint.get("best"),
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "checkpoint_validation_metrics": checkpoint.get("validation_metrics"),
    }
    return model, config, metadata


def load_image(
    path: Path, config: GRMMatcherConfig
) -> tuple[np.ndarray, Tensor, tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        original_rgb = np.asarray(rgb, dtype=np.uint8)
        resized = rgb.resize(
            (config.input_width, config.input_height), Image.Resampling.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if config.input_range == "imagenet_normalized":
        mean = tensor.new_tensor(config.normalization_mean)[None, :, None, None]
        std = tensor.new_tensor(config.normalization_std)[None, :, None, None]
        tensor = (tensor - mean) / std
    scale = (
        config.input_width / original_rgb.shape[1],
        config.input_height / original_rgb.shape[0],
    )
    return cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR), tensor, scale


def infer(
    model: GRMMatcher,
    config: GRMMatcherConfig,
    pair: Pair,
    device: torch.device,
    use_amp: bool,
    candidate_rerank_mode: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[float, float], tuple[float, float], float]:
    _, input_a, scale_a = load_image(pair.image_a, config)
    _, input_b, scale_b = load_image(pair.image_b, config)
    start = time.perf_counter()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=bool(use_amp),
    ):
        output = model(
            input_a.to(device),
            input_b.to(device),
            candidate_rerank_mode=candidate_rerank_mode,
        )
    elapsed = time.perf_counter() - start
    if not isinstance(output, GRMMatcherOutput):
        raise TypeError(f"unexpected matcher output: {type(output)}")
    output = output.to("cpu")
    output.validate()
    src_net = output.src_xy[0].float().numpy()
    dst_net = output.dst_xy[0].float().numpy()
    values = {
        "valid": output.valid_mask[0].numpy().astype(bool),
        "p_match": output.p_match[0].float().numpy(),
        "p_global": output.p_global[0].float().numpy(),
        "p_local": output.p_local[0].float().numpy(),
        "p_false": output.p_false[0].float().numpy(),
        "cycle_error": output.cycle_error[0].float().numpy(),
        "candidate_entropy": output.candidate_entropy[0].float().numpy(),
    }
    return src_net, dst_net, values, scale_a, scale_b, elapsed


def fit_grm_homography(
    src_net: np.ndarray,
    dst_net: np.ndarray,
    selection: np.ndarray,
    scale_a: tuple[float, float],
    scale_b: tuple[float, float],
    threshold_net_px: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if int(selection.sum()) < 4:
        return None, None
    cv2.setRNGSeed(0)
    H_net, inliers = cv2.findHomography(
        src_net[selection].astype(np.float32),
        dst_net[selection].astype(np.float32),
        cv2.USAC_MAGSAC,
        float(threshold_net_px),
        maxIters=10_000,
        confidence=0.999,
    )
    if H_net is None or not np.isfinite(H_net).all():
        return None, inliers
    scale_a_matrix = np.diag([scale_a[0], scale_a[1], 1.0])
    scale_b_matrix = np.diag([scale_b[0], scale_b[1], 1.0])
    H_original = np.linalg.inv(scale_b_matrix) @ H_net @ scale_a_matrix
    if abs(H_original[2, 2]) > 1.0e-12:
        H_original /= H_original[2, 2]
    return H_original, inliers


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.astype(np.float32)[None, :, :], H.astype(np.float64)
    )[0].astype(np.float64)


def summarize(errors: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(errors), dtype=np.float64)
    if not len(array):
        return {
            "point_count": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "p90_error_px": None,
            "max_error_px": None,
            "success_at_5px": None,
            "success_at_10px": None,
            "success_at_20px": None,
        }
    return {
        "point_count": int(len(array)),
        "mean_error_px": float(array.mean()),
        "median_error_px": float(np.median(array)),
        "p90_error_px": float(np.percentile(array, 90)),
        "max_error_px": float(array.max()),
        "success_at_5px": float(np.mean(array <= 5.0)),
        "success_at_10px": float(np.mean(array <= 10.0)),
        "success_at_20px": float(np.mean(array <= 20.0)),
    }


def method_result(
    pair: Pair,
    method: str,
    H: np.ndarray | None,
    selected_matches: int,
    inliers: np.ndarray | None,
    elapsed: float,
    values: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = {
        "pair_id": pair.pair_id,
        "video_id": pair.video_id,
        "scenario": pair.scenario,
        "quality": pair.quality,
        "status": pair.status,
        "method": method,
        "num_annotations": int(len(pair.points_a)),
        "selected_matches": int(selected_matches),
        "inlier_count": int(inliers.sum()) if inliers is not None else None,
        "inlier_ratio": (
            float(inliers.mean()) if inliers is not None and len(inliers) else None
        ),
        "valid_match_count": int(values["valid"].sum()),
        "global_match_count": int(
            (values["valid"] & (values["p_global"] >= 0.5)).sum()
        ),
        "p_match_mean_valid": (
            float(values["p_match"][values["valid"]].mean())
            if values["valid"].any()
            else None
        ),
        "p_global_mean_valid": (
            float(values["p_global"][values["valid"]].mean())
            if values["valid"].any()
            else None
        ),
        "inference_sec": float(elapsed),
    }
    if H is None:
        return {**base, "solver_status": "failed", **summarize([])}, []
    predicted_b = transform_points(pair.points_a, H)
    errors = np.linalg.norm(predicted_b - pair.points_b, axis=1)
    point_rows = []
    for index, (point_a, point_b, predicted, error) in enumerate(
        zip(pair.points_a, pair.points_b, predicted_b, errors)
    ):
        point_rows.append(
            {
                "pair_id": pair.pair_id,
                "video_id": pair.video_id,
                "quality": pair.quality,
                "method": method,
                "point_index": index,
                "x_a": float(point_a[0]),
                "y_a": float(point_a[1]),
                "x_b": float(point_b[0]),
                "y_b": float(point_b[1]),
                "pred_x_b": float(predicted[0]),
                "pred_y_b": float(predicted[1]),
                "error_px": float(error),
            }
        )
    return {
        **base,
        "solver_status": "ok",
        **summarize(errors),
        "H_a_to_b": json.dumps(H.tolist(), ensure_ascii=False),
    }, point_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    pair_rows: list[dict[str, Any]], point_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overall: dict[str, Any] = {}
    groups: list[dict[str, Any]] = []
    for method in METHODS:
        method_pairs = [row for row in pair_rows if row["method"] == method]
        method_points = [row for row in point_rows if row["method"] == method]
        successful_pairs = [row for row in method_pairs if row["solver_status"] == "ok"]
        overall[method] = {
            **summarize(row["error_px"] for row in method_points),
            "pair_count": len(method_pairs),
            "successful_pair_count": len(successful_pairs),
            "failed_pair_count": len(method_pairs) - len(successful_pairs),
            "pair_success_ratio": (
                len(successful_pairs) / len(method_pairs) if method_pairs else None
            ),
            "pair_median_error_median_px": (
                float(np.median([row["median_error_px"] for row in successful_pairs]))
                if successful_pairs
                else None
            ),
        }
        for video_id in sorted({row["video_id"] for row in method_pairs}):
            video_pairs = [row for row in method_pairs if row["video_id"] == video_id]
            video_points = [
                row for row in method_points if row["video_id"] == video_id
            ]
            video_success = [
                row for row in video_pairs if row["solver_status"] == "ok"
            ]
            groups.append(
                {
                    "video_id": video_id,
                    "method": method,
                    "pair_count": len(video_pairs),
                    "successful_pair_count": len(video_success),
                    **summarize(row["error_px"] for row in video_points),
                }
            )
    return overall, groups


def choose_visuals(
    artifacts: list[PairArtifact], per_group: int, save_all: bool = False
) -> list[tuple[PairArtifact, str]]:
    if per_group <= 0 and not save_all:
        return []
    chosen: list[tuple[PairArtifact, str]] = []
    grouped: dict[str, list[PairArtifact]] = defaultdict(list)
    for artifact in artifacts:
        grouped[artifact.pair.video_id].append(artifact)
    for video_id in sorted(grouped):
        ordered = sorted(
            grouped[video_id],
            key=lambda item: float(item.pair_metrics["median_error_px"]),
        )
        if save_all:
            chosen.extend((artifact, "all") for artifact in ordered)
            continue
        count = min(max(1, per_group), len(ordered))
        if count == 1:
            indices = [0]
            labels = ["only"]
        elif count == 2:
            indices = [0, len(ordered) - 1]
            labels = ["best", "worst"]
        else:
            indices = [0, len(ordered) // 2, len(ordered) - 1]
            labels = ["best", "median", "worst"]
        used: set[int] = set()
        for index, label in zip(indices, labels):
            if index not in used:
                chosen.append((ordered[index], label))
                used.add(index)
    return chosen


def render_stitch(
    artifact: PairArtifact,
    label: str,
    output_path: Path,
    max_dimension: int,
) -> None:
    pair = artifact.pair
    H = artifact.homography_a_to_b
    image_a = cv2.imread(str(pair.image_a), cv2.IMREAD_COLOR)
    image_b = cv2.imread(str(pair.image_b), cv2.IMREAD_COLOR)
    if image_a is None or image_b is None:
        raise FileNotFoundError(f"could not read {pair.image_a} or {pair.image_b}")
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]
    corners_a = np.asarray(
        [[0, 0], [w_a - 1, 0], [w_a - 1, h_a - 1], [0, h_a - 1]],
        dtype=np.float64,
    )
    corners_b = np.asarray(
        [[0, 0], [w_b - 1, 0], [w_b - 1, h_b - 1], [0, h_b - 1]],
        dtype=np.float64,
    )
    warped_corners_a = transform_points(corners_a, H)
    all_corners = np.vstack([warped_corners_a, corners_b])
    min_xy = np.floor(all_corners.min(axis=0))
    max_xy = np.ceil(all_corners.max(axis=0))
    width, height = (max_xy - min_xy + 1).astype(int)
    if width <= 0 or height <= 0 or width * height > 80_000_000:
        raise RuntimeError(f"unsafe stitch canvas {width}x{height}")
    translation = np.asarray(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]]
    )
    warp_a = cv2.warpPerspective(image_a, translation @ H, (width, height))
    warp_b = cv2.warpPerspective(image_b, translation, (width, height))
    mask_a = cv2.warpPerspective(
        np.full((h_a, w_a), 255, np.uint8), translation @ H, (width, height)
    )
    mask_b = cv2.warpPerspective(
        np.full((h_b, w_b), 255, np.uint8), translation, (width, height)
    )
    canvas = np.zeros_like(warp_a)
    only_a = (mask_a > 0) & (mask_b == 0)
    only_b = (mask_b > 0) & (mask_a == 0)
    overlap = (mask_a > 0) & (mask_b > 0)
    canvas[only_a] = warp_a[only_a]
    canvas[only_b] = warp_b[only_b]
    canvas[overlap] = (
        0.5 * warp_a[overlap].astype(np.float32)
        + 0.5 * warp_b[overlap].astype(np.float32)
    ).astype(np.uint8)

    display_scale = min(1.0, max_dimension / max(canvas.shape[:2]))
    if display_scale < 1.0:
        canvas = cv2.resize(
            canvas, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA
        )
    predicted_original = transform_points(pair.points_a, translation @ H)
    target_original = transform_points(pair.points_b, translation)
    errors = np.linalg.norm(predicted_original - target_original, axis=1)
    predicted = predicted_original * display_scale
    target = target_original * display_scale
    for index, (pred, truth, error) in enumerate(zip(predicted, target, errors)):
        p = tuple(np.round(pred).astype(int))
        t = tuple(np.round(truth).astype(int))
        color = (40, 190, 40) if error <= 10.0 else ((0, 190, 255) if error <= 20.0 else (30, 30, 230))
        cv2.line(canvas, p, t, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p, 6, (255, 80, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, t, (40, 255, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(
            canvas,
            f"{index}:{error:.1f}",
            (t[0] + 6, t[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{index}:{error:.1f}",
            (t[0] + 6, t[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    banner = np.zeros((86, canvas.shape[1], 3), dtype=np.uint8)
    metrics = artifact.pair_metrics
    cv2.putText(
        banner,
        f"{pair.video_id} {label} | {pair.pair_id}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        (
            f"GRM+MAGSAC A->B | points={len(errors)} "
            f"mean={errors.mean():.2f}px median={np.median(errors):.2f}px "
            f"p90={np.percentile(errors, 90):.2f}px | "
            f"matches={metrics['selected_matches']} inliers={metrics['inlier_count']}"
        ),
        (14, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        "magenta circle=warped A annotation; yellow cross=B annotation; line=residual",
        (14, 79),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    output = np.vstack([banner, canvas])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise RuntimeError(f"failed to write {output_path}")


def main() -> int:
    args = parse_args()
    if args.min_points < 1:
        raise ValueError("--min-points must be >= 1")
    if args.ransac_threshold_net_px <= 0:
        raise ValueError("--ransac-threshold-net-px must be positive")
    if not 0.0 <= args.global_role_threshold <= 1.0:
        raise ValueError("--global-role-threshold must lie in [0,1]")
    if args.visuals_per_group < 0:
        raise ValueError("--visuals-per-group must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stitched_dir.mkdir(parents=True, exist_ok=True)

    pairs, inventory = load_pairs(args)
    if not pairs:
        raise RuntimeError("no annotated pairs matched the filters")
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, config, checkpoint_metadata = load_model(args.checkpoint, device)
    print(
        f"[setup] pairs={len(pairs)} points={inventory['selected_point_count']} "
        f"device={device} amp={args.amp} "
        f"candidate_rerank={args.candidate_rerank_mode or config.candidate_rerank_mode}"
    )

    pair_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    artifacts: list[PairArtifact] = []
    failures: list[dict[str, str]] = []
    run_start = time.perf_counter()
    for pair_index, pair in enumerate(pairs, start=1):
        pair_start = time.perf_counter()
        try:
            src_net, dst_net, values, scale_a, scale_b, inference_sec = infer(
                model,
                config,
                pair,
                device,
                bool(args.amp),
                args.candidate_rerank_mode,
            )
            valid = values["valid"]
            global_selection = valid & (
                values["p_global"] >= float(args.global_role_threshold)
            )
            H_all, inliers_all = fit_grm_homography(
                src_net,
                dst_net,
                valid,
                scale_a,
                scale_b,
                args.ransac_threshold_net_px,
            )
            H_global, inliers_global = fit_grm_homography(
                src_net,
                dst_net,
                global_selection,
                scale_a,
                scale_b,
                args.ransac_threshold_net_px,
            )
            method_specs = [
                ("identity", np.eye(3), 0, None),
                (PRIMARY_METHOD, H_all, int(valid.sum()), inliers_all),
                (
                    "grm_global_magsac",
                    H_global,
                    int(global_selection.sum()),
                    inliers_global,
                ),
            ]
            H_oracle = None
            if len(pair.points_a) >= 4:
                H_oracle, _ = cv2.findHomography(
                    pair.points_a.astype(np.float32),
                    pair.points_b.astype(np.float32),
                    0,
                )
            method_specs.append(
                ("annotation_planar_fit", H_oracle, len(pair.points_a), None)
            )
            primary_metrics = None
            for method, H, selected_count, inliers in method_specs:
                metrics, points = method_result(
                    pair,
                    method,
                    H,
                    selected_count,
                    inliers,
                    inference_sec,
                    values,
                )
                pair_rows.append(metrics)
                point_rows.extend(points)
                if method == PRIMARY_METHOD:
                    primary_metrics = metrics
            if (
                H_all is not None
                and primary_metrics is not None
                and primary_metrics["solver_status"] == "ok"
            ):
                artifacts.append(PairArtifact(pair, H_all, primary_metrics))
            elapsed_pair = time.perf_counter() - pair_start
            primary_text = (
                "failed"
                if primary_metrics is None or primary_metrics["solver_status"] != "ok"
                else f"median={primary_metrics['median_error_px']:.2f}px"
            )
            print(
                f"[pair {pair_index}/{len(pairs)}] {pair.pair_id} "
                f"valid={int(valid.sum())} global={int(global_selection.sum())} "
                f"{primary_text} time={elapsed_pair:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failures.append({"pair_id": pair.pair_id, "error": repr(exc)})
            print(
                f"[pair {pair_index}/{len(pairs)}] FAILED {pair.pair_id}: {exc}",
                flush=True,
            )

    overall, group_rows = aggregate(pair_rows, point_rows)
    selected_visuals = choose_visuals(
        artifacts, args.visuals_per_group, args.save_all_visuals
    )
    visual_rows: list[dict[str, Any]] = []
    visual_failures: list[dict[str, str]] = []
    for artifact, label in selected_visuals:
        filename = f"{label}_{artifact.pair.pair_id}.jpg"
        output_path = args.stitched_dir / artifact.pair.video_id / filename
        try:
            render_stitch(artifact, label, output_path, args.visual_max_dimension)
            visual_rows.append(
                {
                    "video_id": artifact.pair.video_id,
                    "pair_id": artifact.pair.pair_id,
                    "selection": label,
                    "median_error_px": artifact.pair_metrics["median_error_px"],
                    "output_path": str(output_path),
                }
            )
        except Exception as exc:
            visual_failures.append(
                {"pair_id": artifact.pair.pair_id, "error": repr(exc)}
            )

    write_csv(args.output_dir / "pair_metrics.csv", pair_rows)
    write_csv(args.output_dir / "point_errors.csv", point_rows)
    write_csv(args.output_dir / "group_summary.csv", group_rows)
    write_csv(args.output_dir / "visual_selection.csv", visual_rows)
    summary = {
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "annotations": str(args.annotations),
        "pair_list": str(args.pair_list) if args.pair_list else None,
        "image_root": str(args.image_root),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "precision": "amp" if args.amp else "fp32",
        "network_size": [config.input_width, config.input_height],
        "ransac_threshold_net_px": float(args.ransac_threshold_net_px),
        "global_role_threshold": float(args.global_role_threshold),
        "candidate_rerank_mode": (
            args.candidate_rerank_mode or config.candidate_rerank_mode
        ),
        "primary_method": PRIMARY_METHOD,
        "inventory": inventory,
        "checkpoint_metadata": checkpoint_metadata,
        "overall": overall,
        "inference_failures": failures,
        "visual_count": len(visual_rows),
        "visual_failures": visual_failures,
        "elapsed_sec": float(time.perf_counter() - run_start),
        "method_definitions": {
            "identity": "No alignment; annotation A coordinates are compared directly with B.",
            PRIMARY_METHOD: (
                "All GRM valid A->B matches passed to USAC/MAGSAC in 512x512 network "
                "coordinates; the estimated H is converted back to original-image pixels."
            ),
            "grm_global_magsac": (
                "Only GRM valid matches with p_global >= threshold passed to USAC/MAGSAC."
            ),
            "annotation_planar_fit": (
                "In-sample least-squares homography fitted from the annotations themselves; "
                "an optimistic planar-fit reference, not a model result."
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(
        f"[done] outputs={args.output_dir} visuals={len(visual_rows)} "
        f"failures={len(failures)} elapsed={summary['elapsed_sec']:.1f}s"
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
