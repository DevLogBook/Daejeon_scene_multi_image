#!/usr/bin/env python3
"""Generate exact single-source-image homography pairs for matcher training.

Both output images are sampled directly from the original image with exactly one
``cv2.warpPerspective`` call.  The saved homography maps A pixels to B pixels:

    p_b ~ H_gt @ p_a
    H_gt = G_B @ inv(G_A)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_TARGET_SIZES: tuple[tuple[int, int], ...] = (
    (512, 512),
    (768, 768),
    (1024, 1024),
)
# Fixed profiles carry 80% total probability.  The remaining 20% is assigned
# to an adaptive crop covering 80% of the source width and height.
FIXED_CROP_PROFILES: tuple[tuple[int, int, float], ...] = (
    (768, 768, 0.08),
    (1024, 1024, 0.16),
    (1280, 1280, 0.12),
    (1536, 1536, 0.08),
    (768, 1024, 0.06),
    (1024, 768, 0.06),
    (1024, 1280, 0.06),
    (1280, 1024, 0.06),
    (1280, 1536, 0.06),
    (1536, 1280, 0.06),
)
ADAPTIVE_CROP_FRACTION = 0.80
ADAPTIVE_CROP_WEIGHT = 0.20
MIN_CROP_DIMENSION = 768
BASE_OVERLAP_RANGE = (0.58, 0.92)
FINAL_OVERLAP_RANGE = (0.50, 0.95)
ROTATION_RANGE_DEG = (-8.0, 8.0)
SCALE_RANGE = (0.92, 1.08)
SHEAR_RANGE = (-0.025, 0.025)
CORNER_JITTER_RATIO = 0.015
AREA_RATIO_RANGE = (0.70, 1.35)
MAX_PAIR_ATTEMPTS = 200
MAX_WINDOW_ATTEMPTS = 512
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


@dataclass(frozen=True)
class WorkerTask:
    source_path: str
    split: str
    output_root: str
    pairs_per_image: int
    global_seed: int
    target_sizes: tuple[tuple[int, int], ...]


@dataclass
class WorkerResult:
    source_path: str
    split: str
    requested: int
    records: list[dict[str, Any]]
    errors: list[str]


def read_image(path: Path) -> np.ndarray:
    """Read an image robustly, including paths containing non-ASCII text."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(f"cannot read image bytes: {path}: {exc}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"OpenCV cannot decode a color image: {path}")
    return image


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        with temporary.open("wb") as stream:
            written = stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if written != len(data) or temporary.stat().st_size != len(data):
            raise OSError(f"short write for {path}")
        os.replace(temporary, path)
        if not path.is_file() or path.stat().st_size != len(data):
            raise OSError(f"failed to finalize {path}")
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def write_encoded_image(path: Path, image: np.ndarray, params: list[int]) -> None:
    extension = path.suffix.lower()
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok or encoded is None or encoded.size == 0:
        raise OSError(f"cv2.imencode failed for {path}")
    _atomic_write_bytes(path, encoded.tobytes())


def write_npy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, np.asarray(matrix, dtype=np.float64), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        check = np.load(temporary, allow_pickle=False)
        if check.shape != (3, 3) or check.dtype != np.float64:
            raise OSError(f"invalid temporary homography file: {temporary}")
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")
    images = sorted(
        (path.resolve() for path in input_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )
    if not images:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RuntimeError(f"no supported images found in {input_dir}; expected: {extensions}")
    return images


def split_sources(
    sources: list[Path], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[Path]]:
    indices = np.arange(len(sources))
    np.random.default_rng(seed).shuffle(indices)
    shuffled = [sources[int(index)] for index in indices]
    train_count = int(len(sources) * train_ratio)
    val_count = int(len(sources) * val_ratio)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }


def prepare_output_directories(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        for relative in (
            "input1", "input2", "homography", "masks/valid1",
            "masks/valid2", "masks/overlap1",
        ):
            (output_root / split / relative).mkdir(parents=True, exist_ok=True)
        (output_root / split / "buckets").mkdir(parents=True, exist_ok=True)


def save_split_sources(output_root: Path, splits: dict[str, list[Path]]) -> None:
    payload = {
        split: [str(path.resolve()) for path in paths]
        for split, paths in splits.items()
    }
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(output_root / "split_sources.json", data)


def stable_worker_seed(global_seed: int, split: str, source_path: str) -> int:
    material = f"{global_seed}\0{split}\0{source_path}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def source_identifier(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")[:40]
    return f"{safe_stem or 'image'}_{digest}"


def choose_crop_shape(
    width: int, height: int, rng: np.random.Generator
) -> tuple[int, int, str]:
    candidates: list[tuple[int, int, str]] = []
    weights: list[float] = []
    for crop_width, crop_height, weight in FIXED_CROP_PROFILES:
        fits = crop_width <= width and crop_height <= height
        can_shift = crop_width < width or crop_height < height
        if fits and can_shift:
            candidates.append((crop_width, crop_height, "fixed"))
            weights.append(weight)

    adaptive_width = int(round(width * ADAPTIVE_CROP_FRACTION))
    adaptive_height = int(round(height * ADAPTIVE_CROP_FRACTION))
    if min(adaptive_width, adaptive_height) >= MIN_CROP_DIMENSION:
        candidates.append((adaptive_width, adaptive_height, "adaptive_80pct"))
        weights.append(ADAPTIVE_CROP_WEIGHT)

    if not candidates:
        raise ValueError(
            "source cannot contain two distinct supported crop windows: "
            f"{width}x{height}"
        )
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    selected = int(rng.choice(len(candidates), p=probabilities))
    return candidates[selected]


def choose_target_shape(
    crop_width: int, crop_height: int,
    target_sizes: tuple[tuple[int, int], ...], rng: np.random.Generator,
) -> tuple[int, int]:
    supported = [
        (target_width, target_height)
        for target_width, target_height in target_sizes
        if target_width <= crop_width and target_height <= crop_height
    ]
    if not supported:
        raise ValueError(
            f"no target size can be downsampled from crop {crop_width}x{crop_height}"
        )
    # Area weighting favors high-resolution buckets without excluding 512.
    weights = np.asarray(
        [target_width * target_height for target_width, target_height in supported],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return supported[int(rng.choice(len(supported), p=weights))]


def rectangle_overlap(
    crop_width: int, crop_height: int, dx: int, dy: int
) -> float:
    overlap_width = max(0, crop_width - abs(dx))
    overlap_height = max(0, crop_height - abs(dy))
    return float(overlap_width * overlap_height) / float(crop_width * crop_height)


def sample_crop_windows(
    width: int, height: int, crop_width: int, crop_height: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, float]:
    max_x = width - crop_width
    max_y = height - crop_height
    for _ in range(MAX_WINDOW_ATTEMPTS):
        xa = int(rng.integers(0, max_x + 1))
        ya = int(rng.integers(0, max_y + 1))
        desired = float(rng.uniform(*BASE_OVERLAP_RANGE))

        # Factor desired overlap into independent retained x/y fractions.
        retained_x = float(rng.uniform(desired, 1.0))
        retained_y = desired / retained_x
        dx_abs = int(round(crop_width * (1.0 - retained_x)))
        dy_abs = int(round(crop_height * (1.0 - retained_y)))
        if dx_abs == 0 and dy_abs == 0:
            if rng.random() < 0.5:
                dx_abs = 1
            else:
                dy_abs = 1
        signs = rng.choice(np.asarray([-1, 1], dtype=np.int32), size=2)
        dx = int(signs[0]) * dx_abs
        dy = int(signs[1]) * dy_abs
        xb = xa + dx
        yb = ya + dy
        if not (0 <= xb <= max_x and 0 <= yb <= max_y):
            continue
        overlap = rectangle_overlap(crop_width, crop_height, dx, dy)
        if BASE_OVERLAP_RANGE[0] <= overlap <= BASE_OVERLAP_RANGE[1]:
            return xa, ya, xb, yb, dx, dy, overlap
    raise RuntimeError("could not sample two in-bounds crop windows with valid overlap")


def polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_convex_quad(points: np.ndarray) -> bool:
    contour = np.asarray(points, dtype=np.float32).reshape(4, 1, 2)
    return bool(cv2.isContourConvex(contour)) and polygon_area(points) > 1.0


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected_h = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    denominators = projected_h[:, 2]
    projected = projected_h[:, :2] / denominators[:, None]
    return projected, denominators


def sample_local_warp(
    crop_width: int, crop_height: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    rotation_deg = float(rng.uniform(*ROTATION_RANGE_DEG))
    scale = float(rng.uniform(*SCALE_RANGE))
    shear_x = float(rng.uniform(*SHEAR_RANGE))
    shear_y = float(rng.uniform(*SHEAR_RANGE))
    radians = math.radians(rotation_deg)
    rotation = np.asarray(
        [[math.cos(radians), -math.sin(radians)],
         [math.sin(radians), math.cos(radians)]],
        dtype=np.float64,
    )
    shear = np.asarray([[1.0, shear_x], [shear_y, 1.0]], dtype=np.float64)
    linear = scale * (rotation @ shear)
    center = np.asarray([crop_width / 2.0, crop_height / 2.0], dtype=np.float64)
    affine = np.eye(3, dtype=np.float64)
    affine[:2, :2] = linear
    affine[:2, 2] = center - linear @ center

    corners = np.asarray(
        [[0.0, 0.0], [float(crop_width), 0.0],
         [float(crop_width), float(crop_height)], [0.0, float(crop_height)]],
        dtype=np.float64,
    )
    affine_corners, _ = apply_homography(affine, corners)
    max_jitter = CORNER_JITTER_RATIO * min(crop_width, crop_height)
    offsets = rng.uniform(-max_jitter, max_jitter, size=(4, 2))
    destination = affine_corners + offsets
    if not is_convex_quad(destination):
        raise RuntimeError("sampled local warp is not convex")
    area_ratio = polygon_area(destination) / float(crop_width * crop_height)
    if not AREA_RATIO_RANGE[0] <= area_ratio <= AREA_RATIO_RANGE[1]:
        raise RuntimeError(f"local warp area ratio is invalid: {area_ratio:.6f}")

    local_warp = cv2.getPerspectiveTransform(
        corners.astype(np.float32), destination.astype(np.float32)
    ).astype(np.float64)
    if abs(float(local_warp[2, 2])) < 1e-12:
        raise RuntimeError("local warp has an invalid normalization term")
    local_warp /= local_warp[2, 2]
    parameters: dict[str, Any] = {
        "rotation_deg": rotation_deg,
        "scale": scale,
        "shear_x": shear_x,
        "shear_y": shear_y,
        "corner_jitter_ratio": CORNER_JITTER_RATIO,
        "corner_offsets_xy": offsets.tolist(),
        "area_ratio": area_ratio,
    }
    return local_warp, parameters, destination


def crop_transform(x: int, y: int) -> np.ndarray:
    return np.asarray(
        [[1.0, 0.0, -float(x)], [0.0, 1.0, -float(y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scale_transform(
    crop_width: int, crop_height: int, target_width: int, target_height: int
) -> np.ndarray:
    scale_x = target_width / float(crop_width)
    scale_y = target_height / float(crop_height)
    return np.asarray(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def validate_homography(
    h_gt: np.ndarray, local_destination: np.ndarray,
    crop_width: int, crop_height: int, target_width: int, target_height: int,
) -> None:
    if h_gt.shape != (3, 3) or not np.all(np.isfinite(h_gt)):
        raise RuntimeError("H_gt contains NaN/Inf or has an invalid shape")
    if abs(float(h_gt[2, 2])) < 1e-10:
        raise RuntimeError("H_gt[2,2] is too close to zero")
    determinant = float(np.linalg.det(h_gt))
    condition = float(np.linalg.cond(h_gt))
    if not np.isfinite(determinant) or abs(determinant) < 1e-12 or condition > 1e12:
        raise RuntimeError("H_gt is singular or numerically unstable")
    np.linalg.inv(h_gt)

    edge_x = float(target_width - 1)
    edge_y = float(target_height - 1)
    test_points = np.asarray(
        [[0.0, 0.0], [edge_x, 0.0], [edge_x, edge_y], [0.0, edge_y],
         [edge_x / 2.0, edge_y / 2.0]],
        dtype=np.float64,
    )
    projected, denominators = apply_homography(h_gt, test_points)
    denominator_scale = max(1.0, float(np.max(np.abs(h_gt))))
    if np.any(np.abs(denominators) < 1e-9 * denominator_scale):
        raise RuntimeError("projective denominator is too close to zero")
    if not np.all(np.isfinite(projected)) or not is_convex_quad(projected[:4]):
        raise RuntimeError("A corners do not map to a finite convex quadrilateral")

    area_ratio = polygon_area(local_destination) / float(crop_width * crop_height)
    if not AREA_RATIO_RANGE[0] <= area_ratio <= AREA_RATIO_RANGE[1]:
        raise RuntimeError("B local warped quadrilateral area is outside the valid range")


def has_usable_content(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nonblack_fraction = float(np.count_nonzero(gray > 5)) / float(gray.size)
    standard_deviation = float(gray.std())
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return nonblack_fraction >= 0.05 and standard_deviation >= 5.0 and laplacian_variance >= 8.0


def build_pair(
    original: np.ndarray, source_mask: np.ndarray, rng: np.random.Generator,
    target_sizes: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    height, width = original.shape[:2]
    crop_width, crop_height, crop_mode = choose_crop_shape(width, height, rng)
    target_width, target_height = choose_target_shape(
        crop_width, crop_height, target_sizes, rng
    )
    xa, ya, xb, yb, dx, dy, base_overlap = sample_crop_windows(
        width, height, crop_width, crop_height, rng
    )
    local_warp, warp_parameters, local_destination = sample_local_warp(
        crop_width, crop_height, rng
    )
    c_a = crop_transform(xa, ya)
    c_b = crop_transform(xb, yb)
    scaling = scale_transform(
        crop_width, crop_height, target_width, target_height
    )
    g_a = scaling @ c_a
    g_b = scaling @ local_warp @ c_b
    h_gt = g_b @ np.linalg.inv(g_a)
    if abs(float(h_gt[2, 2])) < 1e-10:
        raise RuntimeError("H_gt cannot be normalized")
    h_gt = h_gt / h_gt[2, 2]
    validate_homography(
        h_gt, local_destination, crop_width, crop_height,
        target_width, target_height,
    )

    output_shape = (target_width, target_height)
    image_a = cv2.warpPerspective(
        original, g_a, output_shape, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    image_b = cv2.warpPerspective(
        original, g_b, output_shape, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    valid_a = cv2.warpPerspective(
        source_mask, g_a, output_shape, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid_b = cv2.warpPerspective(
        source_mask, g_b, output_shape, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid_a = np.where(valid_a > 0, 255, 0).astype(np.uint8)
    valid_b = np.where(valid_b > 0, 255, 0).astype(np.uint8)
    valid_b_in_a = cv2.warpPerspective(
        valid_b, np.linalg.inv(h_gt), output_shape, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    overlap_a = cv2.bitwise_and(valid_a, valid_b_in_a)
    valid_a_pixels = int(np.count_nonzero(valid_a))
    if valid_a_pixels == 0:
        raise RuntimeError("valid_a is empty")
    final_overlap = float(np.count_nonzero(overlap_a)) / float(valid_a_pixels)
    if not FINAL_OVERLAP_RANGE[0] <= final_overlap <= FINAL_OVERLAP_RANGE[1]:
        raise RuntimeError(f"final overlap is outside [0.50, 0.95]: {final_overlap:.6f}")
    if not has_usable_content(image_a) or not has_usable_content(image_b):
        raise RuntimeError("sample is nearly black or has insufficient texture")

    return {
        "image_a_array": image_a,
        "image_b_array": image_b,
        "valid_a_array": valid_a,
        "valid_b_array": valid_b,
        "overlap_a_array": overlap_a,
        "h_gt_array": h_gt.astype(np.float64),
        "g_a": g_a,
        "g_b": g_b,
        "local_warp": local_warp,
        "crop_size": [crop_width, crop_height],
        "crop_mode": crop_mode,
        "target_size": [target_width, target_height],
        "crop_a_xy": [xa, ya],
        "crop_b_xy": [xb, yb],
        "crop_shift_xy": [dx, dy],
        "base_overlap": base_overlap,
        "final_overlap": final_overlap,
        "warp": warp_parameters,
    }


def pair_paths(output_root: Path, split: str, pair_id: str) -> dict[str, Path]:
    split_root = output_root / split
    return {
        "image_a": split_root / "input1" / f"{pair_id}.jpg",
        "image_b": split_root / "input2" / f"{pair_id}.jpg",
        "h_gt": split_root / "homography" / f"{pair_id}.npy",
        "valid_a": split_root / "masks" / "valid1" / f"{pair_id}.png",
        "valid_b": split_root / "masks" / "valid2" / f"{pair_id}.png",
        "overlap_a": split_root / "masks" / "overlap1" / f"{pair_id}.png",
    }


def cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def save_pair(
    pair: dict[str, Any], paths: dict[str, Path], pair_id: str, split: str,
    source_path: Path, source_size: tuple[int, int], pair_seed: int,
) -> dict[str, Any]:
    completed: list[Path] = []
    try:
        write_encoded_image(
            paths["image_a"], pair["image_a_array"],
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        completed.append(paths["image_a"])
        write_encoded_image(
            paths["image_b"], pair["image_b_array"],
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        completed.append(paths["image_b"])
        write_npy(paths["h_gt"], pair["h_gt_array"])
        completed.append(paths["h_gt"])
        for key, array_key in (
            ("valid_a", "valid_a_array"), ("valid_b", "valid_b_array"),
            ("overlap_a", "overlap_a_array"),
        ):
            write_encoded_image(paths[key], pair[array_key], [cv2.IMWRITE_PNG_COMPRESSION, 3])
            completed.append(paths[key])
    except Exception:
        cleanup_paths(completed)
        raise

    width, height = source_size
    return {
        "pair_id": pair_id,
        "split": split,
        "source_image": str(source_path.resolve()),
        "image_a": str(paths["image_a"].resolve()),
        "image_b": str(paths["image_b"].resolve()),
        "h_gt_a_to_b": str(paths["h_gt"].resolve()),
        "valid_a": str(paths["valid_a"].resolve()),
        "valid_b": str(paths["valid_b"].resolve()),
        "overlap_a": str(paths["overlap_a"].resolve()),
        "source_size": [width, height],
        "target_size": pair["target_size"],
        "target_bucket": f"{pair['target_size'][0]}x{pair['target_size'][1]}",
        "crop_size": pair["crop_size"],
        "crop_mode": pair["crop_mode"],
        "crop_a_xy": pair["crop_a_xy"],
        "crop_b_xy": pair["crop_b_xy"],
        "crop_shift_xy": pair["crop_shift_xy"],
        "base_overlap": pair["base_overlap"],
        "final_overlap": pair["final_overlap"],
        "warp": pair["warp"],
        "g_a": np.asarray(pair["g_a"], dtype=np.float64).tolist(),
        "g_b": np.asarray(pair["g_b"], dtype=np.float64).tolist(),
        "w_b_local": np.asarray(pair["local_warp"], dtype=np.float64).tolist(),
        "direction": "A_to_B",
        "seed": int(pair_seed),
    }


def process_source(task: WorkerTask) -> WorkerResult:
    source_path = Path(task.source_path)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        original = read_image(source_path)
    except Exception as exc:
        return WorkerResult(
            task.source_path, task.split, task.pairs_per_image, [],
            [f"load failed: {exc}"],
        )

    height, width = original.shape[:2]
    if min(width, height) < MIN_CROP_DIMENSION:
        return WorkerResult(
            task.source_path, task.split, task.pairs_per_image, [],
            [f"source {width}x{height} is smaller than 768x768"],
        )
    source_mask = np.full((height, width), 255, dtype=np.uint8)
    worker_seed = stable_worker_seed(task.global_seed, task.split, task.source_path)
    worker_rng = np.random.default_rng(worker_seed)
    identifier = source_identifier(source_path)
    output_root = Path(task.output_root)

    for pair_index in range(task.pairs_per_image):
        saved = False
        last_error = "unknown sampling failure"
        for _ in range(MAX_PAIR_ATTEMPTS):
            pair_seed = int(worker_rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            pair_rng = np.random.default_rng(pair_seed)
            try:
                pair = build_pair(
                    original, source_mask, pair_rng, task.target_sizes
                )
                pair_id = f"{identifier}_{pair_index:04d}"
                paths = pair_paths(output_root, task.split, pair_id)
                record = save_pair(
                    pair, paths, pair_id, task.split, source_path,
                    (width, height), pair_seed,
                )
                records.append(record)
                saved = True
                break
            except Exception as exc:
                last_error = str(exc)
        if not saved:
            errors.append(f"pair {pair_index}: exhausted retries: {last_error}")

    return WorkerResult(
        task.source_path, task.split, task.pairs_per_image, records, errors
    )


def write_manifest(output_root: Path, split: str, records: list[dict[str, Any]]) -> None:
    records.sort(key=lambda item: item["pair_id"])
    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_write_bytes(output_root / split / "manifest.jsonl", text.encode("utf-8"))

    bucket_root = output_root / split / "buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["target_bucket"], []).append(record)
    bucket_index: dict[str, dict[str, Any]] = {}
    for bucket, bucket_records in sorted(grouped.items()):
        bucket_path = bucket_root / f"{bucket}.jsonl"
        bucket_text = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in bucket_records
        )
        _atomic_write_bytes(bucket_path, bucket_text.encode("utf-8"))
        bucket_index[bucket] = {
            "manifest": str(bucket_path.resolve()),
            "count": len(bucket_records),
        }
    index_data = (
        json.dumps(bucket_index, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(bucket_root / "index.json", index_data)


def load_manifest_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        manifest = output_root / split / "manifest.jsonl"
        if not manifest.is_file():
            continue
        with manifest.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"invalid JSON in {manifest}:{line_number}: {exc}"
                        ) from exc
    return records


def create_self_check_visualization(
    record: dict[str, Any], projected_points: np.ndarray, source_points: np.ndarray,
    output_path: Path,
) -> None:
    image_a = read_image(Path(record["image_a"]))
    image_b = read_image(Path(record["image_b"]))
    target_width, target_height = map(int, record["target_size"])
    if image_a.shape[:2] != (target_height, target_width):
        raise AssertionError("image A dimensions disagree with manifest target_size")
    if image_b.shape[:2] != (target_height, target_width):
        raise AssertionError("image B dimensions disagree with manifest target_size")
    canvas = np.hstack([image_a, image_b])
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (80, 180, 255), (180, 80, 255),
    ]
    for index, (point_a, point_b) in enumerate(zip(source_points, projected_points)):
        color = colors[index % len(colors)]
        a_xy = tuple(np.rint(point_a).astype(int))
        b_xy = tuple(np.rint(point_b).astype(int) + np.asarray([target_width, 0]))
        cv2.circle(canvas, a_xy, 5, color, -1, lineType=cv2.LINE_AA)
        if (
            target_width <= b_xy[0] < 2 * target_width
            and 0 <= b_xy[1] < target_height
        ):
            cv2.circle(canvas, b_xy, 5, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, a_xy, b_xy, color, 1, lineType=cv2.LINE_AA)
    write_encoded_image(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def self_check_dataset(
    output_root: Path, seed: int = 0, visualization_path: Path | None = None
) -> dict[str, Any]:
    records = load_manifest_records(output_root)
    if not records:
        raise RuntimeError(f"no generated samples found below {output_root}")
    rng = np.random.default_rng(seed)
    record = records[int(rng.integers(0, len(records)))]
    h_saved = np.load(record["h_gt_a_to_b"], allow_pickle=False).astype(np.float64)
    g_a = np.asarray(record["g_a"], dtype=np.float64)
    g_b = np.asarray(record["g_b"], dtype=np.float64)
    h_composed = g_b @ np.linalg.inv(g_a)
    h_composed /= h_composed[2, 2]
    matrix_error = float(np.max(np.abs(h_saved - h_composed)))

    target_width, target_height = map(int, record["target_size"])
    edge_x = float(target_width - 1)
    edge_y = float(target_height - 1)
    fixed_points = np.asarray(
        [[0.0, 0.0], [edge_x, 0.0], [edge_x, edge_y], [0.0, edge_y],
         [edge_x / 2.0, edge_y / 2.0]], dtype=np.float64,
    )
    random_points = np.column_stack([
        rng.uniform(0.0, edge_x, size=8),
        rng.uniform(0.0, edge_y, size=8),
    ])
    source_points = np.vstack([fixed_points, random_points])
    projected_saved, denominators = apply_homography(h_saved, source_points)
    projected_composed, _ = apply_homography(h_composed, source_points)
    projection_error = float(np.max(np.abs(projected_saved - projected_composed)))
    if not np.all(np.isfinite(projected_saved)) or np.any(np.abs(denominators) < 1e-9):
        raise AssertionError("self-check found invalid A-to-B projections")
    if matrix_error >= 1e-5 or projection_error >= 1e-5:
        raise AssertionError(
            f"self-check failed: matrix_error={matrix_error:.3e}, "
            f"projection_error={projection_error:.3e}"
        )
    if record.get("direction") != "A_to_B":
        raise AssertionError("manifest direction is not A_to_B")
    if visualization_path is not None:
        create_self_check_visualization(
            record, projected_saved, source_points, visualization_path.resolve()
        )

    print("\nSelf-check passed")
    print(f"pair_id: {record['pair_id']}")
    print(f"direction: A -> B")
    print(f"target_size: {target_width}x{target_height}")
    print(f"base_overlap: {record['base_overlap']:.6f}")
    print(f"final_overlap: {record['final_overlap']:.6f}")
    print(f"max matrix error: {matrix_error:.3e}")
    print(f"max projection error: {projection_error:.3e}")
    print("H_gt:\n" + np.array2string(h_saved, precision=10, suppress_small=False))
    if visualization_path is not None:
        print(f"visualization: {visualization_path.resolve()}")
    return {
        "pair_id": record["pair_id"],
        "matrix_error": matrix_error,
        "projection_error": projection_error,
    }


def parse_target_sizes(value: str) -> tuple[tuple[int, int], ...]:
    sizes: list[tuple[int, int]] = []
    for raw_token in value.replace("×", "x").split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        parts = token.split("x")
        if len(parts) == 1:
            parts = [parts[0], parts[0]]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "target sizes must look like 512x512,768x768,1024x1024"
            )
        try:
            width, height = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid target size: {raw_token}") from exc
        if width < 64 or height < 64:
            raise argparse.ArgumentTypeError("target width and height must be at least 64")
        if width % 32 != 0 or height % 32 != 0:
            raise argparse.ArgumentTypeError(
                f"target size {width}x{height} must be divisible by 32"
            )
        shape = (width, height)
        if shape not in sizes:
            sizes.append(shape)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one target size is required")
    return tuple(sizes)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.pairs_per_image <= 0:
        raise ValueError("--pairs-per-image must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not 0.0 <= args.train_ratio <= 1.0:
        raise ValueError("--train-ratio must be in [0, 1]")
    if not 0.0 <= args.val_ratio <= 1.0:
        raise ValueError("--val-ratio must be in [0, 1]")
    if args.train_ratio + args.val_ratio > 1.0:
        raise ValueError("--train-ratio + --val-ratio must not exceed 1")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate exact A-to-B homography pairs from high-resolution images."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-image", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--target-sizes", type=parse_target_sizes,
        default=DEFAULT_TARGET_SIZES,
        help=(
            "Comma-separated output WxH buckets; only downsampling-compatible "
            "sizes are sampled (default: 512x512,768x768,1024x1024)."
        ),
    )
    parser.add_argument(
        "--self-check-visualization", type=Path, default=None,
        help="Optionally save an A/B projected-point visualization after generation.",
    )
    return parser


def run_generation(args: argparse.Namespace) -> int:
    validate_arguments(args)
    input_dir = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    sources = discover_images(input_dir)
    splits = split_sources(sources, args.train_ratio, args.val_ratio, args.seed)
    prepare_output_directories(output_root)
    save_split_sources(output_root, splits)

    tasks: list[WorkerTask] = []
    for split in ("train", "val", "test"):
        for source in splits[split]:
            tasks.append(WorkerTask(
                source_path=str(source), split=split,
                output_root=str(output_root), pairs_per_image=args.pairs_per_image,
                global_seed=args.seed, target_sizes=args.target_sizes,
            ))

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [], "val": [], "test": []
    }
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_source, task): task for task in tasks}
        with tqdm(total=len(tasks), desc="Source images", unit="image") as progress:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    records_by_split[result.split].extend(result.records)
                    errors.extend(
                        f"[{result.split}] {result.source_path}: {message}"
                        for message in result.errors
                    )
                except Exception:
                    errors.append(
                        f"[{task.split}] {task.source_path}: worker crashed:\n"
                        + traceback.format_exc()
                    )
                progress.update(1)

    for split in ("train", "val", "test"):
        write_manifest(output_root, split, records_by_split[split])
        requested = len(splits[split]) * args.pairs_per_image
        actual = len(records_by_split[split])
        target_counts: dict[str, int] = {}
        for record in records_by_split[split]:
            bucket = record["target_bucket"]
            target_counts[bucket] = target_counts.get(bucket, 0) + 1
        print(
            f"{split}: sources={len(splits[split])}, "
            f"requested_pairs={requested}, successful_pairs={actual}, "
            f"target_buckets={target_counts}"
        )
    if errors:
        print(f"\nWarnings/failures ({len(errors)}):", file=sys.stderr)
        for message in errors:
            print(f"- {message}", file=sys.stderr)

    total_success = sum(len(records) for records in records_by_split.values())
    if total_success == 0:
        raise RuntimeError("generation produced zero valid pairs")
    self_check_dataset(
        output_root, seed=args.seed,
        visualization_path=args.self_check_visualization,
    )
    return 0


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        return run_generation(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
