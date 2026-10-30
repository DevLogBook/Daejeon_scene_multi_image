"""Paired Homography dataset loader for the standalone GRM-Matcher.

The manifest/path resolution and resolution-bucket batching logic are adapted
from ``dense_match/train_homography_gt.py`` without importing that module.  The
generated dataset contract comes from
``dataset/generate_single_image_homography_pairs.py``: every saved homography
maps ``input1`` coordinates to ``input2`` coordinates (A -> B).
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .config import GRMMatcherConfig
from .coordinates import validate_image_size


class GRMHomographyPairDataset(Dataset[dict[str, Tensor]]):
    """Read exact A-to-B Homography pairs and construct dense GRM targets."""

    required_manifest_fields = {
        "image_a",
        "image_b",
        "h_gt_a_to_b",
        "valid_a",
        "valid_b",
        "overlap_a",
        "target_size",
        "target_bucket",
        "direction",
    }

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        config: GRMMatcherConfig,
        selected_buckets: Optional[Sequence[str]] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.config = config
        selected = set(selected_buckets) if selected_buckets else None
        self.records = [
            record
            for record in self._load_manifest(self.dataset_root / split / "manifest.jsonl")
            if selected is None or str(record["target_bucket"]) in selected
        ]
        if not self.records:
            requested = "all" if selected is None else sorted(selected)
            raise RuntimeError(f"No {split!r} Homography pairs remain for buckets={requested}.")

        self.bucket_to_indices: Dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.bucket_to_indices[str(record["target_bucket"])].append(index)
        self.bucket_to_indices = dict(self.bucket_to_indices)
        self._grid_cache: Dict[tuple[int, int], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def source_image_paths(self) -> tuple[Path, ...]:
        """Return unique ``input1`` paths from this manifest-filtered split.

        Stage-3 controlled geometry uses these images as texture sources while
        retaining the original train/validation split.  It deliberately does
        not scan the dataset root, which could leak validation images into
        training.
        """

        paths: list[Path] = []
        seen: set[Path] = set()
        for record in self.records:
            path = self._resolve_record_path(record["image_a"], "input1")
            if path not in seen:
                paths.append(path)
                seen.add(path)
        return tuple(paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        image_a_path = self._resolve_record_path(record["image_a"], "input1")
        image_b_path = self._resolve_record_path(record["image_b"], "input2")
        h_ab_path = self._resolve_record_path(record["h_gt_a_to_b"], "homography")
        valid_a_path = self._resolve_record_path(record["valid_a"], "masks/valid1")
        valid_b_path = self._resolve_record_path(record["valid_b"], "masks/valid2")
        overlap_a_path = self._resolve_record_path(record["overlap_a"], "masks/overlap1")

        image_a = self._read_rgb(image_a_path)
        image_b = self._read_rgb(image_b_path)
        valid_a = self._read_mask(valid_a_path)
        valid_b = self._read_mask(valid_b_path)
        overlap_a = self._read_mask(overlap_a_path) & valid_a
        height, width = image_a.shape[:2]
        self._validate_record_shapes(
            record, image_a, image_b, valid_a, valid_b, overlap_a
        )

        h_ab = np.load(h_ab_path, allow_pickle=False).astype(np.float64, copy=False)
        h_ab = self._validate_homography(h_ab, h_ab_path)
        dense_mapping = self._dense_mapping(h_ab, height, width)
        overlap_b = cv2.warpPerspective(
            overlap_a.astype(np.uint8) * 255,
            h_ab,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        overlap_b &= valid_b

        image_a_tensor = self._image_tensor(image_a)
        image_b_tensor = self._image_tensor(image_b)
        overlap_a_tensor = torch.from_numpy(np.ascontiguousarray(overlap_a)).unsqueeze(0)
        overlap_b_tensor = torch.from_numpy(np.ascontiguousarray(overlap_b)).unsqueeze(0)
        return {
            "image_a": image_a_tensor,
            "image_b": image_b_tensor,
            "H_ab": torch.from_numpy(h_ab.astype(np.float32, copy=True)),
            "gt_dense_mapping": torch.from_numpy(
                np.ascontiguousarray(dense_mapping.transpose(2, 0, 1), dtype=np.float32)
            ),
            "gt_overlap_a": overlap_a_tensor,
            "gt_overlap_b": overlap_b_tensor,
            "gt_global_mask": overlap_a_tensor.clone(),
            "gt_local_mask": torch.zeros_like(overlap_a_tensor),
            "local_flow": torch.zeros(2, height, width, dtype=torch.float32),
            "valid_a": torch.from_numpy(np.ascontiguousarray(valid_a)).unsqueeze(0),
            "valid_b": torch.from_numpy(np.ascontiguousarray(valid_b)).unsqueeze(0),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }

    def _load_manifest(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Paired dataset manifest not found: {path}. The dataset root must "
                "contain train/val/test manifests produced by "
                "generate_single_image_homography_pairs.py."
            )
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
                missing = self.required_manifest_fields - set(record)
                if missing:
                    raise RuntimeError(
                        f"{path}:{line_number} is missing fields: {sorted(missing)}"
                    )
                if record["direction"] != "A_to_B":
                    raise RuntimeError(
                        f"{path}:{line_number} has direction={record['direction']!r}; "
                        "GRM requires H_ab to map A to B."
                    )
                records.append(record)
        if not records:
            raise RuntimeError(f"Paired dataset manifest is empty: {path}")
        return records

    def _resolve_record_path(self, recorded_path: object, relative_directory: str) -> Path:
        raw = str(recorded_path)
        direct = Path(raw)
        if direct.is_file():
            return direct
        normalized = Path(raw.replace("\\", "/"))
        if normalized.is_file():
            return normalized
        local = self.dataset_root / self.split / relative_directory / normalized.name
        if local.is_file():
            return local
        raise FileNotFoundError(
            f"Dataset file not found; recorded={raw!r}, local fallback={local}"
        )

    def _image_tensor(self, image_rgb: np.ndarray) -> Tensor:
        tensor = torch.from_numpy(
            np.ascontiguousarray(image_rgb.transpose(2, 0, 1), dtype=np.float32)
        ) / 255.0
        if self.config.input_range == "imagenet_normalized":
            mean = tensor.new_tensor(self.config.normalization_mean).view(3, 1, 1)
            std = tensor.new_tensor(self.config.normalization_std).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor

    def _dense_mapping(self, h_ab: np.ndarray, height: int, width: int) -> np.ndarray:
        key = (height, width)
        homogeneous = self._grid_cache.get(key)
        if homogeneous is None:
            yy, xx = np.meshgrid(
                np.arange(height, dtype=np.float64),
                np.arange(width, dtype=np.float64),
                indexing="ij",
            )
            homogeneous = np.stack(
                (xx.reshape(-1), yy.reshape(-1), np.ones(height * width)), axis=0
            )
            self._grid_cache[key] = homogeneous
        projected = h_ab @ homogeneous
        denominator = projected[2]
        sign = np.where(denominator < 0.0, -1.0, 1.0)
        safe_denominator = np.where(
            np.abs(denominator) >= 1.0e-8, denominator, sign * 1.0e-8
        )
        mapping = (projected[:2] / safe_denominator[None]).T.reshape(height, width, 2)
        if not np.all(np.isfinite(mapping)):
            raise RuntimeError("H_ab produced non-finite dense mapping coordinates.")
        return mapping

    def _validate_record_shapes(
        self,
        record: Mapping[str, Any],
        image_a: np.ndarray,
        image_b: np.ndarray,
        *masks: np.ndarray,
    ) -> None:
        target_width, target_height = map(int, record["target_size"])
        expected = (target_height, target_width)
        if image_a.shape[:2] != expected or image_b.shape[:2] != expected:
            raise RuntimeError(
                f"Pair {record.get('pair_id')} image size disagrees with manifest "
                f"target_size={target_width}x{target_height}."
            )
        if any(mask.shape != expected for mask in masks):
            raise RuntimeError(f"Pair {record.get('pair_id')} mask size mismatch.")
        expected_bucket = f"{target_width}x{target_height}"
        if str(record["target_bucket"]) != expected_bucket:
            raise RuntimeError(
                f"Pair {record.get('pair_id')} bucket is {record['target_bucket']!r}, "
                f"expected {expected_bucket!r}."
            )
        validate_image_size(target_height, target_width, self.config.coarse_stride)

    @staticmethod
    def _validate_homography(matrix: np.ndarray, path: Path) -> np.ndarray:
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"Invalid Homography matrix: {path}")
        if abs(float(np.linalg.det(matrix))) < 1.0e-10:
            raise RuntimeError(f"Singular Homography matrix: {path}")
        if abs(float(matrix[2, 2])) < 1.0e-10:
            raise RuntimeError(f"Homography cannot be normalized: {path}")
        return matrix / matrix[2, 2]

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read RGB image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        return mask > 0


class GRMResolutionBucketBatchSampler(Sampler[list[int]]):
    """Create homogeneous-resolution batches with optional per-bucket sizes."""

    def __init__(
        self,
        bucket_to_indices: Mapping[str, Sequence[int]],
        default_batch_size: int,
        bucket_batch_sizes: Optional[Mapping[str, int]] = None,
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        if default_batch_size <= 0:
            raise ValueError("default_batch_size must be positive.")
        self.bucket_to_indices = {
            name: list(indices) for name, indices in bucket_to_indices.items() if indices
        }
        if not self.bucket_to_indices:
            raise ValueError("bucket_to_indices must contain at least one sample.")
        self.default_batch_size = default_batch_size
        self.bucket_batch_sizes = dict(bucket_batch_sizes or {})
        for name, size in self.bucket_batch_sizes.items():
            if size <= 0:
                raise ValueError(f"Batch size for bucket {name!r} must be positive.")
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def batch_size_for(self, bucket: str) -> int:
        return int(self.bucket_batch_sizes.get(bucket, self.default_batch_size))

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for bucket, original_indices in sorted(self.bucket_to_indices.items()):
            indices = list(original_indices)
            if self.shuffle:
                generator.shuffle(indices)
            batch_size = self.batch_size_for(bucket)
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                if len(batch) == batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for bucket, indices in self.bucket_to_indices.items():
            batch_size = self.batch_size_for(bucket)
            if self.drop_last:
                total += len(indices) // batch_size
            else:
                total += (len(indices) + batch_size - 1) // batch_size
        return total


def parse_bucket_batch_sizes(values: Optional[Sequence[str]]) -> Dict[str, int]:
    """Parse repeated ``BUCKET:BATCH`` CLI values."""
    result: Dict[str, int] = {}
    for value in values or ():
        if ":" not in value:
            raise ValueError(
                f"Invalid bucket batch size {value!r}; expected e.g. 768x768:2."
            )
        bucket, raw_size = value.rsplit(":", 1)
        try:
            size = int(raw_size)
        except ValueError as error:
            raise ValueError(f"Invalid batch size in {value!r}.") from error
        if not bucket or size <= 0:
            raise ValueError(f"Invalid bucket batch size {value!r}.")
        result[bucket] = size
    return result
