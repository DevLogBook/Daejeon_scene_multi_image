"""Real overlapping image pairs with optional weak Homography supervision."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import GRMMatcherConfig
from .coordinates import validate_image_size


class GRMRealPairDataset(Dataset[dict[str, object]]):
    """Load real A/B pairs from a manifest or matching ``input1/input2`` names.

    Preferred manifest location is ``<root>/<split>/manifest.jsonl``.  Every
    record must contain ``image_a`` and ``image_b`` and may contain ``pair_id``,
    ``valid_a``, ``valid_b``, and ``target_bucket``.  When no manifest exists,
    files with identical names below ``input1`` and ``input2`` are paired.
    """

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        config: GRMMatcherConfig,
        selected_buckets: Optional[Sequence[str]] = None,
        difficulty: str = "all",
        selected_pair_buckets: Optional[Sequence[str]] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.split_root = self.dataset_root / split
        self.config = config
        if not self.split_root.is_dir():
            raise NotADirectoryError(f"Real-pair split does not exist: {self.split_root}")
        manifest = self.split_root / "manifest.jsonl"
        records = self._load_manifest(manifest) if manifest.is_file() else self._scan_pairs()
        selected = set(selected_buckets) if selected_buckets else None
        selected_pairs = set(selected_pair_buckets) if selected_pair_buckets else None
        if difficulty not in {"all", "simple", "hard", "negative"}:
            raise ValueError(
                "difficulty must be 'all', 'simple', 'hard', or 'negative'."
            )
        self.difficulty = difficulty
        self.records = [
            record
            for record in records
            if (selected is None or str(record["target_bucket"]) in selected)
            and (
                selected_pairs is None
                or str(record.get("bucket", "unknown")) in selected_pairs
            )
            and (
                difficulty == "all"
                or (
                    difficulty == "simple"
                    and str(record.get("bucket")) in {"global", "residual"}
                )
                or (
                    difficulty == "hard"
                    and str(record.get("bucket")) == "hard_positive"
                )
                or (
                    difficulty == "negative"
                    and str(record.get("bucket")) == "negative"
                )
            )
        ]
        if not self.records:
            raise RuntimeError(f"No real image pairs found for split={split!r}.")
        self.bucket_to_indices: Dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.bucket_to_indices[str(record["target_bucket"])].append(index)
        self.bucket_to_indices = dict(self.bucket_to_indices)
        self.pair_bucket_counts = dict(
            Counter(str(record.get("bucket", "unknown")) for record in self.records)
        )

    def __len__(self) -> int:
        return len(self.records)

    def source_image_paths(self) -> tuple[Path, ...]:
        """Return unique real-domain images suitable as synthetic texture sources."""

        paths = {
            Path(record[key]).resolve()
            for record in self.records
            for key in ("image_a", "image_b")
        }
        return tuple(sorted(paths, key=lambda path: str(path)))

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        path_a = Path(record["image_a"])
        path_b = Path(record["image_b"])
        image_a = self._read_rgb(path_a)
        image_b = self._read_rgb(path_b)
        if image_a.shape != image_b.shape:
            raise RuntimeError(
                f"Real pair must have identical image shapes: {path_a}={image_a.shape}, "
                f"{path_b}={image_b.shape}."
            )
        height, width = image_a.shape[:2]
        validate_image_size(height, width, self.config.coarse_stride)
        expected_bucket = f"{width}x{height}"
        if str(record["target_bucket"]) != expected_bucket:
            raise RuntimeError(
                f"Pair {record['pair_id']!r} declares bucket={record['target_bucket']!r}, "
                f"but its images are {expected_bucket}."
            )
        valid_a = self._read_optional_mask(record.get("valid_a"), path_a, "valid1")
        valid_b = self._read_optional_mask(record.get("valid_b"), path_b, "valid2")
        overlap_known_a = bool(record.get("overlap_known_a", False))
        overlap_known_b = bool(record.get("overlap_known_b", False))
        overlap_usable_a = bool(record.get("overlap_usable_a", False))
        overlap_usable_b = bool(record.get("overlap_usable_b", False))
        overlap_a = self._read_optional_mask(
            record.get("overlap_a"), path_a, "overlap1"
        ) if overlap_known_a else np.ones((height, width), dtype=np.bool_)
        overlap_b = self._read_optional_mask(
            record.get("overlap_b"), path_b, "overlap2"
        ) if overlap_known_b else np.ones((height, width), dtype=np.bool_)
        h_ab, h_ab_valid = self._read_optional_homography(record.get("h_ab"))
        h_ba, h_ba_valid = self._read_optional_homography(record.get("h_ba"))
        homography_valid = h_ab_valid and h_ba_valid
        return {
            "image_a": self._image_tensor(image_a),
            "image_b": self._image_tensor(image_b),
            "valid_a": torch.from_numpy(np.ascontiguousarray(valid_a)).unsqueeze(0),
            "valid_b": torch.from_numpy(np.ascontiguousarray(valid_b)).unsqueeze(0),
            "overlap_a": torch.from_numpy(np.ascontiguousarray(overlap_a)).unsqueeze(0),
            "overlap_b": torch.from_numpy(np.ascontiguousarray(overlap_b)).unsqueeze(0),
            "overlap_known_a": torch.tensor(overlap_known_a, dtype=torch.bool),
            "overlap_known_b": torch.tensor(overlap_known_b, dtype=torch.bool),
            "overlap_usable_a": torch.tensor(overlap_usable_a, dtype=torch.bool),
            "overlap_usable_b": torch.tensor(overlap_usable_b, dtype=torch.bool),
            # Backward-compatible aliases refer to the A/source mask.
            "overlap_known": torch.tensor(overlap_known_a, dtype=torch.bool),
            "overlap_usable": torch.tensor(overlap_usable_a, dtype=torch.bool),
            "overlap_fraction_a": torch.tensor(
                float(record.get("overlap_fraction_a", 0.0)), dtype=torch.float32
            ),
            "overlap_fraction_b": torch.tensor(
                float(record.get("overlap_fraction_b", 0.0)), dtype=torch.float32
            ),
            "overlap_fraction": torch.tensor(
                float(record.get("overlap_fraction_a", 0.0)), dtype=torch.float32
            ),
            "pair_label": torch.tensor(int(record.get("label", -1)), dtype=torch.long),
            "pair_bucket": str(record.get("bucket", "unknown")),
            "H_ab": torch.from_numpy(h_ab.astype(np.float32, copy=True)),
            "H_ba": torch.from_numpy(h_ba.astype(np.float32, copy=True)),
            "homography_valid": torch.tensor(homography_valid, dtype=torch.bool),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "pair_id": str(record["pair_id"]),
        }

    def _load_manifest(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
                if "image_a" not in raw or "image_b" not in raw:
                    raise RuntimeError(
                        f"{path}:{line_number} must contain image_a and image_b."
                    )
                image_a = self._resolve_path(raw["image_a"], "input1")
                image_b = self._resolve_path(raw["image_b"], "input2")
                height, width = self._read_rgb(image_a).shape[:2]
                record = dict(raw)
                record["image_a"] = image_a
                record["image_b"] = image_b
                record["pair_id"] = str(raw.get("pair_id", image_a.stem))
                record["target_bucket"] = str(raw.get("target_bucket", f"{width}x{height}"))
                for key, directory in (("valid_a", "masks/valid1"), ("valid_b", "masks/valid2")):
                    if raw.get(key) is not None:
                        record[key] = self._resolve_path(raw[key], directory)
                h_ab_value = raw.get("h_ab", raw.get("h_gt_a_to_b"))
                h_ba_value = raw.get("h_ba")
                if h_ab_value is not None:
                    record["h_ab"] = self._resolve_path(h_ab_value, "homography")
                if h_ba_value is not None:
                    record["h_ba"] = self._resolve_path(h_ba_value, "homography_ba")
                for key, directory, image_path, suffix in (
                    ("overlap_a", "masks/overlap1", image_a, "a"),
                    ("overlap_b", "masks/overlap2", image_b, "b"),
                ):
                    overlap_value = raw.get(key)
                    if overlap_value is not None:
                        record[key] = self._resolve_path(overlap_value, directory)
                    else:
                        candidate = self.split_root / directory / image_path.with_suffix(
                            ".png"
                        ).name
                        if candidate.is_file():
                            record[key] = candidate
                    known, fraction = self._overlap_metadata(record.get(key))
                    record[f"overlap_known_{suffix}"] = known
                    record[f"overlap_fraction_{suffix}"] = fraction
                    record[f"overlap_usable_{suffix}"] = fraction > 0.0
                records.append(record)
        if not records:
            raise RuntimeError(f"Real-pair manifest is empty: {path}")
        return records

    def _scan_pairs(self) -> list[dict[str, Any]]:
        input_a = self.split_root / "input1"
        input_b = self.split_root / "input2"
        if not input_a.is_dir() or not input_b.is_dir():
            raise FileNotFoundError(
                f"Expected {input_a}, {input_b}, or a manifest.jsonl file."
            )
        paths_a = sorted(
            path for path in input_a.rglob("*")
            if path.is_file() and path.suffix.lower() in self.image_extensions
        )
        records: list[dict[str, Any]] = []
        missing: list[Path] = []
        for path_a in paths_a:
            relative = path_a.relative_to(input_a)
            path_b = input_b / relative
            if not path_b.is_file():
                missing.append(path_b)
                continue
            height, width = self._read_rgb(path_a).shape[:2]
            mask_relative = relative.with_suffix(".png")
            overlap_path_a = self.split_root / "masks" / "overlap1" / mask_relative
            overlap_path_b = self.split_root / "masks" / "overlap2" / mask_relative
            overlap_known_a, overlap_fraction_a = self._overlap_metadata(overlap_path_a)
            overlap_known_b, overlap_fraction_b = self._overlap_metadata(overlap_path_b)
            label_path = self.split_root / "labels" / relative.with_suffix(".json")
            label_record: dict[str, Any] = {}
            if label_path.is_file():
                try:
                    label_record = json.loads(label_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid real-pair label JSON: {label_path}") from error
            h_ab_path = self.split_root / "homography" / relative.with_suffix(".npy")
            h_ba_path = self.split_root / "homography_ba" / relative.with_suffix(".npy")
            records.append(
                {
                    "image_a": path_a,
                    "image_b": path_b,
                    "pair_id": relative.as_posix(),
                    "target_bucket": f"{width}x{height}",
                    "overlap_known_a": overlap_known_a,
                    "overlap_known_b": overlap_known_b,
                    "overlap_usable_a": overlap_fraction_a > 0.0,
                    "overlap_usable_b": overlap_fraction_b > 0.0,
                    "overlap_fraction_a": overlap_fraction_a,
                    "overlap_fraction_b": overlap_fraction_b,
                    "overlap_a": overlap_path_a,
                    "overlap_b": overlap_path_b,
                    "h_ab": h_ab_path if h_ab_path.is_file() else None,
                    "h_ba": h_ba_path if h_ba_path.is_file() else None,
                    "bucket": str(label_record.get("bucket", "unknown")),
                    "label": int(label_record.get("label", -1)),
                    "label_path": label_path if label_path.is_file() else None,
                }
            )
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} input1 files have no same-name input2 pair; e.g. {preview}. "
                "Use manifest.jsonl when A/B names differ."
            )
        if not records:
            raise RuntimeError(f"No paired images found below {input_a} and {input_b}.")
        return records

    @staticmethod
    def _overlap_metadata(value: object) -> tuple[bool, float]:
        """Return file availability and positive-pixel fraction.

        An all-zero mask is an available annotation but is not usable as a
        positive-overlap constraint. It is therefore routed to the hard
        curriculum without being interpreted as a False label.
        """

        if value is None:
            return False, 0.0
        path = Path(value)
        if not path.is_file():
            return False, 0.0
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read overlap mask: {path}")
        return True, float(np.mean(mask > 0))

    def _resolve_path(self, value: object, fallback_directory: str) -> Path:
        direct = Path(str(value))
        if direct.is_file():
            return direct.resolve()
        local = self.split_root / fallback_directory / direct.name
        if local.is_file():
            return local.resolve()
        relative = self.dataset_root / direct
        if relative.is_file():
            return relative.resolve()
        raise FileNotFoundError(f"Real-pair file not found: {value!r}")

    def _read_optional_mask(
        self, value: object, image_path: Path, directory: str
    ) -> np.ndarray:
        if value is not None:
            path = Path(value)
        else:
            candidate = self.split_root / "masks" / directory / image_path.name
            path = candidate if candidate.is_file() else Path()
        if path.is_file():
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read validity mask: {path}")
            return mask > 0
        image = self._read_rgb(image_path)
        return np.ones(image.shape[:2], dtype=np.bool_)

    @staticmethod
    def _read_optional_homography(value: object) -> tuple[np.ndarray, bool]:
        """Return a normalized finite matrix or an explicit identity fallback."""

        if value is None:
            return np.eye(3, dtype=np.float64), False
        path = Path(value)
        if not path.is_file():
            return np.eye(3, dtype=np.float64), False
        matrix = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"Invalid Homography matrix: {path}")
        if abs(float(np.linalg.det(matrix))) < 1.0e-10:
            raise RuntimeError(f"Singular Homography matrix: {path}")
        if abs(float(matrix[2, 2])) < 1.0e-10:
            raise RuntimeError(f"Homography cannot be normalized: {path}")
        return matrix / matrix[2, 2], True

    def _image_tensor(self, image_rgb: np.ndarray) -> Tensor:
        tensor = torch.from_numpy(
            np.ascontiguousarray(image_rgb.transpose(2, 0, 1), dtype=np.float32)
        ) / 255.0
        if self.config.input_range == "imagenet_normalized":
            mean = tensor.new_tensor(self.config.normalization_mean).view(3, 1, 1)
            std = tensor.new_tensor(self.config.normalization_std).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read RGB image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
