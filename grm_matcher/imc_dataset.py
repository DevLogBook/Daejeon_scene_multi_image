"""IMC 2022 real pairs: scene manifests, letterboxing and epipolar checks.

F always follows x_b.T @ F_ab @ x_a = 0. No homography/depth labels
are fabricated from the camera calibration or covisibility score.
"""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import GRMMatcherConfig
from .stage25 import GRMStage25PseudoLabels


def normalize_fundamental(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    norm = np.linalg.norm(matrix)
    if not np.isfinite(matrix).all() or norm <= 1e-20:
        raise ValueError("Fundamental matrix must be finite and nonzero")
    matrix = matrix / norm
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[1] <= 1e-12 or singular[2] > singular[1] * 0.01:
        raise ValueError("Fundamental matrix must have numerical rank two")
    return matrix


def letterbox(image: np.ndarray, height: int, width: int):
    """Return RGB canvas, valid mask and original->canvas pixel-center map."""
    h, w = image.shape[:2]
    ratio = min(width / w, height / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (width - nw) // 2, (height - nh) // 2
    sx, sy = nw / w, nh / h
    transform = np.array([[sx, 0, left + (sx - 1) / 2],
                          [0, sy, top + (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((1, height, width), dtype=bool)
    mask[:, top:top + nh, left:left + nw] = True
    return canvas, mask, transform


def transform_fundamental(matrix, transform_a, transform_b):
    return normalize_fundamental(
        np.linalg.inv(transform_b).T @ matrix @ np.linalg.inv(transform_a)
    )


def sampson_squared(src_xy, dst_xy, fundamental):
    """Squared Sampson distance in the coordinate system of the input points."""
    a = torch.cat((src_xy.float(), torch.ones_like(src_xy[..., :1])), -1)
    b = torch.cat((dst_xy.float(), torch.ones_like(dst_xy[..., :1])), -1)
    f = fundamental.float()
    f = f / torch.linalg.vector_norm(f, dim=(-2, -1), keepdim=True).clamp_min(1e-20)
    line_b = torch.einsum("bij,bnj->bni", f, a)
    line_a = torch.einsum("bji,bnj->bni", f, b)
    residual = (b * line_b).sum(-1)
    denominator = line_b[..., :2].square().sum(-1) + line_a[..., :2].square().sum(-1)
    distance = residual.square() / denominator.clamp_min(1e-20)
    return torch.where(denominator > 1e-20, distance, torch.full_like(distance, float("inf")))


def filter_epipolar_labels(labels: GRMStage25PseudoLabels, fundamental, threshold_px):
    """Intersect existing Teacher reliability with GT-F consistency; never revive labels."""
    distance = sampson_squared(labels.src_xy, labels.fine_dst_xy, fundamental)
    valid = labels.valid_mask & torch.isfinite(distance) & (distance <= threshold_px ** 2)
    return replace(labels, valid_mask=valid, confidence=labels.confidence * valid.float())


class IMCPairDataset(Dataset):
    def __init__(self, data_dir, split, config: GRMMatcherConfig, *, limit=0, label_source="teacher", epipolar_threshold_px=2., teacher_cache=None):
        self.root = Path(data_dir).resolve()
        self.config = config
        self.label_source = label_source
        self.teacher_cache = Path(teacher_cache) if teacher_cache else None
        if label_source == "roma" and self.teacher_cache is None:
            raise ValueError("RoMa labels require teacher_cache")
        self.epipolar_threshold_px = epipolar_threshold_px
        metadata = json.loads((self.root / "splits.json").read_text(encoding="utf-8"))
        self.image_root = Path(metadata["image_root"])
        self.height, self.width = metadata["height"], metadata["width"]
        if self.height % config.coarse_stride or self.width % config.coarse_stride:
            raise ValueError("Canvas dimensions must be multiples of coarse_stride")
        self.records = [json.loads(line) for line in (self.root / split / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"Empty IMC split: {split}")
        self.bucket_to_indices = {f"{self.width}x{self.height}": list(range(len(self.records)))}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        result = {"pair_id": row["pair_id"]}
        transforms = []
        canvases, masks = [], []
        for side in ("a", "b"):
            path = self.image_root / row[f"image_{side}"]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Cannot decode {path}")
            image, valid, transform = letterbox(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.height, self.width)
            value = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255
            if self.config.input_range == "imagenet_normalized":
                mean = value.new_tensor(self.config.normalization_mean)[:, None, None]
                std = value.new_tensor(self.config.normalization_std)[:, None, None]
                value = (value - mean) / std
            result[f"image_{side}"] = value
            result[f"valid_{side}"] = torch.from_numpy(valid)
            transforms.append(transform)
            canvases.append(image)
            masks.append(valid[0])
        matrix = transform_fundamental(normalize_fundamental(row["F_ab"]), *transforms)
        result["F_ab"] = torch.from_numpy(matrix.astype(np.float32))
        if self.label_source in ("sift", "roma"):
            slots = self.config.max_source_queries
            if self.label_source == "roma":
                from .teacher_cache import load_pair, filter_matches, unique_coarse
                points_a, points_b, confidence = load_pair(self.teacher_cache, row, self.height, self.width)
                points_a, points_b, confidence = filter_matches(points_a, points_b, confidence,
                    masks, matrix, self.epipolar_threshold_px, minimum_confidence=0.)
                points_a, points_b, confidence = unique_coarse(points_a, points_b, confidence, self.config.coarse_stride)
                slots = (self.height // self.config.coarse_stride) * (self.width // self.config.coarse_stride)
            else:
                points_a, points_b, confidence = self._sift_labels(row, canvases, masks, matrix)
            count = min(len(points_a), slots)
            for key, values in (("sparse_a", points_a), ("sparse_b", points_b)):
                padded = np.zeros((slots, 2), dtype=np.float32)
                padded[:count] = values[:count]
                result[key] = torch.from_numpy(padded)
            weights = torch.zeros(slots)
            weights[:count] = torch.from_numpy(confidence[:count])
            result["sparse_confidence"] = weights
            result["sparse_valid"] = torch.arange(slots) < count
        return result

    def _sift_labels(self, row, images, masks, matrix):
        signature = {"version": 1, "row": row, "size": [self.height, self.width],
                     "stride": self.config.coarse_stride, "threshold": self.epipolar_threshold_px,
                     "files": [(str(self.image_root / row[f"image_{s}"]),
                                (self.image_root / row[f"image_{s}"]).stat().st_size,
                                (self.image_root / row[f"image_{s}"]).stat().st_mtime_ns) for s in ("a", "b")]}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        cache = self.root / "sift_cache" / f"{key}.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                return data["a"], data["b"], data["confidence"]
        sift = cv2.SIFT_create(nfeatures=4096)
        features = [sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask.astype(np.uint8) * 255)
                    for image, mask in zip(images, masks)]
        ka, da = features[0]
        kb, db = features[1]
        accepted = []
        if da is not None and db is not None and min(len(da), len(db)) >= 2:
            bf = cv2.BFMatcher(cv2.NORM_L2)
            forward = bf.knnMatch(da, db, k=2)
            backward = bf.knnMatch(db, da, k=2)
            reverse = {m.queryIdx: (m.trainIdx, m.distance / max(n.distance, 1e-12))
                       for m, n in backward if m.distance < .75 * n.distance}
            for m, n in forward:
                other = reverse.get(m.trainIdx)
                if m.distance < .75 * n.distance and other is not None and other[0] == m.queryIdx:
                    accepted.append((ka[m.queryIdx].pt, kb[m.trainIdx].pt,
                                     1 - max(m.distance / max(n.distance, 1e-12), other[1])))
        a = np.array([x[0] for x in accepted], dtype=np.float32).reshape(-1, 2)
        b = np.array([x[1] for x in accepted], dtype=np.float32).reshape(-1, 2)
        weights = np.array([x[2] for x in accepted], dtype=np.float32)
        if len(a):
            d = sampson_squared(torch.from_numpy(a)[None], torch.from_numpy(b)[None], torch.from_numpy(matrix)[None])[0].numpy()
            keep = np.isfinite(d) & (d <= self.epipolar_threshold_px ** 2)
            a, b, weights = a[keep], b[keep], weights[keep]
        # Keep a one-to-one coarse-cell mapping; repeated SIFT orientations must
        # not create contradictory coarse CE labels in the same source cell.
        stride = self.config.coarse_stride
        ca = np.rint((a + .5) / stride - .5).astype(int)
        cb = np.rint((b + .5) / stride - .5).astype(int)
        used_a, used_b, chosen = set(), set(), []
        for index in np.argsort(-weights, kind="stable"):
            x, y = tuple(ca[index]), tuple(cb[index])
            if x not in used_a and y not in used_b:
                used_a.add(x); used_b.add(y); chosen.append(index)
        a, b, weights = a[chosen], b[chosen], weights[chosen]
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, a=a, b=b, confidence=weights)
        os.replace(temporary, cache)
        return a, b, weights


def sparse_bidirectional_labels(batch, device, matcher_config, stage25_config):
    """Wrap cached real SIFT correspondences in the existing refinement contract."""
    a, b = batch["sparse_a"].to(device), batch["sparse_b"].to(device)
    valid = batch["sparse_valid"].to(device)
    confidence = batch["sparse_confidence"].to(device)
    height, width = batch["image_a"].shape[-2:]
    stride = matcher_config.coarse_stride
    def direction(source, target):
        cells = torch.round((target + .5) / stride - .5)
        coarse = (cells + .5) * stride - .5
        source_cells = torch.round((source + .5) / stride - .5).long()
        region_size = stage25_config.coarse_region_size_cells
        region_width = (width // stride + region_size - 1) // region_size
        regions = (source_cells[..., 1] // region_size) * region_width + source_cells[..., 0] // region_size
        zeros = confidence.new_zeros(confidence.shape)
        return GRMStage25PseudoLabels(src_xy=source, coarse_dst_xy=coarse,
            mid_dst_xy=target, fine_dst_xy=target, confidence=confidence,
            cycle_error=zeros, fine_std=zeros, valid_mask=valid.clone(),
            mutual_top1=valid.clone(), base_valid_mask=valid.clone(), region_ids=regions)
    return SimpleNamespace(forward=direction(a, b), backward=direction(b, a))
