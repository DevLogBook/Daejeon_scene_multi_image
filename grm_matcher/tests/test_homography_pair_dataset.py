"""Generated A/B Homography pair loader and bucket sampler tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.homography_pair_dataset import (
    GRMHomographyPairDataset,
    GRMResolutionBucketBatchSampler,
    parse_bucket_batch_sizes,
)


def _write_pair(root: Path, split: str, pair_id: str, *, direction: str = "A_to_B") -> None:
    split_root = root / split
    paths = {
        "image_a": split_root / "input1" / f"{pair_id}.jpg",
        "image_b": split_root / "input2" / f"{pair_id}.jpg",
        "h": split_root / "homography" / f"{pair_id}.npy",
        "valid_a": split_root / "masks" / "valid1" / f"{pair_id}.png",
        "valid_b": split_root / "masks" / "valid2" / f"{pair_id}.png",
        "overlap_a": split_root / "masks" / "overlap1" / f"{pair_id}.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = (20, 140, 220)
    assert cv2.imwrite(str(paths["image_a"]), image)
    assert cv2.imwrite(str(paths["image_b"]), image)
    h_ab = np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.save(paths["h"], h_ab)
    valid = np.full((32, 32), 255, dtype=np.uint8)
    overlap_a = valid.copy()
    overlap_a[:, 28:] = 0
    assert cv2.imwrite(str(paths["valid_a"]), valid)
    assert cv2.imwrite(str(paths["valid_b"]), valid)
    assert cv2.imwrite(str(paths["overlap_a"]), overlap_a)

    # Deliberately record moved Windows paths; the loader must use the known
    # local split layout plus basename fallback on Linux/HPC.
    record = {
        "pair_id": pair_id,
        "image_a": f"Z:\\moved\\input1\\{pair_id}.jpg",
        "image_b": f"Z:\\moved\\input2\\{pair_id}.jpg",
        "h_gt_a_to_b": f"Z:\\moved\\homography\\{pair_id}.npy",
        "valid_a": f"Z:\\moved\\valid1\\{pair_id}.png",
        "valid_b": f"Z:\\moved\\valid2\\{pair_id}.png",
        "overlap_a": f"Z:\\moved\\overlap1\\{pair_id}.png",
        "target_size": [32, 32],
        "target_bucket": "32x32",
        "direction": direction,
    }
    manifest = split_root / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def _config() -> GRMMatcherConfig:
    return GRMMatcherConfig(input_height=32, input_width=32)


def test_pair_dataset_constructs_a_to_b_dense_mapping_and_b_overlap(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "train", "pair_000")
    dataset = GRMHomographyPairDataset(tmp_path, "train", _config())

    sample = dataset[0]

    assert len(dataset) == 1
    assert dataset.bucket_to_indices == {"32x32": [0]}
    assert dataset.source_image_paths() == (
        (tmp_path / "train" / "input1" / "pair_000.jpg").resolve(),
    )
    assert sample["image_a"].shape == (3, 32, 32)
    assert sample["image_a"].min() >= 0.0 and sample["image_a"].max() <= 1.0
    assert torch.allclose(sample["gt_dense_mapping"][:, 0, 0], torch.tensor([4.0, 0.0]))
    assert torch.allclose(sample["gt_dense_mapping"][:, 10, 7], torch.tensor([11.0, 10.0]))
    assert sample["gt_overlap_a"][0, :, :28].all()
    assert not sample["gt_overlap_a"][0, :, 28:].any()
    assert not sample["gt_overlap_b"][0, :, :4].any()
    assert sample["gt_overlap_b"][0, :, 4:].all()
    assert torch.equal(sample["gt_global_mask"], sample["gt_overlap_a"])
    assert not sample["gt_local_mask"].any()
    assert torch.count_nonzero(sample["local_flow"]) == 0


def test_pair_dataset_rejects_wrong_homography_direction(tmp_path: Path) -> None:
    _write_pair(tmp_path, "train", "pair_000", direction="B_to_A")

    with pytest.raises(RuntimeError, match="A to B"):
        GRMHomographyPairDataset(tmp_path, "train", _config())


def test_resolution_bucket_sampler_never_mixes_buckets() -> None:
    sampler = GRMResolutionBucketBatchSampler(
        {"512x512": [0, 1, 2, 3, 4], "768x768": [5, 6, 7]},
        default_batch_size=3,
        bucket_batch_sizes={"768x768": 1},
        shuffle=True,
        drop_last=False,
        seed=7,
    )

    batches = list(sampler)
    assert len(sampler) == 5
    assert sorted(index for batch in batches for index in batch) == list(range(8))
    assert all(all(index < 5 for index in batch) or all(index >= 5 for index in batch) for batch in batches)
    assert all(len(batch) == 1 for batch in batches if batch[0] >= 5)
    sampler.set_epoch(3)
    assert list(sampler) == list(sampler)


def test_parse_bucket_batch_sizes() -> None:
    assert parse_bucket_batch_sizes(["512x512:6", "768x768:2", "1024x1024:1"]) == {
        "512x512": 6,
        "768x768": 2,
        "1024x1024": 1,
    }
    with pytest.raises(ValueError, match="expected"):
        parse_bucket_batch_sizes(["512x512=6"])
