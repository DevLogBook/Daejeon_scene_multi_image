"""Synthetic A-to-B mapping and batch-contract tests."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.seed_geometry import apply_homography
from grm_matcher.synthetic_dataset import (
    GRMControlledGeometryDataset,
    SyntheticGRMDataConfig,
    SyntheticGRMDataset,
)


def _striped_image(height: int = 80, width: int = 96) -> torch.Tensor:
    x = torch.arange(width).float()[None, None].expand(3, height, -1)
    y = torch.arange(height).float()[None, :, None].expand(3, -1, width)
    return torch.stack(
        ((x[0] % 12) / 11.0, (y[0] % 10) / 9.0, ((x[0] + y[0]) % 16) / 15.0)
    )


def test_synthetic_dataset_shapes_roles_and_homography_direction() -> None:
    config = SyntheticGRMDataConfig(
        output_height=64,
        output_width=64,
        samples_per_source=2,
        hard_negative_count=16,
        seed=9,
    )
    dataset = SyntheticGRMDataset([_striped_image()], config)
    sample = dataset[0]
    assert sample["image_a"].shape == sample["image_b"].shape == (3, 64, 64)
    assert sample["gt_dense_mapping"].shape == (2, 64, 64)
    assert sample["local_flow"].shape == (2, 64, 64)
    assert sample["gt_overlap_a"].dtype == torch.bool
    assert not (sample["gt_global_mask"] & sample["gt_local_mask"]).any()
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    points = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2).float()
    global_mapping = apply_homography(points, sample["H_ab"].unsqueeze(0))
    expected = global_mapping.reshape(64, 64, 2) + sample["local_flow"].permute(1, 2, 0)
    torch.testing.assert_close(
        sample["gt_dense_mapping"].permute(1, 2, 0), expected, atol=1.0e-4, rtol=0.0
    )
    assert sample["hard_negative_src_xy"].shape == (16, 2)


def test_synthetic_dataset_batches_without_custom_collate() -> None:
    config = SyntheticGRMDataConfig(
        output_height=32,
        output_width=32,
        samples_per_source=2,
        hard_negative_count=8,
    )
    dataset = SyntheticGRMDataset([_striped_image()], config)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["image_a"].shape == (2, 3, 32, 32)
    assert batch["H_ab"].shape == (2, 3, 3)
    assert batch["gt_dense_mapping"].shape == (2, 2, 32, 32)


def test_synthetic_geometry_is_deterministic_and_controllable() -> None:
    no_flow = SyntheticGRMDataConfig(
        output_height=32,
        output_width=32,
        samples_per_source=1,
        hard_negative_count=8,
        local_flow_probability=0.0,
        occlusion_probability=0.0,
        blur_probability=0.0,
        noise_std=0.0,
        seed=17,
    )
    dataset = SyntheticGRMDataset([_striped_image()], no_flow)
    first = dataset[0]
    second = dataset[0]
    for key in ("image_a", "image_b", "H_ab", "gt_dense_mapping", "local_flow"):
        torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)
    assert not first["local_flow_active"]
    assert torch.count_nonzero(first["local_flow"]) == 0

    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    points = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2).float()
    expected = apply_homography(points, first["H_ab"].unsqueeze(0))
    torch.testing.assert_close(
        first["gt_dense_mapping"].permute(1, 2, 0).reshape(1, -1, 2),
        expected,
        atol=1.0e-4,
        rtol=0.0,
    )

    with_flow = SyntheticGRMDataConfig(
        output_height=32,
        output_width=32,
        samples_per_source=1,
        hard_negative_count=8,
        local_region_min=1,
        local_region_max=1,
        local_flow_probability=1.0,
        local_flow_min_pixels=8.0,
        local_flow_max_pixels=8.0,
        seed=17,
    )
    flow_sample = SyntheticGRMDataset([_striped_image()], with_flow)[0]
    assert flow_sample["local_flow_active"]
    assert float(flow_sample["local_flow"].abs().amax()) > 0.0
    assert flow_sample["gt_local_mask"].any()


def test_controlled_geometry_wrapper_matches_training_contract() -> None:
    data_config = SyntheticGRMDataConfig(
        output_height=32,
        output_width=32,
        samples_per_source=2,
        hard_negative_count=8,
    )
    matcher_config = GRMMatcherConfig(input_height=32, input_width=32)
    dataset = GRMControlledGeometryDataset(
        [_striped_image()], data_config, matcher_config
    )
    sample = dataset[0]
    assert dataset.bucket_to_indices == {"32x32": [0, 1]}
    assert sample["valid_a"].shape == sample["gt_overlap_a"].shape == (1, 32, 32)
    assert sample["valid_b"].dtype == torch.bool
    assert torch.isfinite(sample["image_a"]).all()
