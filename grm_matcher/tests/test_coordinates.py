"""Coordinate direction, stride, and sampling-convention tests."""

from __future__ import annotations

import pytest
import torch

from grm_matcher.coordinates import (
    GRM_ALIGN_CORNERS,
    feature_xy_to_image_xy,
    flat_indices_to_feature_xy,
    flat_indices_to_image_xy,
    image_xy_to_feature_xy,
    image_xy_to_normalized_grid,
    make_window_offsets,
    normalized_grid_to_image_xy,
    validate_image_size,
)


def test_flattened_indices_are_converted_to_xy_not_row_column() -> None:
    indices = torch.tensor([0, 1, 4, 7], dtype=torch.long)
    feature_xy = flat_indices_to_feature_xy(indices, feature_width=4)
    expected = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 1.0]])
    torch.testing.assert_close(feature_xy, expected)


def test_feature_image_stride_round_trip_uses_pixel_centers() -> None:
    feature_xy = torch.tensor([[0.0, 0.0], [3.25, 6.5]], dtype=torch.float64)
    image_xy = feature_xy_to_image_xy(feature_xy, stride=4)
    torch.testing.assert_close(image_xy[0], torch.tensor([1.5, 1.5], dtype=torch.float64))
    torch.testing.assert_close(image_xy_to_feature_xy(image_xy, stride=4), feature_xy)


def test_flat_indices_to_image_xy_stride_conversion() -> None:
    indices = torch.tensor([0, 5], dtype=torch.long)
    image_xy = flat_indices_to_image_xy(indices, feature_width=4, stride=4)
    expected = torch.tensor([[1.5, 1.5], [5.5, 5.5]])
    torch.testing.assert_close(image_xy, expected)


def test_align_corners_false_normalization_round_trip() -> None:
    assert GRM_ALIGN_CORNERS is False
    image_xy = torch.tensor([[0.0, 0.0], [3.0, 1.0]], dtype=torch.float32)
    grid = image_xy_to_normalized_grid(image_xy, height=2, width=4)
    expected = torch.tensor([[-0.75, -0.5], [0.75, 0.5]])
    torch.testing.assert_close(grid, expected)
    torch.testing.assert_close(normalized_grid_to_image_xy(grid, height=2, width=4), image_xy)


def test_centered_window_offsets_are_xy_ordered() -> None:
    offsets = make_window_offsets(3, device=torch.device("cpu"), dtype=torch.float32)
    assert offsets.shape == (3, 3, 2)
    torch.testing.assert_close(offsets[0, 0], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(offsets[1, 2], torch.tensor([1.0, 0.0]))


def test_runtime_image_size_is_not_hardcoded_to_512() -> None:
    validate_image_size(256, 768, divisor=16)
    with pytest.raises(ValueError, match="divisible"):
        validate_image_size(255, 768, divisor=16)

