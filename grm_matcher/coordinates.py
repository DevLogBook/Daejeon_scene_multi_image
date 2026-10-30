"""Centralized coordinate conversions for GRM-Matcher.

All public and internal point tensors use ``(..., 2)`` ordered as ``(x, y)``.
Feature coordinates refer to feature-pixel centers.  For a stride ``s``:

``image_xy = (feature_xy + 0.5) * s - 0.5``.

This convention is exactly compatible with ``grid_sample(...,
align_corners=False)`` and is used by every later refinement module.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


GRM_ALIGN_CORNERS = False


def _validate_xy(xy: Tensor) -> None:
    if xy.ndim < 1 or xy.shape[-1] != 2:
        raise ValueError(f"Expected an (..., 2) (x, y) tensor, got {tuple(xy.shape)}.")
    if not (xy.is_floating_point() or xy.is_complex()):
        raise TypeError("Coordinate tensors must use a floating-point dtype.")


def validate_image_size(height: int, width: int, divisor: int = 16) -> None:
    """Validate a runtime image size without assuming 512 x 512."""

    if height <= 0 or width <= 0:
        raise ValueError(f"Image size must be positive, got {(height, width)}.")
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}.")
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Image size {(height, width)} must be divisible by {divisor}."
        )


def feature_xy_to_image_xy(feature_xy: Tensor, stride: int) -> Tensor:
    """Map feature-pixel coordinates to original-image pixel coordinates."""

    _validate_xy(feature_xy)
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    return (feature_xy + 0.5) * float(stride) - 0.5


def image_xy_to_feature_xy(image_xy: Tensor, stride: int) -> Tensor:
    """Map original-image pixel coordinates to feature-pixel coordinates."""

    _validate_xy(image_xy)
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    return (image_xy + 0.5) / float(stride) - 0.5


def flat_indices_to_feature_xy(
    indices: Tensor,
    feature_width: int,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Convert row-major flattened indices to feature ``(x, y)`` centers."""

    if feature_width <= 0:
        raise ValueError(f"feature_width must be positive, got {feature_width}.")
    if indices.is_floating_point() or indices.is_complex():
        raise TypeError("Flattened feature indices must use an integer dtype.")
    if torch.any(indices < 0):
        raise ValueError("Flattened feature indices must be nonnegative.")
    output_dtype = dtype if dtype is not None else torch.float32
    x = torch.remainder(indices, feature_width).to(dtype=output_dtype)
    y = torch.div(indices, feature_width, rounding_mode="floor").to(dtype=output_dtype)
    return torch.stack((x, y), dim=-1)


def flat_indices_to_image_xy(
    indices: Tensor,
    feature_width: int,
    stride: int,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Convert flattened feature indices directly to image ``(x, y)``."""

    feature_xy = flat_indices_to_feature_xy(indices, feature_width, dtype=dtype)
    return feature_xy_to_image_xy(feature_xy, stride)


def image_xy_to_normalized_grid(image_xy: Tensor, height: int, width: int) -> Tensor:
    """Convert image pixels to a grid for ``align_corners=False`` sampling."""

    _validate_xy(image_xy)
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial size must be positive, got {(height, width)}.")
    scale = image_xy.new_tensor((float(width), float(height)))
    return 2.0 * (image_xy + 0.5) / scale - 1.0


def normalized_grid_to_image_xy(grid_xy: Tensor, height: int, width: int) -> Tensor:
    """Invert :func:`image_xy_to_normalized_grid` for ``align_corners=False``."""

    _validate_xy(grid_xy)
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial size must be positive, got {(height, width)}.")
    scale = grid_xy.new_tensor((float(width), float(height)))
    return ((grid_xy + 1.0) * scale - 1.0) * 0.5


def make_window_offsets(
    window_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a centered ``[window_size, window_size, 2]`` ``(dx, dy)`` grid."""

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}.")
    radius = window_size // 2
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx, yy), dim=-1)

