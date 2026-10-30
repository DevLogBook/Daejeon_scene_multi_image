"""Canonical pixel-coordinate geometry helpers for the GRM stitching path.

All public points use original-image pixels in ``(x, y)`` order and all
homographies map A to B.  Sampling grids follow ``align_corners=False``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def apply_homography_xy(points_xy: Tensor, homography_ab: Tensor, eps: float = 1e-8) -> Tensor:
    """Project ``(x,y)`` pixels through an A-to-B homography."""

    if points_xy.ndim not in {2, 3} or points_xy.shape[-1] != 2:
        raise ValueError("points_xy must be [N,2] or [B,N,2].")
    squeeze = points_xy.ndim == 2
    points = points_xy.unsqueeze(0) if squeeze else points_xy
    homography = homography_ab.unsqueeze(0) if homography_ab.ndim == 2 else homography_ab
    if homography.shape != (points.shape[0], 3, 3):
        raise ValueError("homography_ab must be [3,3] or batch-compatible [B,3,3].")
    work_dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    points_f = points.to(dtype=work_dtype)
    h_f = homography.to(device=points.device, dtype=work_dtype)
    ones = torch.ones_like(points_f[..., :1])
    projected = torch.bmm(torch.cat((points_f, ones), dim=-1), h_f.transpose(1, 2))
    denominator = projected[..., 2:3]
    sign = torch.where(denominator >= 0.0, 1.0, -1.0)
    denominator = torch.where(denominator.abs() >= eps, denominator, sign * eps)
    result = projected[..., :2] / denominator
    return result[0] if squeeze else result


def pixel_xy_to_normalized_grid(points_xy: Tensor, image_hw: Tuple[int, int]) -> Tensor:
    """Convert pixel centers to an ``align_corners=False`` normalized grid."""

    height, width = map(int, image_hw)
    if height <= 0 or width <= 0 or points_xy.shape[-1] != 2:
        raise ValueError("image_hw must be positive and points must end in two coordinates.")
    scale = points_xy.new_tensor((2.0 / width, 2.0 / height))
    return (points_xy + 0.5) * scale - 1.0


def normalized_grid_to_pixel_xy(grid: Tensor, image_hw: Tuple[int, int]) -> Tensor:
    """Convert an ``align_corners=False`` normalized grid to pixel centers."""

    height, width = map(int, image_hw)
    if height <= 0 or width <= 0 or grid.shape[-1] != 2:
        raise ValueError("image_hw must be positive and grid must end in two coordinates.")
    scale = grid.new_tensor((width * 0.5, height * 0.5))
    return (grid + 1.0) * scale - 0.5


def image_pixel_grid(
    image_hw: Tuple[int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return an ``[H,W,2]`` identity grid in pixel ``(x,y)`` coordinates."""

    height, width = map(int, image_hw)
    if height <= 0 or width <= 0:
        raise ValueError("image_hw must be positive.")
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1)


def homography_sampling_grid(
    homography_ab: Tensor,
    image_a_hw: Tuple[int, int],
    image_b_hw: Tuple[int, int],
) -> Tensor:
    """Build the B sampling grid for every A pixel under ``H_ab``."""

    squeeze = homography_ab.ndim == 2
    h = homography_ab.unsqueeze(0) if squeeze else homography_ab
    if h.ndim != 3 or h.shape[-2:] != (3, 3):
        raise ValueError("homography_ab must be [3,3] or [B,3,3].")
    pixel_grid = image_pixel_grid(
        image_a_hw, device=h.device, dtype=h.dtype
    ).reshape(1, -1, 2).expand(h.shape[0], -1, -1)
    projected = apply_homography_xy(pixel_grid, h)
    normalized = pixel_xy_to_normalized_grid(projected, image_b_hw)
    result = normalized.reshape(h.shape[0], image_a_hw[0], image_a_hw[1], 2)
    return result[0] if squeeze else result
