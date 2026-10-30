"""Hard source selection with multiband blending restricted to a seam band."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class NarrowBandBlendConfig:
    band_width_px: int = 12
    pyramid_bands: int = 5

    def __post_init__(self) -> None:
        if self.band_width_px < 0 or self.pyramid_bands < 1:
            raise ValueError("band_width_px must be nonnegative and pyramid_bands positive.")


@dataclass
class NarrowBandBlendResult:
    image: np.ndarray
    band_mask: np.ndarray
    alpha_b: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _laplacian_blend(
    image_a: np.ndarray,
    image_b: np.ndarray,
    alpha_b: np.ndarray,
    levels: int,
) -> np.ndarray:
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    mask = alpha_b.astype(np.float32)
    gaussian_a, gaussian_b, gaussian_m = [a], [b], [mask]
    for _ in range(levels - 1):
        if min(gaussian_a[-1].shape[:2]) < 4:
            break
        gaussian_a.append(cv2.pyrDown(gaussian_a[-1]))
        gaussian_b.append(cv2.pyrDown(gaussian_b[-1]))
        gaussian_m.append(cv2.pyrDown(gaussian_m[-1]))
    laplacian_a, laplacian_b = [], []
    for index in range(len(gaussian_a) - 1):
        size = (gaussian_a[index].shape[1], gaussian_a[index].shape[0])
        laplacian_a.append(gaussian_a[index] - cv2.pyrUp(gaussian_a[index + 1], dstsize=size))
        laplacian_b.append(gaussian_b[index] - cv2.pyrUp(gaussian_b[index + 1], dstsize=size))
    laplacian_a.append(gaussian_a[-1])
    laplacian_b.append(gaussian_b[-1])
    blended_levels = []
    for la, lb, mask_level in zip(laplacian_a, laplacian_b, gaussian_m):
        blended_levels.append(la * (1.0 - mask_level[..., None]) + lb * mask_level[..., None])
    result = blended_levels[-1]
    for level in range(len(blended_levels) - 2, -1, -1):
        size = (blended_levels[level].shape[1], blended_levels[level].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended_levels[level]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def narrow_band_multiband_blend(
    image_a: np.ndarray,
    image_b: np.ndarray,
    valid_mask_a: np.ndarray,
    valid_mask_b: np.ndarray,
    source_selection_mask: np.ndarray,
    *,
    config: NarrowBandBlendConfig | None = None,
) -> NarrowBandBlendResult:
    """Blend only near the binary seam; all other valid pixels are hard-selected."""

    cfg = config or NarrowBandBlendConfig()
    if image_a.shape != image_b.shape or image_a.ndim != 3:
        raise ValueError("image_a and image_b must share [H,W,3].")
    shape = image_a.shape[:2]
    valid_a = np.asarray(valid_mask_a) > 0
    valid_b = np.asarray(valid_mask_b) > 0
    select_b = np.asarray(source_selection_mask, dtype=bool).copy()
    if valid_a.shape != shape or valid_b.shape != shape or select_b.shape != shape:
        raise ValueError("Masks must match image spatial shape.")
    select_b[valid_a & ~valid_b] = False
    select_b[valid_b & ~valid_a] = True
    valid_any = valid_a | valid_b
    overlap = valid_a & valid_b
    boundary = cv2.morphologyEx(
        select_b.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    )
    boundary = (boundary > 0) & overlap
    if cfg.band_width_px > 0 and boundary.any():
        distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
        band = (distance <= float(cfg.band_width_px)) & overlap
    else:
        band = boundary

    hard = np.where(select_b[..., None], image_b, image_a).copy()
    hard[~valid_any] = 0
    sigma = max(cfg.band_width_px / 3.0, 0.5)
    kernel = max(3, 2 * int(round(3.0 * sigma)) + 1)
    alpha = cv2.GaussianBlur(
        select_b.astype(np.float32), (kernel, kernel), sigmaX=sigma, sigmaY=sigma
    )
    alpha[~band] = select_b[~band].astype(np.float32)
    alpha[valid_a & ~valid_b] = 0.0
    alpha[valid_b & ~valid_a] = 1.0
    blended = _laplacian_blend(image_a, image_b, alpha, cfg.pyramid_bands)
    result = hard.copy()
    result[band] = blended[band]
    return NarrowBandBlendResult(
        image=result,
        band_mask=band,
        alpha_b=alpha,
        diagnostics={
            "band_width_px": cfg.band_width_px,
            "pyramid_bands": cfg.pyramid_bands,
            "band_pixel_count": int(band.sum()),
            "band_overlap_fraction": float(band.sum() / max(int(overlap.sum()), 1)),
            "hard_pixel_count": int(valid_any.sum() - band.sum()),
        },
    )
