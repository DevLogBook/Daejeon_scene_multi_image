"""Vectorized local correlation and probability-derived localization moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .coordinates import (
    GRM_ALIGN_CORNERS,
    feature_xy_to_image_xy,
    image_xy_to_feature_xy,
    image_xy_to_normalized_grid,
    make_window_offsets,
)


@dataclass
class GRMLocalCorrelationOutput:
    logits: Tensor
    probability: Tensor
    delta_xy: Tensor
    entropy: Tensor
    covariance: Tensor
    peak_probability: Tensor
    source_descriptor: Tensor
    target_descriptor: Tensor
    window_valid_mask: Tensor


@dataclass
class GRMRefinementOutput:
    src_xy: Tensor
    dst_xy: Tensor
    delta_feature_xy: Tensor
    local_probability: Tensor
    entropy: Tensor
    covariance_feature: Tensor
    covariance_image: Tensor
    peak_probability: Tensor
    source_descriptor: Tensor
    target_descriptor: Tensor
    valid_mask: Tensor


def compute_spatial_expectation(
    probability: Tensor,
    offsets_xy: Optional[Tensor] = None,
    eps: float = 1.0e-8,
) -> Tensor:
    """Return ``[B, N, 2]`` ``(dx, dy)`` expectation from a local distribution."""

    if probability.ndim != 4 or not probability.is_floating_point():
        raise ValueError("probability must be a floating [B, N, H, W] tensor.")
    height, width = probability.shape[-2:]
    if offsets_xy is None:
        if height != width or height % 2 == 0:
            raise ValueError("Implicit offsets require an odd square probability window.")
        offsets_xy = make_window_offsets(
            height, device=probability.device, dtype=probability.dtype
        )
    if offsets_xy.shape != (height, width, 2):
        raise ValueError(
            f"offsets_xy must be {(height, width, 2)}, got {tuple(offsets_xy.shape)}."
        )
    mass = probability.sum(dim=(-2, -1), keepdim=True)
    normalized = torch.where(
        mass > eps, probability / mass.clamp_min(eps), torch.zeros_like(probability)
    )
    return torch.einsum("bnhw,hwd->bnd", normalized, offsets_xy)


def compute_spatial_covariance(
    probability: Tensor,
    expectation_xy: Optional[Tensor] = None,
    offsets_xy: Optional[Tensor] = None,
    eps: float = 1.0e-8,
) -> Tensor:
    """Return probability-derived ``[B, N, 2, 2]`` covariance."""

    if probability.ndim != 4 or not probability.is_floating_point():
        raise ValueError("probability must be a floating [B, N, H, W] tensor.")
    height, width = probability.shape[-2:]
    if offsets_xy is None:
        if height != width or height % 2 == 0:
            raise ValueError("Implicit offsets require an odd square probability window.")
        offsets_xy = make_window_offsets(
            height, device=probability.device, dtype=probability.dtype
        )
    if expectation_xy is None:
        expectation_xy = compute_spatial_expectation(probability, offsets_xy, eps)
    if expectation_xy.shape != (*probability.shape[:2], 2):
        raise ValueError("expectation_xy must be [B, N, 2].")
    mass = probability.sum(dim=(-2, -1), keepdim=True)
    normalized = torch.where(
        mass > eps, probability / mass.clamp_min(eps), torch.zeros_like(probability)
    )
    difference = offsets_xy[None, None] - expectation_xy[:, :, None, None]
    return torch.einsum("bnhw,bnhwi,bnhwj->bnij", normalized, difference, difference)


def sample_feature_windows(
    feature: Tensor,
    centers_xy: Tensor,
    window_size: int,
) -> tuple[Tensor, Tensor]:
    """Vectorized sampling of N windows with one ``grid_sample`` call.

    Returns features ``[B, N, C, S, S]`` and an analytic in-bounds mask
    ``[B, N, S, S]``. No loop is performed over candidates.
    """

    if feature.ndim != 4:
        raise ValueError("feature must be [B, C, H, W].")
    if centers_xy.ndim != 3 or centers_xy.shape[-1] != 2:
        raise ValueError("centers_xy must be [B, N, 2].")
    if feature.shape[0] != centers_xy.shape[0]:
        raise ValueError("feature and centers_xy batches must match.")
    if not centers_xy.is_floating_point():
        raise TypeError("centers_xy must use a floating-point dtype.")

    batch, channels, height, width = feature.shape
    count = centers_xy.shape[1]
    offsets = make_window_offsets(
        window_size, device=feature.device, dtype=centers_xy.dtype
    )
    sample_xy = centers_xy[:, :, None, None, :] + offsets[None, None]
    valid = (
        (sample_xy[..., 0] >= 0.0)
        & (sample_xy[..., 0] <= float(width - 1))
        & (sample_xy[..., 1] >= 0.0)
        & (sample_xy[..., 1] <= float(height - 1))
    )
    grid = image_xy_to_normalized_grid(sample_xy, height=height, width=width)
    grid = grid.reshape(batch, count * window_size, window_size, 2)
    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=GRM_ALIGN_CORNERS,
    )
    sampled = sampled.reshape(batch, channels, count, window_size, window_size)
    return sampled.permute(0, 2, 1, 3, 4).contiguous(), valid


class GRMLocalCorrelation(nn.Module):
    """Trainable shared descriptor projection plus vectorized local matching."""

    def __init__(self, channels: int, temperature: float, eps: float = 1.0e-8) -> None:
        super().__init__()
        if channels <= 0 or temperature <= 0.0 or eps <= 0.0:
            raise ValueError("channels, temperature, and eps must be positive.")
        self.channels = channels
        self.temperature = temperature
        self.eps = eps
        self.peak_localization = False
        self.descriptor_projection = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.dirac_(self.descriptor_projection.weight)

    def forward(
        self,
        feature_source: Tensor,
        feature_target: Tensor,
        source_centers_xy: Tensor,
        target_centers_xy: Tensor,
        source_patch_size: int,
        target_window_size: int,
        valid_mask: Optional[Tensor] = None,
        source_valid_map: Optional[Tensor] = None,
        target_valid_map: Optional[Tensor] = None,
    ) -> GRMLocalCorrelationOutput:
        self._validate_inputs(
            feature_source,
            feature_target,
            source_centers_xy,
            target_centers_xy,
            valid_mask,
        )
        batch, count, _ = source_centers_xy.shape
        if valid_mask is None:
            valid_mask = torch.ones(
                batch, count, device=feature_source.device, dtype=torch.bool
            )

        projected_source = self.descriptor_projection(feature_source)
        projected_target = self.descriptor_projection(feature_target)
        source_patch, source_patch_valid = sample_feature_windows(
            projected_source, source_centers_xy, source_patch_size
        )
        target_window, target_window_valid = sample_feature_windows(
            projected_target, target_centers_xy, target_window_size
        )
        if source_valid_map is not None:
            sampled_mask, _ = sample_feature_windows(source_valid_map[:, None].float(), source_centers_xy, source_patch_size)
            source_patch_valid = source_patch_valid & (sampled_mask[:, :, 0] > .999)
        if target_valid_map is not None:
            sampled_mask, _ = sample_feature_windows(target_valid_map[:, None].float(), target_centers_xy, target_window_size)
            target_window_valid = target_window_valid & (sampled_mask[:, :, 0] > .999)
        source_has_support = source_patch_valid.flatten(-2).any(-1)
        if self.peak_localization:
            source_has_support = source_patch_valid[..., source_patch_size // 2, source_patch_size // 2]
        valid_mask = valid_mask & source_has_support

        source_weights = source_patch_valid[:, :, None].to(source_patch.dtype)
        source_denominator = source_weights.sum(dim=(-2, -1)).clamp_min(1.0)
        source_descriptor = (source_patch * source_weights).sum(dim=(-2, -1))
        source_descriptor = source_descriptor / source_denominator
        if self.peak_localization:
            # Retain the query's identity instead of averaging neighboring
            # repeated structures into the same descriptor.
            center = source_patch_size // 2
            source_descriptor = source_patch[..., center, center]
        source_descriptor = F.normalize(source_descriptor.float(), dim=-1, eps=self.eps)
        target_descriptor_map = F.normalize(target_window.float(), dim=2, eps=self.eps)
        logits = torch.einsum(
            "bnc,bnchw->bnhw", source_descriptor, target_descriptor_map
        ) / self.temperature

        window_valid = target_window_valid & valid_mask[:, :, None, None]
        masked_logits = logits.masked_fill(
            ~window_valid, torch.finfo(logits.dtype).min
        )
        probability = torch.softmax(masked_logits.flatten(-2), dim=-1).reshape_as(logits)
        probability = probability * window_valid.float()
        probability = probability / probability.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(self.eps)

        offsets = make_window_offsets(
            target_window_size, device=probability.device, dtype=probability.dtype
        )
        localization_probability = probability
        if self.peak_localization:
            peak_index = probability.flatten(-2).argmax(-1)
            px, py = peak_index % target_window_size, peak_index // target_window_size
            grid_y, grid_x = torch.meshgrid(
                torch.arange(target_window_size, device=probability.device),
                torch.arange(target_window_size, device=probability.device), indexing="ij")
            neighborhood = ((grid_x-px[..., None, None]).abs() <= 1) & (
                (grid_y-py[..., None, None]).abs() <= 1)
            localization_probability = probability * neighborhood
            localization_probability = localization_probability / localization_probability.sum(
                (-2, -1), keepdim=True).clamp_min(self.eps)
        delta = compute_spatial_expectation(localization_probability, offsets, self.eps)
        covariance = compute_spatial_covariance(
            localization_probability, delta, offsets, self.eps
        )
        entropy = -(
            probability * torch.log(probability.clamp_min(self.eps))
        ).sum(dim=(-2, -1))
        peak = probability.flatten(-2).amax(dim=-1)
        target_descriptor = torch.einsum(
            "bnhw,bnchw->bnc", localization_probability, target_descriptor_map
        )

        mask_float = valid_mask[..., None].float()
        return GRMLocalCorrelationOutput(
            logits=logits,
            probability=probability,
            delta_xy=delta * mask_float,
            entropy=entropy * valid_mask.float(),
            covariance=covariance * valid_mask[..., None, None].float(),
            peak_probability=peak * valid_mask.float(),
            source_descriptor=source_descriptor * mask_float,
            target_descriptor=target_descriptor * mask_float,
            window_valid_mask=window_valid,
        )

    def _validate_inputs(
        self,
        feature_source: Tensor,
        feature_target: Tensor,
        source_centers_xy: Tensor,
        target_centers_xy: Tensor,
        valid_mask: Optional[Tensor],
    ) -> None:
        for name, feature in (
            ("feature_source", feature_source),
            ("feature_target", feature_target),
        ):
            if feature.ndim != 4 or feature.shape[1] != self.channels:
                raise ValueError(
                    f"{name} must be [B, {self.channels}, H, W], got {tuple(feature.shape)}."
                )
        if feature_source.shape[0] != feature_target.shape[0]:
            raise ValueError("Feature batches must match.")
        expected = (feature_source.shape[0], source_centers_xy.shape[1], 2)
        if source_centers_xy.shape != expected or target_centers_xy.shape != expected:
            raise ValueError(f"Center tensors must both be {expected}.")
        if valid_mask is not None:
            if valid_mask.shape != expected[:2] or valid_mask.dtype != torch.bool:
                raise ValueError(f"valid_mask must be bool {expected[:2]}.")


class GRMLocalRefinerCore(nn.Module):
    """Scale-aware wrapper keeping all public coordinates in image pixels."""

    def __init__(
        self,
        channels: int,
        stride: int,
        source_patch_size: int,
        target_window_size: int,
        temperature: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.source_patch_size = source_patch_size
        self.target_window_size = target_window_size
        self.correlation = GRMLocalCorrelation(channels, temperature, eps)

    def forward(
        self,
        feature_source: Tensor,
        feature_target: Tensor,
        src_xy: Tensor,
        initial_dst_xy: Tensor,
        valid_mask: Optional[Tensor] = None,
        source_valid_map: Optional[Tensor] = None,
        target_valid_map: Optional[Tensor] = None,
    ) -> GRMRefinementOutput:
        source_feature_xy = image_xy_to_feature_xy(src_xy, self.stride)
        target_feature_xy = image_xy_to_feature_xy(initial_dst_xy, self.stride)
        local = self.correlation(
            feature_source,
            feature_target,
            source_feature_xy,
            target_feature_xy,
            self.source_patch_size,
            self.target_window_size,
            valid_mask,
            source_valid_map,
            target_valid_map,
        )
        refined_target_feature_xy = target_feature_xy + local.delta_xy
        refined_dst_xy = feature_xy_to_image_xy(
            refined_target_feature_xy, self.stride
        )
        covariance_image = local.covariance * float(self.stride * self.stride)
        if valid_mask is None:
            valid_mask = torch.ones(
                src_xy.shape[:2], device=src_xy.device, dtype=torch.bool
            )
        valid_mask = valid_mask & local.window_valid_mask.flatten(-2).any(-1)
        refined_dst_xy = torch.where(
            valid_mask[..., None], refined_dst_xy, initial_dst_xy
        )
        return GRMRefinementOutput(
            src_xy=src_xy,
            dst_xy=refined_dst_xy,
            delta_feature_xy=local.delta_xy,
            local_probability=local.probability,
            entropy=local.entropy,
            covariance_feature=local.covariance,
            covariance_image=covariance_image,
            peak_probability=local.peak_probability,
            source_descriptor=local.source_descriptor,
            target_descriptor=local.target_descriptor,
            valid_mask=valid_mask,
        )
