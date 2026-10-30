"""Role-guided bounded correction of a non-learning residual mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dense_match.geometry_utils import (
    normalized_grid_to_pixel_xy,
    pixel_xy_to_normalized_grid,
)
from dense_match.refine import make_grid


@dataclass(frozen=True)
class RoleGuidedMeshDecoderConfig:
    """Architecture and correction budgets for the learned mesh stage."""

    feature_channels: int = 128
    hidden_channels: int = 128
    blocks: int = 3
    max_correction_px: float = 4.0
    max_total_residual_px: float = 24.0
    minimum_confidence_bias: float = -1.5

    def __post_init__(self) -> None:
        if min(self.feature_channels, self.hidden_channels, self.blocks) <= 0:
            raise ValueError("Decoder channel and block counts must be positive.")
        if self.max_correction_px <= 0.0 or self.max_total_residual_px <= 0.0:
            raise ValueError("Decoder residual budgets must be positive.")


@dataclass
class GRMMeshRaster:
    """Dense or mesh-resolution GRM evidence maps."""

    residual_mean_xy: Tensor
    residual_variance: Tensor
    support_count: Tensor
    mean_p_match: Tensor
    mean_w_h: Tensor
    mean_w_deform: Tensor
    mean_cycle_error: Tensor
    mean_uncertainty: Tensor

    def as_channels(self, size: Tuple[int, int]) -> Tensor:
        maps = []
        batch = self.residual_mean_xy.shape[0]
        values = (
            ("residual_mean_xy", self.residual_mean_xy, 2),
            ("residual_variance", self.residual_variance, 1),
            ("support_count", self.support_count, 1),
            ("mean_p_match", self.mean_p_match, 1),
            ("mean_w_h", self.mean_w_h, 1),
            ("mean_w_deform", self.mean_w_deform, 1),
            ("mean_cycle_error", self.mean_cycle_error, 1),
            ("mean_uncertainty", self.mean_uncertainty, 1),
        )
        for name, value, channels in values:
            if value.ndim == 3:
                value = value.unsqueeze(1)
            elif value.ndim == 4 and value.shape[-1] == channels:
                value = value.permute(0, 3, 1, 2)
            if value.ndim != 4 or value.shape[:2] != (batch, channels):
                raise ValueError(f"{name} must resolve to [B,{channels},H,W].")
            value = torch.nan_to_num(value.float(), nan=0.0, posinf=1e3, neginf=-1e3)
            if value.shape[-2:] != size:
                value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            maps.append(value)
        return torch.cat(maps, dim=1)


@dataclass
class RoleGuidedMeshDecoderOutput:
    """Bounded decoder correction and final geometry."""

    mesh_correction: Tensor
    final_mesh_offset: Tensor
    deformation_confidence: Tensor
    deformation_uncertainty: Tensor
    dense_residual: Tensor
    final_mapping_xy: Tensor
    final_grid: Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = next(group for group in (8, 4, 2, 1) if channels % group == 0)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.gelu(value + self.block(value))


class RoleGuidedResidualMeshDecoder(nn.Module):
    """Predict a limited static mesh correction, never a complete warp."""

    grm_raster_channels = 9
    geometry_channels = 8  # H grid, base flow, overlap, valid, initial residual

    def __init__(self, config: Optional[RoleGuidedMeshDecoderConfig] = None) -> None:
        super().__init__()
        self.config = config or RoleGuidedMeshDecoderConfig()
        cfg = self.config
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(9, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GELU(),
        )
        input_channels = (
            32 + 4 * cfg.feature_channels + self.grm_raster_channels + self.geometry_channels
        )
        groups = next(group for group in (8, 4, 2, 1) if cfg.hidden_channels % group == 0)
        self.fuse = nn.Sequential(
            nn.Conv2d(input_channels, cfg.hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, cfg.hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(cfg.hidden_channels) for _ in range(cfg.blocks)]
        )
        self.correction_head = nn.Conv2d(cfg.hidden_channels, 2, 3, padding=1)
        self.confidence_head = nn.Conv2d(cfg.hidden_channels, 1, 3, padding=1)
        self.uncertainty_head = nn.Conv2d(cfg.hidden_channels, 1, 3, padding=1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.constant_(self.confidence_head.bias, cfg.minimum_confidence_bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    @staticmethod
    def _mask(value: Tensor, batch: int, size: Tuple[int, int]) -> Tensor:
        if value.ndim == 3:
            value = value.unsqueeze(1)
        elif value.ndim == 4 and value.shape[-1] == 1:
            value = value.permute(0, 3, 1, 2)
        if value.ndim != 4 or value.shape[:2] != (batch, 1):
            raise ValueError("Mask inputs must resolve to [B,1,H,W].")
        if value.shape[-2:] != size:
            value = F.interpolate(value.float(), size=size, mode="bilinear", align_corners=False)
        return torch.nan_to_num(value.float(), nan=0.0).clamp(0.0, 1.0)

    def forward(
        self,
        image_a: Tensor,
        image_b_warped_by_h: Tensor,
        feature_a: Tensor,
        feature_b_warped_by_h: Tensor,
        h_base_grid: Tensor,
        overlap_mask: Tensor,
        valid_mask: Tensor,
        grm_raster: GRMMeshRaster,
        initial_mesh_offset: Tensor,
        *,
        image_b_hw: Optional[Tuple[int, int]] = None,
    ) -> RoleGuidedMeshDecoderOutput:
        if image_a.ndim != 4 or image_a.shape != image_b_warped_by_h.shape:
            raise ValueError("image_a and image_b_warped_by_h must share [B,3,H,W].")
        batch, channels, height, width = image_a.shape
        if channels != 3 or h_base_grid.shape != (batch, height, width, 2):
            raise ValueError("Expected RGB images and h_base_grid [B,H,W,2].")
        if (
            feature_a.ndim != 4
            or feature_a.shape != feature_b_warped_by_h.shape
            or feature_a.shape[0] != batch
            or feature_a.shape[1] != self.config.feature_channels
        ):
            raise ValueError("Feature inputs must share [B,feature_channels,Hf,Wf].")
        if initial_mesh_offset.ndim != 4 or initial_mesh_offset.shape[0] != batch:
            raise ValueError("initial_mesh_offset must be [B,Gh,Gw,2] or [B,2,Gh,Gw].")
        if initial_mesh_offset.shape[-1] == 2:
            initial_mesh = initial_mesh_offset.permute(0, 3, 1, 2).float()
        elif initial_mesh_offset.shape[1] == 2:
            initial_mesh = initial_mesh_offset.float()
        else:
            raise ValueError("initial_mesh_offset must have a two-channel coordinate axis.")
        mesh_hw = initial_mesh.shape[-2:]
        feature_hw = feature_a.shape[-2:]

        difference = (image_a.float() - image_b_warped_by_h.float()).abs()
        rgb = self.rgb_encoder(torch.cat((image_a, image_b_warped_by_h, difference), dim=1).float())
        if rgb.shape[-2:] != feature_hw:
            rgb = F.interpolate(rgb, size=feature_hw, mode="bilinear", align_corners=False)
        feature_evidence = torch.cat(
            (
                feature_a.float(),
                feature_b_warped_by_h.float(),
                (feature_a.float() - feature_b_warped_by_h.float()).abs(),
                feature_a.float() * feature_b_warped_by_h.float(),
            ),
            dim=1,
        )
        identity = make_grid(batch, height, width, image_a.device, torch.float32)
        base_flow = h_base_grid.float() - identity
        initial_dense = F.interpolate(
            initial_mesh, size=(height, width), mode="bilinear", align_corners=True
        )
        geometry = torch.cat(
            (
                h_base_grid.float().permute(0, 3, 1, 2),
                base_flow.permute(0, 3, 1, 2),
                self._mask(overlap_mask, batch, (height, width)),
                self._mask(valid_mask, batch, (height, width)),
                initial_dense,
            ),
            dim=1,
        )
        geometry = F.interpolate(geometry, size=feature_hw, mode="bilinear", align_corners=False)
        raster = grm_raster.as_channels(feature_hw).to(device=image_a.device)
        hidden = self.blocks(self.fuse(torch.cat((rgb, feature_evidence, geometry, raster), dim=1)))
        mesh_features = F.adaptive_avg_pool2d(hidden, mesh_hw)

        correction = (
            torch.tanh(self.correction_head(mesh_features).float())
            * self.config.max_correction_px
        )
        confidence_mesh = torch.sigmoid(self.confidence_head(mesh_features).float())
        uncertainty_mesh = F.softplus(self.uncertainty_head(mesh_features).float())
        final_mesh = initial_mesh + correction
        magnitude = torch.linalg.vector_norm(final_mesh, dim=1, keepdim=True)
        final_mesh = final_mesh * torch.clamp(
            self.config.max_total_residual_px / magnitude.clamp_min(1e-8), max=1.0
        )
        dense = F.interpolate(
            final_mesh, size=(height, width), mode="bilinear", align_corners=True
        ).permute(0, 2, 3, 1)
        confidence = F.interpolate(
            confidence_mesh, size=(height, width), mode="bilinear", align_corners=True
        )[:, 0]
        uncertainty = F.interpolate(
            uncertainty_mesh, size=(height, width), mode="bilinear", align_corners=True
        )[:, 0]
        confidence = confidence * self._mask(valid_mask, batch, (height, width))[:, 0]
        target_hw = image_b_hw or (height, width)
        base_mapping = normalized_grid_to_pixel_xy(h_base_grid.float(), target_hw)
        final_mapping = base_mapping + dense
        final_grid = pixel_xy_to_normalized_grid(final_mapping, target_hw)
        return RoleGuidedMeshDecoderOutput(
            mesh_correction=correction.permute(0, 2, 3, 1),
            final_mesh_offset=final_mesh.permute(0, 2, 3, 1),
            deformation_confidence=confidence,
            deformation_uncertainty=uncertainty,
            dense_residual=dense,
            final_mapping_xy=final_mapping,
            final_grid=final_grid,
            diagnostics={
                "max_correction_px": self.config.max_correction_px,
                "max_total_residual_px": self.config.max_total_residual_px,
                "mesh_hw": mesh_hw,
            },
        )
