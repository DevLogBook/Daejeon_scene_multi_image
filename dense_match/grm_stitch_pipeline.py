"""End-to-end two-image GRM geometry, final-warp seam, and narrow-band blend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from dense_match.geometry_pipeline import (
    GeometryPipelineConfig,
    GeometryPipelineResult,
    run_geometry_pipeline,
)
from dense_match.geometry_utils import apply_homography_xy, pixel_xy_to_normalized_grid
from dense_match.grm_adapter import GRMAdapterResult, GRMMatcherAdapter
from dense_match.instability_graphcut import (
    InstabilitySeamConfig,
    InstabilitySeamResult,
    estimate_instability_seam,
)
from dense_match.narrow_band_blend import (
    NarrowBandBlendConfig,
    NarrowBandBlendResult,
    narrow_band_multiband_blend,
)
from dense_match.role_guided_mesh_decoder import (
    GRMMeshRaster,
    RoleGuidedMeshDecoderOutput,
    RoleGuidedResidualMeshDecoder,
)
from grm_matcher.routing_head import (
    GRMGeometryRoutingHead,
    GRMGeometryRoutingOutput,
    fallback_geometry_routing,
)


@dataclass(frozen=True)
class GRMStitchPipelineConfig:
    geometry: GeometryPipelineConfig = field(default_factory=GeometryPipelineConfig)
    seam_mode: str = "graphcut_multiband"
    seam: InstabilitySeamConfig = field(default_factory=InstabilitySeamConfig)
    blend: NarrowBandBlendConfig = field(default_factory=NarrowBandBlendConfig)
    maximum_canvas_megapixels: float = 80.0

    def __post_init__(self) -> None:
        if self.seam_mode not in {"none", "graphcut", "graphcut_multiband"}:
            raise ValueError("seam_mode must be none, graphcut, or graphcut_multiband.")
        if self.maximum_canvas_megapixels <= 0.0:
            raise ValueError("maximum_canvas_megapixels must be positive.")


@dataclass
class CanvasWarp:
    image_a: np.ndarray
    image_b: np.ndarray
    valid_a: np.ndarray
    valid_b: np.ndarray
    offset_xy: tuple[int, int]
    mapping_b_xy: np.ndarray


@dataclass
class GRMStitchingResult:
    grm_correspondences: Any
    grm_adapter_result: GRMAdapterResult
    routing: GRMGeometryRoutingOutput
    geometry: GeometryPipelineResult
    learned_decoder_output: Optional[RoleGuidedMeshDecoderOutput]
    final_warp_grid: Tensor
    deformation_confidence: Tensor
    deformation_uncertainty: Tensor
    instability_map: np.ndarray
    binary_seam_mask: np.ndarray
    seam: InstabilitySeamResult
    final_stitched_image: np.ndarray
    h_only_stitched_image: np.ndarray
    residual_mesh_stitched_image: np.ndarray
    learned_mesh_stitched_image: np.ndarray
    fallback_reason: str
    diagnostics: dict[str, Any]
    debug_data: dict[str, Any] = field(default_factory=dict)


def _routing_for_matches(
    adapter_result: GRMAdapterResult,
    routing_mode: str,
    routing_head: Optional[GRMGeometryRoutingHead],
) -> tuple[GRMGeometryRoutingOutput, Tensor, Tensor, str]:
    matches = adapter_result.correspondences[0]
    valid = matches.valid_mask[None]
    reliability = matches.reliability[None]
    if routing_mode == "legacy_role":
        output = fallback_geometry_routing(
            valid,
            reliability,
            legacy_p_local=matches.legacy_p_local[None],
            prefer_legacy_local=True,
        )
        output.w_h = matches.legacy_p_global[None]
        return output, output.w_h[0], output.w_deform[0], "legacy_role_ablation"
    if routing_mode == "soft_routing" and routing_head is not None:
        auxiliary = adapter_result.auxiliary
        tokens = getattr(auxiliary, "selected_graph_tokens", None)
        if tokens is not None:
            padded = routing_head(tokens, adapter_result.raw_output.valid_mask)
            indices = matches.source_indices
            output = GRMGeometryRoutingOutput(
                h_support_logits=padded.h_support_logits[:, indices],
                deform_support_logits=padded.deform_support_logits[:, indices],
                w_h=padded.w_h[:, indices],
                w_deform=padded.w_deform[:, indices],
                valid_mask=padded.valid_mask[:, indices],
            )
            return output, output.w_h[0], output.w_deform[0], ""
    output = fallback_geometry_routing(
        valid,
        reliability,
        legacy_p_local=matches.legacy_p_local[None],
    )
    reason = (
        "routing_checkpoint_or_tokens_missing:w_h=1,w_deform=reliability"
        if routing_mode == "soft_routing" else "routing_disabled"
    )
    return output, output.w_h[0], output.w_deform[0], reason


def _grm_raster(
    geometry: GeometryPipelineResult,
    w_h: Tensor,
    w_deform: Tensor,
) -> GRMMeshRaster:
    matches = geometry.correspondences
    mesh = geometry.initial_residual_mesh
    rows, cols = mesh.support_weight.shape
    height, width = mesh.base_mapping_xy.shape[:2]
    residual = matches.dst_xy - apply_homography_xy(matches.src_xy, geometry.H_base.H_ab)
    gx = (matches.src_xy[:, 0] * (cols - 1) / max(width - 1, 1)).round().long().clamp(0, cols - 1)
    gy = (matches.src_xy[:, 1] * (rows - 1) / max(height - 1, 1)).round().long().clamp(0, rows - 1)
    index = gy * cols + gx
    base_weight = matches.reliability.float()
    support = torch.zeros(rows * cols, device=index.device)
    support.scatter_add_(0, index, base_weight)

    def aggregate(value: Tensor) -> Tensor:
        channels = value.shape[-1] if value.ndim == 2 else 1
        value_2d = value if value.ndim == 2 else value[:, None]
        output = torch.zeros(rows * cols, channels, device=value.device)
        output.scatter_add_(0, index[:, None].expand(-1, channels), base_weight[:, None] * value_2d)
        output = output / support[:, None].clamp_min(1e-8)
        return output.reshape(rows, cols, channels)

    residual_mean = aggregate(residual)
    residual_variance = aggregate(
        torch.linalg.vector_norm(residual - residual_mean[gy, gx], dim=-1).square()
    )[..., 0]
    uncertainty = matches.fine_uncertainty
    if uncertainty.ndim == 3:
        uncertainty_value = torch.sqrt(
            uncertainty[..., 0, 0].clamp_min(0.0)
            + uncertainty[..., 1, 1].clamp_min(0.0)
        )
    else:
        uncertainty_value = torch.linalg.vector_norm(uncertainty, dim=-1)
    return GRMMeshRaster(
        residual_mean_xy=residual_mean[None],
        residual_variance=residual_variance[None],
        support_count=support.reshape(1, rows, cols),
        mean_p_match=aggregate(matches.p_match)[..., 0][None],
        mean_w_h=aggregate(w_h)[..., 0][None],
        mean_w_deform=aggregate(w_deform)[..., 0][None],
        mean_cycle_error=aggregate(matches.cycle_error)[..., 0][None],
        mean_uncertainty=aggregate(uncertainty_value)[..., 0][None],
    )


def _canvas_bounds(
    h_ab: Tensor,
    image_a_hw: tuple[int, int],
    image_b_hw: tuple[int, int],
    maximum_megapixels: float,
) -> tuple[int, int, int, int]:
    ha, wa = image_a_hw
    hb, wb = image_b_hw
    b_corners = h_ab.new_tensor(
        [[0.0, 0.0], [wb - 1.0, 0.0], [wb - 1.0, hb - 1.0], [0.0, hb - 1.0]]
    )
    inverse = torch.linalg.inv(h_ab.double()).to(h_ab.dtype)
    b_in_a = apply_homography_xy(b_corners, inverse)
    a_corners = h_ab.new_tensor(
        [[0.0, 0.0], [wa - 1.0, 0.0], [wa - 1.0, ha - 1.0], [0.0, ha - 1.0]]
    )
    all_points = torch.cat((a_corners, b_in_a), dim=0)
    min_x = int(torch.floor(all_points[:, 0].min()).item())
    min_y = int(torch.floor(all_points[:, 1].min()).item())
    max_x = int(torch.ceil(all_points[:, 0].max()).item()) + 1
    max_y = int(torch.ceil(all_points[:, 1].max()).item()) + 1
    canvas_w, canvas_h = max_x - min_x, max_y - min_y
    if canvas_w <= 0 or canvas_h <= 0:
        raise RuntimeError("Invalid panorama canvas bounds.")
    if canvas_w * canvas_h > maximum_megapixels * 1_000_000:
        raise RuntimeError(
            f"Panorama canvas {canvas_w}x{canvas_h} exceeds "
            f"{maximum_megapixels:.1f} MP."
        )
    return min_x, min_y, canvas_w, canvas_h


def _warp_to_canvas(
    image_a: np.ndarray,
    image_b: np.ndarray,
    h_ab: Tensor,
    mapping_a_to_b: Tensor,
    maximum_megapixels: float,
) -> CanvasWarp:
    ha, wa = image_a.shape[:2]
    hb, wb = image_b.shape[:2]
    min_x, min_y, canvas_w, canvas_h = _canvas_bounds(
        h_ab, (ha, wa), (hb, wb), maximum_megapixels
    )
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    a_xy = torch.from_numpy(
        np.stack((xx + min_x, yy + min_y), axis=-1).reshape(-1, 2)
    ).to(device=h_ab.device, dtype=h_ab.dtype)
    mapping = apply_homography_xy(a_xy, h_ab).reshape(canvas_h, canvas_w, 2).cpu().numpy()
    offset_x, offset_y = -min_x, -min_y
    mapping[offset_y:offset_y + ha, offset_x:offset_x + wa] = (
        mapping_a_to_b.detach().float().cpu().numpy()
    )
    warped_b = cv2.remap(
        image_b,
        mapping[..., 0].astype(np.float32),
        mapping[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid_b = cv2.remap(
        np.full((hb, wb), 255, dtype=np.uint8),
        mapping[..., 0].astype(np.float32),
        mapping[..., 1].astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    canvas_a = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    valid_a = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_a[offset_y:offset_y + ha, offset_x:offset_x + wa] = image_a
    valid_a[offset_y:offset_y + ha, offset_x:offset_x + wa] = 255
    return CanvasWarp(
        image_a=canvas_a,
        image_b=warped_b,
        valid_a=valid_a,
        valid_b=valid_b,
        offset_xy=(offset_x, offset_y),
        mapping_b_xy=mapping,
    )


def _hard_stitch(canvas: CanvasWarp) -> np.ndarray:
    choose_b = (canvas.valid_b > 0) & (canvas.valid_a == 0)
    output = np.where(choose_b[..., None], canvas.image_b, canvas.image_a)
    output[(canvas.valid_a == 0) & (canvas.valid_b == 0)] = 0
    return output.astype(np.uint8)


def _place_map(value: Tensor, canvas: CanvasWarp) -> np.ndarray:
    output = np.zeros(canvas.valid_a.shape, dtype=np.float32)
    x, y = canvas.offset_xy
    array = value.detach().float().cpu().numpy()
    output[y:y + array.shape[0], x:x + array.shape[1]] = array
    return output


def stitch_grm_pair(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    grm_input_a: Tensor,
    grm_input_b: Tensor,
    matcher_adapter: GRMMatcherAdapter,
    *,
    config: Optional[GRMStitchPipelineConfig] = None,
    routing_head: Optional[GRMGeometryRoutingHead] = None,
    learned_decoder: Optional[RoleGuidedResidualMeshDecoder] = None,
) -> GRMStitchingResult:
    """Run the complete pair pipeline with seam strictly after final warp."""

    cfg = config or GRMStitchPipelineConfig()
    ha, wa = image_a_bgr.shape[:2]
    hb, wb = image_b_bgr.shape[:2]
    scale_a = (grm_input_a.shape[-1] / wa, grm_input_a.shape[-2] / ha)
    scale_b = (grm_input_b.shape[-1] / wb, grm_input_b.shape[-2] / hb)
    adapter_result = matcher_adapter(
        grm_input_a,
        grm_input_b,
        scale_a_xy=scale_a,
        scale_b_xy=scale_b,
        return_auxiliary=True,
    )
    matches = adapter_result.correspondences[0]
    routing, w_h, w_deform, routing_fallback = _routing_for_matches(
        adapter_result, cfg.geometry.routing_mode, routing_head
    )
    base_config = (
        replace(cfg.geometry, geometry_mode="residual_mesh")
        if cfg.geometry.geometry_mode == "learned_residual_mesh"
        else cfg.geometry
    )
    geometry = run_geometry_pipeline(
        matches,
        (ha, wa),
        (hb, wb),
        config=base_config,
        w_h=w_h,
        w_deform=w_deform,
    )
    learned_output = None
    learned_fallback = ""
    if cfg.geometry.geometry_mode == "learned_residual_mesh":
        auxiliary = adapter_result.auxiliary
        feature_a = getattr(auxiliary, "feature_a_fine_map", None)
        feature_b = getattr(auxiliary, "feature_b_fine_map", None)
        if learned_decoder is None or feature_a is None or feature_b is None:
            learned_fallback = "learned_decoder_or_features_missing:using_initial_mesh"
        else:
            h_grid = pixel_xy_to_normalized_grid(
                geometry.initial_residual_mesh.base_mapping_xy, (hb, wb)
            )[None]
            image_a_tensor = torch.from_numpy(image_a_bgr[..., ::-1].copy()).to(
                device=h_grid.device, dtype=torch.float32
            ).permute(2, 0, 1)[None] / 255.0
            image_b_tensor = torch.from_numpy(image_b_bgr[..., ::-1].copy()).to(
                device=h_grid.device, dtype=torch.float32
            ).permute(2, 0, 1)[None] / 255.0
            image_b_warped = F.grid_sample(
                image_b_tensor, h_grid, align_corners=False
            )
            feature_grid = F.interpolate(
                h_grid.permute(0, 3, 1, 2),
                size=feature_b.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            feature_b_warped = F.grid_sample(
                feature_b, feature_grid, align_corners=False, padding_mode="border"
            )
            valid = ((h_grid.abs() <= 1.0).all(dim=-1)).float()
            learned_output = learned_decoder(
                image_a_tensor,
                image_b_warped,
                feature_a,
                feature_b_warped,
                h_grid,
                valid,
                valid,
                _grm_raster(geometry, w_h, w_deform),
                geometry.initial_residual_mesh.mesh_offset[None],
                image_b_hw=(hb, wb),
            )
            geometry.learned_mesh_correction = learned_output.mesh_correction
            geometry.final_mapping_xy = learned_output.final_mapping_xy[0]
            geometry.final_grid = learned_output.final_grid[0]
            geometry.deformation_confidence = learned_output.deformation_confidence[0]
            geometry.deformation_uncertainty = learned_output.deformation_uncertainty[0]

    h_canvas = _warp_to_canvas(
        image_a_bgr, image_b_bgr, geometry.H_base.H_ab,
        geometry.initial_residual_mesh.base_mapping_xy, cfg.maximum_canvas_megapixels
    )
    residual_canvas = _warp_to_canvas(
        image_a_bgr, image_b_bgr, geometry.H_base.H_ab,
        geometry.initial_residual_mesh.final_mapping_xy, cfg.maximum_canvas_megapixels
    )
    final_canvas = _warp_to_canvas(
        image_a_bgr, image_b_bgr, geometry.H_base.H_ab,
        geometry.final_mapping_xy, cfg.maximum_canvas_megapixels
    )
    uncertainty_canvas = _place_map(geometry.deformation_uncertainty, final_canvas)
    confidence_canvas = _place_map(geometry.deformation_confidence, final_canvas)
    dense = geometry.final_mapping_xy - geometry.initial_residual_mesh.base_mapping_xy
    dx = F.pad(dense[:, 1:] - dense[:, :-1], (0, 0, 0, 1))
    dy = F.pad(dense[1:] - dense[:-1], (0, 0, 0, 0, 0, 1))
    strain = torch.sqrt(dx.square().sum(-1) + dy.square().sum(-1) + 1e-8)
    strain_canvas = _place_map(strain, final_canvas)
    sparse_instability = np.clip(1.0 - confidence_canvas, 0.0, 1.0)

    if cfg.seam_mode == "none":
        selection = (final_canvas.valid_b > 0) & (final_canvas.valid_a == 0)
        seam = estimate_instability_seam(
            final_canvas.image_a,
            final_canvas.image_b,
            final_canvas.valid_a,
            final_canvas.valid_b,
            deformation_uncertainty=uncertainty_canvas,
            mesh_strain=strain_canvas,
            sparse_instability=sparse_instability,
            config=replace(cfg.seam, backend="dynamic_programming"),
        )
        seam.source_selection_mask = selection
        seam.backend = "none"
        seam.seam_boundary[:] = False
    else:
        seam = estimate_instability_seam(
            final_canvas.image_a,
            final_canvas.image_b,
            final_canvas.valid_a,
            final_canvas.valid_b,
            deformation_uncertainty=uncertainty_canvas,
            mesh_strain=strain_canvas,
            sparse_instability=sparse_instability,
            config=cfg.seam,
        )
    blend_result: Optional[NarrowBandBlendResult] = None
    if cfg.seam_mode == "graphcut_multiband":
        blend_result = narrow_band_multiband_blend(
            final_canvas.image_a,
            final_canvas.image_b,
            final_canvas.valid_a,
            final_canvas.valid_b,
            seam.source_selection_mask,
            config=cfg.blend,
        )
        final_image = blend_result.image
    else:
        final_image = np.where(
            seam.source_selection_mask[..., None],
            final_canvas.image_b,
            final_canvas.image_a,
        ).astype(np.uint8)
    fallback_reason = ";".join(
        reason for reason in (
            routing_fallback,
            geometry.fallback_reason,
            learned_fallback,
            seam.diagnostics.get("fallback_reason", ""),
        ) if reason
    )
    diagnostics = {
        "geometry": geometry.json_report(),
        "routing_fallback": routing_fallback,
        "learned_fallback": learned_fallback,
        "seam_backend": seam.backend,
        "seam": seam.diagnostics,
        "blend": blend_result.diagnostics if blend_result else {"mode": "hard"},
        "fallback_reason": fallback_reason,
    }
    return GRMStitchingResult(
        grm_correspondences=matches,
        grm_adapter_result=adapter_result,
        routing=routing,
        geometry=geometry,
        learned_decoder_output=learned_output,
        final_warp_grid=geometry.final_grid,
        deformation_confidence=geometry.deformation_confidence,
        deformation_uncertainty=geometry.deformation_uncertainty,
        instability_map=seam.unstable_map,
        binary_seam_mask=seam.source_selection_mask,
        seam=seam,
        final_stitched_image=final_image,
        h_only_stitched_image=_hard_stitch(h_canvas),
        residual_mesh_stitched_image=_hard_stitch(residual_canvas),
        learned_mesh_stitched_image=_hard_stitch(final_canvas),
        fallback_reason=fallback_reason,
        diagnostics=diagnostics,
        debug_data={
            "h_canvas": h_canvas,
            "residual_canvas": residual_canvas,
            "final_canvas": final_canvas,
            "confidence_canvas": confidence_canvas,
            "uncertainty_canvas": uncertainty_canvas,
            "strain_canvas": strain_canvas,
            "blend_result": blend_result,
        },
    )
