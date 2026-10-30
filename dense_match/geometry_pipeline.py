"""Composable GRM global-H and residual-mesh geometry pipeline.

This module contains no seam or blending logic.  It provides the common
geometry modes and ablations used before final-warp seam selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Tuple

import torch
from torch import Tensor

from dense_match.geometry_utils import pixel_xy_to_normalized_grid
from dense_match.global_homography import (
    GlobalHomographyConfig,
    GlobalHomographyResult,
    estimate_global_homography,
)
from dense_match.grm_adapter import GRMCorrespondenceSet
from dense_match.residual_mesh import (
    ResidualMeshConfig,
    ResidualMeshResult,
    estimate_residual_mesh,
)


GeometryMode = Literal["h_only", "residual_mesh", "learned_residual_mesh"]
RoutingMode = Literal["none", "legacy_role", "soft_routing"]
HMatchMode = Literal["all_reliable", "legacy_global_only"]
MeshMatchMode = Literal["all_reliable", "legacy_local_only"]


@dataclass(frozen=True)
class GeometryPipelineConfig:
    """Mode controls for experiments A–F without changing legacy APIs."""

    geometry_mode: GeometryMode = "residual_mesh"
    routing_mode: RoutingMode = "none"
    h_match_mode: HMatchMode = "all_reliable"
    mesh_match_mode: MeshMatchMode = "all_reliable"
    legacy_role_threshold: float = 0.5
    homography: GlobalHomographyConfig = field(default_factory=GlobalHomographyConfig)
    residual_mesh: ResidualMeshConfig = field(default_factory=ResidualMeshConfig)

    def __post_init__(self) -> None:
        if self.geometry_mode not in {"h_only", "residual_mesh", "learned_residual_mesh"}:
            raise ValueError(f"Unsupported geometry_mode={self.geometry_mode!r}.")
        if self.routing_mode not in {"none", "legacy_role", "soft_routing"}:
            raise ValueError(f"Unsupported routing_mode={self.routing_mode!r}.")
        if self.h_match_mode not in {"all_reliable", "legacy_global_only"}:
            raise ValueError(f"Unsupported h_match_mode={self.h_match_mode!r}.")
        if self.mesh_match_mode not in {"all_reliable", "legacy_local_only"}:
            raise ValueError(f"Unsupported mesh_match_mode={self.mesh_match_mode!r}.")
        if not 0.0 <= self.legacy_role_threshold <= 1.0:
            raise ValueError("legacy_role_threshold must lie in [0,1].")

    @classmethod
    def experiment(cls, name: str, **kwargs: Any) -> "GeometryPipelineConfig":
        """Construct the requested A–F geometry ablation."""

        presets: dict[str, dict[str, str]] = {
            "A": dict(geometry_mode="h_only", routing_mode="none",
                      h_match_mode="all_reliable", mesh_match_mode="all_reliable"),
            "B": dict(geometry_mode="h_only", routing_mode="legacy_role",
                      h_match_mode="legacy_global_only", mesh_match_mode="all_reliable"),
            "C": dict(geometry_mode="residual_mesh", routing_mode="none",
                      h_match_mode="all_reliable", mesh_match_mode="all_reliable"),
            "D": dict(geometry_mode="residual_mesh", routing_mode="legacy_role",
                      h_match_mode="all_reliable", mesh_match_mode="legacy_local_only"),
            "E": dict(geometry_mode="residual_mesh", routing_mode="soft_routing",
                      h_match_mode="all_reliable", mesh_match_mode="all_reliable"),
            "F": dict(geometry_mode="learned_residual_mesh", routing_mode="soft_routing",
                      h_match_mode="all_reliable", mesh_match_mode="all_reliable"),
        }
        key = name.upper()
        if key not in presets:
            raise ValueError(f"Unknown experiment {name!r}; expected A through F.")
        values: dict[str, Any] = dict(presets[key])
        values.update(kwargs)
        return cls(**values)


@dataclass
class GeometryPipelineResult:
    """Geometry-only state passed to decoder, seam, and debug layers."""

    correspondences: GRMCorrespondenceSet
    H_base: GlobalHomographyResult
    initial_residual_mesh: ResidualMeshResult
    learned_mesh_correction: Optional[Tensor]
    final_mapping_xy: Tensor
    final_grid: Tensor
    deformation_confidence: Tensor
    deformation_uncertainty: Tensor
    fallback_reason: str
    diagnostics: dict[str, Any]

    def json_report(self) -> dict[str, Any]:
        """Return a JSON-serializable metric report."""

        def convert(value: Any) -> Any:
            if torch.is_tensor(value):
                if value.numel() == 1:
                    return value.detach().cpu().item()
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(self.diagnostics)


def _mesh_uncertainty(mesh: ResidualMeshResult) -> Tensor:
    variance = mesh.residual_variance[None, None].float()
    dense = torch.nn.functional.interpolate(
        variance,
        size=mesh.deformation_confidence.shape,
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    uncertainty = torch.sqrt(dense.clamp_min(0.0))
    return uncertainty * (1.0 - mesh.deformation_confidence.float())


def run_geometry_pipeline(
    correspondences: GRMCorrespondenceSet,
    image_a_hw: Tuple[int, int],
    image_b_hw: Tuple[int, int],
    *,
    config: Optional[GeometryPipelineConfig] = None,
    previous_h_ab: Optional[Tensor] = None,
    overlap_mask: Optional[Tensor] = None,
    w_h: Optional[Tensor] = None,
    w_deform: Optional[Tensor] = None,
    learned_decoder: Optional[Any] = None,
    learned_decoder_inputs: Optional[dict[str, Any]] = None,
) -> GeometryPipelineResult:
    """Run all-reliable H and optional non-learning residual mesh."""

    cfg = config or GeometryPipelineConfig()
    correspondences.validate()
    count = correspondences.count
    if w_h is not None and w_h.shape != (count,):
        raise ValueError("w_h must be [N].")
    if w_deform is not None and w_deform.shape != (count,):
        raise ValueError("w_deform must be [N].")

    h_mask = torch.ones(count, device=correspondences.src_xy.device, dtype=torch.bool)
    if cfg.h_match_mode == "legacy_global_only":
        h_mask &= correspondences.legacy_p_global >= cfg.legacy_role_threshold
    h_weights = correspondences.reliability[h_mask]
    # Soft H support is a ranking/refinement weight only; no low-w_h point is
    # removed from MAGSAC.
    if cfg.routing_mode == "soft_routing" and w_h is not None:
        h_weights = h_weights * (0.25 + 0.75 * w_h[h_mask].clamp(0.0, 1.0))
    h_result = estimate_global_homography(
        correspondences.src_xy[h_mask],
        correspondences.dst_xy[h_mask],
        h_weights,
        image_a_hw=image_a_hw,
        image_b_hw=image_b_hw,
        previous_h_ab=previous_h_ab,
        config=cfg.homography,
    )

    fallback_reasons: list[str] = []
    if h_result.failure_reason:
        fallback_reasons.append(h_result.failure_reason)
    if cfg.geometry_mode == "h_only":
        mesh_mode = "h_only"
    elif cfg.mesh_match_mode == "legacy_local_only":
        mesh_mode = "legacy_local_only_mesh"
    else:
        mesh_mode = "all_reliable_mesh"

    effective_w_deform = w_deform
    if cfg.routing_mode == "soft_routing" and effective_w_deform is None:
        effective_w_deform = correspondences.reliability
        fallback_reasons.append("routing_checkpoint_missing:w_deform=reliability")
    mesh_result = estimate_residual_mesh(
        correspondences.src_xy,
        correspondences.dst_xy,
        h_result.H_ab,
        image_a_hw,
        correspondences.reliability,
        overlap_mask=overlap_mask,
        w_deform=effective_w_deform if cfg.routing_mode == "soft_routing" else None,
        legacy_p_local=correspondences.legacy_p_local,
        mode=mesh_mode,
        config=cfg.residual_mesh,
    )
    if mesh_result.diagnostics.get("fallback_reason"):
        fallback_reasons.append(str(mesh_result.diagnostics["fallback_reason"]))

    learned_correction = None
    final_mapping = mesh_result.final_mapping_xy
    deformation_confidence = mesh_result.deformation_confidence
    deformation_uncertainty = _mesh_uncertainty(mesh_result)
    if cfg.geometry_mode == "learned_residual_mesh":
        if learned_decoder is None:
            fallback_reasons.append("learned_decoder_unavailable:using_initial_mesh")
        elif learned_decoder_inputs is None:
            fallback_reasons.append("learned_decoder_inputs_missing:using_initial_mesh")
        else:
            decoder_kwargs = dict(learned_decoder_inputs)
            decoder_kwargs.setdefault(
                "h_base_grid",
                pixel_xy_to_normalized_grid(mesh_result.base_mapping_xy, image_b_hw)[None],
            )
            decoder_kwargs.setdefault(
                "initial_mesh_offset", mesh_result.mesh_offset[None]
            )
            decoder_output = learned_decoder(**decoder_kwargs)
            learned_correction = decoder_output.mesh_correction
            final_mapping = decoder_output.final_mapping_xy[0]
            deformation_confidence = decoder_output.deformation_confidence[0]
            deformation_uncertainty = decoder_output.deformation_uncertainty[0]
    final_grid = pixel_xy_to_normalized_grid(final_mapping, image_b_hw)

    match_prediction = mesh_result.final_mapping_xy[
        correspondences.src_xy[:, 1].round().long().clamp(0, image_a_hw[0] - 1),
        correspondences.src_xy[:, 0].round().long().clamp(0, image_a_hw[1] - 1),
    ]
    final_error = torch.linalg.vector_norm(
        match_prediction - correspondences.dst_xy, dim=-1
    )
    diagnostics = {
        "config": {
            "geometry_mode": cfg.geometry_mode,
            "routing_mode": cfg.routing_mode,
            "h_match_mode": cfg.h_match_mode,
            "mesh_match_mode": cfg.mesh_match_mode,
            "legacy_role_threshold": cfg.legacy_role_threshold,
        },
        "match_count": count,
        "h_input_match_count": int(h_mask.sum()),
        "h_inlier_count": h_result.inlier_count,
        "h_valid": h_result.valid,
        "h_model": h_result.diagnostics.get("model"),
        "h_reprojection_error_median_px": (
            float(h_result.reprojection_error.median())
            if h_result.reprojection_error.numel() else 0.0
        ),
        "final_match_error_median_px": (
            float(final_error.median()) if final_error.numel() else 0.0
        ),
        "final_match_error_p90_px": (
            float(torch.quantile(final_error.float(), 0.9))
            if final_error.numel() else 0.0
        ),
        "mesh": mesh_result.diagnostics,
        "fallback_reason": ";".join(reason for reason in fallback_reasons if reason),
    }
    return GeometryPipelineResult(
        correspondences=correspondences,
        H_base=h_result,
        initial_residual_mesh=mesh_result,
        learned_mesh_correction=learned_correction,
        final_mapping_xy=final_mapping,
        final_grid=final_grid,
        deformation_confidence=deformation_confidence,
        deformation_uncertainty=deformation_uncertainty,
        fallback_reason=diagnostics["fallback_reason"],
        diagnostics=diagnostics,
    )
