"""Inference-only adapter from stable GRM outputs to stitching correspondences.

The adapter deliberately does not use the legacy Global/Local/False role as a
hard reliability filter.  Public coordinates are returned in original-image
pixel coordinates in ``(x, y)`` order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

from grm_matcher.outputs import GRMMatcherOutput


@dataclass(frozen=True)
class GRMAdapterConfig:
    """Reliability filtering and uncertainty calibration for GRM inference."""

    minimum_match_probability: float = 0.05
    maximum_cycle_error: Optional[float] = None
    cycle_sigma_px: float = 4.0
    uncertainty_sigma_px: float = 4.0
    minimum_reliability: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_match_probability <= 1.0:
            raise ValueError("minimum_match_probability must lie in [0, 1].")
        if self.maximum_cycle_error is not None and self.maximum_cycle_error <= 0.0:
            raise ValueError("maximum_cycle_error must be positive or None.")
        if self.cycle_sigma_px <= 0.0 or self.uncertainty_sigma_px <= 0.0:
            raise ValueError("Reliability sigmas must be positive.")
        if not 0.0 <= self.minimum_reliability <= 1.0:
            raise ValueError("minimum_reliability must lie in [0, 1].")


@dataclass
class GRMCorrespondenceSet:
    """One unpadded, reliable correspondence set for a single image pair."""

    src_xy: Tensor
    dst_xy: Tensor
    p_match: Tensor
    cycle_error: Tensor
    fine_uncertainty: Tensor
    candidate_entropy: Tensor
    reliability: Tensor
    uncertainty_score: Tensor
    legacy_p_global: Tensor
    legacy_p_local: Tensor
    legacy_p_false: Tensor
    valid_mask: Tensor
    source_indices: Tensor

    @property
    def count(self) -> int:
        return int(self.src_xy.shape[0])

    def validate(self) -> None:
        """Validate the unbatched public tensor contract."""

        if self.src_xy.ndim != 2 or self.src_xy.shape[-1] != 2:
            raise ValueError(f"src_xy must be [N,2], got {tuple(self.src_xy.shape)}.")
        count = self.src_xy.shape[0]
        if self.dst_xy.shape != (count, 2):
            raise ValueError("dst_xy must have shape [N,2].")
        for name in (
            "p_match",
            "cycle_error",
            "candidate_entropy",
            "reliability",
            "uncertainty_score",
            "legacy_p_global",
            "legacy_p_local",
            "legacy_p_false",
            "valid_mask",
            "source_indices",
        ):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must have shape [N].")
        if self.fine_uncertainty.shape not in {(count, 2), (count, 2, 2)}:
            raise ValueError("fine_uncertainty must be [N,2] or [N,2,2].")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean.")
        if self.source_indices.dtype != torch.long:
            raise TypeError("source_indices must use torch.long.")
        floating = (
            self.src_xy,
            self.dst_xy,
            self.p_match,
            self.cycle_error,
            self.fine_uncertainty,
            self.candidate_entropy,
            self.reliability,
        )
        if any(not torch.isfinite(value).all() for value in floating):
            raise ValueError("GRMCorrespondenceSet contains NaN or Inf.")


@dataclass
class GRMAdapterResult:
    """Raw padded GRM result plus reliable per-sample correspondence sets."""

    raw_output: GRMMatcherOutput
    correspondences: list[GRMCorrespondenceSet]
    auxiliary: Optional[Any] = None


def uncertainty_score(
    uncertainty: Tensor,
    sigma_px: float,
) -> Tensor:
    """Convert covariance/axis uncertainty into a bounded confidence score."""

    if uncertainty.ndim < 2:
        raise ValueError("uncertainty must end in [2] or [2,2].")
    value = torch.nan_to_num(
        uncertainty.float(), nan=float("inf"), posinf=float("inf"), neginf=float("inf")
    )
    if value.shape[-2:] == (2, 2):
        trace = value[..., 0, 0].clamp_min(0.0) + value[..., 1, 1].clamp_min(0.0)
        rms = torch.sqrt(0.5 * trace)
    elif value.shape[-1] == 2:
        rms = torch.sqrt(value.square().mean(dim=-1))
    else:
        raise ValueError("uncertainty must end in [2] or [2,2].")
    return torch.exp(-rms / float(sigma_px)).clamp(0.0, 1.0)


def unpack_grm_output(
    output: GRMMatcherOutput,
    config: GRMAdapterConfig,
    *,
    scale_a_xy: Sequence[float] = (1.0, 1.0),
    scale_b_xy: Sequence[float] = (1.0, 1.0),
) -> list[GRMCorrespondenceSet]:
    """Remove padding, rescale coordinates/covariance, and filter reliability."""

    output.validate(check_finite=False)
    if len(scale_a_xy) != 2 or len(scale_b_xy) != 2:
        raise ValueError("scale_a_xy and scale_b_xy must contain (sx, sy).")
    if min(*scale_a_xy, *scale_b_xy) <= 0.0:
        raise ValueError("Resize scales must be positive.")

    device = output.src_xy.device
    src_scale = output.src_xy.new_tensor(scale_a_xy)
    dst_scale = output.dst_xy.new_tensor(scale_b_xy)
    src_xy = output.src_xy / src_scale
    dst_xy = output.dst_xy / dst_scale

    covariance = output.fine_uncertainty
    if covariance.shape[-2:] == (2, 2):
        inv_scale = covariance.new_tensor(
            [[1.0 / scale_b_xy[0], 0.0], [0.0, 1.0 / scale_b_xy[1]]]
        )
        covariance = torch.einsum("ij,bnjk,lk->bnil", inv_scale, covariance, inv_scale)
    else:
        covariance = covariance / dst_scale

    uncertainty_confidence = uncertainty_score(covariance, config.uncertainty_sigma_px)
    cycle_score = torch.exp(
        -torch.nan_to_num(
            output.cycle_error.float(),
            nan=float("inf"),
            posinf=float("inf"),
            neginf=float("inf"),
        ).clamp_min(0.0)
        / config.cycle_sigma_px
    )
    reliability = (
        output.p_match.float().clamp(0.0, 1.0)
        * cycle_score
        * uncertainty_confidence
    ).clamp(0.0, 1.0)
    finite = (
        torch.isfinite(src_xy).all(dim=-1)
        & torch.isfinite(dst_xy).all(dim=-1)
        & torch.isfinite(output.p_match)
        & torch.isfinite(output.cycle_error)
        & torch.isfinite(covariance).flatten(2).all(dim=-1)
    )
    reliable = (
        output.valid_mask
        & finite
        & (output.p_match >= config.minimum_match_probability)
        & (reliability >= config.minimum_reliability)
    )
    if config.maximum_cycle_error is not None:
        reliable &= output.cycle_error <= config.maximum_cycle_error

    result: list[GRMCorrespondenceSet] = []
    for batch_index in range(output.batch_size):
        indices = torch.nonzero(reliable[batch_index], as_tuple=False).squeeze(-1)
        selected = GRMCorrespondenceSet(
            src_xy=src_xy[batch_index, indices],
            dst_xy=dst_xy[batch_index, indices],
            p_match=output.p_match[batch_index, indices],
            cycle_error=output.cycle_error[batch_index, indices],
            fine_uncertainty=covariance[batch_index, indices],
            candidate_entropy=output.candidate_entropy[batch_index, indices],
            reliability=reliability[batch_index, indices],
            uncertainty_score=uncertainty_confidence[batch_index, indices],
            legacy_p_global=output.p_global[batch_index, indices],
            legacy_p_local=output.p_local[batch_index, indices],
            legacy_p_false=output.p_false[batch_index, indices],
            valid_mask=torch.ones(indices.numel(), device=device, dtype=torch.bool),
            source_indices=indices.to(dtype=torch.long),
        )
        selected.validate()
        result.append(selected)
    return result


class GRMMatcherAdapter:
    """Call a GRM matcher under ``no_grad`` and expose stitching-ready matches."""

    def __init__(
        self,
        matcher: nn.Module,
        config: Optional[GRMAdapterConfig] = None,
    ) -> None:
        self.matcher = matcher
        self.config = config or GRMAdapterConfig()

    def __call__(
        self,
        image_a: Tensor,
        image_b: Tensor,
        *,
        scale_a_xy: Sequence[float] = (1.0, 1.0),
        scale_b_xy: Sequence[float] = (1.0, 1.0),
        return_auxiliary: bool = False,
    ) -> GRMAdapterResult:
        if image_a.ndim != 4 or image_b.ndim != 4:
            raise ValueError("image_a and image_b must be [B,C,H,W].")
        with torch.no_grad():
            output = self.matcher(
                image_a, image_b, return_auxiliary=return_auxiliary
            ) if return_auxiliary else self.matcher(image_a, image_b)
        auxiliary = None
        if isinstance(output, tuple):
            output, auxiliary = output
        if not isinstance(output, GRMMatcherOutput):
            raise TypeError("GRM matcher must return GRMMatcherOutput.")
        correspondences = unpack_grm_output(
            output,
            self.config,
            scale_a_xy=scale_a_xy,
            scale_b_xy=scale_b_xy,
        )
        return GRMAdapterResult(
            raw_output=output,
            correspondences=correspondences,
            auxiliary=auxiliary,
        )
