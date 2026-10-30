"""Stable public output contract for GRM-Matcher."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class GRMMatcherAuxiliary:
    """Training-only tensors; never required by the public inference API."""

    coarse_similarity: Tensor
    coarse_probability: Tensor
    matchability_a: Tensor
    matchability_b: Tensor
    coarse_height: int
    coarse_width: int

    query_indices: Tensor
    candidate_indices: Tensor
    candidate_prior: Tensor
    candidate_valid_mask: Tensor
    candidate_src_xy: Tensor
    candidate_initial_dst_xy: Tensor

    mid_dst_xy: Tensor
    mid_probability: Tensor
    fine_initial_dst_xy: Tensor
    fine_dst_xy: Tensor
    fine_probability: Tensor

    rerank_logits: Tensor
    rerank_probabilities: Tensor
    selected_indices: Tensor
    role_logits: Tensor
    inlier_logits: Optional[Tensor] = None
    # Optional extension for separately checkpointed geometry routing.  This is
    # training-only and does not change the stable GRMMatcherOutput contract.
    selected_graph_tokens: Optional[Tensor] = None
    feature_a_fine_map: Optional[Tensor] = None
    feature_b_fine_map: Optional[Tensor] = None
    # Ablation diagnostics.  ``selected_graph_tokens`` contains pre-graph
    # tokens when the selected plugin bypasses geometry context.
    candidate_rerank_mode: str = "geometry_graph"
    candidate_rerank_used_geometry: bool = True
    candidate_rerank_used_learned_head: bool = True


@dataclass
class GRMMatcherOutput:
    """Public matcher result.

    ``src_xy`` and ``dst_xy`` are original-image pixel coordinates in ``(x,
    y)`` order.  The candidate axis ``N`` is padded across a batch and selected
    by ``valid_mask``.  In GRM v1 each retained source query owns one public
    result and ``K`` retained hypotheses, so ``M == N``.
    """

    src_xy: Tensor
    dst_xy: Tensor

    p_match: Tensor

    p_global: Tensor
    p_local: Tensor
    p_false: Tensor

    candidate_entropy: Tensor
    cycle_error: Tensor
    fine_uncertainty: Tensor

    overlap_src: Tensor
    overlap_dst: Tensor

    candidate_indices: Tensor
    candidate_probs: Tensor

    valid_mask: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.src_xy.shape[0])

    @property
    def num_matches(self) -> int:
        return int(self.src_xy.shape[1])

    @property
    def topk(self) -> int:
        return int(self.candidate_indices.shape[-1])

    def as_dict(self) -> Dict[str, Tensor]:
        """Return tensors keyed by the stable public field names."""

        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def p_inlier(self) -> Tensor:
        """Binary model: learned-head score; legacy model: Global+Local mass.

        Binary mode retains p_global=p_inlier, p_local=0, p_false=1-p_inlier
        solely for old tensor consumers. No global-plane membership is predicted.
        """
        return self.p_global + self.p_local

    def detach(self) -> "GRMMatcherOutput":
        """Detach every output tensor while preserving the dataclass contract."""

        return GRMMatcherOutput(**{name: value.detach() for name, value in self.as_dict().items()})

    def to(self, *args: Any, **kwargs: Any) -> "GRMMatcherOutput":
        """Move/cast outputs with ``nn.Module.to``-compatible dtype semantics.

        Device changes apply to every tensor, while explicit floating-point
        dtype changes do not corrupt integer indices or the boolean valid mask.
        """

        converted: Dict[str, Tensor] = {}
        for name, value in self.as_dict().items():
            moved = value.to(*args, **kwargs)
            if not (value.is_floating_point() or value.is_complex()):
                moved = moved.to(dtype=value.dtype)
            converted[name] = moved
        return GRMMatcherOutput(**converted)

    def validate(
        self,
        *,
        check_finite: bool = True,
        check_probabilities: bool = True,
        atol: float = 1.0e-5,
    ) -> None:
        """Validate shapes, dtypes, finiteness, and probability semantics."""

        if self.src_xy.ndim != 3 or self.src_xy.shape[-1] != 2:
            raise ValueError(f"src_xy must be [B, N, 2], got {tuple(self.src_xy.shape)}.")
        batch, count, _ = self.src_xy.shape
        if self.dst_xy.shape != (batch, count, 2):
            raise ValueError(f"dst_xy must be {(batch, count, 2)}, got {tuple(self.dst_xy.shape)}.")

        per_match_names = (
            "p_match",
            "p_global",
            "p_local",
            "p_false",
            "candidate_entropy",
            "cycle_error",
            "valid_mask",
        )
        for name in per_match_names:
            value = getattr(self, name)
            if value.shape != (batch, count):
                raise ValueError(f"{name} must be [B, N], got {tuple(value.shape)}.")

        uncertainty_shape = tuple(self.fine_uncertainty.shape)
        valid_uncertainty_shapes = {(batch, count, 2), (batch, count, 2, 2)}
        if uncertainty_shape not in valid_uncertainty_shapes:
            raise ValueError(
                "fine_uncertainty must be [B, N, 2] or [B, N, 2, 2], "
                f"got {uncertainty_shape}."
            )

        for name in ("overlap_src", "overlap_dst"):
            value = getattr(self, name)
            if value.ndim != 4 or value.shape[0] != batch or value.shape[1] != 1:
                raise ValueError(f"{name} must be [B, 1, Hc, Wc], got {tuple(value.shape)}.")
        if self.overlap_src.shape != self.overlap_dst.shape:
            raise ValueError("overlap_src and overlap_dst must have identical shapes.")

        if self.candidate_indices.ndim != 3:
            raise ValueError("candidate_indices must be [B, N, K].")
        if self.candidate_probs.shape != self.candidate_indices.shape:
            raise ValueError("candidate_probs must have the same [B, N, K] shape as indices.")
        if self.candidate_indices.shape[:2] != (batch, count):
            raise ValueError("The candidate query dimension must match public match count N.")
        if self.candidate_indices.shape[-1] <= 0:
            raise ValueError("The candidate K dimension must be positive.")
        if self.candidate_indices.dtype != torch.long:
            raise TypeError("candidate_indices must use torch.long.")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must use torch.bool.")

        if check_finite:
            for name, value in self.as_dict().items():
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"{name} contains NaN or Inf.")

        if check_probabilities:
            probability_names = (
                "p_match",
                "p_global",
                "p_local",
                "p_false",
                "candidate_probs",
                "overlap_src",
                "overlap_dst",
            )
            for name in probability_names:
                value = getattr(self, name)
                if torch.any(value < -atol) or torch.any(value > 1.0 + atol):
                    raise ValueError(f"{name} contains values outside [0, 1].")

            valid = self.valid_mask
            role_sum = self.p_global + self.p_local + self.p_false
            if valid.any() and not torch.allclose(
                role_sum[valid], torch.ones_like(role_sum[valid]), atol=atol, rtol=0.0
            ):
                raise ValueError("Role probabilities must sum to one for valid matches.")

            candidate_sum = self.candidate_probs.sum(dim=-1)
            if valid.any() and not torch.allclose(
                candidate_sum[valid],
                torch.ones_like(candidate_sum[valid]),
                atol=atol,
                rtol=0.0,
            ):
                raise ValueError("Conditional top-k probabilities must sum to one for valid queries.")


def select_global_matches(
    output: GRMMatcherOutput,
    *,
    minimum_role_probability: float = 0.5,
    minimum_match_probability: float = 0.0,
) -> List[Dict[str, Tensor]]:
    """Return unpadded per-batch matches suitable for an external global solver."""

    return _select_role_matches(
        output,
        output.p_global,
        minimum_role_probability,
        minimum_match_probability,
    )


def select_local_matches(
    output: GRMMatcherOutput,
    *,
    minimum_role_probability: float = 0.5,
    minimum_match_probability: float = 0.0,
) -> List[Dict[str, Tensor]]:
    """Return unpadded per-batch matches for external rematch/residual modules."""

    return _select_role_matches(
        output,
        output.p_local,
        minimum_role_probability,
        minimum_match_probability,
    )


def _select_role_matches(
    output: GRMMatcherOutput,
    role_probability: Tensor,
    minimum_role_probability: float,
    minimum_match_probability: float,
) -> List[Dict[str, Tensor]]:
    if not 0.0 <= minimum_role_probability <= 1.0:
        raise ValueError("minimum_role_probability must lie in [0, 1].")
    if not 0.0 <= minimum_match_probability <= 1.0:
        raise ValueError("minimum_match_probability must lie in [0, 1].")
    selected_batches: List[Dict[str, Tensor]] = []
    mask = (
        output.valid_mask
        & (role_probability >= minimum_role_probability)
        & (output.p_match >= minimum_match_probability)
    )
    for batch_index in range(output.batch_size):
        selected = mask[batch_index]
        selected_batches.append(
            {
                "src_xy": output.src_xy[batch_index, selected],
                "dst_xy": output.dst_xy[batch_index, selected],
                "p_match": output.p_match[batch_index, selected],
                "p_role": role_probability[batch_index, selected],
                "fine_uncertainty": output.fine_uncertainty[batch_index, selected],
            }
        )
    return selected_batches
