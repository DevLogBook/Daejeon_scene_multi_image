"""Explicit normalized correlation, matchability, and Dual Softmax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import GRMMatcherConfig


@dataclass
class GRMCorrelationOutput:
    """Correlation tensors; all probability computations are FP32."""

    similarity: Tensor
    dual_probability: Tensor
    probability: Tensor
    matchability_a: Tensor
    matchability_b: Tensor


class GRMMatchabilityHead(nn.Module):
    """Per-token overlap/matchability head, not a replacement for correlation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        # Keep overlap logits in FP32 even when surrounding feature extraction
        # runs under AMP; this also avoids CPU half/float linear mismatches.
        tokens_fp32 = tokens.float()
        return torch.sigmoid(self.projection(self.norm(tokens_fp32)).squeeze(-1))


def grm_dual_softmax(similarity: Tensor, valid_pairs: Optional[Tensor] = None) -> Tensor:
    """Compute stable FP32 row-softmax times column-softmax."""

    logits = similarity.float()
    if valid_pairs is not None:
        if valid_pairs.shape != logits.shape or valid_pairs.dtype != torch.bool:
            raise ValueError("valid_pairs must be a bool tensor matching similarity.")
        logits = logits.masked_fill(~valid_pairs, torch.finfo(logits.dtype).min)
    p_ab = torch.softmax(logits, dim=-1)
    p_ba = torch.softmax(logits, dim=-2)
    dual = p_ab * p_ba
    if valid_pairs is not None:
        dual = dual * valid_pairs.to(dtype=dual.dtype)
    return dual


class GRMExplicitCorrelation(nn.Module):
    """L2-normalized all-pairs correlation with two matchability heads."""

    def __init__(self, config: GRMMatcherConfig) -> None:
        super().__init__()
        self.dim = config.coarse_dim
        self.temperature = config.temperature
        self.eps = config.probability_eps
        self.matchability_head_a = GRMMatchabilityHead(config.coarse_dim)
        self.matchability_head_b = GRMMatchabilityHead(config.coarse_dim)

    def forward(
        self,
        tokens_a: Tensor,
        tokens_b: Tensor,
        valid_mask_a: Optional[Tensor] = None,
        valid_mask_b: Optional[Tensor] = None,
    ) -> GRMCorrelationOutput:
        self._validate_tokens(tokens_a, tokens_b)
        batch, count_a, _ = tokens_a.shape
        count_b = tokens_b.shape[1]
        mask_a = self._prepare_mask(valid_mask_a, batch, count_a, tokens_a.device)
        mask_b = self._prepare_mask(valid_mask_b, batch, count_b, tokens_b.device)
        valid_pairs = mask_a[:, :, None] & mask_b[:, None, :]

        normalized_a = F.normalize(tokens_a.float(), dim=-1, eps=self.eps)
        normalized_b = F.normalize(tokens_b.float(), dim=-1, eps=self.eps)
        similarity = torch.einsum("bic,bjc->bij", normalized_a, normalized_b)
        similarity = similarity / self.temperature
        similarity = similarity.masked_fill(
            ~valid_pairs, torch.finfo(similarity.dtype).min
        )

        dual_probability = grm_dual_softmax(similarity, valid_pairs)
        matchability_a = self.matchability_head_a(tokens_a) * mask_a.float()
        matchability_b = self.matchability_head_b(tokens_b) * mask_b.float()
        probability = (
            dual_probability
            * matchability_a[:, :, None]
            * matchability_b[:, None, :]
        )
        return GRMCorrelationOutput(
            similarity=similarity,
            dual_probability=dual_probability,
            probability=probability,
            matchability_a=matchability_a,
            matchability_b=matchability_b,
        )

    def _validate_tokens(self, tokens_a: Tensor, tokens_b: Tensor) -> None:
        for name, tokens in (("tokens_a", tokens_a), ("tokens_b", tokens_b)):
            if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
                raise ValueError(
                    f"{name} must be [B, N, {self.dim}], got {tuple(tokens.shape)}."
                )
        if tokens_a.shape[0] != tokens_b.shape[0]:
            raise ValueError("Token batches must match.")

    @staticmethod
    def _prepare_mask(
        mask: Optional[Tensor], batch: int, count: int, device: torch.device
    ) -> Tensor:
        if mask is None:
            return torch.ones(batch, count, device=device, dtype=torch.bool)
        if mask.shape != (batch, count) or mask.dtype != torch.bool:
            raise ValueError(f"Mask must be bool {(batch, count)}, got {tuple(mask.shape)}.")
        return mask.to(device=device)
