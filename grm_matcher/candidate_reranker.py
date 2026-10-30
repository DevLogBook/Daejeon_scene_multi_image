"""Pluggable within-query candidate reranking for GRM-Matcher.

The router and built-in plugins intentionally own no trainable parameters.
Existing trainable modules remain registered on :class:`GRMMatcher` under the
legacy ``geometry_graph.*`` and ``rerank_head.*`` state-dict prefixes.  This
keeps checkpoints produced before the plugin abstraction strictly loadable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, TYPE_CHECKING

import torch
from torch import Tensor

from .config import GRMCandidateRerankMode

if TYPE_CHECKING:
    from .geometry_graph import GRMGeometryGraph
    from .role_head import GRMCandidateRerankHead


@dataclass(frozen=True)
class GRMCandidateRerankContext:
    """Inputs shared by every candidate reranking plugin.

    ``candidate_tokens`` are pre-graph tokens with shape ``[B, M, K, C]``.
    Coordinates and seed errors use the flattened candidate shape
    ``[B, M*K, ...]`` expected by :class:`GRMGeometryGraph`.
    """

    candidate_tokens: Tensor
    candidate_prior: Tensor
    candidate_reliability: Tensor
    candidate_valid_mask: Tensor
    flat_src_xy: Tensor
    flat_dst_xy: Tensor
    flat_seed_error: Tensor
    image_height: int
    image_width: int


@dataclass
class GRMCandidateRerankResult:
    """Normalized plugin result with a stable matcher-facing contract."""

    logits: Tensor
    probabilities: Tensor
    selected_indices: Tensor
    candidate_tokens: Tensor
    selected_probability_multiplier: Tensor
    mode: str
    used_geometry_graph: bool
    used_learned_rerank: bool


class GRMCandidateRerankPlugin(Protocol):
    """Structural interface for built-in or experiment-specific rerankers."""

    mode: str
    used_geometry_graph: bool
    used_learned_rerank: bool

    def __call__(
        self,
        context: GRMCandidateRerankContext,
        *,
        geometry_graph: "GRMGeometryGraph",
        rerank_head: "GRMCandidateRerankHead",
        eps: float,
    ) -> GRMCandidateRerankResult:
        ...


class _BuiltInReranker:
    def __init__(
        self,
        mode: GRMCandidateRerankMode,
        *,
        use_geometry_graph: bool,
        use_learned_rerank: bool,
        deterministic_score: str = "reliability",
    ) -> None:
        self.mode = mode
        self.used_geometry_graph = use_geometry_graph
        self.used_learned_rerank = use_learned_rerank
        self.deterministic_score = deterministic_score

    def __call__(
        self,
        context: GRMCandidateRerankContext,
        *,
        geometry_graph: "GRMGeometryGraph",
        rerank_head: "GRMCandidateRerankHead",
        eps: float,
    ) -> GRMCandidateRerankResult:
        candidate_tokens = context.candidate_tokens
        batch, queries, topk, hidden = candidate_tokens.shape
        if self.used_geometry_graph:
            flat_tokens = candidate_tokens.reshape(batch, queries * topk, hidden)
            flat_valid = context.candidate_valid_mask.reshape(batch, queries * topk)
            candidate_tokens = geometry_graph(
                flat_tokens,
                context.flat_src_xy,
                context.flat_dst_xy,
                context.flat_seed_error,
                flat_valid,
                image_height=context.image_height,
                image_width=context.image_width,
            ).reshape(batch, queries, topk, hidden)

        if self.used_learned_rerank:
            learned = rerank_head(
                candidate_tokens,
                context.candidate_prior,
                context.candidate_valid_mask,
            )
            selected_probability = torch.gather(
                learned.probabilities,
                2,
                learned.selected_indices[..., None],
            ).squeeze(-1)
            return GRMCandidateRerankResult(
                logits=learned.logits,
                probabilities=learned.probabilities,
                selected_indices=learned.selected_indices,
                candidate_tokens=candidate_tokens,
                # Preserve the exact legacy p_match fusion for learned modes.
                selected_probability_multiplier=selected_probability,
                mode=self.mode,
                used_geometry_graph=self.used_geometry_graph,
                used_learned_rerank=True,
            )

        if self.deterministic_score == "coarse_prior":
            scores = context.candidate_prior
        else:
            scores = context.candidate_reliability
        probabilities, logits = _masked_conditional_distribution(
            scores,
            context.candidate_valid_mask,
            fallback=context.candidate_prior,
            eps=eps,
        )
        selected = probabilities.argmax(dim=-1)
        # Deterministic plugins use reliability only to select K; multiplying
        # by its conditional distribution again would square the same signal.
        multiplier = context.candidate_valid_mask.any(dim=-1).to(
            dtype=context.candidate_reliability.dtype
        )
        return GRMCandidateRerankResult(
            logits=logits,
            probabilities=probabilities,
            selected_indices=selected,
            candidate_tokens=candidate_tokens,
            selected_probability_multiplier=multiplier,
            mode=self.mode,
            used_geometry_graph=self.used_geometry_graph,
            used_learned_rerank=False,
        )


class GRMCandidateRerankRouter:
    """Runtime-selectable reranker registry with parameter-free built-ins.

    Built-in modes isolate the two learned components:

    - ``geometry_graph``: legacy graph plus learned rerank head.
    - ``graph_reliability``: graph context retained, deterministic reliability
      selection replaces the learned rerank head.
    - ``token_mlp``: graph bypassed, the existing learned head sees raw tokens.
    - ``reliability``: graph and learned head bypassed; the explicit
      coarse/mid/fine/cycle reliability selects the candidate.
    - ``coarse_prior``: graph and learned head bypassed; coarse top-k
      conditional probability alone selects the candidate.

    The registry is a plain Python object rather than an ``nn.Module`` so it
    introduces no new state-dict keys.
    """

    def __init__(self, eps: float) -> None:
        self.eps = float(eps)
        self._plugins: Dict[str, GRMCandidateRerankPlugin] = {}
        self.register(
            _BuiltInReranker(
                "geometry_graph",
                use_geometry_graph=True,
                use_learned_rerank=True,
            )
        )
        self.register(
            _BuiltInReranker(
                "graph_reliability",
                use_geometry_graph=True,
                use_learned_rerank=False,
            )
        )
        self.register(
            _BuiltInReranker(
                "token_mlp",
                use_geometry_graph=False,
                use_learned_rerank=True,
            )
        )
        self.register(
            _BuiltInReranker(
                "reliability",
                use_geometry_graph=False,
                use_learned_rerank=False,
            )
        )
        self.register(
            _BuiltInReranker(
                "coarse_prior",
                use_geometry_graph=False,
                use_learned_rerank=False,
                deterministic_score="coarse_prior",
            )
        )

    @property
    def available_modes(self) -> tuple[str, ...]:
        return tuple(self._plugins)

    def register(
        self,
        plugin: GRMCandidateRerankPlugin,
        *,
        replace: bool = False,
    ) -> None:
        """Register a parameter-free experimental strategy by its ``mode``."""

        mode = str(plugin.mode)
        if not mode:
            raise ValueError("Candidate rerank plugin mode must be non-empty.")
        if mode in self._plugins and not replace:
            raise ValueError(f"Candidate rerank mode {mode!r} is already registered.")
        self._plugins[mode] = plugin

    def __call__(
        self,
        mode: str,
        context: GRMCandidateRerankContext,
        *,
        geometry_graph: "GRMGeometryGraph",
        rerank_head: "GRMCandidateRerankHead",
    ) -> GRMCandidateRerankResult:
        if mode not in self._plugins:
            raise ValueError(
                f"Unknown candidate rerank mode {mode!r}; "
                f"available modes are {self.available_modes}."
            )
        _validate_context(context)
        result = self._plugins[mode](
            context,
            geometry_graph=geometry_graph,
            rerank_head=rerank_head,
            eps=self.eps,
        )
        _validate_result(result, context)
        return result


def _masked_conditional_distribution(
    scores: Tensor,
    valid_mask: Tensor,
    *,
    fallback: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Normalize nonnegative scores and retain a finite fallback distribution."""

    finite_scores = torch.nan_to_num(
        scores.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    finite_scores = finite_scores * valid_mask.float()
    score_sum = finite_scores.sum(dim=-1, keepdim=True)

    fallback_scores = torch.nan_to_num(
        fallback.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    fallback_scores = fallback_scores * valid_mask.float()
    fallback_sum = fallback_scores.sum(dim=-1, keepdim=True)
    uniform = valid_mask.float() / valid_mask.float().sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    fallback_probabilities = torch.where(
        fallback_sum > eps,
        fallback_scores / fallback_sum.clamp_min(eps),
        uniform,
    )
    probabilities = torch.where(
        score_sum > eps,
        finite_scores / score_sum.clamp_min(eps),
        fallback_probabilities,
    )
    probabilities = probabilities * valid_mask.float()
    logits = torch.log(finite_scores.clamp_min(eps))
    logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    return probabilities, logits


def _validate_context(context: GRMCandidateRerankContext) -> None:
    tokens = context.candidate_tokens
    if tokens.ndim != 4:
        raise ValueError("candidate_tokens must be [B, M, K, C].")
    if context.candidate_prior.shape != tokens.shape[:3]:
        raise ValueError("candidate_prior must be [B, M, K].")
    if context.candidate_reliability.shape != tokens.shape[:3]:
        raise ValueError("candidate_reliability must be [B, M, K].")
    if context.candidate_valid_mask.shape != tokens.shape[:3]:
        raise ValueError("candidate_valid_mask must be [B, M, K].")
    if context.candidate_valid_mask.dtype != torch.bool:
        raise TypeError("candidate_valid_mask must use torch.bool.")
    batch, queries, topk, _ = tokens.shape
    flat_count = queries * topk
    if context.flat_src_xy.shape != (batch, flat_count, 2):
        raise ValueError("flat_src_xy must be [B, M*K, 2].")
    if context.flat_dst_xy.shape != (batch, flat_count, 2):
        raise ValueError("flat_dst_xy must be [B, M*K, 2].")
    if context.flat_seed_error.shape != (batch, flat_count):
        raise ValueError("flat_seed_error must be [B, M*K].")


def _validate_result(
    result: GRMCandidateRerankResult,
    context: GRMCandidateRerankContext,
) -> None:
    candidate_shape = context.candidate_prior.shape
    query_shape = candidate_shape[:2]
    if result.logits.shape != candidate_shape:
        raise ValueError("Rerank plugin logits must be [B, M, K].")
    if result.probabilities.shape != candidate_shape:
        raise ValueError("Rerank plugin probabilities must be [B, M, K].")
    if result.selected_indices.shape != query_shape:
        raise ValueError("Rerank plugin selected_indices must be [B, M].")
    if result.candidate_tokens.shape != context.candidate_tokens.shape:
        raise ValueError("Rerank plugin candidate_tokens must preserve [B, M, K, C].")
    if result.selected_probability_multiplier.shape != query_shape:
        raise ValueError("selected_probability_multiplier must be [B, M].")
    if result.selected_indices.dtype != torch.long:
        raise TypeError("selected_indices must use torch.long.")
    if torch.any(result.probabilities < 0.0) or torch.any(
        result.probabilities > 1.0
    ):
        raise ValueError("Rerank probabilities must lie in [0, 1].")
