"""Pluggable geometry/rerank ablation and compatibility tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from grm_matcher.candidate_reranker import (
    GRMCandidateRerankContext,
    GRMCandidateRerankResult,
    GRMCandidateRerankRouter,
)
from grm_matcher.config import (
    GRM_BUILTIN_CANDIDATE_RERANK_MODES,
    GRMMatcherConfig,
)
from grm_matcher.matcher import GRMMatcher
from grm_matcher.role_head import GRMCandidateRerankHead


class _OffsetGraph(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        tokens: Tensor,
        src_xy: Tensor,
        dst_xy: Tensor,
        seed_error: Tensor,
        valid_mask: Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> Tensor:
        del src_xy, dst_xy, seed_error, valid_mask, image_height, image_width
        self.calls += 1
        return tokens + 0.25


def _context() -> GRMCandidateRerankContext:
    torch.manual_seed(4)
    batch, queries, topk, hidden = 2, 3, 4, 16
    valid = torch.ones(batch, queries, topk, dtype=torch.bool)
    valid[0, 2] = False
    prior = torch.tensor(
        [
            [[0.6, 0.2, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1], [0.0, 0.0, 0.0, 0.0]],
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]],
        ],
        dtype=torch.float32,
    )
    reliability = torch.tensor(
        [
            [[0.1, 0.7, 0.1, 0.1], [0.8, 0.1, 0.05, 0.05], [0.0, 0.0, 0.0, 0.0]],
            [[0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1], [0.2, 0.3, 0.1, 0.4]],
        ],
        dtype=torch.float32,
    )
    flat_count = queries * topk
    return GRMCandidateRerankContext(
        candidate_tokens=torch.randn(batch, queries, topk, hidden),
        candidate_prior=prior,
        candidate_reliability=reliability,
        candidate_valid_mask=valid,
        flat_src_xy=torch.randn(batch, flat_count, 2),
        flat_dst_xy=torch.randn(batch, flat_count, 2),
        flat_seed_error=torch.rand(batch, flat_count),
        image_height=64,
        image_width=64,
    )


def _run(mode: str) -> tuple[GRMCandidateRerankResult, _OffsetGraph]:
    config = GRMMatcherConfig(
        input_height=32,
        input_width=32,
        graph_hidden_dim=16,
        graph_heads=4,
    )
    router = GRMCandidateRerankRouter(config.probability_eps)
    graph = _OffsetGraph()
    result = router(
        mode,
        _context(),
        geometry_graph=graph,  # type: ignore[arg-type]
        rerank_head=GRMCandidateRerankHead(config),
    )
    return result, graph


@pytest.mark.parametrize("mode", GRM_BUILTIN_CANDIDATE_RERANK_MODES)
def test_all_builtin_rerank_modes_preserve_probability_contract(mode: str) -> None:
    result, _ = _run(mode)
    context = _context()
    valid_queries = context.candidate_valid_mask.any(dim=-1)
    torch.testing.assert_close(
        result.probabilities.sum(dim=-1)[valid_queries],
        torch.ones_like(result.probabilities.sum(dim=-1)[valid_queries]),
    )
    assert torch.count_nonzero(result.probabilities[~valid_queries]) == 0
    assert result.selected_indices.shape == valid_queries.shape
    assert result.selected_indices.dtype == torch.long


def test_graph_and_learned_head_can_be_ablated_independently() -> None:
    full, full_graph = _run("geometry_graph")
    graph_reliability, reliability_graph = _run("graph_reliability")
    token_mlp, token_graph = _run("token_mlp")
    reliability, bypass_graph = _run("reliability")

    assert full_graph.calls == 1
    assert reliability_graph.calls == 1
    assert token_graph.calls == 0
    assert bypass_graph.calls == 0
    assert full.used_learned_rerank
    assert not graph_reliability.used_learned_rerank
    assert token_mlp.used_learned_rerank
    assert not reliability.used_learned_rerank


def test_reliability_and_coarse_prior_select_different_expected_candidates() -> None:
    reliability, _ = _run("reliability")
    coarse, _ = _run("coarse_prior")
    assert reliability.selected_indices[0, :2].tolist() == [1, 0]
    assert coarse.selected_indices[0, :2].tolist() == [0, 2]
    # Deterministic selection must not multiply p_match by the same normalized
    # reliability a second time.
    torch.testing.assert_close(
        reliability.selected_probability_multiplier[0, :2],
        torch.ones(2),
    )


def test_zero_reliability_uses_finite_prior_fallback() -> None:
    context = _context()
    context = replace(
        context,
        candidate_reliability=torch.zeros_like(context.candidate_reliability),
    )
    config = GRMMatcherConfig(
        input_height=32,
        input_width=32,
        graph_hidden_dim=16,
        graph_heads=4,
    )
    result = GRMCandidateRerankRouter(config.probability_eps)(
        "reliability",
        context,
        geometry_graph=_OffsetGraph(),  # type: ignore[arg-type]
        rerank_head=GRMCandidateRerankHead(config),
    )
    assert torch.isfinite(result.probabilities).all()
    assert result.selected_indices[0, :2].tolist() == [0, 2]


def test_router_adds_no_checkpoint_keys_and_config_defaults_to_legacy_mode() -> None:
    config = GRMMatcherConfig(
        input_height=32,
        input_width=32,
        backbone_stem_dim=8,
        backbone_stage_blocks=(1, 1, 1),
        coarse_dim=32,
        mid_dim=32,
        fine_dim=32,
        transformer_heads=4,
        transformer_ffn_dim=64,
        max_source_queries=4,
        graph_hidden_dim=32,
        graph_heads=4,
        graph_neighbors=2,
        role_appearance_dim=8,
    )
    model = GRMMatcher(config)
    assert config.candidate_rerank_mode == "geometry_graph"
    assert not any(
        key.startswith("candidate_reranker.") for key in model.state_dict()
    )
    clone = GRMMatcher(GRMMatcherConfig.from_dict(config.to_dict()))
    clone.load_state_dict(model.state_dict(), strict=True)
    assert clone.available_candidate_rerank_modes == GRM_BUILTIN_CANDIDATE_RERANK_MODES


def test_config_rejects_unknown_rerank_mode() -> None:
    with pytest.raises(ValueError, match="candidate_rerank_mode"):
        GRMMatcherConfig(
            input_height=32,
            input_width=32,
            candidate_rerank_mode="unknown",  # type: ignore[arg-type]
        )
