"""Adaptive reliability-density candidate sampling tests."""

from __future__ import annotations

import math

import pytest
import torch

from grm_matcher.candidate_sampler import (
    GRMCandidateSampler,
    GRMTopKCandidateSampler,
)
from grm_matcher.config import GRMCandidateSamplerConfig, GRMMatcherConfig


def _config(**overrides: object) -> GRMCandidateSamplerConfig:
    values = {
        "max_source_queries": 16,
        "topk_candidates": 4,
        "region_height_cells": 4,
        "region_width_cells": 4,
        "fixed_score_threshold": 1.0e-4,
    }
    values.update(overrides)
    return GRMCandidateSamplerConfig(**values)


def _confident_probabilities(
    top1: torch.Tensor,
    *,
    ambiguous_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create [B, S, 4] rows with controlled full-matrix top-1 mass."""

    if top1.ndim == 1:
        top1 = top1.unsqueeze(0)
    probability = torch.stack(
        (top1, 0.03 * top1, 0.02 * top1, 0.01 * top1), dim=-1
    )
    if ambiguous_mask is not None:
        if ambiguous_mask.ndim == 1:
            ambiguous_mask = ambiguous_mask.unsqueeze(0)
        ambiguous_rows = top1[..., None].expand_as(probability)
        probability = torch.where(
            ambiguous_mask[..., None], ambiguous_rows, probability
        )
    return probability


def _region_mask(height: int, width: int, region_id: int) -> torch.Tensor:
    grid_cols = math.ceil(width / 4)
    y = torch.arange(height)[:, None]
    x = torch.arange(width)[None, :]
    ids = torch.div(y, 4, rounding_mode="floor") * grid_cols + torch.div(
        x, 4, rounding_mode="floor"
    )
    return ids.reshape(-1) == region_id


def test_topk_distinguishes_global_and_conditional_probabilities() -> None:
    sampler = GRMTopKCandidateSampler(_config(max_source_queries=4))
    probability = torch.tensor([[[0.4, 0.2, 0.1, 0.05, 0.01]]])
    result = sampler.topk_candidates(probability)
    torch.testing.assert_close(
        result.global_probs[0, 0], torch.tensor([0.4, 0.2, 0.1, 0.05])
    )
    torch.testing.assert_close(result.conditional_probs.sum(dim=-1), torch.ones(1, 1))
    assert result.entropy.item() > 0.0
    torch.testing.assert_close(
        result.top1_top2_margin, torch.tensor([[(0.4 - 0.2) / 0.75]])
    )
    torch.testing.assert_close(
        result.global_top1_top2_margin, torch.tensor([[0.2]])
    )


def test_topk_pads_when_target_has_fewer_than_k_cells() -> None:
    sampler = GRMTopKCandidateSampler(_config(max_source_queries=4))
    result = sampler.topk_candidates(torch.tensor([[[0.8, 0.2]]]))
    assert result.indices.shape == (1, 1, 4)
    torch.testing.assert_close(result.global_probs[0, 0, 2:], torch.zeros(2))
    torch.testing.assert_close(result.conditional_probs.sum(dim=-1), torch.ones(1, 1))


def test_uniform_scores_give_uniform_quota_and_full_coverage() -> None:
    sampler = GRMCandidateSampler(_config())
    probability = _confident_probabilities(torch.full((64,), 0.5))
    result = sampler(probability, torch.ones(1, 64), 8, 8)

    assert result.valid_mask.sum().item() == 16
    assert result.region_quota[0].tolist() == [4, 4, 4, 4]
    assert result.diagnostics["region_coverage_ratio"].item() == pytest.approx(1.0)
    assert result.diagnostics["selected_region_gini"].item() == pytest.approx(0.0)


def test_single_dense_region_gets_more_but_is_capped() -> None:
    sampler = GRMCandidateSampler(_config())
    score = torch.full((64,), 0.1)
    score[_region_mask(8, 8, 0)] = 1.0
    result = sampler(_confident_probabilities(score), torch.ones(1, 64), 8, 8)

    quota = result.region_quota[0]
    maximum = result.diagnostics["region_max_quota"][0]
    assert quota[0] > quota[1:].max()
    assert quota[0] <= maximum[0]
    assert torch.all(quota[1:] >= 1)
    assert result.diagnostics["region_coverage_ratio"].item() == pytest.approx(1.0)


def test_ambiguous_dense_region_does_not_inflate_reliable_density() -> None:
    sampler = GRMCandidateSampler(_config())
    top1 = torch.full((64,), 0.2)
    ambiguous = _region_mask(8, 8, 0)
    top1[ambiguous] = 1.0
    result = sampler(
        _confident_probabilities(top1, ambiguous_mask=ambiguous),
        torch.ones(1, 64),
        8,
        8,
    )

    assert result.region_density[0, 0].item() == pytest.approx(0.0, abs=1.0e-6)
    assert result.region_valid_count[0, 0].item() == 0
    assert result.region_quota[0, 0].item() == 0
    assert torch.all(result.region_quota[0, 1:] > 0)


def test_sparse_valid_regions_receive_budget_without_invalid_region_quota() -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=12))
    source_valid = _region_mask(8, 8, 0) | _region_mask(8, 8, 3)
    result = sampler(
        _confident_probabilities(torch.full((64,), 0.4)),
        torch.ones(1, 64),
        8,
        8,
        source_valid_mask=source_valid.unsqueeze(0),
    )

    assert result.valid_mask.sum().item() == 12
    assert result.region_quota[0, 1].item() == 0
    assert result.region_quota[0, 2].item() == 0
    assert result.region_quota[0].sum().item() == 12


def test_budget_is_conserved_when_capacity_is_sufficient() -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=13))
    score = torch.linspace(0.1, 0.9, 64)
    result = sampler(_confident_probabilities(score), torch.ones(1, 64), 8, 8)

    assert result.valid_mask.sum().item() == 13
    assert result.region_quota.sum().item() == 13
    assert result.region_quota.sum().item() <= sampler.max_queries
    assert result.diagnostics["selected_count"].item() == 13


def test_quantile_threshold_really_underfills_instead_of_restoring_all() -> None:
    sampler = GRMCandidateSampler(
        _config(
            max_source_queries=100,
            minimum_query_count=16,
            score_quantile=0.5,
            use_dynamic_query_budget=False,
        )
    )
    score = torch.linspace(0.01, 1.0, 100)
    result = sampler(
        _confident_probabilities(score), torch.ones(1, 100), 10, 10
    )

    assert result.diagnostics["high_quality_count"].item() == 50
    assert result.actual_budget.item() == 50
    assert result.valid_mask.sum().item() == 50


def test_quantile_fallback_only_restores_minimum_query_count() -> None:
    sampler = GRMCandidateSampler(
        _config(
            max_source_queries=100,
            minimum_query_count=16,
            score_quantile=0.95,
            use_dynamic_query_budget=False,
        )
    )
    score = torch.linspace(0.01, 1.0, 100)
    result = sampler(
        _confident_probabilities(score), torch.ones(1, 100), 10, 10
    )

    assert result.diagnostics["high_quality_count"].item() == 16
    assert result.valid_mask.sum().item() == 16


def test_disabling_underfill_restores_previous_fill_behavior() -> None:
    sampler = GRMCandidateSampler(
        _config(
            max_source_queries=100,
            allow_underfilled_budget=False,
            score_quantile=0.5,
            use_dynamic_query_budget=False,
        )
    )
    score = torch.linspace(0.01, 1.0, 100)
    result = sampler(
        _confident_probabilities(score), torch.ones(1, 100), 10, 10
    )

    assert result.diagnostics["high_quality_count"].item() == 100
    assert result.valid_mask.sum().item() == 100


def test_dynamic_budget_reduces_weak_texture_support() -> None:
    config = _config(
        max_source_queries=256,
        minimum_query_count=16,
        dynamic_budget_target_mean_score=0.1,
    )
    sampler = GRMCandidateSampler(config)
    weak = sampler(
        _confident_probabilities(torch.full((256,), 0.02)),
        torch.ones(1, 256),
        16,
        16,
    )
    strong = sampler(
        _confident_probabilities(torch.full((256,), 0.5)),
        torch.ones(1, 256),
        16,
        16,
    )

    assert 16 <= weak.actual_budget.item() < 256
    assert weak.valid_mask.sum().item() == weak.actual_budget.item()
    assert strong.actual_budget.item() == 256
    assert weak.diagnostics["mean_valid_score"] < strong.diagnostics["mean_valid_score"]


def test_insufficient_geometry_support_sets_low_quality_flag() -> None:
    sampler = GRMCandidateSampler(
        _config(max_source_queries=64, min_geometry_queries=16)
    )
    source_valid = torch.zeros(64, dtype=torch.bool)
    source_valid[:10] = True
    result = sampler(
        _confident_probabilities(torch.full((64,), 0.5)),
        torch.ones(1, 64),
        8,
        8,
        source_valid_mask=source_valid.unsqueeze(0),
    )

    assert result.actual_budget.item() == 10
    assert result.low_quality_sample.item()
    assert not result.diagnostics["geometry_ready"].item()


@pytest.mark.parametrize(
    ("height", "width", "expected_rows", "expected_cols"),
    [(32, 32, 8, 8), (40, 40, 10, 10), (48, 64, 12, 16), (64, 64, 16, 16)],
)
def test_dynamic_grid_uses_region_cell_size(
    height: int, width: int, expected_rows: int, expected_cols: int
) -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=8))
    source_count = height * width
    result = sampler(
        _confident_probabilities(torch.full((source_count,), 0.5)),
        torch.ones(1, source_count),
        height,
        width,
    )

    assert result.region_density.shape == (1, expected_rows * expected_cols)
    assert result.diagnostics["grid_rows"].item() == expected_rows
    assert result.diagnostics["grid_cols"].item() == expected_cols
    assert result.coverage_budget_clipped.item()


def test_partial_border_regions_are_counted_without_padding_assumptions() -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=8))
    result = sampler(
        _confident_probabilities(torch.full((110,), 0.5)),
        torch.ones(1, 110),
        10,
        11,
    )

    assert result.region_density.shape == (1, 9)
    assert result.region_valid_count[0].tolist() == [16, 16, 12, 16, 16, 12, 8, 8, 6]


def test_dynamic_nms_radius_spreads_equal_score_points() -> None:
    sampler = GRMCandidateSampler(
        _config(max_source_queries=2, use_quantile_threshold=False)
    )
    result = sampler(
        _confident_probabilities(torch.full((16,), 0.5)),
        torch.ones(1, 16),
        4,
        4,
    )

    points = result.source_xy_coarse[0, result.valid_mask[0]]
    expected_radius = 0.5 * math.sqrt(16 / 2)
    assert torch.pdist(points).min().item() >= expected_radius


def test_batch_allocations_are_independent() -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=12))
    score = torch.full((2, 64), 0.1)
    score[0, _region_mask(8, 8, 0)] = 0.9
    score[1, _region_mask(8, 8, 3)] = 0.9
    source_valid = torch.ones(2, 64, dtype=torch.bool)
    source_valid[1, _region_mask(8, 8, 0)] = False
    result = sampler(
        _confident_probabilities(score),
        torch.ones(2, 64),
        8,
        8,
        source_valid_mask=source_valid,
    )

    assert result.valid_mask.sum(dim=1).tolist() == [12, 12]
    assert result.region_quota[0].argmax().item() == 0
    assert result.region_quota[1].argmax().item() == 3
    assert result.region_valid_count[1, 0].item() == 0


def test_empty_candidates_return_finite_canonical_padding() -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=7))
    result = sampler(torch.zeros(2, 16, 4), torch.zeros(2, 16), 4, 4)

    assert result.query_indices.shape == (2, 7)
    assert result.candidate_probs.shape == (2, 7, 4)
    assert not result.valid_mask.any()
    assert torch.count_nonzero(result.candidate_probs) == 0
    assert torch.count_nonzero(result.region_quota) == 0
    assert all(
        torch.isfinite(value).all()
        for value in result.diagnostics.values()
        if value.is_floating_point()
    )


def test_nonfinite_probability_entries_are_safely_excluded() -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=4))
    probability = _confident_probabilities(torch.full((16,), 0.5))
    probability[0, 0, 0] = torch.nan
    probability[0, 1, 0] = torch.inf
    result = sampler(probability, torch.ones(1, 16), 4, 4)

    for value in (
        result.candidate_global_probs,
        result.candidate_probs,
        result.query_scores,
        result.region_density,
    ):
        assert torch.isfinite(value).all()


def test_sampler_is_deterministic() -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=12))
    score = torch.linspace(0.1, 0.9, 64)
    args = (_confident_probabilities(score), torch.ones(1, 64), 8, 8)
    first = sampler(*args)
    second = sampler(*args)

    torch.testing.assert_close(first.query_indices, second.query_indices)
    torch.testing.assert_close(first.region_quota, second.region_quota)
    torch.testing.assert_close(first.query_scores, second.query_scores)


def test_density_schedule_uses_explicit_global_step() -> None:
    sampler = GRMCandidateSampler(
        _config(
            max_source_queries=4,
            use_density_schedule=True,
            density_power_warmup=0.3,
            density_power=0.5,
            density_power_max=0.75,
            density_schedule_warmup_steps=5_000,
        )
    )
    probability = _confident_probabilities(torch.full((16,), 0.5))
    powers = [
        sampler(probability, torch.ones(1, 16), 4, 4, global_step=step)
        .effective_density_power.item()
        for step in (0, 2_500, 5_000)
    ]
    assert powers == pytest.approx([0.3, 0.4, 0.5])


def test_clipped_coverage_never_exceeds_budget() -> None:
    sampler = GRMCandidateSampler(
        _config(
            max_source_queries=4,
            region_height_cells=2,
            region_width_cells=2,
        )
    )
    result = sampler(
        _confident_probabilities(torch.full((64,), 0.5)),
        torch.ones(1, 64),
        8,
        8,
    )

    assert result.coverage_budget_clipped.item()
    assert result.valid_mask.sum().item() == 4
    assert result.region_quota.sum().item() == 4
    assert result.diagnostics["covered_region_count"].item() == 4
    assert result.diagnostics["region_coverage_ratio"].item() == pytest.approx(0.25)
    assert result.diagnostics[
        "achievable_region_coverage_ratio"
    ].item() == pytest.approx(1.0)
    assert result.diagnostics["max_coverable_region_count"].item() == 4


def test_pairwise_distance_diagnostics_are_diagonal_normalized() -> None:
    sampler = GRMCandidateSampler(_config(max_source_queries=8))
    result = sampler(
        _confident_probabilities(torch.full((64,), 0.5)),
        torch.ones(1, 64),
        8,
        8,
    )
    diagnostics = result.diagnostics
    diagonal = math.hypot(7, 7)

    assert diagnostics["mean_selected_pairwise_distance"].item() == pytest.approx(
        diagnostics["mean_selected_pairwise_distance_cells"].item() / diagonal
    )
    assert diagnostics["median_selected_pairwise_distance"].item() == pytest.approx(
        diagnostics["median_selected_pairwise_distance_cells"].item() / diagonal
    )
    assert 0.0 <= diagnostics["mean_selected_pairwise_distance"].item() <= 1.0


def test_new_sampler_adapts_more_than_legacy_equal_quota() -> None:
    common = {
        "input_height": 32,
        "input_width": 32,
        "max_source_queries": 16,
        "sampling_grid_rows": 2,
        "sampling_grid_cols": 2,
        "fixed_score_threshold": 1.0e-4,
    }
    adaptive = GRMCandidateSampler(GRMMatcherConfig(**common))
    legacy = GRMCandidateSampler(
        GRMMatcherConfig(**common, use_adaptive_density_sampling=False)
    )
    score = torch.full((64,), 0.1)
    score[_region_mask(8, 8, 0)] = 1.0
    probability = _confident_probabilities(score)
    adaptive_result = adaptive(probability, torch.ones(1, 64), 8, 8)
    legacy_result = legacy(probability, torch.ones(1, 64), 8, 8)

    assert legacy_result.region_quota[0].tolist() == [4, 4, 4, 4]
    assert adaptive_result.region_quota[0, 0] > legacy_result.region_quota[0, 0]
    assert adaptive_result.region_quota.sum() == legacy_result.region_quota.sum()


def test_sampler_config_rejects_invalid_weights_and_warmup() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        GRMCandidateSamplerConfig(entropy_weight=0.0, margin_weight=0.0)
    with pytest.raises(ValueError, match="effective target"):
        GRMCandidateSamplerConfig(
            density_power=0.2,
            density_power_warmup=0.3,
            use_density_schedule=True,
        )
