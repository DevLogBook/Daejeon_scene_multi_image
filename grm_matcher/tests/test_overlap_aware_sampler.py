"""Low-support sampling contracts; these fixtures do not measure model accuracy."""
import pytest
import torch

from grm_matcher.config import GRMCandidateSamplerConfig, GRMMatcherConfig
from grm_matcher.candidate_sampler import GRMTopKCandidateSampler


def sampler(**overrides):
    config = dict(sampling_mode="overlap_aware", max_source_queries=32,
                  use_dynamic_query_budget=False, fixed_score_threshold=0.01)
    config.update(overrides)
    return GRMTopKCandidateSampler(GRMCandidateSamplerConfig(**config))


def diagonal(scores):
    return torch.diag_embed(scores.unsqueeze(0))


def test_narrow_strip_no_background_fill_and_no_region_cap():
    scores = torch.full((64,), 0.03)
    strip = torch.arange(64) % 8 == 0
    scores[strip] = 0.8
    probability = diagonal(scores)
    new = sampler(max_quota_upper_bound=2, max_quota_lower_bound=1)(
        probability, torch.ones(1, 64), 8, 8)
    old = sampler(sampling_mode="adaptive_density")(
        probability, torch.ones(1, 64), 8, 8)
    kept = new.query_indices[0, new.valid_mask[0]]
    assert len(kept) == 8
    assert torch.all(strip[kept])
    assert new.actual_budget.item() == 8
    assert old.valid_mask.sum() > 8
    assert not torch.all(strip[old.query_indices[0, old.valid_mask[0]]])
    assert new.region_quota.max() > 2  # Reliable narrow support can exceed old caps.
    assert torch.all(new.candidate_global_probs[~new.valid_mask] == 0)
    assert torch.all(new.region_ids[~new.valid_mask] == -1)


def test_empty_and_tiny_support_never_promoted_to_minimum_count():
    for n in (0, 2):
        scores = torch.zeros(16)
        scores[:n] = 0.8
        result = sampler(minimum_query_count=16, allow_underfilled_budget=False)(
            diagonal(scores), torch.ones(1, 16), 4, 4)
        assert result.valid_mask.sum() == n
        assert result.low_quality_sample.item()


def test_reciprocity_rejects_many_to_one_even_with_tiny_reciprocal_alternative():
    probability = torch.eye(8).unsqueeze(0) * 1e-5
    probability[0, :, 0] = torch.linspace(0.8, 0.5, 8)
    result = sampler()(probability, torch.ones(1, 8), 2, 4)
    selected = result.query_indices[0, result.valid_mask[0]]
    assert set(selected.tolist()) == {0, 1, 2, 3}


def test_matchability_masks_and_nonfinite_values_are_not_backfilled():
    probability = diagonal(torch.full((16,), 0.8))
    probability[0, 1, 1] = float('nan')
    matchability = torch.ones(1, 16)
    matchability[0, 2] = float('nan')
    matchability[0, 3] = 0.1
    source = torch.ones(1, 16, dtype=torch.bool); source[0, 4] = False
    target = torch.ones(1, 16, dtype=torch.bool); target[0, 5] = False
    result = sampler()(probability, matchability, 4, 4, source, target)
    selected = set(result.query_indices[0, result.valid_mask[0]].tolist())
    assert selected == set(range(16)) - {1, 2, 3, 4, 5}
    assert torch.isfinite(result.query_scores).all()


def test_low_quality_budget_has_no_coverage_floor():
    result = sampler(use_dynamic_query_budget=True, dynamic_budget_target_mean_score=0.5)(
        diagonal(torch.full((64,), 0.02)), torch.ones(1, 64), 8, 8)
    assert result.actual_budget.item() == 2
    assert result.valid_mask.sum() == 2


def test_support_local_diversity_is_deterministic_and_batch_independent():
    probability = diagonal(torch.full((64,), 0.8)).repeat(2, 1, 1)
    probability[1] = 0
    module = sampler(max_source_queries=4)
    a = module(probability, torch.ones(2, 64), 8, 8)
    b = module(probability[:1], torch.ones(1, 64), 8, 8)
    assert torch.equal(a.query_indices[0], b.query_indices[0])
    assert a.valid_mask.sum(dim=1).tolist() == [4, 0]
    assert a.query_indices[0, :2].tolist() == [0, 63]
    assert len(set(a.query_indices[0].tolist())) == 4


def test_legacy_checkpoint_config_and_new_config_roundtrip():
    original = GRMMatcherConfig().to_dict()
    for name in ('sampling_mode', 'support_min_matchability', 'support_relative_score'):
        original.pop(name)
    legacy = GRMMatcherConfig.from_dict(original)
    assert legacy.sampling_mode == 'adaptive_density'
    new = GRMMatcherConfig(sampling_mode='overlap_aware', support_relative_score=0.25)
    restored = GRMMatcherConfig.from_dict(new.to_dict())
    assert restored.candidate_sampler_config().support_relative_score == 0.25
    with pytest.raises(ValueError):
        GRMMatcherConfig(support_relative_score=float('nan'))
    with pytest.raises(ValueError):
        GRMCandidateSamplerConfig(sampling_mode='unknown')
