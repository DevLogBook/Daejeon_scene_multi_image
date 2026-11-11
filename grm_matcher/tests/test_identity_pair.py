"""Deterministic identity logic test independent of untrained image features."""

from __future__ import annotations

import torch

from grm_matcher.candidate_sampler import GRMTopKCandidateSampler
from grm_matcher.config import GRMMatcherConfig
from grm_matcher.coordinates import flat_indices_to_image_xy
from grm_matcher.cycle_consistency import compute_cycle_consistency
from grm_matcher.seed_geometry import GRMSeedHomographyEstimator


def test_deterministic_identity_candidates_cycle_and_seed_h() -> None:
    config = GRMMatcherConfig(
        input_height=64,
        input_width=64,
        max_source_queries=16,
        sampling_grid_rows=4,
        sampling_grid_cols=4,
    )
    sampler = GRMTopKCandidateSampler(config)
    probability = torch.eye(16).unsqueeze(0) * 0.9
    sampled = sampler(
        probability,
        torch.ones(1, 16),
        source_height=4,
        source_width=4,
    )
    src = flat_indices_to_image_xy(sampled.query_indices, 4, 16)
    dst = flat_indices_to_image_xy(sampled.candidate_indices[..., 0], 4, 16)
    torch.testing.assert_close(src, dst)
    cycle = compute_cycle_consistency(src, dst, src, dst, sigma=4.0)
    assert cycle.cycle_error.max() == 0.0
    estimator = GRMSeedHomographyEstimator(config)
    result = estimator(
        src,
        dst,
        torch.full((1, 16), 0.9),
        sampled.candidate_entropy,
        cycle.cycle_error,
        sampled.valid_mask,
    )
    assert result.homography_valid.item()
    torch.testing.assert_close(result.homography_ab, torch.eye(3).unsqueeze(0), atol=1.0e-4, rtol=0.0)

