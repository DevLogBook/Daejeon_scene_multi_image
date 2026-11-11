"""Cycle and A-to-B seed geometry tests (graph/role tests are added in stage 7)."""

from __future__ import annotations

import torch

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.cycle_consistency import compute_cycle_consistency
from grm_matcher.seed_geometry import GRMSeedHomographyEstimator, apply_homography
from grm_matcher.role_head import GRMCandidateRerankHead, GRMGeometryRoleHead
from grm_matcher.role_targets import build_geometry_role_targets


def _grid_points() -> torch.Tensor:
    return torch.tensor(
        [[[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [0.0, 20.0],
          [20.0, 20.0], [40.0, 20.0], [0.0, 40.0], [40.0, 40.0]]]
    )


def test_cycle_consistency_uses_b_to_a_direction() -> None:
    src = _grid_points()
    translation = torch.tensor([5.0, -3.0])
    dst = src + translation
    backward_src = dst.flip(1)
    backward_dst = src.flip(1)
    result = compute_cycle_consistency(
        src, dst, backward_src, backward_dst, sigma=4.0
    )
    torch.testing.assert_close(result.cycle_error, torch.zeros_like(result.cycle_error))
    torch.testing.assert_close(result.cycle_score, torch.ones_like(result.cycle_score))


def test_apply_homography_direction_is_strictly_a_to_b() -> None:
    src = _grid_points()
    h_ab = torch.tensor([[[1.0, 0.0, 7.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]]])
    dst = apply_homography(src, h_ab)
    torch.testing.assert_close(dst, src + torch.tensor([7.0, -4.0]))


def test_seed_homography_recovers_a_to_b_transform_and_residuals() -> None:
    config = GRMMatcherConfig(input_height=64, input_width=64)
    estimator = GRMSeedHomographyEstimator(config)
    src = _grid_points()
    h_ab = torch.tensor([[[1.02, 0.03, 4.0], [-0.01, 0.98, 6.0], [0.0002, -0.0001, 1.0]]])
    dst = apply_homography(src, h_ab)
    shape = src.shape[:2]
    result = estimator(
        src,
        dst,
        p_match=torch.full(shape, 0.9),
        candidate_entropy=torch.full(shape, 0.1),
        cycle_error=torch.zeros(shape),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        mutual_top1=torch.ones(shape, dtype=torch.bool),
    )
    assert result.homography_valid.item()
    torch.testing.assert_close(
        result.homography_ab / result.homography_ab[:, 2:3, 2:3],
        h_ab,
        atol=2.0e-3,
        rtol=2.0e-3,
    )
    assert result.symmetric_transfer_error.max() < 0.05
    assert result.failure_reasons == ("",)


def test_seed_homography_fewer_than_four_returns_reported_identity_fallback() -> None:
    config = GRMMatcherConfig(input_height=64, input_width=64)
    estimator = GRMSeedHomographyEstimator(config)
    src = _grid_points()[:, :3]
    shape = src.shape[:2]
    result = estimator(
        src,
        src,
        torch.ones(shape),
        torch.zeros(shape),
        torch.zeros(shape),
        torch.ones(shape, dtype=torch.bool),
    )
    assert not result.homography_valid.item()
    torch.testing.assert_close(result.homography_ab, torch.eye(3).unsqueeze(0))
    assert "need 4" in result.failure_reasons[0]


def test_detached_seed_h_does_not_block_point_residual_gradients() -> None:
    config = GRMMatcherConfig(input_height=64, input_width=64, detach_seed_homography=True)
    estimator = GRMSeedHomographyEstimator(config)
    src = _grid_points().requires_grad_(True)
    dst = (_grid_points() + torch.tensor([3.0, 2.0])).requires_grad_(True)
    shape = src.shape[:2]
    result = estimator(
        src,
        dst,
        torch.ones(shape),
        torch.zeros(shape),
        torch.zeros(shape),
        torch.ones(shape, dtype=torch.bool),
    )
    assert not result.homography_ab.requires_grad
    result.symmetric_transfer_error.sum().backward()
    assert src.grad is not None and torch.isfinite(src.grad).all()
    assert dst.grad is not None and torch.isfinite(dst.grad).all()


def test_rerank_and_geometry_role_probability_contracts() -> None:
    config = GRMMatcherConfig(input_height=32, input_width=32)
    rerank = GRMCandidateRerankHead(config)
    role = GRMGeometryRoleHead(config)
    graph_tokens = torch.randn(2, 3, 4, config.graph_hidden_dim)
    prior = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    candidate_valid = torch.ones(2, 3, 4, dtype=torch.bool)
    candidate_valid[0, 2] = False
    reranked = rerank(graph_tokens, prior, candidate_valid)
    assert reranked.logits.shape == (2, 3, 4)
    torch.testing.assert_close(
        reranked.probabilities[1].sum(dim=-1), torch.ones(3)
    )
    assert torch.count_nonzero(reranked.probabilities[0, 2]) == 0

    selected = graph_tokens[:, :, 0]
    query_valid = candidate_valid.any(dim=-1)
    roles = role(selected, query_valid)
    assert roles.probabilities.shape == (2, 3, 3)
    torch.testing.assert_close(
        roles.probabilities.sum(dim=-1), torch.ones(2, 3)
    )
    torch.testing.assert_close(
        roles.probabilities[0, 2], torch.tensor([0.0, 0.0, 1.0])
    )


def test_role_correctness_is_independent_from_homography_residual_band() -> None:
    height = width = 32
    yy, xx = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )
    dense = torch.stack((xx, yy), dim=0).float().unsqueeze(0)
    global_mask = torch.zeros(1, 1, height, width, dtype=torch.bool)
    local_mask = torch.zeros_like(global_mask)
    global_mask[..., 8, 8] = True
    local_mask[..., 16, 16] = True
    overlap = global_mask | local_mask
    src = torch.tensor([[[8.0, 8.0], [16.0, 16.0], [16.0, 16.0]]])
    # 6 px is inside the old 24 px Local residual band but outside the exact
    # correspondence threshold, so the first prediction must be False.
    dst = torch.tensor([[[14.0, 8.0], [20.0, 16.0], [22.0, 16.0]]])
    targets = {
        "gt_dense_mapping": dense,
        "gt_overlap_a": overlap,
        "gt_global_mask": global_mask,
        "gt_local_mask": local_mask,
    }

    result = build_geometry_role_targets(
        src,
        dst,
        targets,
        correspondence_correct_threshold=5.0,
    )

    assert result.labels.tolist() == [[2, 1, 2]]
    assert result.correct_mask.tolist() == [[False, True, False]]
    torch.testing.assert_close(
        result.correspondence_error,
        torch.tensor([[6.0, 4.0, 6.0]]),
    )
