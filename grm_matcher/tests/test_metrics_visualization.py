"""Metric correctness and headless visualization smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch

from grm_matcher.metrics import (
    calibration_metrics,
    candidate_recall_at_k,
    homography_quality_metrics,
    localization_error_metrics,
    role_classification_metrics,
    spatial_coverage,
    threshold_matching_metrics,
)
from grm_matcher.visualization import (
    visualize_candidate_entropy,
    visualize_geometry_roles,
    visualize_matches,
    visualize_role_ground_truth_comparison,
    visualize_seed_h_residuals,
    visualize_spatial_coverage,
    visualize_topk_candidates,
)


def test_matching_candidate_role_coverage_and_calibration_metrics() -> None:
    probability = torch.tensor(
        [[[0.9, 0.1, 0.0], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]]]
    )
    recall = candidate_recall_at_k(
        probability,
        torch.tensor([[0, 2, 1]]),
        torch.ones(1, 3, dtype=torch.bool),
        ks=(1, 2),
    )
    assert recall == {"recall_at_1": 1.0, "recall_at_2": 1.0}

    gt = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]])
    predicted = gt + torch.tensor([[[0.0, 0.0], [0.5, 0.0], [4.0, 0.0], [20.0, 0.0]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    threshold = threshold_matching_metrics(predicted, gt, valid, valid)
    assert threshold["precision_at_1px"] == 0.5
    location = localization_error_metrics(predicted, gt, valid)
    assert location["median_pixel_error"] == 0.5
    coverage = spatial_coverage(
        torch.tensor([[[2.0, 2.0], [14.0, 2.0], [2.0, 14.0], [14.0, 14.0]]]),
        valid,
        16,
        16,
        2,
        2,
    )
    assert coverage == 1.0
    roles = role_classification_metrics(
        torch.tensor([[0, 1, 2, 2]]), torch.tensor([[0, 1, 2, 0]]), valid
    )
    assert roles["local_f1"] == 1.0
    calibration = calibration_metrics(
        torch.tensor([[1.0, 0.0, 0.8, 0.2]]),
        torch.tensor([[True, False, True, False]]),
        valid,
        bins=4,
    )
    assert calibration["brier_score"] < 0.021


def test_homography_quality_is_exact_for_translation() -> None:
    src = torch.tensor([[[0.0, 0.0], [31.0, 0.0], [31.0, 31.0], [0.0, 31.0], [15.0, 15.0]]])
    dst = src + torch.tensor([4.0, -2.0])
    h = torch.tensor([[[1.0, 0.0, 4.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]])
    metrics = homography_quality_metrics(
        src,
        dst,
        torch.ones(1, 5),
        torch.ones(1, 5, dtype=torch.bool),
        h,
        32,
        32,
    )
    assert metrics["homography_corner_error"] < 1.0e-4
    assert metrics["homography_success_rate"] == 1.0


def test_all_visualizations_write_headless_pngs(tmp_path: Path) -> None:
    image = torch.rand(3, 32, 32)
    src = torch.tensor([[[4.0, 4.0], [12.0, 8.0], [20.0, 20.0], [28.0, 26.0]]])
    dst = src + torch.tensor([2.0, 1.0])
    p = torch.tensor([[0.9, 0.7, 0.5, 0.2]])
    roles = torch.softmax(torch.randn(1, 4, 3), dim=-1)
    entropy = torch.tensor([[0.1, 0.4, 0.8, 1.2]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    candidate_dst = dst[:, :, None].expand(-1, -1, 4, -1) + torch.randn(1, 4, 4, 2)
    candidate_probs = torch.softmax(torch.randn(1, 4, 4), dim=-1)
    functions = [
        ("matches.png", lambda path: visualize_matches(image, image, src, dst, p, roles[..., 0], roles[..., 1], roles[..., 2], entropy, valid, save_path=path)),
        ("topk.png", lambda path: visualize_topk_candidates(image, image, src, candidate_dst, candidate_probs, valid, save_path=path)),
        ("entropy.png", lambda path: visualize_candidate_entropy(image, src, entropy, valid, save_path=path)),
        ("roles.png", lambda path: visualize_geometry_roles(image, image, src, dst, roles[..., 0], roles[..., 1], roles[..., 2], valid, save_path=path)),
        ("residuals.png", lambda path: visualize_seed_h_residuals(image, dst, torch.ones_like(dst), valid, save_path=path)),
        ("coverage.png", lambda path: visualize_spatial_coverage(image, src, valid, grid_rows=2, grid_cols=2, save_path=path)),
        (
            "role_gt.png",
            lambda path: visualize_role_ground_truth_comparison(
                image,
                image,
                src,
                dst,
                dst - torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [8.0, 0.0]]]),
                roles[..., 0],
                roles[..., 1],
                roles[..., 2],
                torch.tensor([[0, 1, 2, 2]]),
                torch.tensor([[0.0, 2.0, 0.0, 8.0]]),
                valid,
                save_path=path,
            ),
        ),
    ]
    from matplotlib import pyplot as plt

    for filename, render in functions:
        path = tmp_path / filename
        figure = render(path)
        plt.close(figure)
        assert path.is_file() and path.stat().st_size > 0
