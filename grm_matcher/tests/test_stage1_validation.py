"""Stage 1-only validation logic tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.validate_stage1 import (
    GRMStage1BatchOutput,
    _binary_metrics,
    _compare_metric_means,
    _stage1_batch_metrics,
    forward_stage1,
    parse_args,
)


def _small_config() -> GRMMatcherConfig:
    return GRMMatcherConfig(
        input_height=32,
        input_width=32,
        max_source_queries=4,
        fixed_score_threshold=0.0,
        backbone_stage_blocks=(1, 1, 1),
        graph_layers=1,
        graph_neighbors=3,
    )


def test_stage1_forward_stops_at_coarse_outputs() -> None:
    model = GRMMatcher(_small_config()).eval()
    image_a = torch.rand(2, 3, 32, 32)
    image_b = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        output = forward_stage1(model, image_a, image_b)

    assert output.probability.shape == (2, 4, 4)
    assert output.matchability_a.shape == (2, 4)
    assert output.matchability_b.shape == (2, 4)
    assert output.sampled.valid_mask.shape == (2, 4)


def test_perfect_coarse_predictions_have_perfect_stage1_metrics() -> None:
    probability = torch.eye(4).unsqueeze(0)
    sampled = SimpleNamespace(
        valid_mask=torch.tensor([[True, True, False, False]]),
        query_indices=torch.tensor([[0, 1, 0, 0]]),
        candidate_indices=torch.tensor(
            [[[0, 1, 2, 3], [1, 0, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]]]
        ),
        diagnostics={"selected_count": torch.tensor([2])},
    )
    output = GRMStage1BatchOutput(
        probability=probability,
        matchability_a=torch.tensor([[0.9, 0.8, 0.2, 0.1]]),
        matchability_b=torch.tensor([[0.9, 0.8, 0.2, 0.1]]),
        sampled=sampled,  # type: ignore[arg-type]
        coarse_height=2,
        coarse_width=2,
    )
    feature_xy = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    )
    valid = torch.tensor([[True, True, False, False]])
    ground_truth = {
        "target_index": torch.tensor([[0, 1, 2, 3]]),
        "target_feature_xy": feature_xy,
        "valid_source": valid,
        "overlap_a": valid,
        "overlap_b": valid,
    }

    metrics = _stage1_batch_metrics(output, ground_truth)[0]

    assert metrics["recall_at_1"] == pytest.approx(1.0)
    assert metrics["recall_at_4"] == pytest.approx(1.0)
    assert metrics["top1_error_cells_mean"] == pytest.approx(0.0)
    assert metrics["selected_top1_precision"] == pytest.approx(1.0)
    assert metrics["selected_top4_precision"] == pytest.approx(1.0)
    assert metrics["matchability_a_auroc"] == pytest.approx(1.0)


def test_matchability_metrics_remain_finite_for_empty_class() -> None:
    metrics = _binary_metrics(
        torch.tensor([0.2, 0.4, 0.6]),
        torch.tensor([True, True, True]),
        prefix="matchability",
    )

    assert metrics["matchability_precision"] == pytest.approx(1.0)
    assert metrics["matchability_auroc"] != metrics["matchability_auroc"]


def test_stage1_validation_cli_supports_untrained_comparison() -> None:
    args = parse_args(
        [
            "--data_dir",
            "held_out_images",
            "--checkpoint",
            "best_candidate_recall.pt",
            "--compare_untrained",
            "--visualize_samples",
            "0",
        ]
    )

    assert args.compare_untrained
    assert args.baseline_checkpoint is None
    assert args.visualize_samples == 0


def test_checkpoint_comparison_marks_metric_direction() -> None:
    comparison = _compare_metric_means(
        {"recall_at_4": 0.8, "top1_error_cells_mean": 0.4},
        {"recall_at_4": 0.5, "top1_error_cells_mean": 0.9},
    )

    assert comparison["recall_at_4"]["preferred_direction"] == "higher"
    assert comparison["recall_at_4"]["improved"] is True
    assert comparison["top1_error_cells_mean"]["preferred_direction"] == "lower"
    assert comparison["top1_error_cells_mean"]["improved"] is True
