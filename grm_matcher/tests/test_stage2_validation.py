"""Stage-2 trainability and refinement-only metric tests."""

from __future__ import annotations

import torch

from grm_matcher import GRMMatcher, GRMMatcherConfig
from grm_matcher.losses import GRMMatcherLoss
from grm_matcher.stage2_metrics import (
    GRMStage2Predictions,
    compute_stage2_metric_components,
)
from grm_matcher.train_matcher import _configure_stage
from grm_matcher.validate_stage2 import forward_stage2


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


def test_stage2_unfreezes_only_refinement_fpn_branches() -> None:
    model = GRMMatcher(_small_config())
    _configure_stage(model, "stage2")

    assert any(parameter.requires_grad for parameter in model.fpn.mid_lateral.parameters())
    assert any(parameter.requires_grad for parameter in model.fpn.mid_output.parameters())
    assert any(parameter.requires_grad for parameter in model.fpn.fine_lateral.parameters())
    assert any(parameter.requires_grad for parameter in model.fpn.fine_output.parameters())
    assert model.fpn._fine_alpha_logit.requires_grad
    assert any(parameter.requires_grad for parameter in model.mid_refiner.parameters())
    assert any(parameter.requires_grad for parameter in model.fine_refiner.parameters())

    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.fpn.coarse_lateral.parameters())
    assert not any(parameter.requires_grad for parameter in model.fpn.coarse_output.parameters())
    assert not any(parameter.requires_grad for parameter in model.coarse_transformer.parameters())
    assert not any(parameter.requires_grad for parameter in model.correlation.parameters())
    assert not any(parameter.requires_grad for parameter in model.geometry_graph.parameters())
    assert not any(parameter.requires_grad for parameter in model.rerank_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.role_head.parameters())


def test_stage2_metrics_follow_one_coarse_oracle_candidate() -> None:
    config = _small_config()
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    dense_mapping = torch.stack((xx, yy), dim=0).float().unsqueeze(0)
    targets = {
        "gt_dense_mapping": dense_mapping,
        "gt_overlap_a": torch.ones(1, 1, 32, 32),
    }
    source = torch.tensor([[[[8.0, 8.0], [8.0, 8.0]]]])
    predictions = GRMStage2Predictions(
        src_xy=source,
        coarse_dst_xy=torch.tensor([[[[16.0, 8.0], [28.0, 8.0]]]]),
        mid_dst_xy=torch.tensor([[[[12.0, 8.0], [20.0, 8.0]]]]),
        fine_dst_xy=torch.tensor([[[[9.0, 8.0], [10.0, 8.0]]]]),
        candidate_valid_mask=torch.ones(1, 1, 2, dtype=torch.bool),
    )

    metrics = compute_stage2_metric_components(predictions, targets, config).metrics()
    assert metrics["coarse_top1_pixel_error"] == 8.0
    assert metrics["mid_top1_pixel_error"] == 4.0
    assert metrics["fine_top1_pixel_error"] == 1.0
    assert metrics["coarse_oracle_pixel_error"] == 8.0
    assert metrics["mid_oracle_pixel_error"] == 4.0
    assert metrics["fine_oracle_pixel_error"] == 1.0
    assert metrics["mid_top1_reachable_ratio"] == 1.0
    assert metrics["fine_top1_reachable_ratio"] == 1.0
    assert metrics["mid_oracle_reachable_ratio"] == 1.0
    assert metrics["fine_oracle_reachable_ratio"] == 1.0
    assert metrics["mid_candidate_reachable_ratio"] == 1.0
    assert metrics["fine_candidate_reachable_ratio"] == 0.5
    assert metrics["fine_top1_reachable_given_mid_ratio"] == 1.0
    assert metrics["coarse_top1_pck_5"] == 0.0
    assert metrics["mid_top1_pck_5"] == 1.0
    assert metrics["fine_top1_pck_1"] == 1.0
    assert metrics["fine_oracle_pck_3"] == 1.0


def test_stage2_forward_stops_before_geometry_and_rerank() -> None:
    model = GRMMatcher(_small_config()).eval()

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("Stage-3 module must not run during Stage-2 validation.")

    model.seed_geometry.forward = forbidden  # type: ignore[method-assign]
    model.geometry_graph.forward = forbidden  # type: ignore[method-assign]
    model.rerank_head.forward = forbidden  # type: ignore[method-assign]
    model.role_head.forward = forbidden  # type: ignore[method-assign]
    image_a = torch.rand(2, 3, 32, 32)
    image_b = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        predictions = forward_stage2(model, image_a, image_b)

    assert predictions.src_xy.shape == (2, 4, config_topk := model.config.topk_candidates, 2)
    assert predictions.coarse_dst_xy.shape == predictions.src_xy.shape
    assert predictions.mid_dst_xy.shape == predictions.src_xy.shape
    assert predictions.fine_dst_xy.shape == predictions.src_xy.shape
    assert predictions.candidate_valid_mask.shape == (2, 4, config_topk)


def test_stage2_loss_reaches_mid_and_fine_fpn_but_not_coarse() -> None:
    torch.manual_seed(17)
    config = _small_config()
    model = GRMMatcher(config)
    _configure_stage(model, "stage2")
    image = torch.rand(1, 3, 32, 32)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    dense_mapping = torch.stack((xx, yy), dim=0).float().unsqueeze(0)
    overlap = torch.ones(1, 1, 32, 32)
    targets = {
        "gt_dense_mapping": dense_mapping,
        "gt_overlap_a": overlap,
        "gt_overlap_b": overlap,
        "gt_global_mask": overlap,
        "gt_local_mask": torch.zeros_like(overlap),
    }

    output, auxiliary = model(image, image, return_auxiliary=True)
    losses = GRMMatcherLoss(config)(output, targets, auxiliary, stage="stage2")
    assert torch.isfinite(losses["total"])
    losses["total"].backward()

    assert model.fpn.mid_lateral[0].weight.grad is not None
    assert model.fpn.mid_output.conv1[0].weight.grad is not None
    assert model.fpn.fine_lateral[0].weight.grad is not None
    assert model.fpn.fine_output.conv1[0].weight.grad is not None
    assert model.fpn._fine_alpha_logit.grad is not None
    assert model.mid_refiner.correlation.descriptor_projection.weight.grad is not None
    assert model.fine_refiner.correlation.descriptor_projection.weight.grad is not None
    assert model.fpn.coarse_lateral[0].weight.grad is None
    assert model.backbone.stem[0].weight.grad is None
    assert model.coarse_transformer.layers[0].attention.in_proj_weight.grad is None
