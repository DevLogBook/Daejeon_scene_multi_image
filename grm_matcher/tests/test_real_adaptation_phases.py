"""Focused gradient and loss tests for real-domain adaptation phases."""

from __future__ import annotations

import torch

from grm_matcher import GRMMatcher, GRMMatcherConfig
from grm_matcher.real_adaptation import (
    GRMStage25BCoarseAndRefinementLoss,
    GRMStage35GeometryLoss,
    GRMStage35PseudoLabels,
    forward_student_coarse,
)
from grm_matcher.stage25 import (
    GRMStage25Config,
    GRMStage25PseudoLabels,
    forward_student_refinement,
)
from grm_matcher.train_stage25 import _configure_real_phase


def _matcher_config() -> GRMMatcherConfig:
    return GRMMatcherConfig(
        input_height=32,
        input_width=32,
        max_source_queries=4,
        fixed_score_threshold=0.0,
        backbone_stage_blocks=(1, 1, 1),
        graph_layers=1,
        graph_neighbors=3,
    )


def _identity_labels() -> GRMStage25PseudoLabels:
    points = torch.tensor([[[7.5, 7.5], [23.5, 7.5], [7.5, 23.5], [23.5, 23.5]]])
    return GRMStage25PseudoLabels(
        src_xy=points,
        coarse_dst_xy=points.clone(),
        mid_dst_xy=points.clone(),
        fine_dst_xy=points.clone(),
        confidence=torch.ones(1, 4),
        cycle_error=torch.zeros(1, 4),
        fine_std=torch.ones(1, 4),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
        mutual_top1=torch.ones(1, 4, dtype=torch.bool),
        region_ids=torch.arange(4).reshape(1, 4),
    )


def test_stage25b_executes_coarse_and_refinement_with_two_scopes() -> None:
    torch.manual_seed(3)
    matcher_config = _matcher_config()
    adaptation = GRMStage25Config(
        min_pseudo_matches=1,
        coarse_min_pseudo_matches=1,
        coarse_min_covered_regions=1,
        min_mid_peak_probability=0.0,
        min_fine_peak_probability=0.0,
    )
    model = GRMMatcher(matcher_config)
    _configure_real_phase(model, "stage2.5b")
    image = torch.rand(1, 3, 32, 32)
    labels = _identity_labels()
    coarse = forward_student_coarse(model, image, image)
    refinement = forward_student_refinement(model, image, image, labels)
    losses = GRMStage25BCoarseAndRefinementLoss(matcher_config, adaptation)(
        coarse, refinement, labels, labels
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.backbone.stage_c4[0].conv1[0].weight.grad is not None
    assert model.coarse_transformer.layers[0].attention.in_proj_weight.grad is not None
    assert model.fpn.mid_lateral[0].weight.grad is not None
    assert model.backbone.stem[0].weight.grad is None
    assert model.geometry_graph.layers[0].query.weight.grad is None


def test_stage35_loss_reaches_graph_rerank_and_role_only() -> None:
    torch.manual_seed(5)
    matcher_config = _matcher_config()
    adaptation = GRMStage25Config(
        min_pseudo_matches=1,
        geometry_source_alignment_radius=1.0,
        geometry_rerank_target_radius=1.0,
    )
    model = GRMMatcher(matcher_config)
    _configure_real_phase(model, "stage3.5")
    image = torch.rand(1, 3, 32, 32)
    output, auxiliary = model(image, image, return_auxiliary=True)
    selected = auxiliary.selected_indices
    teacher_target = torch.gather(
        auxiliary.fine_dst_xy,
        2,
        selected[..., None, None].expand(-1, -1, 1, 2),
    ).squeeze(2).detach()
    source = auxiliary.candidate_src_xy[:, :, 0].detach()
    count = source.shape[1]
    labels = GRMStage25PseudoLabels(
        src_xy=source,
        coarse_dst_xy=teacher_target,
        mid_dst_xy=teacher_target,
        fine_dst_xy=teacher_target,
        confidence=torch.ones(1, count),
        cycle_error=torch.zeros(1, count),
        fine_std=torch.ones(1, count),
        valid_mask=torch.ones(1, count, dtype=torch.bool),
    )
    pseudo = GRMStage35PseudoLabels(
        matches=labels,
        role_labels=torch.zeros(1, count, dtype=torch.long),
        homography_valid=torch.ones(1, dtype=torch.bool),
        homography_residual=torch.zeros(1, count),
        local_flow_consistent=torch.ones(1, count, dtype=torch.bool),
    )
    losses = GRMStage35GeometryLoss(adaptation)(output, auxiliary, pseudo)
    assert torch.isfinite(losses["total"])
    assert losses["role_count"] > 0
    losses["total"].backward()
    assert model.geometry_graph.layers[0].query.weight.grad is not None
    assert model.rerank_head.score[1].weight.grad is not None
    assert model.role_head.classifier[1].weight.grad is not None
    assert model.backbone.stem[0].weight.grad is None
