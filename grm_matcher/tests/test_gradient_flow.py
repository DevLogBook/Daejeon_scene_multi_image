"""Geometry graph, role, and end-to-end matcher gradient tests."""

from __future__ import annotations

import torch

from grm_matcher import GRMMatcher, GRMMatcherConfig
from grm_matcher.geometry_graph import GRMGeometryGraph
from grm_matcher.losses import GRMMatcherLoss
from grm_matcher.role_head import GRMGeometryRoleHead
from grm_matcher.synthetic_dataset import SyntheticGRMDataConfig, SyntheticGRMDataset


def test_role_loss_flows_through_sparse_geometry_graph() -> None:
    config = GRMMatcherConfig(input_height=32, input_width=32, graph_neighbors=3)
    graph = GRMGeometryGraph(config)
    role = GRMGeometryRoleHead(config)
    tokens = torch.randn(2, 6, config.graph_hidden_dim, requires_grad=True)
    src = torch.rand(2, 6, 2) * 31.0
    dst = src + torch.randn_like(src)
    valid = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]]
    )
    contextual = graph(
        tokens,
        src,
        dst,
        torch.rand(2, 6),
        valid,
        image_height=32,
        image_width=32,
    )
    output = role(contextual, valid)
    loss = -torch.log(output.probabilities[..., 0][valid]).mean()
    loss.backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert graph.layers[0].query.weight.grad is not None
    assert role.classifier[-1].weight.grad is not None
    assert graph.last_memory_estimate_bytes > 0


def test_end_to_end_matcher_batch_shapes_probabilities_and_gradients() -> None:
    torch.manual_seed(23)
    config = GRMMatcherConfig(
        input_height=32,
        input_width=32,
        max_source_queries=4,
        sampling_grid_rows=2,
        sampling_grid_cols=2,
        backbone_stage_blocks=(1, 1, 1),
        mobilevit_depth=1,
        graph_layers=1,
        graph_neighbors=3,
    )
    model = GRMMatcher(config)
    image_a = torch.rand(2, 3, 32, 32)
    image_b = torch.rand(2, 3, 32, 32)
    output = model(image_a, image_b)
    output.validate()
    assert output.src_xy.shape == (2, 4, 2)
    assert output.candidate_indices.shape == (2, 4, 4)
    role_sum = output.p_global + output.p_local + output.p_false
    torch.testing.assert_close(role_sum, torch.ones_like(role_sum))
    loss = output.p_match.mean() + output.p_global.mean() + output.dst_xy.mean() * 1.0e-3
    loss.backward()
    assert model.backbone.stem[0].weight.grad is not None
    assert model.mid_refiner.correlation.descriptor_projection.weight.grad is not None
    assert model.fine_refiner.correlation.descriptor_projection.weight.grad is not None
    assert model.role_head.classifier[-1].weight.grad is not None


def test_joint_loss_components_are_finite_on_identity_synthetic_pair() -> None:
    config = GRMMatcherConfig(
        input_height=32,
        input_width=32,
        max_source_queries=4,
        sampling_grid_rows=2,
        sampling_grid_cols=2,
        backbone_stage_blocks=(1, 1, 1),
        graph_layers=1,
        graph_neighbors=3,
    )
    source = torch.rand(3, 40, 40)
    data_config = SyntheticGRMDataConfig(
        output_height=32,
        output_width=32,
        samples_per_source=1,
        max_rotation_degrees=0.0,
        min_scale=1.0,
        max_scale=1.0,
        max_translation_fraction=0.0,
        max_perspective=0.0,
        local_region_min=0,
        local_region_max=0,
        max_occlusions=0,
        hard_negative_count=8,
    )
    sample = SyntheticGRMDataset([source], data_config)[0]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = GRMMatcher(config)
    output, auxiliary = model(batch["image_a"], batch["image_b"], return_auxiliary=True)
    losses = GRMMatcherLoss(config)(output, batch, auxiliary, stage="joint")
    for name, value in losses.items():
        assert torch.isfinite(value), name
    losses["total"].backward()
    assert model.correlation.matchability_head_a.projection.weight.grad is not None
    assert model.rerank_head.score[-1].weight.grad is not None


def test_mixed_precision_autocast_forward_has_no_nan_or_inf() -> None:
    config = GRMMatcherConfig(
        input_height=32,
        input_width=32,
        max_source_queries=4,
        sampling_grid_rows=2,
        sampling_grid_cols=2,
        backbone_stage_blocks=(1, 1, 1),
        graph_layers=1,
        graph_neighbors=3,
    )
    model = GRMMatcher(config).eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(image, image)
    output.validate()
