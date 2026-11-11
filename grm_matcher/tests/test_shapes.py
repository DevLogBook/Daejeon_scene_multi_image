"""Stage-one configuration and public-output contract tests."""

from __future__ import annotations

import pytest
import torch

from grm_matcher import (
    GRMMatcherConfig,
    GRMMatcherOutput,
    select_global_matches,
    select_local_matches,
)
from grm_matcher.backbone import GRMConvBackbone, GRMHybridMobileViTBackbone
from grm_matcher.feature_pyramid import GRMResidualFPN


def _make_output(batch: int = 2, count: int = 5, topk: int = 4) -> GRMMatcherOutput:
    roles = torch.softmax(torch.randn(batch, count, 3), dim=-1)
    candidate_probs = torch.softmax(torch.randn(batch, count, topk), dim=-1)
    return GRMMatcherOutput(
        src_xy=torch.rand(batch, count, 2) * 127.0,
        dst_xy=torch.rand(batch, count, 2) * 127.0,
        p_match=torch.rand(batch, count),
        p_global=roles[..., 0],
        p_local=roles[..., 1],
        p_false=roles[..., 2],
        candidate_entropy=torch.rand(batch, count),
        cycle_error=torch.rand(batch, count),
        fine_uncertainty=torch.eye(2).reshape(1, 1, 2, 2).repeat(batch, count, 1, 1),
        overlap_src=torch.rand(batch, 1, 8, 8),
        overlap_dst=torch.rand(batch, 1, 8, 8),
        candidate_indices=torch.randint(0, 64, (batch, count, topk), dtype=torch.long),
        candidate_probs=candidate_probs,
        valid_mask=torch.ones(batch, count, dtype=torch.bool),
    )


def test_config_defaults_and_serialization() -> None:
    config = GRMMatcherConfig()
    assert (config.coarse_stride, config.mid_stride, config.fine_stride) == (16, 8, 4)
    assert config.input_range == "zero_one"
    assert config.topk_candidates == 4
    assert GRMMatcherConfig.from_dict(config.to_dict()) == config


def test_config_rejects_invalid_or_unknown_values() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GRMMatcherConfig(input_height=510)
    with pytest.raises(ValueError, match="positive odd"):
        GRMMatcherConfig(fine_window_size=4)
    with pytest.raises(ValueError, match="Unknown"):
        GRMMatcherConfig.from_dict({"coarse_dims": 128})


def test_output_shape_contract_and_helpers() -> None:
    output = _make_output()
    output.validate()
    assert output.batch_size == 2
    assert output.num_matches == 5
    assert output.topk == 4
    assert set(output.as_dict()) == set(GRMMatcherOutput.__dataclass_fields__)
    assert output.detach().src_xy.grad_fn is None
    assert output.to(dtype=torch.float64).src_xy.dtype == torch.float64
    assert output.to(dtype=torch.float64).candidate_indices.dtype == torch.long


def test_output_rejects_non_normalized_conditional_candidates() -> None:
    output = _make_output()
    output.candidate_probs.fill_(0.5)
    with pytest.raises(ValueError, match="Conditional top-k"):
        output.validate()


def test_output_accepts_diagonal_uncertainty() -> None:
    output = _make_output()
    output.fine_uncertainty = torch.ones(2, 5, 2)
    output.validate()


def test_minimal_external_solver_selection_interface() -> None:
    output = _make_output(batch=2, count=5)
    output.p_global.fill_(0.8)
    output.p_local.fill_(0.1)
    output.p_false.fill_(0.1)
    global_batches = select_global_matches(output, minimum_role_probability=0.5)
    local_batches = select_local_matches(output, minimum_role_probability=0.5)
    assert len(global_batches) == len(local_batches) == 2
    assert global_batches[0]["src_xy"].shape == (5, 2)
    assert local_batches[0]["src_xy"].shape == (0, 2)


@pytest.mark.parametrize("backbone_class", [GRMConvBackbone, GRMHybridMobileViTBackbone])
def test_backbone_and_fpn_shapes_for_batch_and_non_square_input(backbone_class: type) -> None:
    config = GRMMatcherConfig(input_height=64, input_width=96)
    backbone = backbone_class(config)
    fpn = GRMResidualFPN(config)
    images = torch.rand(2, 3, 64, 96)

    features = backbone(images)
    assert features["c2"].shape == (2, 64, 16, 24)
    assert features["c3"].shape == (2, 128, 8, 12)
    assert features["c4"].shape == (2, 192, 4, 6)

    pyramid = fpn(features)
    assert pyramid["F_fine"].shape == (2, config.fine_dim, 16, 24)
    assert pyramid["F_mid"].shape == (2, config.mid_dim, 8, 12)
    assert pyramid["F_coarse"].shape == (2, config.coarse_dim, 4, 6)
    torch.testing.assert_close(fpn.fine_alpha.detach(), torch.tensor(0.25))


def test_backbone_fpn_gradient_reaches_learnable_fine_fusion() -> None:
    config = GRMMatcherConfig(input_height=32, input_width=32)
    backbone = GRMConvBackbone(config)
    fpn = GRMResidualFPN(config)
    pyramid = fpn(backbone(torch.rand(1, 3, 32, 32)))
    pyramid["F_fine"].mean().backward()
    assert fpn._fine_alpha_logit.grad is not None
    assert torch.isfinite(fpn._fine_alpha_logit.grad).all()
