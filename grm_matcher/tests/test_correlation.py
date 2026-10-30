"""Coarse context and explicit-correlation tests."""

from __future__ import annotations

import torch

from grm_matcher.coarse_transformer import GRMCoarseContextTransformer
from grm_matcher.config import GRMMatcherConfig
from grm_matcher.correlation import GRMExplicitCorrelation, grm_dual_softmax


def test_dual_softmax_identity_has_diagonal_maxima() -> None:
    similarity = torch.eye(4).unsqueeze(0) * 20.0
    dual = grm_dual_softmax(similarity)
    assert dual.shape == (1, 4, 4)
    assert torch.equal(dual.argmax(dim=-1), torch.arange(4).unsqueeze(0))
    assert torch.isfinite(dual).all()


def test_explicit_correlation_shapes_padding_and_gradients() -> None:
    config = GRMMatcherConfig(input_height=32, input_width=32)
    module = GRMExplicitCorrelation(config)
    tokens_a = torch.randn(2, 4, config.coarse_dim, requires_grad=True)
    tokens_b = torch.randn(2, 6, config.coarse_dim, requires_grad=True)
    mask_a = torch.tensor([[True, True, True, False], [True, True, True, True]])
    mask_b = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, True, True, True]]
    )
    output = module(tokens_a, tokens_b, mask_a, mask_b)
    assert output.similarity.shape == (2, 4, 6)
    assert output.probability.shape == (2, 4, 6)
    assert output.matchability_a.shape == (2, 4)
    assert output.matchability_b.shape == (2, 6)
    assert torch.count_nonzero(output.probability[0, 3]) == 0
    assert torch.count_nonzero(output.probability[0, :, 4:]) == 0
    assert torch.isfinite(output.probability).all()
    output.probability.sum().backward()
    assert tokens_a.grad is not None and torch.isfinite(tokens_a.grad).all()
    assert tokens_b.grad is not None and torch.isfinite(tokens_b.grad).all()
    assert module.matchability_head_a.projection.weight.grad is not None


def test_coarse_transformer_is_bilateral_and_respects_padding() -> None:
    torch.manual_seed(7)
    config = GRMMatcherConfig(input_height=32, input_width=32)
    transformer = GRMCoarseContextTransformer(config).eval()
    feature_a = torch.randn(2, config.coarse_dim, 2, 2)
    feature_b = torch.randn(2, config.coarse_dim, 2, 2)
    mask_a = torch.tensor(
        [[[True, True], [True, False]], [[False, False], [False, False]]]
    )
    mask_b = torch.ones(2, 2, 2, dtype=torch.bool)
    tokens_a, tokens_b = transformer(feature_a, feature_b, mask_a, mask_b)
    assert tokens_a.shape == tokens_b.shape == (2, 4, config.coarse_dim)
    assert torch.count_nonzero(tokens_a[0, 3]) == 0
    assert torch.count_nonzero(tokens_a[1]) == 0
    assert torch.isfinite(tokens_a).all()
    assert torch.isfinite(tokens_b).all()


def test_correlation_softmax_stays_fp32() -> None:
    config = GRMMatcherConfig(input_height=32, input_width=32)
    module = GRMExplicitCorrelation(config)
    tokens = torch.randn(1, 4, config.coarse_dim, dtype=torch.float16)
    output = module(tokens, tokens)
    assert output.similarity.dtype == torch.float32
    assert output.probability.dtype == torch.float32
    assert torch.isfinite(output.probability).all()

