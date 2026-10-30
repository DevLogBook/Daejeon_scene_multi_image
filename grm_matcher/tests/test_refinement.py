"""Local expectation, translation, border, and refinement-gradient tests."""

from __future__ import annotations

import torch

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.coordinates import feature_xy_to_image_xy
from grm_matcher.fine_refiner import GRMFineRefiner
from grm_matcher.local_correlation import (
    GRMLocalCorrelation,
    compute_spatial_covariance,
    compute_spatial_expectation,
)
from grm_matcher.mid_refiner import GRMMidRefiner


def test_spatial_expectation_and_covariance_are_probability_derived() -> None:
    probability = torch.zeros(1, 1, 5, 5)
    probability[0, 0, 2, 3] = 1.0
    expectation = compute_spatial_expectation(probability)
    covariance = compute_spatial_covariance(probability, expectation)
    torch.testing.assert_close(expectation, torch.tensor([[[1.0, 0.0]]]))
    torch.testing.assert_close(covariance, torch.zeros(1, 1, 2, 2))


def test_local_correlation_recovers_known_xy_translation() -> None:
    torch.manual_seed(11)
    channels = 16
    feature_a = torch.randn(1, channels, 8, 8)
    feature_a = torch.nn.functional.normalize(feature_a, dim=1)
    feature_b = torch.zeros_like(feature_a)
    # A(x, y) appears in B(x + 1, y + 2): direction is A -> B.
    feature_b[:, :, 2:, 1:] = feature_a[:, :, :-2, :-1]
    module = GRMLocalCorrelation(channels, temperature=0.01)
    center = torch.tensor([[[3.0, 3.0]]])
    result = module(feature_a, feature_b, center, center, 1, 5)
    torch.testing.assert_close(
        result.delta_xy, torch.tensor([[[1.0, 2.0]]]), atol=0.08, rtol=0.0
    )


def test_mid_refiner_converts_feature_delta_to_original_pixels() -> None:
    torch.manual_seed(19)
    config = GRMMatcherConfig(
        input_height=64,
        input_width=64,
        mid_source_patch_size=1,
        mid_window_size=5,
        local_temperature=0.01,
    )
    feature_a = torch.randn(1, config.mid_dim, 8, 8)
    feature_a = torch.nn.functional.normalize(feature_a, dim=1)
    feature_b = torch.zeros_like(feature_a)
    feature_b[:, :, 2:, 1:] = feature_a[:, :, :-2, :-1]
    refiner = GRMMidRefiner(config)
    src_xy = feature_xy_to_image_xy(torch.tensor([[[3.0, 3.0]]]), config.mid_stride)
    output = refiner(feature_a, feature_b, src_xy, src_xy)
    expected = src_xy + torch.tensor([[[8.0, 16.0]]])
    torch.testing.assert_close(output.dst_xy, expected, atol=0.7, rtol=0.0)


def test_border_windows_are_finite_and_normalized() -> None:
    config = GRMMatcherConfig(input_height=32, input_width=32)
    refiner = GRMFineRefiner(config)
    features = torch.randn(2, config.fine_dim, 8, 8)
    border_xy = feature_xy_to_image_xy(
        torch.tensor([[[0.0, 0.0]], [[7.0, 7.0]]]), config.fine_stride
    )
    output = refiner(features, features, border_xy, border_xy)
    assert torch.isfinite(output.local_probability).all()
    torch.testing.assert_close(
        output.local_probability.sum(dim=(-2, -1)), torch.ones(2, 1)
    )
    assert torch.isfinite(output.covariance_image).all()


def test_mid_and_fine_refiner_parameters_receive_gradients() -> None:
    config = GRMMatcherConfig(input_height=32, input_width=32)
    mid = GRMMidRefiner(config)
    fine = GRMFineRefiner(config)
    mid_a = torch.randn(1, config.mid_dim, 4, 4, requires_grad=True)
    mid_b = torch.randn(1, config.mid_dim, 4, 4, requires_grad=True)
    fine_a = torch.randn(1, config.fine_dim, 8, 8, requires_grad=True)
    fine_b = torch.randn(1, config.fine_dim, 8, 8, requires_grad=True)
    src_xy = torch.tensor([[[11.5, 11.5], [19.5, 19.5]]])
    mid_output = mid(mid_a, mid_b, src_xy, src_xy)
    fine_output = fine(fine_a, fine_b, src_xy, mid_output.dst_xy)
    loss = (
        mid_output.peak_probability.mean()
        + fine_output.peak_probability.mean()
        + fine_output.dst_xy.mean()
    )
    loss.backward()
    assert mid.correlation.descriptor_projection.weight.grad is not None
    assert fine.correlation.descriptor_projection.weight.grad is not None
    assert mid_a.grad is not None and torch.isfinite(mid_a.grad).all()
    assert fine_b.grad is not None and torch.isfinite(fine_b.grad).all()

