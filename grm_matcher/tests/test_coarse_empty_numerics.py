import pytest
import torch

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.imc_dataset import sparse_bidirectional_labels
from grm_matcher.real_adaptation import GRMStage25CoarsePseudoLoss
from grm_matcher.stage25 import GRMStage25Config, GRMStage25CoarseOutput


@pytest.mark.parametrize("valid_count", [0, 4, 8])
def test_masked_coarse_logits_remain_finite_when_supervision_underfills(valid_count):
    config = GRMMatcherConfig()
    adaptation = GRMStage25Config(coarse_min_pseudo_matches=8, coarse_min_covered_regions=1)
    xy = torch.tensor([[[7.5 + 16 * (i % 3), 7.5 + 16 * (i // 3)] for i in range(8)]])
    batch = dict(image_a=torch.zeros(1, 3, 64, 64), sparse_a=xy, sparse_b=xy,
                 sparse_valid=(torch.arange(8)[None] < valid_count),
                 sparse_confidence=torch.ones(1, 8))
    labels = sparse_bidirectional_labels(batch, "cpu", config, adaptation)
    raw = torch.randn(1, 16, 16, requires_grad=True)
    # Simulate letterboxing: the last coarse column has masked target logits.
    padding = (torch.arange(16) % 4 == 3)[None, None].expand_as(raw)
    logits = raw.masked_fill(padding, torch.finfo(torch.float32).min)
    assert not torch.isfinite(logits.sum() * 0)  # original failure mechanism
    probability = torch.full((1, 16), .5, requires_grad=True)
    student = GRMStage25CoarseOutput(similarity=logits,
        matchability_a=probability, matchability_b=probability,
        coarse_height=4, coarse_width=4)
    result = GRMStage25CoarsePseudoLoss(config, adaptation)(student, labels.forward, labels.backward)
    assert all(torch.isfinite(value).all() for value in result.values())
    result["total"].backward()
    assert torch.isfinite(raw.grad).all()
    if valid_count < 8:
        assert result["total"].item() == 0
        assert torch.count_nonzero(raw.grad) == 0
    else:
        assert result["total"].item() > 0


@pytest.mark.parametrize("minimum", [8, 10])
def test_backward_targets_on_padding_are_excluded_before_count_gate(minimum):
    config = GRMMatcherConfig()
    adaptation = GRMStage25Config(coarse_min_pseudo_matches=minimum, coarse_min_covered_regions=1)
    xy = torch.tensor([[[7.5 + 16 * (i % 4), 7.5 + 16 * (i // 4)] for i in range(12)]])
    batch = dict(image_a=torch.zeros(1, 3, 64, 64), sparse_a=xy, sparse_b=xy,
                 sparse_valid=torch.ones(1, 12, dtype=torch.bool), sparse_confidence=torch.ones(1, 12))
    labels = sparse_bidirectional_labels(batch, "cpu", config, adaptation)
    raw = torch.randn(1, 16, 16, requires_grad=True)
    padding = (torch.arange(16) % 4 == 3)[None, :, None].expand_as(raw)
    logits = raw.masked_fill(padding, torch.finfo(torch.float32).min)
    old_backward = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2)[0, :12], torch.arange(12), reduction="none")
    assert torch.isinf(old_backward.sum())  # nonempty CE overflow, unlike the earlier empty sum bug
    probability = torch.full((1, 16), .5, requires_grad=True)
    student = GRMStage25CoarseOutput(similarity=logits, matchability_a=probability,
        matchability_b=probability, coarse_height=4, coarse_width=4)
    objective = GRMStage25CoarsePseudoLoss(config, adaptation)
    result = objective(student, labels.forward, labels.backward)
    assert all(torch.isfinite(value).all() for value in result.values())
    assert result["valid_count"].item() == (18 if minimum == 8 else 0)
    if minimum == 8:
        good = torch.arange(12)[torch.arange(12) % 4 != 3]
        expected = torch.nn.functional.cross_entropy(logits.transpose(1, 2)[0, good], good)
        torch.testing.assert_close(result["coarse_backward"], expected)
    else:
        assert result["total"].item() == 0
    result["total"].backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad[padding]) == 0
