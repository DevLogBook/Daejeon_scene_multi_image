from types import SimpleNamespace
import pytest
import torch
from grm_matcher.config import GRMMatcherConfig
from grm_matcher.losses import GRMMatcherLoss, _masked_mean


@pytest.mark.parametrize("all_masked", [False, True])
def test_homography_coarse_ignores_masked_gt_cells(all_masked):
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    targets = {"gt_dense_mapping": torch.stack((x, y)).float()[None],
               "gt_overlap_a": torch.ones(1, 1, 32, 32, dtype=torch.bool)}
    raw = torch.randn(1, 4, 4, requires_grad=True)
    mask = torch.ones_like(raw, dtype=torch.bool) if all_masked else torch.zeros_like(raw, dtype=torch.bool)
    mask[:, :, 2:] = True
    logits = raw.masked_fill(mask, torch.finfo(raw.dtype).min)
    aux = SimpleNamespace(coarse_similarity=logits, coarse_height=2, coarse_width=2)
    loss = GRMMatcherLoss(GRMMatcherConfig())._coarse_loss(targets, aux)
    assert torch.isfinite(loss)
    if all_masked:
        assert loss.item() == 0
    else:
        expected = torch.nn.functional.cross_entropy(logits[0, :2], torch.arange(2))
        torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad[mask]) == 0


def test_masked_reduction_excludes_invalid_values_but_preserves_valid_nan():
    x = torch.tensor([2., float("inf"), float("nan")], requires_grad=True)
    loss = _masked_mean(x, torch.tensor([True, False, False]))
    assert loss.item() == 2
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor([1., 0., 0.]))
    assert _masked_mean(x, torch.zeros(3, dtype=torch.bool)).item() == 0
    assert torch.isnan(_masked_mean(x, torch.tensor([False, False, True])))
