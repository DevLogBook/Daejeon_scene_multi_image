"""Selection equivalence, including ties and per-image padding, after acceleration."""
import pytest
import torch

from grm_matcher.architecture_v3 import select_queries


@torch.no_grad()
def reference_select_queries(reliability, valid, width, count, bonus=.2,
                             sigma=2., underfill=True):
    """Original device-side Gaussian greedy policy, retained as a regression oracle."""
    b, total = reliability.shape
    count = min(count, total)
    if underfill:
        count = min(count, int(valid.sum(-1).max().item()))
    if count == 0:
        return (torch.empty(b, 0, dtype=torch.long, device=valid.device),
                torch.empty(b, 0, dtype=torch.bool, device=valid.device))
    ids = torch.arange(total, device=reliability.device)
    xy = torch.stack((ids % width, ids // width), -1).float()
    remaining = valid.clone()
    diversity = torch.ones_like(reliability)
    chosen, masks = [], []
    for _ in range(count):
        scores = (reliability * (1 + bonus * diversity)).masked_fill(~remaining, -1)
        index = scores.argmax(-1)
        active = remaining.any(-1)
        chosen.append(index)
        masks.append(active)
        distance = (xy[None] - xy[index, None]).square().sum(-1)
        diversity = torch.minimum(diversity, 1-torch.exp(-distance/(2*sigma*sigma)))
        remaining.scatter_(1, index[:, None], False)
    return torch.stack(chosen, 1), torch.stack(masks, 1)


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('underfill', [False, True])
@pytest.mark.parametrize('bonus', [0., .2, .37])
def test_gaussian_greedy_matches_reference_with_batch_padding(device, underfill, bonus):
    g = torch.Generator().manual_seed(2026)
    reliability = torch.rand(3, 100, generator=g).to(device)
    valid = torch.rand(3, 100, generator=g).gt(.3).to(device)
    valid[1] = False
    valid[2] = False
    valid[2, 87] = True
    valid_before = valid.clone()
    reliability_before = reliability.clone()
    args = (reliability, valid, 10, 32, bonus, 1.7, underfill)
    expected = reference_select_queries(*args)
    actual = select_queries(*args)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(valid, valid_before)
    assert torch.equal(reliability, reliability_before)
    assert actual[0][1].eq(0).all()  # Preserve the original inactive padding index.
    assert not actual[1][1].any()
    assert actual[0][2, 0] == 87
    assert actual[1][2].sum() == 1


@pytest.mark.parametrize('device', DEVICES)
def test_uniform_scores_preserve_greedy_tie_order_and_empty_support(device):
    reliability = torch.ones(1, 16, device=device)
    valid = torch.ones_like(reliability, dtype=torch.bool)
    query, mask = select_queries(reliability, valid, 4, 4)
    assert query.tolist() == [[0, 15, 3, 12]]
    assert mask.all()
    query, mask = select_queries(reliability, ~valid, 4, 8)
    assert query.shape == mask.shape == (1, 0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('support_count', [1, 2, 3])
def test_tiny_support_keeps_exact_underfill(device, support_count):
    reliability = torch.linspace(.01, .5, 1024, device=device)[None]
    valid = torch.zeros_like(reliability, dtype=torch.bool)
    valid[:, -support_count:] = True
    actual = select_queries(reliability, valid, 32, 512)
    expected = reference_select_queries(reliability, valid, 32, 512)
    assert actual[0].shape == (1, support_count)
    assert actual[1].all()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA implementation')
@pytest.mark.parametrize('budget', [128, 256, 512])
def test_512_input_budget_and_near_ties_are_exact(budget):
    # Coarse-grid FP32 values separated by a few ULPs exercise argmax decisions.
    g = torch.Generator().manual_seed(2026 + budget)
    reliability = (torch.rand(1, 1024, generator=g) * 4e-7 + .1).cuda()
    valid = torch.rand(1, 1024, generator=g).gt(.1).cuda()
    expected = reference_select_queries(reliability, valid, 32, budget)
    actual = select_queries(reliability, valid, 32, budget)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA implementation')
@pytest.mark.parametrize('total,dtype', [(1056, torch.float32), (100, torch.float64),
                                       (100, torch.float16)])
def test_larger_grids_and_other_dtypes_keep_device_policy(total, dtype):
    reliability = torch.linspace(.01, .5, total, dtype=dtype, device='cuda')[None]
    valid = torch.ones_like(reliability, dtype=torch.bool)
    expected = reference_select_queries(reliability, valid, 32, 32)
    actual = select_queries(reliability, valid, 32, 32)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
