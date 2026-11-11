import pytest
import torch
from grm_matcher.imc_sampling import IMCCurriculumSampler


def test_scene_balance_missing_buckets_and_curriculum():
    records = ([dict(scene="large", covisibility=.2)] * 20
        + [dict(scene="large", covisibility=.5)] * 3
        + [dict(scene="large", covisibility=.8)] * 2
        + [dict(scene="small", covisibility=.8)])
    sampler = IMCCurriculumSampler(records)
    weights, _ = sampler.weights()
    assert weights.sum().item() == pytest.approx(1)
    assert weights[:25].sum().item() == pytest.approx(.5)
    assert weights[25].item() == pytest.approx(.5)
    assert weights[:20].sum().item() == pytest.approx(.05)
    sampler.set_epoch(9)
    weights, _ = sampler.weights()
    assert weights[:20].sum().item() == pytest.approx(.15)


def test_epoch_resume_reproducible_and_does_not_consume_global_rng():
    records = [dict(scene="a", covisibility=.2 + .3 * (i % 3)) for i in range(100)]
    first, resumed = IMCCurriculumSampler(records), IMCCurriculumSampler(records)
    first.set_epoch(7)
    resumed.set_epoch(7)
    state = torch.get_rng_state().clone()
    assert list(first) == list(resumed)
    assert torch.equal(state, torch.get_rng_state())
    assert sum(first.last_summary["sampled_low_mid_high"]) == len(records)
    resumed.set_epoch(8)
    assert list(first) != list(resumed)


@pytest.mark.parametrize("value", [float("nan"), 0, 1.1])
def test_invalid_covisibility_rejected(value):
    with pytest.raises(ValueError):
        IMCCurriculumSampler([dict(scene="a", covisibility=value)])
