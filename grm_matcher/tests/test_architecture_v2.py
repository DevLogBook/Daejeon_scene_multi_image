"""Behavioral regression checks for the opt-in GRM v2 architecture."""
from types import SimpleNamespace
import pytest
import torch
from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.architecture_upgrade import load_v1_weights
from grm_matcher.local_correlation import GRMLocalCorrelation, GRMLocalRefinerCore
from grm_matcher.scratch_transfer import ScratchTransferObjective, configure_optimizer


def config(version="v2"):
    return GRMMatcherConfig(input_height=64, input_width=64, max_source_queries=16,
        backbone_stage_blocks=(1, 1, 1), graph_layers=1,
        architecture_version=version, candidate_peak_radius=int(version == "v2"),
        classification_mode="inlier" if version == "v2" else "role")


def test_old_weights_only_allow_explicit_new_parameters():
    old, new = GRMMatcher(config("v1")), GRMMatcher(config())
    assert not any(k.startswith("context_bridge") or k == "geometry_gain" for k in old.state_dict())
    load_v1_weights(new, old.state_dict())
    for key, value in old.state_dict().items():
        if not key.startswith("role_head."):
            assert torch.equal(value, new.state_dict()[key])
    assert not any(key.startswith("role_head.") for key in new.state_dict())
    broken = dict(old.state_dict()); broken.pop(next(iter(broken)))
    with pytest.raises(ValueError):
        load_v1_weights(new, broken)
    GRMMatcher(config()).load_state_dict(new.state_dict(), strict=True)


def test_context_reaches_fine_features_and_receives_gradients():
    torch.manual_seed(3)
    model = GRMMatcher(config()).eval()
    image = torch.rand(1, 3, 64, 64)
    a, _, _, _ = model.encode_pair(image, image)
    changed, _, _, _ = model.encode_pair(image, 1-image)
    assert (a["F_fine"]-changed["F_fine"]).abs().max() > 1e-7
    changed["F_fine"].square().mean().backward()
    for parameter in (model.context_bridge.fine.weight, model.context_bridge.gains):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_distinct_candidates_do_not_spend_budget_on_one_peak():
    model = GRMMatcher(config())
    probability = torch.zeros(1, 1, 25)
    probability[0, 0, torch.tensor([0, 1, 5, 24])] = torch.tensor([.9, .8, .7, .6])
    result = model.candidate_sampler.topk_candidates(probability, target_width=5)
    assert result.indices[0, 0, :2].tolist() == [0, 24]
    assert result.global_probs[0, 0, 2:].sum() == 0


def test_two_distant_peaks_do_not_average_to_an_empty_midpoint():
    corr = GRMLocalCorrelation(2, .05)
    source = torch.zeros(1, 2, 5, 5); source[:, 0] = 1
    target = -source.clone()
    target[0, :, 2, 0] = torch.tensor([1., 0.])
    target[0, :, 2, 4] = torch.tensor([1., .1])
    point = torch.tensor([[[2., 2.]]])
    old = corr(source, target, point, point, 1, 5)
    corr.peak_localization = True
    new = corr(source, target, point, point, 1, 5)
    assert abs(old.delta_xy[0, 0, 0]) < .2
    assert new.delta_xy[0, 0, 0] < -1.9
    assert torch.allclose(new.probability, old.probability)


@pytest.mark.parametrize("outside", [False, True])
def test_no_target_support_is_invalid(outside):
    core = GRMLocalRefinerCore(4, 1, 3, 5, .1, 1e-8)
    feature = torch.rand(1, 4, 8, 8)
    source = torch.tensor([[[3., 3.]]])
    target = source + (100 if outside else 0)
    mask = None if outside else torch.zeros(1, 8, 8, dtype=torch.bool)
    result = core(feature, feature, source, target, target_valid_map=mask)
    assert not result.valid_mask.any()
    assert result.local_probability.sum() == 0


def test_all_masked_forward_and_backward_are_finite():
    model = GRMMatcher(config())
    image = torch.rand(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    output, aux = model(image, image, True, valid_a=mask, valid_b=mask)
    assert not output.valid_mask.any() and not aux.candidate_valid_mask.any()
    assert torch.isfinite(output.dst_xy).all()
    output.dst_xy.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_geometry_quality_rejects_lines_but_accepts_compact_overlap():
    mask = torch.ones(1, 4, dtype=torch.bool)
    seed = SimpleNamespace(selected_mask=mask, inlier_mask=mask, homography_valid=torch.tensor([True]))
    square = torch.tensor([[[0., 0.], [1., 0.], [0., 1.], [1., 1.]]])
    assert GRMMatcher.geometry_quality(seed, square).item() == pytest.approx(1.)
    line = square.clone(); line[..., 1] = 0
    assert GRMMatcher.geometry_quality(seed, line).item() == 0
    seed.homography_valid[:] = False
    assert GRMMatcher.geometry_quality(seed, square).item() == 0


def test_direct_cycle_queries_actual_endpoint_and_independent_reverse(monkeypatch):
    model = GRMMatcher(config())
    features = {"F_coarse": torch.zeros(1, 128, 4, 4)}
    probability = torch.eye(16)[None]
    source = torch.tensor([[[7.5, 7.5]]])
    destination = torch.tensor([[[39.5, 39.5]]])
    def refine(a, b, query, initial, valid):
        assert torch.equal(query, destination)
        assert torch.equal(initial, destination)  # reverse coarse winner is NOT the supplied original source
        return None, SimpleNamespace(dst_xy=initial, valid_mask=valid)
    monkeypatch.setattr(model, "refine_points", refine)
    cycle = model.direct_cycle(features, features, probability, source, destination, torch.ones(1, 1, dtype=torch.bool))
    assert cycle.cycle_error.item() > 40 and cycle.cycle_score.item() < .01


def test_imc_objective_trains_context_bridge():
    torch.manual_seed(5)
    model = GRMMatcher(config()); configure_optimizer(model, "imc")
    y, x = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    points = (torch.stack((x,y),-1).float().reshape(1,-1,2)+.5)*16-.5
    batch = dict(image_a=torch.rand(1,3,64,64), image_b=torch.rand(1,3,64,64),
        sparse_a=points, sparse_b=points+torch.tensor([1.,0.]),
        sparse_valid=torch.ones(1,16,dtype=torch.bool), sparse_confidence=torch.ones(1,16),
        valid_a=torch.ones(1,1,64,64,dtype=torch.bool), valid_b=torch.ones(1,1,64,64,dtype=torch.bool),
        F_ab=torch.tensor([[[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]]))
    loss, _ = ScratchTransferObjective(model.config).imc(model, batch, True)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.context_bridge.mid.weight.grad.abs().sum() > 0
    assert model.context_bridge.fine.weight.grad.abs().sum() > 0


def test_inlier_targets_need_no_roles_and_do_not_backpropagate_labels():
    from grm_matcher.losses import GRMMatcherLoss
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    mapping = torch.stack((x,y))[None].float()
    source = torch.tensor([[[8.,8.],[12.,12.],[20.,20.],[24.,24.]]])
    destination = (source + torch.tensor([[[0.,0.],[5.,0.],[6.,0.],[0.,0.]]])).requires_grad_()
    overlap = torch.ones(1,1,32,32); overlap[:,:,24,24] = 0
    targets = dict(gt_dense_mapping=mapping, gt_overlap_a=overlap, gt_overlap_b=overlap)
    loss = GRMMatcherLoss(config()); loss._validate_targets(targets, 1)
    labels = loss._inlier_targets(SimpleNamespace(src_xy=source,dst_xy=destination), targets)
    assert labels.tolist() == [[1.,1.,0.,0.]]  # default GT threshold is 5 input pixels
    assert not labels.requires_grad


def test_binary_score_is_used_without_role_or_extra_threshold():
    model = GRMMatcher(config()).eval()
    image = torch.rand(1,3,64,64)
    output, auxiliary = model(image,image,True)
    assert not hasattr(model, "role_head")
    assert auxiliary.inlier_logits.shape == output.valid_mask.shape
    expected = auxiliary.inlier_logits.sigmoid()*output.valid_mask.float()
    assert torch.allclose(output.p_match, expected)
    assert torch.allclose(output.p_inlier, expected)
    assert output.p_local.count_nonzero() == 0
    output.validate(); output.detach().validate(); output.to(dtype=torch.float64).validate()


def test_four_point_homography_translation_and_gradients():
    from grm_matcher.seed_geometry import estimate_dlt_homography, apply_homography
    source = torch.tensor([[0.,0.],[20.,0.],[20.,20.],[0.,20.]])
    target = (source + torch.tensor([3.,7.])).requires_grad_()
    h = estimate_dlt_homography(source, target)
    assert torch.allclose(apply_homography(source[None], h[None])[0], target, atol=1e-4)
    h.square().sum().backward()
    assert torch.isfinite(target.grad).all() and target.grad.abs().sum() > 0


def test_agricultural_objective_backward_with_gt():
    from grm_matcher.architecture_upgrade import v2_config
    cfg = v2_config(config()); cfg.max_source_queries = 16
    cfg.classification_mode = "inlier"  # regression coverage for historical binary checkpoints
    model = GRMMatcher(cfg); configure_optimizer(model, "agri")
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    image = torch.rand(1,3,64,64)
    mask = torch.ones(1,1,64,64,dtype=torch.bool)
    batch = dict(image_a=image, image_b=image, valid_a=mask, valid_b=mask,
        H_ab=torch.eye(3)[None], gt_dense_mapping=torch.stack((x,y))[None].float(),
        gt_overlap_a=mask, gt_overlap_b=mask, local_flow=torch.zeros(1,2,64,64))
    loss, metrics = ScratchTransferObjective(cfg).agri(model, batch, True, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.context_bridge.fine.weight.grad.abs().sum() > 0
    assert model.inlier_head.classifier[-1].weight.grad.abs().sum() > 0
    assert "inlier" in metrics and "role" not in metrics
