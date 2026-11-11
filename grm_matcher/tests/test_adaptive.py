"""Adaptive routing, coordinate, migration and gradient contracts."""
from dataclasses import replace
import pytest
import torch

from grm_matcher.adaptive import (SearchRiskHead, VarianceCalibrator, budget_route, configure_stage,
    ground_truth_at, grouped_mid, load_adaptive_state, oracle_forward, risk_loss, search_labels, variance_nll)
from grm_matcher.adaptive_metrics import covariance_metrics, risk_metrics
from grm_matcher.matcher import GRMMatcher
from grm_matcher.scratch_transfer import ScratchTransferObjective
from grm_matcher.tests.test_architecture_v3 import config, data


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_risk_shape_and_labels_square_stride():
    head = SearchRiskHead()
    assert head(torch.randn(2, 13, 8)).shape == (2, 13)
    initial = torch.full((1, 7, 2), 47.5)
    offset = torch.tensor([[[16.,16.],[16.01,0.],[32.,32.],[32.01,0.],[0.,0.],[0.,0.],[float('nan'),0.]]])
    visible = torch.tensor([[True,True,True,True,False,True,True]])
    labels = search_labels(initial, initial+offset, visible, 8)
    assert labels.category.tolist() == [[0,1,1,2,3,0,3]]
    logits = torch.zeros(1, 7, requires_grad=True)
    loss = risk_loss(logits, labels, gamma=2)
    loss.backward()
    assert logits.grad[0, 3] == logits.grad[0, 4] == logits.grad[0, 6] == 0
    assert logits.grad[0, 1] < 0 and logits.grad[0, 0] > 0


@pytest.mark.parametrize('fraction', [.1,.2,.3])
def test_per_image_topk_excludes_padding(fraction):
    c = config(adaptive_enabled=True, adaptive_budget=fraction)
    scores = torch.arange(20).float()[None].expand(2, -1).clone()
    valid = torch.ones(2, 20, dtype=torch.bool); valid[1, 10:] = False
    scores[1, 10:] = 100
    route = budget_route(scores, valid, c)
    assert route.sum(1).tolist() == [int(20*fraction), int(10*fraction)]
    assert not route[1, 10:].any()
    assert route[0, -int(20*fraction):].all()


def test_dynamic_budget_and_empty():
    c=config(adaptive_enabled=True, adaptive_dynamic_budget=True)
    mask=torch.ones(2,100,dtype=torch.bool)
    score=torch.stack((torch.zeros(100),torch.ones(100)))
    route=budget_route(score,mask,c)
    assert route.sum(1).tolist()==[5,30]
    assert not budget_route(score,mask & False,c).any()
    assert budget_route(score[:,:0],mask[:,:0],c).shape==(2,0)


def features(b=2):
    torch.manual_seed(45)
    m=GRMMatcher(config(adaptive_enabled=True)).eval()
    image=torch.rand(b,3,64,64)
    with torch.no_grad():
        fa,fb,_,_=m.encode_pair(image,image)
    return m,fa,fb


def test_grouped_windows_match_independent_calls_and_query_order():
    m,fa,fb=features()
    source=torch.tensor([[[15.5,15.5],[31.5,31.5],[47.5,47.5],[7.5,7.5]]]).expand(2,-1,-1)
    valid=torch.tensor([[True,True,True,True],[True,False,True,False]])
    expand=torch.tensor([[False,True,False,True],[True,False,False,False]])
    grouped=grouped_mid(m.mid_refiner,fa,fb,source,source,valid,expand)
    for row in range(2):
        for slot in range(4):
            if not valid[row,slot]:
                assert grouped.local_probability[row,slot].sum()==0
                continue
            size=9 if expand[row,slot] else 5
            one=m.mid_refiner(fa['F_mid'][row:row+1],fb['F_mid'][row:row+1],
                source[row:row+1,slot:slot+1],source[row:row+1,slot:slot+1],
                target_window_size=size,source_valid_map=fa['valid_mid'][row:row+1],target_valid_map=fb['valid_mid'][row:row+1])
            assert torch.allclose(grouped.dst_xy[row,slot],one.dst_xy[0,0],atol=2e-5)
            p=grouped.local_probability[row,slot]
            if size==5:
                assert p[:2].sum()==0 and p[:,:2].sum()==0
                p=p[2:7,2:7]
            assert torch.allclose(p,one.local_probability[0,0],atol=1e-6)
    grouped.dst_xy.sum().backward()
    assert m.mid_refiner.correlation.cost_net[0].weight.grad.abs().sum()>0


def test_grouping_reduces_real_cost_cnn_cells():
    m,fa,fb=features(1)
    point=torch.full((1,10,2),31.5);valid=torch.ones(1,10,dtype=torch.bool)
    expand=torch.zeros_like(valid);expand[:,:2]=True
    shapes=[]
    hook=m.mid_refiner.correlation.cost_net.register_forward_pre_hook(lambda _,args:shapes.append(args[0].shape))
    grouped_mid(m.mid_refiner,fa,fb,point,point,valid,expand)
    hook.remove()
    assert [(s[0],s[-1]) for s in shapes]==[(8,5),(2,9)]
    assert sum(s[0]*s[-1]**2 for s in shapes)==362 < 10*49


@pytest.mark.parametrize('size',[5,7,9])
def test_window_coordinate_mapping_and_boundaries(size):
    from grm_matcher.local_correlation import GRMLocalRefinerCore
    from grm_matcher.coordinates import feature_xy_to_image_xy
    torch.manual_seed(15)
    # Orthogonal per-cell descriptors make a known remote peak unambiguous.
    f=torch.eye(81).reshape(1,81,9,9)
    refiner=GRMLocalRefinerCore(81,8,1,7,.01,1e-8)
    refiner.correlation.peak_localization=True
    source=feature_xy_to_image_xy(torch.tensor([[[4.,4.],[0.,0.]]]),8)
    initial=source.clone();initial[0,0,0]-=(size//2)*8
    out=refiner(f,f,source,initial,target_window_size=size)
    assert torch.allclose(out.dst_xy,source,atol=1e-3)
    assert torch.allclose(out.local_probability.sum((-1,-2)),torch.ones(1,2))
    assert out.local_probability[0,1,:size//2].sum()==0
    blank=torch.zeros(1,9,9,dtype=torch.bool)
    none=refiner(f,f,source,initial,target_window_size=size,target_valid_map=blank)
    assert not none.valid_mask.any() and none.local_probability.sum()==0


def test_covariance_spd_nll_and_calibration_gradient():
    m,fa,fb=features(1)
    point=torch.full((1,4,2),31.5);valid=torch.ones(1,4,dtype=torch.bool)
    _,fine=m.refine_points(fa,fb,point,point,valid)
    head=VarianceCalibrator(config())
    covariance=head(fine)
    assert torch.linalg.eigvalsh(covariance).min()>0
    assert torch.allclose(covariance,covariance.transpose(-1,-2))
    loss=variance_nll(covariance,torch.ones_like(point)*3,valid)
    loss.backward()
    assert torch.isfinite(head.net[-1].weight.grad).all() and head.net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.parameters())
    stats=covariance_metrics(covariance[valid],torch.ones(4,2))
    assert 0 <= stats['coverage95'] <= 1


def test_disabled_baseline_and_explicit_checkpoint_upgrade():
    torch.manual_seed(32)
    base=GRMMatcher(config()).eval();image=data()['image_a']
    saved=base.state_dict()
    legacy_values={k:v for k,v in base.config.to_dict().items() if not k.startswith('adaptive_')}
    from grm_matcher.config import GRMMatcherConfig
    restored=GRMMatcher(GRMMatcherConfig.from_dict(legacy_values)).eval()
    restored.load_state_dict(saved,strict=True)
    assert not any('search_risk' in k for k in saved)
    with torch.no_grad():
        expected=base(image,image)
        actual=restored(image,image)
    for k,v in expected.as_dict().items():
        assert torch.equal(v,actual.as_dict()[k]),k
    adaptive=GRMMatcher(config(adaptive_enabled=True,adaptive_mode='fixed_7')).eval()
    with pytest.warns(UserWarning,match='migration'):
        load_adaptive_state(adaptive,saved,upgrade=True)
    with torch.no_grad(): other=adaptive(image,image)
    for k,v in expected.as_dict().items(): assert torch.equal(v,other.as_dict()[k]),k
    with pytest.raises(RuntimeError): load_adaptive_state(adaptive,saved)
    broken=dict(saved);del broken['backbone.stem.0.weight']
    with pytest.raises(RuntimeError): load_adaptive_state(adaptive,broken,upgrade=True)


def test_oracle_is_explicit_and_uses_true_inverse_dense_field():
    c=config(adaptive_enabled=True,adaptive_mode='oracle_5_9');m=GRMMatcher(c).eval();b=data()
    with pytest.raises(ValueError,match='Oracle'): m(b['image_a'],b['image_b'])
    y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
    grid=torch.stack((x,y),0).float()[None]
    b['gt_dense_mapping']=grid+4
    b['gt_dense_mapping_ba']=grid-7  # Deliberately not inverse H or swapped AB labels.
    points=torch.tensor([[[31.5,31.5]]])
    truth,valid=ground_truth_at(b,points,'b')
    assert torch.allclose(truth,points-7) and valid.all()
    with torch.no_grad(): out,aux=oracle_forward(m,b,True)
    assert aux.expanded_mask.shape==out.valid_mask.shape
    truth,visible=ground_truth_at(b,out.src_xy)
    initial=(torch.stack((aux.target_indices%4,aux.target_indices//4),-1)+.5)*16-.5
    # Mid invalidation may remove routed points; oracle labels are made before Fine.
    labels=search_labels(initial,truth,visible,8)
    assert not (aux.expanded_mask & ~labels.expand).any()


def test_learned_inference_needs_no_gt_including_reverse():
    m=GRMMatcher(config(adaptive_enabled=True,adaptive_variance=True)).eval()
    image=torch.rand(2,3,64,64)
    with torch.no_grad(): out,aux=m(image,image,return_auxiliary=True)
    out.validate()
    assert aux.risk_logits.shape==out.valid_mask.shape
    assert aux.calibrated_covariance is not None
    assert torch.linalg.eigvalsh(out.fine_uncertainty).min()>0


@pytest.mark.parametrize('stage',list('ABCDE'))
def test_stage_training_has_finite_correct_head_gradients(stage):
    c=config(adaptive_enabled=True,adaptive_variance=stage in 'DE')
    m=GRMMatcher(c);params=configure_stage(m,stage);b=data()
    loss,metrics=ScratchTransferObjective(c).agri(m,b,True,0)
    assert torch.isfinite(loss),metrics
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)
    assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    if stage=='A': assert all(not p.requires_grad for n,p in m.named_parameters() if not n.startswith('search_risk_head.'))
    if stage=='D': assert all(not p.requires_grad for n,p in m.named_parameters() if not n.startswith('variance_calibrator.'))


def test_ranking_metrics_and_ties():
    stats=risk_metrics(torch.tensor([.1,.2,.8,.9]),torch.tensor([0,0,1,1]))
    assert stats['auroc']==stats['auprc']==1
    tied=risk_metrics(torch.ones(4),torch.tensor([0,1,0,1]))
    assert tied['auroc']==.5 and tied['auprc']==.5
    assert risk_metrics(torch.ones(3),torch.zeros(3))['auroc'] is None


def test_inverse_dense_field_cannot_be_silently_replaced_by_inverse_h():
    b=data()
    y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
    b['gt_dense_mapping']=torch.stack((x,y),0).float()[None]+4
    with pytest.raises(ValueError,match='true gt_dense_mapping_ba'):
        ground_truth_at(b,torch.tensor([[[31.5,31.5]]]),'b')


def test_dense_flow_supervision_uses_true_correspondences():
    b=data()
    y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
    b['gt_dense_mapping']=torch.stack((x+2,y),0).float()[None]
    b['gt_dense_mapping_ba']=torch.stack((x-2,y),0).float()[None]
    b['augmentation_strength']=torch.tensor([1.])
    b['flow_active']=torch.tensor([True]);b['inverse_residual_max']=torch.tensor([0.])
    m=GRMMatcher(config(adaptive_enabled=True,adaptive_mode='oracle_5_9'))
    loss,metrics=ScratchTransferObjective(m.config,exact_dense=True).agri(m,b,True,0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.search_risk_head.net[-1].weight.grad.abs().sum()>0


def test_scenarios_preserve_shapes_and_occlusion_labels():
    from grm_matcher.adaptive_data import ScenarioDataset, SCENARIOS
    class Base:
        def __len__(self):return 1
        def __getitem__(self,index):
            b=data()
            b={k:v[0].clone() if torch.is_tensor(v) else v for k,v in b.items()}
            y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
            b['gt_dense_mapping']=torch.stack((x,y),0).float()
            b['local_flow']=torch.zeros(2,64,64)
            return b
    for scenario in SCENARIOS:
        b=ScenarioDataset(Base(),config(),scenario)[0]
        assert b['image_a'].shape==b['image_b'].shape==(3,64,64)
        assert b['gt_overlap_a'].dtype==torch.bool
        if scenario=='occlusion_low_overlap':assert b['gt_overlap_a'].float().mean()<.6
        if scenario=='local_parallax':
            assert b['flow_active'] and 'gt_dense_mapping_ba' in b
