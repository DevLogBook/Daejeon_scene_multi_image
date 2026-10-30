"""Behavior and gradient contracts for opt-in V3; no learned accuracy claims."""
import pytest
import torch
from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.architecture_v3 import peak_moments, select_queries
from grm_matcher.scratch_transfer import ScratchTransferObjective, configure_optimizer


def config(**overrides):
    values = dict(input_height=64, input_width=64, max_source_queries=16,
        backbone_stage_blocks=(1,1,1), architecture_version='v3', classification_mode='none',
        topk_candidates=1, v3_mid_hidden=16, v3_fine_hidden=8, v3_revision=1)
    return GRMMatcherConfig(**(values | overrides))


def data():
    y,x = torch.meshgrid(torch.arange(4),torch.arange(4),indexing='ij')
    points = (torch.stack((x,y),-1).float().reshape(1,-1,2)+.5)*16-.5
    a = torch.rand(1,3,64,64)
    mask = torch.ones(1,1,64,64,dtype=torch.bool)
    return dict(image_a=a, image_b=a.clone(), sparse_a=points, sparse_b=points,
        sparse_valid=torch.ones(1,16,dtype=torch.bool), sparse_confidence=torch.ones(1,16),
        F_ab=torch.tensor([[[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]]),
        valid_a=mask,valid_b=mask,gt_overlap_a=mask,gt_overlap_b=mask,H_ab=torch.eye(3)[None])


def test_peak_location_retains_remote_ambiguity():
    p = torch.zeros(1,1,7,7)
    p[0,0,3,0]=.51; p[0,0,3,6]=.49
    location,covariance=peak_moments(p)
    assert location[0,0,0] == -3
    assert covariance[0,0,0,0] > 8


def test_selection_no_quotas_duplicates_or_confidence_threshold():
    scores=torch.tensor([[1.,.001,.95,.002,.9,.003]])
    ids, valid=select_queries(scores,torch.ones_like(scores,dtype=torch.bool),3,3)
    assert set(ids[0].tolist())=={0,2,4}
    assert valid.all()
    _, valid=select_queries(scores,torch.zeros_like(scores,dtype=torch.bool),3,3)
    assert not valid.any()


def test_confidence_does_not_filter_and_invalid_support_is_finite():
    torch.manual_seed(31)
    m=GRMMatcher(config()).eval(); b=data()
    assert not any(k.startswith(('geometry_graph','seed_geometry','role_head','inlier_head','rerank_head')) for k in m.state_dict())
    with torch.no_grad():
        before=m(b['image_a'],b['image_b'])
        m.confidence_head[-1].bias.fill_(-100)
        after=m(b['image_a'],b['image_b'])
        assert torch.equal(before.valid_mask,after.valid_mask)
        assert torch.equal(before.dst_xy,after.dst_xy)
        assert after.p_match.max()<1e-20
        mask=torch.zeros_like(b['valid_a'])
        empty=m(b['image_a'],b['image_b'],valid_b=mask)
        empty.validate(); assert not empty.valid_mask.any()
    GRMMatcher(config()).load_state_dict(m.state_dict(),strict=True)


@pytest.mark.parametrize('stage',['imc','agri'])
def test_all_new_paths_receive_finite_gradients(stage):
    torch.manual_seed(4)
    c=config(); m=GRMMatcher(c); b=data()
    optimizer=configure_optimizer(m,stage)
    objective=ScratchTransferObjective(c)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics=(objective.imc(m,b,False) if stage=='imc' else objective.agri(m,b,True,0))
        assert torch.isfinite(loss), metrics
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        for prefix in ('backbone.','coarse_transformer.','correlation.', 'context_bridge.',
                       'mid_refiner.correlation.cost_net.','fine_refiner.correlation.cost_net.', 'confidence_head.'):
            assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.startswith(prefix)),prefix
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        optimizer.step()
    assert metrics['confidence_supervised_count']>0


def test_configuration_cannot_silently_enable_role_or_multiple_candidates():
    with pytest.raises(ValueError): config(topk_candidates=4)
    with pytest.raises(ValueError): config(classification_mode='inlier')


def test_augmented_imc_and_empty_supervision():
    torch.manual_seed(10)
    c=config(); m=GRMMatcher(c); b=data(); objective=ScratchTransferObjective(c)
    loss, _=objective.imc(m,b,True)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    b['sparse_valid'].zero_()
    loss, metrics=objective.imc(m,b,False)
    assert loss is None and metrics['valid_count']==0


def test_optional_confidence_head_uses_same_points():
    c=config(); m=GRMMatcher(c).eval(); b=data()
    other=GRMMatcher(config(v3_confidence_head=False)).eval()
    other.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('confidence_head.')},strict=True)
    with torch.no_grad():
        a=m(b['image_a'],b['image_b']); z=other(b['image_a'],b['image_b'])
    assert torch.equal(a.src_xy,z.src_xy) and torch.equal(a.dst_xy,z.dst_xy)
    assert torch.equal(a.valid_mask,z.valid_mask)


def test_saved_v3_works_with_agricultural_evaluation_adapter(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from agri_eval_adapter import Matcher
    c=config(); model=GRMMatcher(c)
    path=tmp_path/'random_test_fixture.pt'
    torch.save(dict(config=c.to_dict(),model=model.state_dict()),path)
    args=SimpleNamespace(checkpoint=path,device='cpu',size=64)
    matcher=Matcher('GRM',tmp_path,args)
    image=np.zeros((64,64,3),dtype=np.uint8)
    source,target,score=matcher(image,image)
    assert source.shape==target.shape==(16,2) and score.shape==(16,)
    args.grm_rerank_mode='coarse_prior'
    with pytest.raises(ValueError,match='legacy'):
        Matcher('GRM',tmp_path,args)


def test_v3_mask_centers_match_public_half_pixel_coordinates():
    m=GRMMatcher(config()).eval(); b=data()
    mask=torch.zeros_like(b['valid_a']); mask[:,:,:,4:12]=True
    with torch.no_grad():
        fa,_,_,_=m.encode_pair(b['image_a'],b['image_b'],mask,mask)
    # First coarse center is x=7.5, not the top-left x=0 sampled by legacy nearest.
    assert fa['valid_coarse'][0,:,0].all()
    assert not fa['valid_coarse'][0,:,1:].any()


def test_target_argmax_uses_target_matchability_before_selection():
    from types import SimpleNamespace
    from grm_matcher.architecture_v3 import target_support
    corr=SimpleNamespace(dual_probability=torch.tensor([[[.4,.36]]]),
        matchability_a=torch.ones(1,1),matchability_b=torch.tensor([[.1,.9]]))
    score,target,valid=target_support(corr,torch.ones(1,1,dtype=torch.bool),torch.ones(1,2,dtype=torch.bool),config())
    assert target.item()==1 and score.item()==pytest.approx(.324) and valid.item()


def test_underfill_really_shortens_refiner_input_and_empty_skips_it():
    m=GRMMatcher(config()).eval();b=data();counts=[]
    hook=m.mid_refiner.register_forward_pre_hook(lambda module,args: counts.append(args[2].shape[1]))
    def support(module,args,corr):
        corr.matchability_a=torch.full_like(corr.matchability_a,.01)
        corr.matchability_a[:,:3]=.9
        corr.matchability_b=torch.full_like(corr.matchability_b,.9)
        return corr
    ch=m.correlation.register_forward_hook(support)
    with torch.no_grad(): out=m(b['image_a'],b['image_b'])
    assert out.num_matches==3 and counts==[3,3]
    counts.clear();m.config.v3_min_matchability=.99
    with torch.no_grad(): out=m(b['image_a'],b['image_b'])
    out.validate();assert out.num_matches==0 and counts==[]
    hook.remove();ch.remove()


def test_mixed_batch_padding_only_to_largest_support():
    score=torch.ones(2,16); mask=torch.zeros_like(score,dtype=torch.bool)
    mask[0,:3]=True;mask[1,:1]=True
    ids,valid=select_queries(score,mask,4,16)
    assert ids.shape==(2,3) and valid.sum(1).tolist()==[3,1]


def test_patch_neighbors_reach_cost_net_and_receive_gradients():
    from grm_matcher.architecture_v3 import CostVolumeCorrelation
    torch.manual_seed(23)
    module=CostVolumeCorrelation(4,.1,1e-8,8,patch_size=3)
    source=torch.randn(1,4,9,9,requires_grad=True);target=torch.randn(1,4,9,9)
    point=torch.tensor([[[4.,4.]]]); inputs=[]
    hook=module.cost_net.register_forward_pre_hook(lambda m,args: inputs.append(args[0].detach().clone()))
    first=module(source,target,point,point,3,5)
    changed=source.detach().clone();changed[:,:,3,3]*=-1
    module(changed,target,point,point,3,5)
    assert inputs[0].shape[1]==22
    assert torch.equal(inputs[0][:,:4],inputs[1][:,:4])
    assert not torch.equal(inputs[0][:,4:13],inputs[1][:,4:13])
    first.logits.sum().backward()
    assert source.grad[:,:,3,3].abs().sum()>0
    hook.remove()


def test_no_predicted_support_does_not_starve_supervised_training():
    c=config(v3_min_matchability=.999);m=GRMMatcher(c);b=data()
    loss,metrics=ScratchTransferObjective(c).agri(m,b,True,0)
    assert torch.isfinite(loss) and metrics['predicted_matches']==0
    loss.backward()
    assert m.mid_refiner.correlation.cost_net[0].weight.grad.abs().sum()>0
    assert m.confidence_head[-1].weight.grad.abs().sum()>0


def test_revision_zero_keeps_four_channel_checkpoint_schema():
    m=GRMMatcher(config(v3_revision=0))
    assert m.mid_refiner.correlation.cost_net[0].in_channels==4
    GRMMatcher(config(v3_revision=0)).load_state_dict(m.state_dict(),strict=True)
    with pytest.raises(RuntimeError):
        GRMMatcher(config()).load_state_dict(m.state_dict(),strict=True)


def test_synthetic_scratch_cli_and_resume_are_exclusive():
    from grm_matcher.train_scratch_transfer import parse_args
    base=['--stage','agri','--data-dir','dataset','--output-dir','output']
    args=parse_args(base+['--scratch','--config','grm_matcher/configs/grm_v3.json'])
    assert args.scratch and args.init_checkpoint is None and args.resume is None
    with pytest.raises(SystemExit): parse_args(base+['--scratch','--resume','latest.pt'])
    with pytest.raises(SystemExit): parse_args(base+['--scratch','--init-checkpoint','old.pt'])


def test_split_identity_preserves_source_directory(tmp_path):
    import json
    from grm_matcher.scratch_transfer import check_homography_splits
    for split,source in [('train','Corn/images/1.JPG'),('val','Oat/images/1.JPG')]:
        path=tmp_path/split;path.mkdir()
        (path/'manifest.jsonl').write_text(json.dumps({'source_image':source})+'\n')
    assert check_homography_splits(tmp_path)['train']['pairs']==1
    (tmp_path/'val/manifest.jsonl').write_text(json.dumps({'source_image':'Corn/images/../images/1.jpg'})+'\n')
    with pytest.raises(ValueError,match='overlap'):check_homography_splits(tmp_path)
