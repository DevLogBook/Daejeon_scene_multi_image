"""F8 integration, masked aggregation geometry, cache stride and migration tests."""
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.hra import HierarchicalReliabilityAggregation
from grm_matcher.depth_curriculum import parameter_group, phase_mode, CurriculumObjective
from grm_matcher.depth_finetune import DepthCacheDataset
from grm_matcher.depth_geometry import Frame, coarse_grid, direction
from grm_matcher.coordinates import flat_indices_to_image_xy
from tools.upgrade_grm_f8_checkpoint import upgrade


def config(mode='native16'):
    return GRMMatcherConfig(input_height=64, input_width=64, architecture_version='v3',
        v3_revision=1, classification_mode='none', topk_candidates=1, coarse_dim=32,
        mid_dim=32, fine_dim=32, transformer_ffn_dim=64, v3_mid_hidden=8, v3_fine_hidden=8,
        max_source_queries=8, backbone_stage_blocks=(1,1,1), v3_min_matchability=0,
        v3_relative_reliability=0, coarse_stride=16 if mode=='native16' else 8,
        v3_coarse_mode=mode, hra_hidden_dim=8)


def test_masked_hra_moments_and_hierarchical_total_covariance():
    torch.manual_seed(7)
    features = torch.randn(1, 8, 4, 4)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    first = HierarchicalReliabilityAggregation(8, input_stride=8)(features, mask)
    second = HierarchicalReliabilityAggregation(8, input_stride=16)(
        first.features, first.valid, first.positions, first.covariance)
    torch.testing.assert_close(first.positions[0,0,0], torch.tensor([7.5,7.5]))
    torch.testing.assert_close(first.covariance[0,0,0], torch.eye(2)*16)
    torch.testing.assert_close(second.positions[0,0,0], torch.tensor([15.5,15.5]))
    # Four positions per axis at 3.5,11.5,19.5,27.5 have variance 80.
    torch.testing.assert_close(second.covariance[0,0,0], torch.eye(2)*80)


def test_padding_cannot_change_features_or_weights_even_when_nan():
    pool = HierarchicalReliabilityAggregation(8)
    features = torch.randn(1,8,3,5)
    mask = torch.zeros(1,3,5,dtype=torch.bool); mask[:,0,0] = True
    expected = pool(features,mask)
    poisoned = features.masked_fill(~mask[:,None], float('nan'))
    actual = pool(poisoned,mask)
    for name in ('features','positions','covariance','weights'):
        torch.testing.assert_close(getattr(expected,name),getattr(actual,name))
        assert torch.isfinite(getattr(actual,name)).all()
    assert actual.features.shape == (1,8,2,3)
    torch.testing.assert_close(actual.weights[0,0,0],torch.tensor([1.,0,0,0]))
    assert not actual.features.permute(0,2,3,1)[~actual.valid].any()


@pytest.mark.parametrize('mode',['f8','f8_hra'])
def test_f8_forward_backward_and_checkpoint_round_trip(mode):
    torch.manual_seed(12)
    cfg = config(mode)
    model = GRMMatcher(cfg)
    a,b = torch.rand(1,3,64,64),torch.rand(1,3,64,64)
    phase_mode(model,'A',True)
    fa,fb,ta,tb = model.encode_pair(a,b)
    assert fa['F_coarse'].shape == (1,32,8,8)
    corr = model.correlation(ta,tb)
    truth = torch.arange(64)
    loss = -corr.probability[0,truth,truth].clamp_min(1e-8).log().mean()
    loss.backward()
    assert all(parameter_group(name) in ('backbone','coarse') for name,p in model.named_parameters() if p.grad is not None)
    assert model.fpn.coarse_f8_lateral.weight.grad.abs().sum() > 0
    if mode == 'f8_hra':
        assert model.coarse_transformer.last_token_counts == (64,16,4)
        assert model.coarse_transformer.pool16.score[-1].weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        output = model(a,b)
        assert output.src_xy.shape == (1,8,2)
        assert torch.isfinite(output.dst_xy).all()
        # Actual source addresses remain fixed F8 centers, not HRA centroids.
        torch.testing.assert_close((output.src_xy+.5).remainder(8),torch.full_like(output.src_xy,4))
        restored = GRMMatcher(GRMMatcherConfig.from_dict(cfg.to_dict())).eval()
        restored.load_state_dict(model.state_dict(),strict=True)
        again = restored(a,b)
        torch.testing.assert_close(output.dst_xy,again.dst_xy,rtol=0,atol=0)


def test_legacy_default_has_exact_state_compatibility_and_migration_copies_weights():
    cfg = config(); model = GRMMatcher(cfg)
    legacy = cfg.to_dict()
    for key in ('v3_coarse_mode','hra_hidden_dim','hra_max_tokens'):
        legacy.pop(key)
    restored = GRMMatcher(GRMMatcherConfig.from_dict(legacy))
    restored.load_state_dict(model.state_dict(),strict=True)
    assert not any('pool16' in key or 'coarse_f8' in key for key in model.state_dict())
    raw = dict(config=legacy,model=model.state_dict(),optimizer={'should':'not migrate'})
    for mode in ('f8','f8_hra'):
        converted = upgrade(raw,mode)
        assert 'optimizer' not in converted and not converted['confidence_head_trained']
        candidate = GRMMatcher(GRMMatcherConfig.from_dict(converted['config']))
        candidate.load_state_dict(converted['model'],strict=True)
        for key,value in raw['model'].items():
            mapped = key.replace('coarse_transformer.','coarse_transformer.context.',1) if mode=='f8_hra' and key.startswith('coarse_transformer.') else key
            torch.testing.assert_close(value,converted['model'][mapped],rtol=0,atol=0)


def test_all_invalid_hra_full_forward_is_finite_and_empty():
    model=GRMMatcher(config('f8_hra')).eval()
    image=torch.rand(1,3,64,64); mask=torch.zeros(1,1,64,64,dtype=torch.bool)
    with torch.no_grad():
        output=model(image,image,valid_a=mask,valid_b=mask)
    assert output.src_xy.shape[1]==0


def test_global_budget_is_enforced_before_attention():
    cfg=config('f8_hra'); cfg.hra_max_tokens=3
    model=GRMMatcher(cfg)
    with pytest.raises(ValueError,match='budget'):
        model(torch.rand(1,3,64,64),torch.rand(1,3,64,64))


def test_geometry_cache_refuses_legacy_stride_for_f8_before_reading_manifest():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        (root/'preparation.json').write_text(json.dumps(dict(recipe='v3_depth_cache_v1',size=64)))
        with pytest.raises(ValueError,match='stride 16 differs'):
            DepthCacheDataset(root,'train','megadepth',config('f8'))


def test_depth_projection_recomputes_exact_f8_grid(monkeypatch):
    import grm_matcher.depth_geometry as module
    # Identity cameras/depth should project every new F8 center to itself.
    monkeypatch.setattr(module,'read_depth',lambda ignored: np.ones((64,64),np.float32)*10)
    frame=Frame('identity',Path('image.jpg'),Path('depth.h5'),np.array([[50.,0,32],[0,50,32],[0,0,1]]),
                np.zeros(5),np.eye(4),64,64)
    fine=direction(frame,frame,size=64,stride=8)
    old=direction(frame,frame,size=64)
    assert fine['target'].shape==(64,2) and old['target'].shape==(16,2)
    assert fine['valid'].all()
    np.testing.assert_allclose(fine['target'],coarse_grid(64,8),atol=1e-5)


@pytest.mark.parametrize('phase',['A','B','C'])
def test_hra_curriculum_loss_has_real_gradients_in_each_active_group(phase):
    torch.manual_seed(42)
    cfg=config('f8_hra'); model=GRMMatcher(cfg)
    image=torch.rand(1,3,64,64)
    mask=torch.ones(1,1,64,64,dtype=torch.bool)
    points=flat_indices_to_image_xy(torch.arange(64)[None],8,8)
    batch=dict(domain=['megadepth'],image_a=image,image_b=image.clone(),valid_a=mask,valid_b=mask,
        ab_target=points,ba_target=points,ab_valid=torch.ones(1,64,dtype=torch.bool),
        ba_valid=torch.ones(1,64,dtype=torch.bool),ab_weight=torch.ones(1,64),ba_weight=torch.ones(1,64))
    phase_mode(model,phase,True)
    objective=CurriculumObjective(cfg,phase,anchor_points=8,anchor_jitter=0)
    loss,metrics=objective.agri(model,batch,True)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    from grm_matcher.train_depth_curriculum import gradient_audit
    audit=gradient_audit(model,phase)
    assert audit['coarse' if phase=='A' else 'fine']['squared_norm'] > 0


def test_f8_depth_cache_reads_every_new_source_center():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); (root/'train').mkdir()
        (root/'preparation.json').write_text(json.dumps(dict(recipe='v3_depth_cache_v1',size=64,coarse_stride=8)))
        np.savez(root/'rgb.npz',rgb=np.zeros((64,64,3),np.uint8),valid=np.ones((64,64),bool))
        labels={}
        for prefix in ('ab','ba'):
            labels.update({prefix+'_target':coarse_grid(64,8).astype(np.float32),
                prefix+'_valid':np.ones(64,bool),prefix+'_weight':np.ones(64,np.float32)})
        np.savez(root/'labels.npz',**labels)
        record=dict(domain='megadepth',pair_id='identity',scene='0001',group='megadepth/0001',
                    image_a='rgb.npz',image_b='rgb.npz',labels='labels.npz',source_sha256_a='a',source_sha256_b='b')
        (root/'train/manifest.jsonl').write_text(json.dumps(record)+'\n')
        data=DepthCacheDataset(root,'train','megadepth',config('f8_hra'))
        assert data[0]['ab_target'].shape==(64,2)


def test_prepare_stride8_uses_existing_split_and_loads_generated_labels():
    from tests.test_depth_finetune import fixture_data
    from tools.prepare_grm_depth_data import main as prepare
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); fixture_data(root)
        splits={'megadepth/0001':'train','megadepth/0002':'val','uav/corn1':'train','uav/corn2':'val'}
        (root/'old_preparation.json').write_text(json.dumps(dict(group_split=splits)))
        out=root/'f8_cache'
        prepare(['--megadepth-scenes',str(root/'md/scenes'),'--uav-rgb',str(root/'rgb'),
            '--uav-depth',str(root/'depth'),'--output',str(out),'--size','64','--coarse-stride','8',
            '--split-json',str(root/'old_preparation.json'),'--max-pairs-per-scene','1',
            '--min-points','2','--min-overlap','0','--max-overlap','1'])
        metadata=json.loads((out/'preparation.json').read_text())
        assert metadata['coarse_stride']==8 and metadata['group_split']==splits
        for domain in ('megadepth','uav'):
            for split in ('train','val'):
                data=DepthCacheDataset(out,split,domain,config('f8_hra'))
                assert data[0]['ab_target'].shape==(64,2)
