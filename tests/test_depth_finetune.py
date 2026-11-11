"""Geometry conventions, unknown masks, staged reachability and gradient scopes."""
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import h5py
import numpy as np
import pytest
import tifffile
import torch
from torch.utils.data import DataLoader

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.depth_geometry import Frame,direction,discover_disk,group_splits
from grm_matcher.depth_finetune import DepthCacheDataset,DepthObjective,refinement_terms,FINE_PREFIXES
from tools.prepare_grm_depth_data import main as prepare


def fixture_data(root):
    k=np.array([[80.,0,31.5],[0,80.,31.5],[0,0,1.]])
    catalog={}
    for number in (1,2):
        scene=root/'md'/'scenes'/f'{number:04d}'
        for sub in ('images','calibration','depth_maps'): (scene/sub).mkdir(parents=True)
        for index in (0,1):
            rgb=np.random.default_rng(number*10+index).integers(0,255,(64,64,3),dtype=np.uint8)
            cv2.imwrite(str(scene/'images'/f'{index}.png'),rgb)
            with h5py.File(scene/'calibration'/f'calibration_{index}.png.h5','w') as f:
                f['K']=k; f['R']=np.eye(3); f['T']=[-.2*index,0,0]
            with h5py.File(scene/'depth_maps'/f'{index}.h5','w') as f: f['depth']=np.full((32,32),2.,np.float32)
        catalog[scene.name]=dict(images=['0.png','1.png'],tuples=[[0,1]])
    (root/'md'/'dataset.json').write_text(json.dumps(catalog))
    for number in (1,2):
        name=f'Day001_Corn{number}'; scene=root/'rgb'/'2024'/name
        ds=root/'depth'/'2024'/(name+'_depth')
        (scene/'images').mkdir(parents=True); ds.mkdir(parents=True)
        frames=[]
        for index in (0,1):
            cv2.imwrite(str(scene/'images'/f'{index}.JPG'),np.random.default_rng(100+number*10+index).integers(0,255,(64,64,3),dtype=np.uint8))
            tifffile.imwrite(ds/f'{index}_depth.tif',np.full((64,64),2.,np.float32))
            pose=np.eye(4); pose[0,3]=.2*index; pose=pose@np.diag([1.,-1.,-1.,1.])
            frames.append(dict(file_path=f'images/{index}.JPG',transform_matrix=pose.tolist()))
        (scene/'transforms.json').write_text(json.dumps(dict(w=64,h=64,fl_x=80,fl_y=80,cx=31.5,cy=31.5,
            camera_model='OPENCV',frames=frames)))
    return root


def prepare_fixture(root):
    fixture_data(root)
    out=root/'cache'
    prepare(['--megadepth-scenes',str(root/'md'/'scenes'),'--uav-rgb',str(root/'rgb'),
        '--uav-depth',str(root/'depth'),'--output',str(out),'--size','64',
        '--max-pairs-per-scene','1','--min-points','2','--min-overlap','0','--max-overlap','1'])
    return out


def config():
    return GRMMatcherConfig(input_height=64,input_width=64,architecture_version='v3',v3_revision=1,
        topk_candidates=1,classification_mode='none',
        coarse_dim=32,mid_dim=32,fine_dim=32,transformer_layers=4,transformer_heads=4,
        v3_mid_hidden=8,v3_fine_hidden=8,max_source_queries=16)


def loss_args():
    return SimpleNamespace(anchor_weight=.25,anchor_points=8,freeze_backbone_epochs=2,
        freeze_fine_epochs=2,real_fine_weight=0.,local_nll_weight=.1,confidence_weight=.1,
        positive_matchability_weight=.1,overlap_weight=.1,cycle_weight=0.,
        mid_coordinate_weight=.5,fine_coordinate_weight=1.)


def test_disk_pose_depth_resize_and_unknown_occlusion(tmp_path):
    fixture_data(tmp_path)
    scenes,_=discover_disk(tmp_path/'md'/'scenes')
    a,b=scenes[0].frames
    assert np.allclose(b.c2w[:3,3],[.2,0,0])
    labels=direction(a,b,size=64,cycle_px=.01)
    from grm_matcher.depth_geometry import coarse_grid
    grid=coarse_grid(64)
    assert labels['valid'].sum()>0
    assert np.allclose(labels['target'][labels['valid']],grid[labels['valid']]-[8,0],atol=1e-4)
    occluded=tmp_path/'occluded.tif'; tifffile.imwrite(occluded,np.ones((64,64),np.float32))
    bad=Frame(b.key,b.image,occluded,b.k,b.distortion,b.c2w,b.width,b.height)
    assert not direction(a,bad,size=64)['valid'].any()
    unknown=tmp_path/'unknown.tif'; tifffile.imwrite(unknown,np.zeros((64,64),np.float32))
    bad.depth=unknown
    result=direction(a,bad,size=64)
    assert not result['valid'].any() and not result['weight'].any()


def test_preparation_split_isolation_and_relocation(tmp_path):
    out=prepare_fixture(tmp_path)
    cfg=config()
    train=DepthCacheDataset(out,'train','uav',cfg)
    val=DepthCacheDataset(out,'val','uav',cfg)
    assert not train.groups&val.groups and not train.source_hashes&val.source_hashes
    assert train[0]['image_a'].shape==(3,64,64)
    assert train[0]['ab_target'].shape==(16,2)
    assert len(list((out/'previews').glob('*.jpg')))==4


def test_24_32_annulus_has_reachable_mid_target_and_fine_nll():
    cfg=config()
    initial=torch.tensor([[[7.5,7.5]]]); gt=initial+torch.tensor([[[28.,0.]]])
    mid=SimpleNamespace(dst_xy=initial+torch.tensor([[[24.,0.]]]),valid_mask=torch.ones(1,1,dtype=torch.bool),
        local_probability=torch.full((1,1,7,7),1/49,requires_grad=True))
    fine=SimpleNamespace(dst_xy=gt.clone().requires_grad_(),valid_mask=mid.valid_mask,
        local_probability=torch.full((1,1,5,5),1/25,requires_grad=True))
    loss,metrics=refinement_terms(mid,fine,gt,initial,mid.valid_mask,torch.ones(1,1),cfg)
    assert metrics['local_supervised_count']==1 and metrics['fine_nll_supervised_count']==1
    assert metrics['mid_coordinate']==0
    loss.backward(); assert mid.local_probability.grad is not None
    loss,metrics=refinement_terms(mid,fine,initial+33,initial,mid.valid_mask,torch.ones(1,1),cfg)
    assert metrics['local_supervised_count']==0 and loss==0


def test_real_unknown_masks_and_fine_gradient_scope(tmp_path):
    torch.set_num_threads(2); torch.manual_seed(1)
    out=prepare_fixture(tmp_path); cfg=config(); model=GRMMatcher(cfg)
    objective=DepthObjective(cfg,loss_args())
    batch=next(iter(DataLoader(DepthCacheDataset(out,'train','megadepth',cfg),batch_size=1)))
    model.train(); loss,metrics=objective.agri(model,batch,True,3)
    assert torch.isfinite(loss),metrics
    loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(FINE_PREFIXES))
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith('mid_refiner.'))
    assert metrics['overlap']==0  # Unknown depth must not produce full-map negative BCE.
    model.zero_grad(set_to_none=True)
    batch['domain']=['synthetic']; grid_y,grid_x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
    batch['gt_dense_mapping']=torch.stack((grid_x,grid_y)).float()[None]
    batch['H_ab']=torch.eye(3)[None]
    batch['gt_overlap_a']=batch['gt_overlap_b']=torch.ones(1,1,64,64,dtype=torch.bool)
    loss,metrics=objective.agri(model,batch,True,3)
    assert torch.isfinite(loss),metrics
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith('fine_refiner.'))


def test_checkpoint_cannot_win_by_collapsing_query_outputs():
    from grm_matcher.train_depth_finetune import real_validation_guard
    row=dict(output_matches=100,reference_scored_matches=80,reference_coverage_of_outputs=.8,coarse_recall4=.6)
    baseline={d:row.copy() for d in ('megadepth','uav')}
    args=SimpleNamespace(min_output_ratio=.5,reference_coverage_guard=.05,real_recall_guard=.02)
    assert real_validation_guard(baseline,baseline,args)
    collapsed={d:dict(row,output_matches=0,reference_scored_matches=0) for d in baseline}
    assert not real_validation_guard(collapsed,baseline,args)
