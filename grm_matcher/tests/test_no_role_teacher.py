import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.scratch_transfer import ScratchTransferObjective
from grm_matcher.teacher_cache import filter_matches, unique_coarse, load_pair, pair_key


def test_no_classifier_gt_training():
    cfg=GRMMatcherConfig(input_height=64,input_width=64,max_source_queries=16,
        backbone_stage_blocks=(1,1,1),graph_layers=1,architecture_version="v2",classification_mode="none")
    model=GRMMatcher(cfg)
    assert not hasattr(model,"role_head") and not hasattr(model,"inlier_head")
    image=torch.rand(1,3,64,64); mask=torch.ones(1,1,64,64,dtype=torch.bool)
    y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing="ij")
    batch=dict(image_a=image,image_b=image,valid_a=mask,valid_b=mask,H_ab=torch.eye(3)[None],
        gt_dense_mapping=torch.stack((x,y))[None].float(),gt_overlap_a=mask,gt_overlap_b=mask)
    loss,metrics=ScratchTransferObjective(cfg).agri(model,batch,True,0)
    assert not {"role","inlier","calibration","global_gt_count"}&metrics.keys()
    loss.backward()
    assert torch.isfinite(loss) and model.context_bridge.fine.weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    output=model(image,image); output.validate()


def test_f_filter_and_unique_cells():
    a=np.array([[8,8],[9,9],[40,40],[20,20]],np.float32)
    b=a.copy(); b[2,1]+=10
    confidence=np.array([.9,.8,.99,.1],np.float32)
    masks=[np.ones((64,64),bool)]*2
    f=np.array([[0,0,0],[0,0,-1],[0,1,0]],np.float32)
    a,b,c=filter_matches(a,b,confidence,masks,f)
    assert len(a)==2
    a,b,c=unique_coarse(a,b,c,16)
    assert len(a)==1 and c[0]==pytest.approx(.9)


def test_cache_signature_and_missing(tmp_path):
    row={"pair_id":"x"}; key=pair_key(row,64,64)
    with pytest.raises(FileNotFoundError): load_pair(tmp_path,row,64,64)
    np.savez(tmp_path/f"{key}.npz",key=key,a=np.zeros((0,2)),b=np.zeros((0,2)),confidence=np.zeros(0))
    assert len(load_pair(tmp_path,row,64,64)[0])==0
    with pytest.raises(FileNotFoundError): load_pair(tmp_path,row,128,64)


def test_cached_teacher_labels_reach_scratch_training(tmp_path):
    import cv2
    from torch.utils.data import DataLoader
    from grm_matcher.imc_dataset import IMCPairDataset
    root=tmp_path; (root/"train").mkdir(); cache=root/"cache"; cache.mkdir()
    cv2.imwrite(str(root/"a.png"),np.random.default_rng(1).integers(0,256,(64,64,3),dtype=np.uint8))
    row=dict(pair_id="sample",image_a="a.png",image_b="a.png",F_ab=[[0,0,0],[0,0,-1],[0,1,0]])
    (root/"splits.json").write_text(json.dumps(dict(image_root=str(root),height=64,width=64)))
    (root/"train/manifest.jsonl").write_text(json.dumps(row)+"\n")
    y,x=np.meshgrid(np.arange(4),np.arange(4),indexing="ij")
    a=(np.stack((x,y),-1).reshape(-1,2)*16+7.5).astype(np.float32)
    key=pair_key(row,64,64)
    np.savez(cache/f"{key}.npz",key=key,a=a,b=a,confidence=np.ones(16,np.float32))
    cfg=GRMMatcherConfig(input_height=64,input_width=64,max_source_queries=4,
        backbone_stage_blocks=(1,1,1),graph_layers=1,architecture_version="v2",classification_mode="none")
    dataset=IMCPairDataset(root,"train",cfg,label_source="roma",teacher_cache=cache)
    batch=next(iter(DataLoader(dataset,batch_size=1)))
    assert batch["sparse_valid"].sum()==16  # independent of the inference query cap
    model=GRMMatcher(cfg)
    loss,_=ScratchTransferObjective(cfg).imc(model,batch,False)
    loss.backward()
    assert torch.isfinite(loss) and model.context_bridge.fine.weight.grad.abs().sum()>0
