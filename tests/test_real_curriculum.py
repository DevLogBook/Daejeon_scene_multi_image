"""Real R1/R2 traversal, supervision semantics and executable resume checks."""
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher
from grm_matcher.depth_curriculum import parameter_group
from grm_matcher.real_curriculum import (FullCoverageEpoch, RealCurriculumObjective,
    TeacherGeometryData, RealGeometryData, teacher_grid, audit_geometry)
from grm_matcher.train_real_curriculum import main, selection_scores
from tools.prepare_grm_depth_data import main as prepare
_curriculum_fixture=runpy.run_path(str(Path(__file__).with_name('test_depth_curriculum.py')))
IdentityData, tiny_config = _curriculum_fixture['IdentityData'], _curriculum_fixture['tiny_config']
fixture_data=runpy.run_path(str(Path(__file__).with_name('test_depth_finetune.py')))['fixture_data']


class FineIdentity(IdentityData):
    def __getitem__(self, i):
        result=super().__getitem__(i)
        if self.domain!='synthetic':
            for d in ('ab','ba'):result[d+'_fine_valid']=result[d+'_valid'].clone()
        return result


def test_epoch_full_traversal_and_deterministic_cursor():
    sources={'megadepth':list(range(19)),'uav':list(range(20)),'udis':list(range(7))}
    md=FullCoverageEpoch(sources,'R1',22,3)
    assert len(md)==19 and sorted(i for d,i in md.plan)==list(range(19))
    mixed=FullCoverageEpoch(sources,'R2',22,3,.6)
    assert mixed.counts=={'uav':20,'udis':13}
    assert mixed.unique_counts=={'uav':20,'udis':7}
    resumed=FullCoverageEpoch(sources,'R2',22,3,.6,offset=9)
    assert resumed.plan==mixed.plan and len(resumed)==len(mixed)-9
    assert resumed[0]==mixed[9]
    assert FullCoverageEpoch(sources,'R2',22,4,.6).plan!=mixed.plan


def sparse_teacher(size=64,stride=8):
    axis=(torch.arange(size//stride)+.5)*stride-.5
    y,x=torch.meshgrid(axis,axis,indexing='ij');source=torch.stack((x,y),-1).reshape(-1,2)
    sample={'pair_id':'pair','image_a':torch.zeros(3,size,size),'image_b':torch.zeros(3,size,size)}
    for d in ('ab','ba'):
        sample.update({d+'_source':source.clone(),d+'_target':source.clone(),
                       d+'_confidence':torch.ones(len(source)),d+'_cycle':torch.zeros(len(source))})
    return sample


def test_teacher_grid_unknown_and_strict_fine_gate():
    cfg=tiny_config();cfg.coarse_stride=8
    sample=sparse_teacher()
    for d in ('ab','ba'):
        for key in ('source','target','confidence','cycle'):sample[d+'_'+key]=sample[d+'_'+key][2:]
    sample['ab_confidence'][0]=.85
    sample['ab_cycle'][1]=1.5
    grid=teacher_grid(sample,cfg,64)
    assert not grid['ab_valid'][:2].any() and not grid['ab_weight'][:2].any()
    assert grid['ab_valid'][2:4].all() and not grid['ab_fine_valid'][2:4].any()
    assert grid['ba_fine_valid'][2:].all()
    sample['ab_source'][0,0]+=.5
    with pytest.raises(ValueError,match='cell centers'):teacher_grid(sample,cfg,64)


def make_teacher(root, split='train'):
    folder=root/split;folder.mkdir(parents=True)
    size=64;y,x=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
    warp=(torch.stack((x,y),-1).float()+.5)*2/size-1
    payload={'img_a':np.zeros((size,size,3),dtype=np.uint8),
             'img_b':np.full((size,size,3),64,dtype=np.uint8),
             'metadata':dict(cache_version=2,warp='normalized_target_xy_align_corners_false',
                 teacher='RoMaV2',weights_sha256='f'*64,teacher_setting='precise',teacher_input=512,
                 source_sha256=[('a' if split=='train' else 'c')*64,('b' if split=='train' else 'd')*64])}
    for d in ('AB','BA'):
        payload['warp_'+d]=warp.clone();payload['confidence_'+d]=torch.ones(size,size)
        payload['confidence_raw_'+d]=torch.ones(size,size)
    torch.save(payload,folder/'000.pt')
    return folder


def test_teacher_extraction_not_truncated_by_student_query_budget(tmp_path):
    cfg=tiny_config();cfg.coarse_stride=8;cfg.v3_coarse_mode='f8';cfg.max_source_queries=4
    data=TeacherGeometryData(make_teacher(tmp_path),cfg,64,min_points=2)
    assert int(data[0]['ab_valid'].sum())==64
    assert cfg.max_source_queries==4
    assert data[0]['domain']=='udis'


def test_single_domain_full_preparation_and_resume(tmp_path):
    fixture_data(tmp_path)
    out=tmp_path/'only_md'
    args=['--domains','megadepth','--megadepth-scenes',str(tmp_path/'md/scenes'),
          '--output',str(out),'--size','64','--coarse-stride','16',
          '--max-pairs-per-scene','0','--max-attempts-per-scene','0',
          '--min-points','2','--min-overlap','0','--max-overlap','1',
          '--exclude-megadepth-scenes','0015','0022']
    prepare(args);prepare(args+['--resume'])
    cfg=tiny_config()
    report=audit_geometry(out,'megadepth',cfg,64)
    assert report['counts']=={'train':1,'val':1}
    data=RealGeometryData(out,'train','megadepth',cfg)
    assert data[0]['ab_fine_valid'].any()
    assert set(json.loads((out/'preparation.json').read_text())['group_split'].values())=={'train','val'}


def test_existing_split_is_preserved_with_single_domain(tmp_path):
    fixture_data(tmp_path)
    split=tmp_path/'split.json'
    split.write_text(json.dumps({'group_split':{'megadepth/0001':'val','megadepth/0002':'train',
                                               'uav/corn1':'train','uav/corn2':'val'}}))
    out=tmp_path/'cache'
    prepare(['--domains','megadepth','--megadepth-scenes',str(tmp_path/'md/scenes'),
             '--output',str(out),'--size','64','--split-json',str(split),
             '--max-pairs-per-scene','0','--max-attempts-per-scene','0',
             '--min-points','2','--min-overlap','0','--max-overlap','1'])
    assert json.loads((out/'preparation.json').read_text())['group_split']=={'megadepth/0001':'val','megadepth/0002':'train'}


def test_real_objective_fine_mask_and_freeze():
    torch.set_num_threads(1);torch.manual_seed(7)
    cfg=tiny_config();model=GRMMatcher(cfg)
    obj=RealCurriculumObjective(cfg,anchor_points=8,local_freeze_updates=1)
    b=default_collate([FineIdentity('udis')[0]])
    b['ab_fine_valid'].zero_();b['ba_fine_valid'].zero_()
    loss,metrics=obj.agri(model,b,True)
    assert torch.isfinite(loss) and metrics['fine_weight']==.1
    assert metrics['anchor_fine_coordinate_supervised_count']==0
    assert metrics['pred_fine_coordinate_supervised_count']==0
    loss.backward()
    for n,p in model.named_parameters():
        if parameter_group(n) in ('mid','fine','bridge','confidence'):
            assert not p.requires_grad and p.grad is None
    obj.update=1
    obj.agri(model,b,True)
    assert all(p.requires_grad for p in model.parameters())


def fake_data(args, config):
    focus='megadepth' if args.phase=='R1' else 'uav'
    sources={focus:FineIdentity(focus,rows=5)}
    val={focus:FineIdentity(focus,rows=1),'synthetic':FineIdentity('synthetic',rows=1)}
    if args.phase=='R2':
        sources['udis']=FineIdentity('udis',rows=3);val['udis']=FineIdentity('udis',rows=1)
    return sources,val,dict(audit={},signatures={'toy':'v1'},validation_records={'fixed':'toy'})


def test_actual_training_mid_epoch_resume_and_phase_init(tmp_path,monkeypatch):
    import grm_matcher.train_real_curriculum as trainer
    monkeypatch.setattr(trainer,'build_data',fake_data)
    cfg=tiny_config();cfg.input_height=cfg.input_width=64
    cfg.v3_min_matchability=0;cfg.v3_relative_reliability=0
    torch.manual_seed(7)
    initial=tmp_path/'initial.pt';torch.save(dict(config=cfg.to_dict(),model=GRMMatcher(cfg).state_dict()),initial)
    common=['--phase','R1','--geometry-cache',str(tmp_path/'unused'), '--synthetic-data',str(tmp_path/'unused'),
            '--epochs','2','--warmup-epochs','0','--accumulation-steps','2','--size','64',
            '--local-freeze-updates','0','--num-workers','0','--device','cpu','--log-every','1',
            '--eval-every','2','--save-every','1','--anchor-points','8']
    full=tmp_path/'full';part=tmp_path/'part'
    main(common+['--init-checkpoint',str(initial),'--output-dir',str(full)])
    main(common+['--init-checkpoint',str(initial),'--output-dir',str(part),'--stop-after-updates','1'])
    paused=torch.load(part/'latest.pt',weights_only=False)
    assert paused['next_epoch']==0 and paused['next_pair_offset']==2
    main(common+['--resume',str(part/'latest.pt'),'--output-dir',str(part)])
    a,b=[torch.load(p/'latest.pt',weights_only=False) for p in (full,part)]
    assert a['schedule_step']==b['schedule_step']==6
    for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],atol=0,rtol=0)
    r2=common.copy();r2[1]='R2'
    result=main(r2+['--teacher-cache',str(tmp_path/'teacher'),'--init-checkpoint',str(full/'best_phase.pt'),
                   '--output-dir',str(tmp_path/'r2'),'--stop-after-updates','1'])
    assert result['step']==1
    with pytest.raises(ValueError,match='exact phase'):
        main(r2+['--teacher-cache',str(tmp_path/'teacher'),'--resume',str(part/'latest.pt'),
                 '--output-dir',str(part)])


def test_selection_is_not_teacher_agreement():
    vals={'megadepth':{'macro':{'coarse_recall1':.4,'reference_f1_3':.5}},
          'uav':{'macro':{'coarse_recall1':.3,'reference_f1_3':.6}},
          'udis':{'macro':{'reference_f1_3':.99}},'synthetic':{'macro':{'pck1':.9}}}
    assert selection_scores(vals,'R1')['phase']==.4
    assert selection_scores(vals,'R2')['phase']==.6


def synthetic_fixture(root):
    import cv2
    for split in ('train','val'):
        path=root/split;path.mkdir(parents=True)
        rgb=np.random.default_rng(19 if split=='train' else 20).integers(0,255,(64,64,3),dtype=np.uint8)
        cv2.imwrite(str(path/'a.png'),rgb);cv2.imwrite(str(path/'b.png'),rgb)
        cv2.imwrite(str(path/'mask.png'),np.full((64,64),255,np.uint8))
        np.save(path/'h.npy',np.eye(3))
        row=dict(pair_id=split,source_image=split+'_source',image_a=str(path/'a.png'),image_b=str(path/'b.png'),
            h_gt_a_to_b=str(path/'h.npy'),valid_a=str(path/'mask.png'),valid_b=str(path/'mask.png'),
            overlap_a=str(path/'mask.png'),target_size=[64,64],target_bucket='64x64',direction='A_to_B')
        (path/'manifest.jsonl').write_text(json.dumps(row)+'\n')
    return root


def test_real_cache_and_teacher_pipeline_without_mocked_data(tmp_path):
    fixture_data(tmp_path)
    cache=tmp_path/'geometry'
    prepare(['--megadepth-scenes',str(tmp_path/'md/scenes'),'--uav-rgb',str(tmp_path/'rgb'),
        '--uav-depth',str(tmp_path/'depth'),'--output',str(cache),'--size','64',
        '--max-pairs-per-scene','1','--min-points','2','--min-overlap','0','--max-overlap','1'])
    synth=synthetic_fixture(tmp_path/'synthetic')
    make_teacher(tmp_path/'teacher','train');make_teacher(tmp_path/'teacher','val')
    cfg=tiny_config();cfg.input_height=cfg.input_width=64
    initial=tmp_path/'init.pt';torch.save(dict(config=cfg.to_dict(),model=GRMMatcher(cfg).state_dict()),initial)
    common=['--geometry-cache',str(cache),'--synthetic-data',str(synth),
        '--init-checkpoint',str(initial),'--epochs','2','--warmup-epochs','0','--size','64',
        '--local-freeze-updates','0','--anchor-points','8','--accumulation-steps','2',
        '--num-workers','0','--device','cpu','--val-limit','0','--stop-after-updates','1']
    for phase in ('R1','R2'):
        out=tmp_path/phase
        result=main(common+['--phase',phase,'--teacher-cache',str(tmp_path/'teacher'),
                           '--min-teacher-points','2','--output-dir',str(out)])
        assert result['step']==1 and (out/'latest.pt').exists()
        saved=torch.load(out/'latest.pt',weights_only=False)
        assert set(saved['validation_metrics'])==({'megadepth','synthetic'} if phase=='R1' else {'uav','udis','synthetic'})


def test_f8_hra_real_objective_backward():
    from dataclasses import replace
    from grm_matcher.coordinates import flat_indices_to_image_xy
    cfg=replace(tiny_config(),v3_coarse_mode='f8_hra',coarse_stride=8,input_height=64,input_width=64,
                v3_query_selection='physical_density',v3_min_matchability=0,v3_relative_reliability=0)
    model=GRMMatcher(cfg)
    sample=FineIdentity('uav')[0]
    grid=flat_indices_to_image_xy(torch.arange(64)[None],8,8)[0]
    for d in ('ab','ba'):
        sample.update({d+'_target':grid.clone(),d+'_valid':torch.ones(64,dtype=torch.bool),
                       d+'_weight':torch.ones(64),d+'_fine_valid':torch.ones(64,dtype=torch.bool)})
    obj=RealCurriculumObjective(cfg,anchor_points=8,local_freeze_updates=0)
    loss,metrics=obj.agri(model,default_collate([sample]),True)
    assert torch.isfinite(loss) and metrics['reference_count']==64
    loss.backward()
    for group in ('backbone','coarse','mid','fine','confidence'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if parameter_group(n)==group)


def test_termination_saves_before_long_validation(tmp_path,monkeypatch):
    import grm_matcher.train_real_curriculum as trainer
    monkeypatch.setattr(trainer,'build_data',fake_data)
    handlers={}
    monkeypatch.setattr(trainer.signal,'signal',lambda sig,callback:handlers.update({sig:callback}))
    original_agri=RealCurriculumObjective.agri
    def send_stop(self,model,batch,train,epoch=0):
        result=original_agri(self,model,batch,train,epoch)
        if train:handlers[trainer.signal.SIGTERM](trainer.signal.SIGTERM,None)
        return result
    monkeypatch.setattr(RealCurriculumObjective,'agri',send_stop)
    original_eval=trainer.evaluate;eval_steps=[]
    def count_eval(*args):
        eval_steps.append(args[-1]);return original_eval(*args)
    monkeypatch.setattr(trainer,'evaluate',count_eval)
    cfg=tiny_config();cfg.input_height=cfg.input_width=64
    initial=tmp_path/'initial.pt';torch.save(dict(config=cfg.to_dict(),model=GRMMatcher(cfg).state_dict()),initial)
    out=tmp_path/'signal_run'
    main(['--phase','R1','--geometry-cache',str(tmp_path/'unused'),'--synthetic-data',str(tmp_path/'unused'),
          '--init-checkpoint',str(initial),'--output-dir',str(out),'--epochs','2','--warmup-epochs','0',
          '--accumulation-steps','2','--size','64','--local-freeze-updates','0','--num-workers','0',
          '--device','cpu','--anchor-points','8'])
    raw=torch.load(out/'latest.pt',weights_only=False)
    assert raw['schedule_step']==1 and raw['next_pair_offset']==2
    assert raw['validation_step']==0 and eval_steps==[0]
