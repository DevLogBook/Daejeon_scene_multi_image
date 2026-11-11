"""R3 routing-free replay: quotas, LR, scopes, actual updates and exact resume."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from grm_matcher.coordinates import flat_indices_to_image_xy
from grm_matcher.depth_curriculum import parameter_group
from grm_matcher.matcher import GRMMatcher
from grm_matcher.real_replay import ReplayPlan, ReplayObjective, schedule_factor, set_learning_rates, guarded_scores
import grm_matcher.train_real_replay as trainer

fixtures=runpy.run_path(str(Path(__file__).with_name('test_real_curriculum.py')))
tiny_config=fixtures['tiny_config']
IdentityData=fixtures['IdentityData']


def config():
    return replace(tiny_config(),coarse_stride=8,v3_coarse_mode='f8_hra',
                   v3_query_selection='physical_density')


class F8Data(IdentityData):
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        grid=flat_indices_to_image_xy(torch.arange(64)[None],8,8)[0]
        if self.domain!='synthetic':
            for d in ('ab','ba'):
                sample.update({d+'_target':grid.clone(),d+'_valid':torch.ones(64,dtype=torch.bool),
                               d+'_fine_valid':torch.ones(64,dtype=torch.bool),d+'_weight':torch.ones(64)})
        return sample


def fake_data(args,cfg):
    sources={d:F8Data(d,rows=5) for d in ('uav','udis','megadepth','synthetic')}
    validation={d:F8Data(d,rows=1) for d in sources}
    return sources,validation,dict(audit={},signatures={'toy':'v1'},validation_records={'fixed':'toy'})


def test_plan_exact_quota_and_resume_cursor():
    sources=fake_data(None,None)[0]
    plan=ReplayPlan(sources,2026,100,8)
    assert Counter(d for d,i in plan.plan)==dict(uav=400,udis=160,megadepth=160,synthetic=80)
    for start in range(0,800,80):
        assert Counter(d for d,i in plan.plan[start:start+80])==dict(uav=40,udis=16,megadepth=16,synthetic=8)
    other=ReplayPlan(sources,2026,100,8,offset=72)
    assert other.signature==plan.signature and other[0]['pair_id']==plan[72]['pair_id']
    for d in sources:
        # Global block shuffling can reorder visits from adjacent bucket cycles.
        counts=Counter(i for domain,i in plan.plan[:80] if domain==d)
        assert len(counts)==5 and max(counts.values())-min(counts.values())<=1
    assert plan.statistics()['uav']['unique_pairs']==5
    with pytest.raises(ValueError,match='cursor'): ReplayPlan(sources,2026,100,8,offset=3)


def test_lr_endpoints_and_unfreeze_transition():
    assert schedule_factor(1,8000,300)==pytest.approx(.1)
    assert schedule_factor(300,8000,300)==pytest.approx(1)
    assert schedule_factor(8000,8000,300)==pytest.approx(.1)
    groups=[dict(name='coarse',peak_lr=3e-6),dict(name='fine',peak_lr=5e-7)]
    opt=SimpleNamespace(param_groups=groups)
    args=SimpleNamespace(total_updates=8000,warmup_updates=300,min_lr_ratio=.1,
                         local_freeze_updates=1000,local_warmup_updates=200)
    set_learning_rates(opt,args,1000);assert groups[1]['lr']==0
    set_learning_rates(opt,args,1001)
    assert groups[1]['lr']==pytest.approx(5e-8*schedule_factor(1001,8000,300))
    set_learning_rates(opt,args,1200)
    assert groups[1]['lr']==pytest.approx(5e-7*schedule_factor(1200,8000,300))
    set_learning_rates(opt,args,8000);assert groups[1]['lr']==pytest.approx(5e-8)


def test_stage_scopes_and_no_backbone_update():
    torch.set_num_threads(1); torch.manual_seed(7)
    model=GRMMatcher(config())
    args=SimpleNamespace(coarse_lr=3e-6,refinement_lr=5e-7,confidence_lr=5e-7,weight_decay=1e-4)
    opt=trainer.make_optimizer(model,args)
    assert 'backbone' not in {g['name'] for g in opt.param_groups}
    obj=ReplayObjective(model.config,anchor_points=8,local_freeze_updates=1)
    sample=default_collate([F8Data('uav')[0]])
    before={k:v.clone() for k,v in model.state_dict().items() if k.startswith('backbone.')}
    loss,metrics=obj.agri(model,sample,True);loss.backward();opt.step()
    assert obj.phase=='A' and metrics['anchor']==metrics['refinement']==metrics['confidence']==0
    for n,p in model.named_parameters():
        if parameter_group(n)!='coarse': assert not p.requires_grad and p.grad is None
    opt.zero_grad(set_to_none=True);obj.update=1
    loss,metrics=obj.agri(model,sample,True);loss.backward();opt.step()
    assert obj.phase=='C' and metrics['anchor_fine_coordinate_supervised_count']>0
    for group in ('coarse','mid','fine','confidence'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if parameter_group(n)==group)
    for k,v in before.items(): torch.testing.assert_close(v,model.state_dict()[k],atol=0,rtol=0)


def test_guard_and_independent_selection():
    baseline={'uav':{'macro':dict(pck1=.6,reference_f1_3=.4)},
              'megadepth':{'macro':dict(coarse_recall1=.5)},'synthetic':{'macro':dict(pck1=.95)}}
    args=SimpleNamespace(md_guard=.01,uav_guard=.005,synthetic_guard=.005)
    failures,scores=guarded_scores(baseline,baseline,args)
    assert not failures and scores['balanced']==(.4,.6) and scores['localization']==(.6,.4)
    result=json.loads(json.dumps(baseline)); result['megadepth']['macro']['coarse_recall1']=.48
    assert guarded_scores(result,baseline,args)[0]==['megadepth/coarse_recall1']


def initial_args(root):
    torch.manual_seed(7)
    initial=root/'initial.pt'
    torch.save(dict(config=config().to_dict(),model=GRMMatcher(config()).state_dict(),schedule_step=9000),initial)
    args=['--geometry-cache',str(root/'unused'),'--megadepth-cache',str(root/'unused'),
          '--teacher-cache',str(root/'unused'),'--synthetic-data',str(root/'unused'),
          '--total-updates','4','--warmup-updates','1','--accumulation-steps','2',
          '--local-freeze-updates','1','--local-warmup-updates','1','--anchor-points','8',
          '--size','64','--num-workers','0','--device','cpu','--eval-every','2',
          '--save-every','1','--log-every','3','--md-guard','1','--uav-guard','1','--synthetic-guard','1']
    return initial,args


def test_actual_updates_exact_resume_and_logging(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer,'build_data',fake_data)
    initial,args=initial_args(tmp_path)
    full,part=tmp_path/'full',tmp_path/'part'
    trainer.main(args+['--init-checkpoint',str(initial),'--output-dir',str(full)])
    trainer.main(args+['--init-checkpoint',str(initial),'--output-dir',str(part),'--stop-after-updates','2'])
    paused=torch.load(part/'latest.pt',weights_only=False)
    assert paused['schedule_step']==2 and paused['next_pair_offset']==4
    assert paused['provenance']['parent_step']==9000
    trainer.main(args+['--resume',str(part/'latest.pt'),'--output-dir',str(part)])
    a,b=[torch.load(p/'latest.pt',weights_only=False) for p in (full,part)]
    for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],atol=0,rtol=0)
    assert a['best_scores']==b['best_scores'] and a['guard_streak']==b['guard_streak']
    assert a['contract']['validation_records']==b['contract']['validation_records']
    assert (part/'previous.pt').exists() and (part/'best_balanced.pt').exists()
    events=[json.loads(s) for s in (full/'progress.jsonl').read_text().splitlines()]
    e=next(e for e in events if e['step']==3)
    assert e['interval_start']==2 and e['updates']==2 and e['microbatches']==4
    changed=args.copy();changed[changed.index('--total-updates')+1]='5'
    with pytest.raises(ValueError,match='contract'):
        trainer.main(changed+['--resume',str(part/'latest.pt'),'--output-dir',str(part)])


def test_best_phase_initialization_without_checksum(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer,'build_data',fake_data)
    initial,args=initial_args(tmp_path)
    best=initial.with_name('best_phase.pt');initial.rename(best)
    result=trainer.main(args+['--init-checkpoint',str(best),'--output-dir',str(tmp_path/'out'),'--audit-only'])
    assert 'expected_init_sha256' not in result['contract']
    assert 'source_sha256' not in result['contract']


def test_real_four_domain_cache_pipeline(tmp_path):
    fixtures['fixture_data'](tmp_path)
    cache=tmp_path/'cache'
    fixtures['prepare'](['--megadepth-scenes',str(tmp_path/'md/scenes'),'--uav-rgb',str(tmp_path/'rgb'),
        '--uav-depth',str(tmp_path/'depth'),'--output',str(cache),'--size','64','--coarse-stride','8',
        '--max-pairs-per-scene','1','--min-points','2','--min-overlap','0','--max-overlap','1'])
    synth=fixtures['synthetic_fixture'](tmp_path/'synthetic')
    fixtures['make_teacher'](tmp_path/'teacher','train');fixtures['make_teacher'](tmp_path/'teacher','val')
    initial,args=initial_args(tmp_path)
    replacements={'--geometry-cache':cache,'--megadepth-cache':cache,'--teacher-cache':tmp_path/'teacher',
                  '--synthetic-data':synth,'--total-updates':10,'--accumulation-steps':8}
    for flag,value in replacements.items():args[args.index(flag)+1]=str(value)
    out=tmp_path/'actual'
    result=trainer.main(args+['--init-checkpoint',str(initial),'--output-dir',str(out),
                             '--min-teacher-points','2','--stop-after-updates','2'])
    assert result['step']==2
    run=json.loads((out/'run.json').read_text())
    assert set(run['training_pairs'])=={'uav','udis','megadepth','synthetic'}
    assert {d:v['visits'] for d,v in run['planned_visits'].items()}==dict(uav=40,udis=16,megadepth=16,synthetic=8)
    raw=torch.load(out/'latest.pt',weights_only=False)
    assert set(raw['validation_metrics'])==set(run['training_pairs'])


def test_signal_saved_before_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer,'build_data',fake_data)
    handlers={}
    monkeypatch.setattr(trainer.signal,'signal',lambda sig,fn:handlers.update({sig:fn}))
    original=ReplayObjective.agri
    def stop(self,model,batch,train,epoch=0):
        result=original(self,model,batch,train,epoch)
        if train:handlers[trainer.signal.SIGTERM](trainer.signal.SIGTERM,None)
        return result
    monkeypatch.setattr(ReplayObjective,'agri',stop)
    initial,args=initial_args(tmp_path);out=tmp_path/'signal'
    trainer.main(args+['--init-checkpoint',str(initial),'--output-dir',str(out)])
    raw=torch.load(out/'latest.pt',weights_only=False)
    assert raw['schedule_step']==1 and raw['validation_step']==0


def test_repeated_guard_failure_halts_and_does_not_replace_best(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer,'build_data',fake_data)
    baseline={'uav':{'macro':dict(pck1=.6,reference_f1_3=.4)},
              'megadepth':{'macro':dict(coarse_recall1=.5)},'synthetic':{'macro':dict(pck1=.95)}}
    def degrade(model,validation,args,device,step):
        result=json.loads(json.dumps(baseline))
        if step:
            result['megadepth']['macro']['coarse_recall1']=.4
            result['uav']['macro']['reference_f1_3']=.99
        return result
    monkeypatch.setattr(trainer,'evaluate',degrade)
    initial,args=initial_args(tmp_path);out=tmp_path/'guards'
    args[args.index('--md-guard')+1]='.01'
    result=trainer.main(args+['--init-checkpoint',str(initial),'--output-dir',str(out)])
    assert result['halted'] and result['step']==2
    raw=torch.load(out/'latest.pt',weights_only=False)
    selected=torch.load(out/'best_balanced.pt',weights_only=False)
    assert raw['guard_streak']['megadepth/coarse_recall1']==2 and selected['schedule_step']==0
    with pytest.raises(ValueError,match='guard failures'):
        trainer.main(args+['--resume',str(out/'latest.pt'),'--output-dir',str(out)])


def test_legacy_checksum_fields_do_not_block_same_training_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(trainer,'build_data',fake_data)
    initial,args=initial_args(tmp_path)
    full,part=tmp_path/'full',tmp_path/'part'
    trainer.main(args+['--init-checkpoint',str(initial),'--output-dir',str(full)])
    trainer.main(args+['--init-checkpoint',str(initial),'--output-dir',str(part),'--stop-after-updates','2'])
    path=part/'latest.pt';raw=torch.load(path,weights_only=False)
    for key in ('source_sha256','expected_init_sha256','plan_sha256','fixed_validation_sha256'):
        raw['contract'][key]='legacy value'
    raw['contract'].pop('validation_records')
    torch.save(raw,path)
    trainer.main(args+['--resume',str(path),'--output-dir',str(part)])
    a,b=[torch.load(p/'latest.pt',weights_only=False) for p in (full,part)]
    for k,v in a['model'].items():torch.testing.assert_close(v,b['model'][k],atol=0,rtol=0)
