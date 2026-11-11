"""Serial, resumable four-baseline evaluation. Reuses existing scoring functions."""
from pathlib import Path
import sys, os, json, time, argparse, subprocess, platform, csv, math
from types import SimpleNamespace as NS
from copy import deepcopy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.name == 'nt':
    for p in (ROOT/'.cache/edm_eval_dependencies', ROOT/'.cache/megadepth_eval_dependencies'):
        sys.path.insert(0, str(p))
import numpy as np
import cv2
import agri_eval_common as ag
from tools import evaluate_grm_megadepth1500 as md
from tools import evaluate_grm_uav2025 as uv
MODELS = ['EDM','LoFTR','EfficientLoFTR','XFeat']
SETS = ['real','synthetic','megadepth1500','uav2025']
WEIGHTS = {'EDM':ROOT.parent/'EDM/weights/edm_outdoor.ckpt',
 'LoFTR':Path.home()/'.cache/torch/hub/checkpoints/loftr_outdoor.ckpt',
 'EfficientLoFTR':ROOT.parent/'EfficientLoFTR/weight/eloftr_outdoor.ckpt',
 'XFeat':ROOT.parent/'accelerated_features/weights/xfeat.pt'}
SOURCES = {'EDM':'https://github.com/chicleee/EDM', 'LoFTR':'https://github.com/zju3dv/LoFTR',
 'EfficientLoFTR':'https://github.com/zju3dv/EfficientLoFTR','XFeat':'https://github.com/verlab/accelerated_features'}
OUT = ROOT/'evaluation_outputs/four_baselines_native_20261104'

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(uv.clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def inventory():
    data={}; meta={}
    for mode in ('synthetic','real'):
        args=ag.parser('LoFTR',mode).parse_args([])
        data[mode],meta[mode]=ag.load_pairs(args,mode)
    assert len(data['synthetic'])==2420, len(data['synthetic'])
    assert len(data['real'])==200, len(data['real'])
    assert sum(p['has_ground_truth'] for p in data['real'])==185
    assert sum(len(p['points_a']) for p in data['real'])==802
    roots=list((ROOT.parent/'Dataset/megadepth1500').rglob('pairs_calibrated.txt'))
    if len(roots)!=1: raise ValueError(f'Ambiguous MD root: {roots}')
    data['megadepth1500'],meta['megadepth1500']=md.inventory(roots[0].parent)
    prepared=ROOT/'benchmarks_uav2025_20261023/prepared'
    data['uav2025']=json.loads((prepared/'pairs.json').read_bytes())
    assert len(data['uav2025'])==111
    frames=uv.load_frames(ROOT/'dataset/UAV3DCrop_2025_validation')
    for pair in data['uav2025']:
        for k in ('frame_a','frame_b'):
            f=frames[pair[k]]
            assert f.image.is_file() and f.depth.is_file(), f.key
        assert (prepared/'geometry'/f'{pair["pair_id"]}.npz').is_file()
    meta['uav2025']=dict(prepared=str(prepared),preparation=json.loads((prepared/'preparation.json').read_bytes()),pairs=111)
    return data,meta,frames,prepared

class Matcher:
    def __init__(self,name):
        import torch
        self.torch=torch;self.name=name;self.device=torch.device('cuda:0')
        torch.cuda.set_device(self.device);torch.set_num_threads(1);cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not WEIGHTS[name].is_file(): raise FileNotFoundError(WEIGHTS[name])
        self.metadata=dict(model=name,checkpoint=str(WEIGHTS[name]),source=SOURCES[name],
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            python=sys.executable,python_version=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,
            precision='FP32',batch_size=1,size=512,external_cap=0,external_confidence_threshold=None,
            warmup_calls_per_dataset=3,tf32=False)
        from tools.evaluate_baseline_megadepth1500 import rectangle_attention
        if name in ('LoFTR','EfficientLoFTR'):
            from tools.uav2025_baseline_adapters import BaselineMatcher
            self.native=BaselineMatcher(NS(model=name,device='cuda:0',size=512,checkpoint=WEIGHTS[name]))
            self.model=self.native.model;self.metadata.update(self.native.metadata)
            self.metadata['variant']='full; strict load then reparameter' if name=='EfficientLoFTR' else 'official outdoor dual-softmax; cvpr_ds_config'
        elif name=='EDM':
            sys.path.insert(0,str(ROOT.parent/'EDM'))
            from src.edm import EDM
            from src.config.default import get_cfg_defaults
            from src.utils.misc import lower_config
            from src.edm.neck.loftr_module import transformer
            transformer.Attention._forward=rectangle_attention
            cfg=get_cfg_defaults();cfg.merge_from_file(str(ROOT.parent/'EDM/configs/edm/outdoor/edm_base.py'))
            cfg.EDM.TEST_RES_H=cfg.EDM.TEST_RES_W=512
            cfg.EDM.NECK.NPE=[cfg.EDM.TRAIN_RES_H,cfg.EDM.TRAIN_RES_W,512,512]
            cfg.EDM.COARSE.TOPK=min(cfg.EDM.COARSE.TOPK,4096)
            cfg.EDM.MP=cfg.EDM.HALF=False
            config=lower_config(cfg)['edm'];self.model=EDM(config)
            raw=torch.load(WEIGHTS[name],map_location='cpu',weights_only=False)
            state={k.removeprefix('module.'):v for k,v in raw.get('state_dict',raw).items()}
            self.model.load_state_dict(state,strict=True);self.model=self.model.float().to(self.device).eval()
            self.metadata.update(config=config,position_encoding='832 to 512 NPE; official outdoor edm_base',mask_adapter='existing rectangle_attention')
        else:
            sys.path.insert(0,str(ROOT.parent/'accelerated_features'))
            from modules.xfeat import XFeat
            self.model=XFeat(weights=str(WEIGHTS[name]),top_k=4096,detection_threshold=.05).float().eval()
            self.metadata.update(variant='standard sparse XFeat',interface='detectAndCompute + match (same calls as match_xfeat)',
                top_k=4096,detection_threshold=.05,min_cossim=-1,matching='mutual nearest neighbors',
                confidence='cosine similarity of matched normalized descriptors; native matching unchanged')
        self.metadata['parameters']=sum(p.numel() for p in self.model.parameters())
        assert all(p.dtype==torch.float32 for p in self.model.parameters() if p.is_floating_point())
    def sync(self): self.torch.cuda.synchronize(self.device)
    def __call__(self,a,b,va,vb):
        t=self.torch
        with t.inference_mode(), t.autocast('cuda',enabled=False):
            if self.name in ('LoFTR','EfficientLoFTR'): return self.native(a,b,va,vb)
            if self.name=='XFeat':
                da,db=[self.model.detectAndCompute(self.model.parse_input(x),top_k=4096)[0] for x in (a,b)]
                i,j=self.model.match(da['descriptors'],db['descriptors'],min_cossim=-1)
                values=(da['keypoints'][i],db['keypoints'][j],(da['descriptors'][i]*db['descriptors'][j]).sum(-1))
            else:
                import torch.nn.functional as F
                batch={}
                for i,(im,valid) in enumerate(((a,va),(b,vb))):
                    batch[f'image{i}']=t.from_numpy(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)).to(self.device,dtype=t.float32)[None,None]/255
                    batch[f'mask{i}']=F.interpolate(t.from_numpy(valid.copy()).to(self.device)[None,None].float(),size=(64,64),mode='nearest')[:,0].bool()
                self.model(batch);values=tuple(batch[k] for k in ('mkpts0_f','mkpts1_f','mconf'))
            return tuple(x.detach().float().cpu().numpy() for x in values)

def show(path,ia,ib,pa,pb,conf,max_matches=None):
    # Optionally draw a confidence shortlist with joint source/target spatial coverage.
    canvas=np.concatenate((ia,ib),axis=1).copy()
    if max_matches is None or len(pa)<=max_matches:
        chosen=np.arange(len(pa))
    else:
        shortlist=np.argsort(-conf,kind='stable')[:min(len(conf),max_matches*5)]
        chosen=[int(shortlist[0])]
        da=np.sum((pa[shortlist]-pa[chosen[0]])**2,axis=1)
        db=np.sum((pb[shortlist]-pb[chosen[0]])**2,axis=1)
        for _ in range(min(max_matches,len(shortlist))-1):
            distance=np.minimum(da,db); distance[np.isin(shortlist,chosen)]=-1
            current=int(shortlist[np.argmax(distance)]); chosen.append(current)
            da=np.minimum(da,np.sum((pa[shortlist]-pa[current])**2,axis=1))
            db=np.minimum(db,np.sum((pb[shortlist]-pb[current])**2,axis=1))
    for k in chosen:
        a=tuple(np.rint(pa[k]).astype(int));b=tuple(np.rint(pb[k]+[512,0]).astype(int))
        color=(20,255,57)  # fluorescent green in RGB, visible on crop imagery
        cv2.line(canvas,a,b,color,1,cv2.LINE_AA)
        for point in (a,b):
            cv2.circle(canvas,point,1,color,-1,cv2.LINE_AA)
    cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))[1].tofile(str(path))

def evaluate(name,out,stop_after,*,benchmark=None,matcher=None,datasets=SETS,checkpoint=None,visual_matches=None):
    data,meta,frames,prepared=inventory() if benchmark is None else benchmark
    matcher=Matcher(name) if matcher is None else matcher
    checkpoint=WEIGHTS[name] if checkpoint is None else checkpoint
    t=matcher.torch
    for mode in datasets:
        dest=out/name/mode
        for d in ('matches','stitches','visualizations'): (dest/d).mkdir(parents=True,exist_ok=True)
        contract=dict(protocol_version='native_v1',model=name,weight=str(checkpoint),weight_bytes=checkpoint.stat().st_size,
            weight_mtime_ns=checkpoint.stat().st_mtime_ns,metadata=matcher.metadata,data=meta.get(mode),
            pair_ids=[p['pair_id'] for p in data[mode]],size=512,batch=1,precision='FP32',external_cap=0,
            preprocess='stretch' if mode in ('real','synthetic') else 'letterbox',seed=2026,
            geometry='MAGSAC 3px/10000/.999' if mode in ('real','synthetic') else 'E RANSAC .5px/1000/.99999')
        contract=uv.clean(contract)
        if (dest/'settings.json').exists():
            if json.loads((dest/'settings.json').read_bytes())!=contract: raise ValueError(f'Resume protocol mismatch: {dest}')
        else: write(dest/'settings.json',contract)
        write(dest/'model_metadata.json',matcher.metadata)
        journal=dest/'pairs.jsonl';rows=[json.loads(s) for s in journal.read_text(encoding='utf-8').splitlines() if s.strip()] if journal.exists() else []
        done={r['pair_id'] for r in rows}
        assert len(done)==len(rows)
        warm=False
        for index,pair in enumerate(data[mode]):
            if pair['pair_id'] in done: continue
            if stop_after and len(rows)>=stop_after: break
            if mode=='uav2025':
                a,b=[frames[pair[k]] for k in ('frame_a','frame_b')]
                ia,ta,va=uv.prepare_image(a.image,a.w,a.h,512);ib,tb,vb=uv.prepare_image(b.image,b.w,b.h,512)
            else:
                a,b=[ag.read_image(pair[k]) for k in ('image_a','image_b')]
                prep='stretch' if mode in ('real','synthetic') else 'letterbox'
                ia,ta,va=md.preprocess_image(a,512,prep);ib,tb,vb=md.preprocess_image(b,512,prep)
                if va is None: va=np.ones((512,512),bool);vb=va.copy()
            if not warm:
                t.manual_seed(2026)
                for _ in range(3): matcher(ia,ib,va,vb)
                matcher.sync();warm=True
            t.manual_seed(2026+index);t.cuda.reset_peak_memory_stats(matcher.device)
            matcher.sync();tick=time.perf_counter();pa,pb,conf=matcher(ia,ib,va,vb);matcher.sync()
            elapsed=time.perf_counter()-tick;peak=t.cuda.max_memory_allocated(matcher.device)/1024**2
            assert len(pa)==len(pb)==len(conf) and all(np.isfinite(v).all() for v in (pa,pb,conf))
            raw=len(pa);keep=ag.in_bounds(pa,ia.shape)&ag.in_bounds(pb,ib.shape)
            oa,ob=ag.project(pa,np.linalg.inv(ta)),ag.project(pb,np.linalg.inv(tb))
            if mode in ('megadepth1500','uav2025'):
                if mode=='uav2025': keep &= uv.in_image(oa,a)&uv.in_image(ob,b)
                else: keep &= ag.in_bounds(oa,a.shape)&ag.in_bounds(ob,b.shape)
            pa,pb,conf,oa,ob=(v[keep] for v in (pa,pb,conf,oa,ob))
            row=dict(pair_id=pair['pair_id'],index=index,matches=len(pa),raw_matches=raw,invalid_or_padding_removed=raw-len(pa),
                matching_seconds=elapsed,peak_allocated_mib=peak)
            arrays=dict(points_a_input=pa,points_b_input=pb,points_a_original=oa,points_b_original=ob,confidence=conf,transform_a=ta,transform_b=tb)
            if mode in ('real','synthetic'):
                tick=time.perf_counter();hn,n,status=ag.fit_homography(pa,pb,2026+index,3)
                h=np.linalg.inv(tb)@hn@ta if hn is not None else None
                row.update(geometry_seconds=time.perf_counter()-tick,status=status,group=pair['group'],ransac_inliers=n,
                    pipeline_seconds=elapsed+time.perf_counter()-tick,stitch_file=None,stitch_error=None)
                gt=ag.load_ground_truth(pair,mode,a,b)
                metrics,error=ag.score_real(h,pair) if mode=='real' else ag.score_synthetic(oa,ob,h,gt,a.shape,b.shape)
                row.update(metrics);arrays.update(homography_a_to_b=h if h is not None else np.full((3,3),np.nan),evaluation_errors_px=error)
                if mode=='real' and h is not None:
                    try:
                        ag.save_stitch(dest/'stitches'/f'{index:05d}.jpg',a,b,h);row['stitch_file']=f'stitches/{index:05d}.jpg'
                    except (ValueError,RuntimeError,OSError,cv2.error) as e: row['stitch_error']=str(e)
            elif mode=='megadepth1500':
                tick=time.perf_counter();pose=md.estimate_pose(pa,pb,ta@pair['ka'],tb@pair['kb'],.5,2026+index)
                row['geometry_seconds']=time.perf_counter()-tick
                pr,pt,inliers=pose if pose is not None else (np.full((3,3),np.nan),np.full(3,np.nan),np.zeros(len(pa),bool))
                re,te=md.angular_errors(pr,pt,pair['rotation'],pair['translation']) if pose is not None else (math.inf,math.inf)
                epi=md.epi_error(oa,ob,pair['ka'],pair['kb'],pair['rotation'],pair['translation'])
                gt,error,eligible,discrepancy=md.depth_errors(oa,ob,pair,a.shape,b.shape,.05)
                row.update(status='ok' if pose is not None else 'pose_failed',scene=pair['scene'],rotation_error_deg=re,translation_error_deg=te,pose_error_deg=max(re,te),
                    epi_correct_1e_4=int((epi<1e-4).sum()),depth_eligible=int(eligible.sum()),
                    **{f'depth_correct_{v}px':int((error<=v).sum()) for v in (1,3,5,10)})
                arrays.update(estimated_rotation=pr,estimated_translation=pt,pose_inlier_mask=inliers,epipolar_error=epi,depth_error_px=error,depth_eligible=eligible,depth_target=gt,relative_depth_discrepancy=discrepancy)
            else:
                # Reuse pose_metrics verbatim; capture its chosen R/t without a second solve.
                capture=[];original=cv2.recoverPose
                def record(*args,**kw):
                    result=original(*args,**kw);capture.append(result);return result
                tick=time.perf_counter();cv2.recoverPose=record
                try: epi,re,te=uv.pose_metrics(oa,ob,a,b,512,2026+pair['index'])
                finally: cv2.recoverPose=original
                row['geometry_seconds']=time.perf_counter()-tick
                best=max(capture,key=lambda x:x[0]) if capture else None
                arrays.update(estimated_rotation=best[1] if best is not None and best[0]>0 else np.full((3,3),np.nan),
                    estimated_translation=best[2] if best is not None and best[0]>0 else np.full((3,1),np.nan))
                ref=uv.reference(a,b,oa,512,.01,1.);eligible=ref['valid'];error=np.linalg.norm(pb-ref['target_input'],axis=1);oe=np.linalg.norm(ob-ref['target_original'],axis=1)
                error[~eligible]=np.nan;oe[~eligible]=np.nan
                with np.load(prepared/'geometry'/f'{pair["pair_id"]}.npz') as z: visible=np.flatnonzero(z['valid_ab'])
                cells=np.floor(pa/16).astype(int);covered=np.intersect1d(np.unique((cells[:,1]*32+cells[:,0])[error<=3]),visible)
                row.update(status='ok' if np.isfinite(max(re,te)) else 'pose_failed',scene=pair['scene'],depth_supported=int(eligible.sum()),
                    depth_error_median_input_px=float(np.median(error[eligible])) if eligible.any() else None,
                    visible_grid_coverage_3px=len(covered)/len(visible) if len(visible) else 0.,
                    **{f'correct_input_{v}px':int((error<=v).sum()) for v in (1,3,5,10)},
                    **{f'correct_original_{v}px':int((oe<=v).sum()) for v in (1,3,5,10)},
                    epipolar_correct=int((epi<1e-4).sum()),rotation_error_deg=re,translation_error_deg=te,pose_error_deg=max(re,te),matching_ms=elapsed*1000)
                arrays.update(depth_supported=eligible,depth_error_input_px=error,depth_error_original_px=oe,epipolar_error=epi,**ref)
            row['failure_reason']=None if row['status']=='ok' else ('fewer_than_five_matches' if mode not in ('real','synthetic') and len(pa)<5 else row['status'])
            np.savez_compressed(dest/'matches'/f'{index:05d}.npz',**arrays)
            if (mode=='uav2025' and (name in ('EDM','JamMa') or visual_matches is not None)) or index<6:
                show(dest/'visualizations'/f'{index:05d}.jpg',ia,ib,pa,pb,conf,
                     max_matches=200 if name=='EDM' and mode=='uav2025' else visual_matches)
            row=uv.clean(row)
            with journal.open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            rows.append(row)
            if len(rows)%25==0 or len(rows)<=2: print(f'{name}/{mode}: {len(rows)}/{len(data[mode])} match={elapsed:.3f}s n={len(pa)}',flush=True)
        summarize_dataset(dest,mode,rows,len(data[mode]))
    print('MODEL FINISHED',name,flush=True)

def summarize_dataset(dest,mode,rows,expected):
    if not rows:return
    full=[dict(r) for r in rows]
    for r in full:
        for k in ('landmark_mean_px','corner_mean_px','landmark_pck_1px','landmark_pck_3px','landmark_pck_5px','landmark_pck_10px'):
            if k in r and r[k] is None and (mode=='synthetic' or r.get('landmarks',0)):r[k]=math.inf
    if mode in ('real','synthetic'): result=ag.aggregate(full,mode)
    elif mode=='uav2025': result=uv.summarize(full)
    else:
        n=sum(r['matches'] for r in rows);support=sum(r['depth_eligible'] for r in rows)
        result=dict(pairs=len(rows),epipolar_precision_micro=sum(r['epi_correct_1e_4'] for r in rows)/max(n,1),depth_support=support/max(n,1),
            **{f'pose_auc_{v}deg':md.pose_auc([r['pose_error_deg'] if r['pose_error_deg'] is not None else math.inf for r in rows],v) for v in (5,10,20)},
            **{f'depth_pck_{v}px_micro':sum(r[f'depth_correct_{v}px'] for r in rows)/max(support,1) for v in (1,3,5,10)})
    if mode=='real':
        errors=[r['landmark_mean_px'] if r['landmark_mean_px'] is not None else math.inf for r in rows if r['landmarks']]
        result.update(pair_mean_error_median_px=float(np.median(errors)),severe_failures=sum(e>50 for e in errors),landmark_denominator=sum(r['landmarks'] for r in rows))
        result['stitch_failure_reasons']={s:sum(r.get('stitch_error')==s for r in rows) for s in {r.get('stitch_error') for r in rows} if s}
    result.update(complete=len(rows)==expected,expected=expected,failures=sum(r['status']!='ok' for r in rows),
        failure_reasons={s:sum(r['failure_reason']==s for r in rows) for s in {r['failure_reason'] for r in rows} if s},
        matches_total=sum(r['matches'] for r in rows),mean_matches=float(np.mean([r['matches'] for r in rows])))
    for key in ('matching_seconds','geometry_seconds','peak_allocated_mib'):
        vals=[r[key] for r in rows]
        for label,func in [('mean',np.mean),('median',np.median),('p90',lambda v:np.quantile(v,.9)),('max',np.max)]: result[f'{label}_{key}']=float(func(vals))
    write(dest/'summary.json',result)
    uv.write_csv(dest/'pairs.csv',rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--model',choices=MODELS)
    p.add_argument('--smoke',action='store_true');p.add_argument('--inventory-only',action='store_true');p.add_argument('--report-only',action='store_true');p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.verify_only: verify(args.output);return
    if args.report_only: report(args.output);return
    if args.inventory_only:
        data,meta,_,_=inventory();write(args.output/'inventory.json',dict(counts={k:len(v) for k,v in data.items()},metadata=meta));print({k:len(v) for k,v in data.items()});return
    if args.model: evaluate(args.model,args.output,2 if args.smoke else 0);return
    # Separate process prevents src package collisions and releases every model's GPU allocations.
    status={}
    for smoke in (True,False):
        for model in MODELS:
            if not smoke and status.get(model,{}).get('smoke_exit')!=0:continue
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--model',model,'--output',str(args.output)]+(['--smoke'] if smoke else [])
            with (args.output/f'{model}_{"smoke" if smoke else "full"}.log').open('a',encoding='utf-8') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            status.setdefault(model,{})['smoke_exit' if smoke else 'full_exit']=result.returncode
            write(args.output/'run_status.json',status);print(model,'smoke' if smoke else 'full',result.returncode,flush=True)
            report(args.output)
    print('SERIAL RUN FINISHED',flush=True)

def verify(out):
    checks=[]
    for name in MODELS:
        for mode,total in zip(('real','synthetic','megadepth1500','uav2025'),(200,2420,1500,111)):
            dest=out/name/mode;settings=json.loads((dest/'settings.json').read_bytes())
            rows=[json.loads(s) for s in (dest/'pairs.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
            assert len(rows)==len({r['pair_id'] for r in rows})==total,(name,mode,len(rows))
            assert [r['pair_id'] for r in sorted(rows,key=lambda r:r['index'])]==settings['pair_ids']
            assert len(list((dest/'matches').glob('*.npz')))==total
            for r in rows:
                with np.load(dest/'matches'/f'{r["index"]:05d}.npz') as z:
                    assert len(z['confidence'])==len(z['points_a_input'])==len(z['points_b_input'])==r['matches']
                    assert all(np.isfinite(z[k]).all() for k in ('confidence','points_a_input','points_b_input'))
                    if mode in ('real','synthetic'):
                        e=z['evaluation_errors_px']
                        if mode=='real' and not r['landmarks']: continue
                        for v in (1,3,5,10): assert int((e<=v).sum())==r[f'landmark_correct_{v}px' if mode=='real' else f'correct_matches_{v}px']
                    elif mode=='megadepth1500':
                        assert int(z['depth_eligible'].sum())==r['depth_eligible']
                        for v in (1,3,5,10): assert int((z['depth_error_px']<=v).sum())==r[f'depth_correct_{v}px']
                    else:
                        assert int(z['depth_supported'].sum())==r['depth_supported']
                        for v in (1,3,5,10): assert int((z['depth_error_input_px']<=v).sum())==r[f'correct_input_{v}px']
            if mode=='real':
                assert sum(r['landmarks'] for r in rows)==802
                assert sum(r['landmarks']>0 for r in rows)==185
                assert sum(r['stitch_file'] is not None for r in rows)==len(list((dest/'stitches').glob('*.jpg')))
            summarize_dataset(dest,mode,rows,total)
            checks.append(dict(model=name,dataset=mode,pairs=total,matches=sum(r['matches'] for r in rows),passed=True))
            print('VERIFIED',name,mode,total,flush=True)
    write(out/'verification.json',dict(complete=True,checks=checks));report(out)

def report(out):
    summaries={};speed=[];provenance=[]
    for name in MODELS:
        summaries[name]={}
        for mode in SETS:
            dest=out/name/mode
            if not (dest/'summary.json').exists(): continue
            s=json.loads((dest/'summary.json').read_bytes());summaries[name][mode]=s
            m=json.loads((dest/'model_metadata.json').read_bytes())
            speed.append(dict(model=name,dataset=mode,pairs=s['pairs'],complete=s['complete'],parameters=m['parameters'],
                mean_matches=s['mean_matches'],**{f'{a}_{k}':s[f'{a}_{k}'] for a in ('mean','median','p90','max') for k in ('matching_seconds','geometry_seconds','peak_allocated_mib')}))
        provenance.append(dict(model=name,weight=str(WEIGHTS[name]),source=SOURCES[name],result_directory=str(out/name)))
    write(out/'results.json',dict(results=summaries,models=provenance,units=dict(pck_auc_precision_support_yield_coverage='fraction 0..1; Markdown tables multiply by 100',time='seconds in CSV/JSON; milliseconds in Markdown',memory='PyTorch allocated MiB',parameters='count',agriculture_pck='target original image pixels',megadepth_pose='degrees',megadepth_depth_pck='target original image pixels',uav_pck='input pixels')))
    if speed: uv.write_csv(out/'time_memory_parameters_matches.csv',speed)
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
    def number(s,k,percent=False):
        v=s.get(k)
        if v is not None and k in ('failures','severe_failures','stitches_saved'): return str(int(v))
        return '—' if v is None else f'{v*(100 if percent else 1):.2f}'
    complete=all(summaries[n].get(d,{}).get('complete',False) for n in MODELS for d in SETS)
    lines=['# 四基线本地实测（固定 512 协议）','',
        ('本轮 16 个方法与数据集组合全部完成，共 16,924 对次；每个方法均为真实农业 200、合成 2420、MD1500 1500、UAV 111 对。' if complete else '本轮尚未全部完成，请以各表完成数为准。'), '',
        '精度值均来自本轮本地推理。表内“完成数/总数”必须达到全量后才可用于论文主表。所有 AUC 将求解失败记为无穷大，保留全部 pair 分母。CSV/JSON 中精度、AUC、支持率和覆盖率保留 0–1 比例原值；Markdown 显示百分比。', '',
        'GPU：NVIDIA GeForce GTX 1650 Ti 4GB；CUDA:0；batch=1；FP32；禁用 TF32。每套数据预热 3 次，匹配计时前后 CUDA 同步，包含输入张量转换、网络、匹配提取和 CPU 取回。几何计时为既有求解函数；UAV 的既有函数同时计算极线误差。解码、GT、评分、可视化和写盘不进入匹配计时。显存为含模型参数的 PyTorch peak allocated MiB。', '',
        '农业直接拉伸；MD/UAV 居中 letterbox + RGB mask；半像素变换及内参同步。外部匹配上限从旧农业的 2048 统一改为 0（不限），不额外筛 confidence、不按 GT 或空间分布筛推理输出。模型原生阈值保留。MAGSAC=3 输入 px / 10000 / .999；E-RANSAC=.5 输入 px / 1000 / .99999。', '',
        '## 农业合成', '', 'PCK 为 micro，分母为全部保留匹配；GT 无效和非重叠计错；单位为目标原图 px。H-AUC 基于四角平均投影误差；覆盖率为 pair macro。', '']
    configs=[('synthetic',['方法','完成/2420','PCK@1 %','@3 %','@5 %','@10 %','H-AUC@5 %','@10 %','覆盖率@3 %','平均匹配','H失败'],
        [('micro_match_precision_1px',True),('micro_match_precision_3px',True),('micro_match_precision_5px',True),('micro_match_precision_10px',True),('pair_error_auc_5px',True),('pair_error_auc_10px',True),('macro_coverage_3px',True),('mean_matches',False),('failures',False)],2420),
      ('real',['方法','完成/200','人工PCK@3 %','@5 %','@10 %','pair均差中位 px','严重失败/185','H失败/200','拼接/200'],
        [('micro_landmark_pck_3px',True),('micro_landmark_pck_5px',True),('micro_landmark_pck_10px',True),('pair_mean_error_median_px',False),('severe_failures',False),('failures',False),('stitches_saved',False)],200),
      ('megadepth1500',['方法','完成/1500','Pose-AUC@5° %','@10° %','@20° %','位姿失败','极线精度 %','深度支持 %','深度PCK@3原图px %'],
        [('pose_auc_5deg',True),('pose_auc_10deg',True),('pose_auc_20deg',True),('failures',False),('epipolar_precision_micro',True),('depth_support',True),('depth_pck_3px_micro',True)],1500),
      ('uav2025',['方法','完成/111','PCK@1 %','@3 %','@5 %','@10 %','深度支持 %','verified yield@3 %','可见覆盖@3 %','Pose-AUC@5° %','@10° %','@20° %','位姿失败'],
        [('pck_input_1px_micro',True),('pck_input_3px_micro',True),('pck_input_5px_micro',True),('pck_input_10px_micro',True),('depth_support',True),('verified_yield_input_3px',True),('mean_visible_grid_coverage_3px',True),('pose_auc_5deg',True),('pose_auc_10deg',True),('pose_auc_20deg',True),('failures',False)],111)]
    notes={'real':('真实农业','人工点 micro；仅有效标注 185 对、802 组点进入分母，单位为目标原图 px；无标注 15 对仍统计 H 和拼接。严重失败为有效标注 pair 的均差 >50 px 或 H 失败。'),
     'megadepth1500':('MegaDepth1500','角度单位 °；Pose error=max(R,t方向角)。极线精度为全部匹配 micro（归一化平方对称极线误差 <1e-4）。补充深度 PCK 使用支持匹配分母、目标原图 px 和 5% 深度一致性。'),
     'uav2025':('UAV2025','PCK 为深度支持匹配上的 micro，单位为输入 px。支持率=支持匹配/全部保留匹配；verified yield=正确且支持/全部保留匹配；可见网格覆盖为 pair macro。复用固定 prepared 111 对、1% 深度一致性、1 输入 px 回环和相机畸变处理。支持判断仅依赖源点和参考几何。')}
    accuracy=[]
    for mode,headers,keys,total in configs:
        if mode in notes: lines+=['','## '+notes[mode][0],'',notes[mode][1],'']
        rows=[];csvrows=[]
        for name in MODELS:
            s=summaries[name].get(mode,{})
            rows.append([name,f'{s.get("pairs",0)}/{total}'+('（未完成）' if not s.get('complete') else ''),*[number(s,k,pc) for k,pc in keys]])
            csvrows.append(dict(model=name,completed=s.get('pairs',0),expected=total,complete=s.get('complete',False),**{k:s.get(k) for k,_ in keys}))
            accuracy.extend(dict(model=name,dataset=mode,complete=s.get('complete',False),metric=k,value=s.get(k)) for k,_ in keys)
        lines+=[table(headers,rows),''];uv.write_csv(out/f'{mode}_accuracy.csv',csvrows)
    uv.write_csv(out/'accuracy_long.csv',accuracy)
    lines+=['## 时间、显存、参数量与匹配数','', '匹配及几何耗时：均值 / 中位数 / P90（ms）；峰值 allocated 显存：均值 / 中位数 / P90 / 最大值（MiB）。参数量为实际构建模型，EfficientLoFTR 为重参数化后计数。','',
        table(['方法','数据集','完成数','匹配 ms','几何 ms','峰值 allocated MiB','参数量','平均匹配'],[[r['model'],r['dataset'],r['pairs'],
            ' / '.join(f'{1000*r[a+"_matching_seconds"]:.2f}' for a in ('mean','median','p90')),
            ' / '.join(f'{1000*r[a+"_geometry_seconds"]:.2f}' for a in ('mean','median','p90')),
            ' / '.join(f'{r[a+"_peak_allocated_mib"]:.1f}' for a in ('mean','median','p90','max')),r['parameters'],f'{r["mean_matches"]:.1f}'] for r in speed]),'',
        '## 模型与结果位置','',table(['方法','实际权重','来源','结果目录'],[[r['model'],r['weight'],r['source'],r['result_directory']] for r in provenance]),'',
        'EDM：官方 edm_base，NPE 832→512，原生 TOPK 仅限于物理网格 4096，FP32。LoFTR：官方 outdoor DS 缓存权重，cvpr_ds_config（TEMP_BUG_FIX=False、coarse threshold=.2）。EfficientLoFTR：full_default_cfg，严格加载后 reparameter，FP32；沿用既有居中 mask 适配。XFeat：标准稀疏，parse_input → detectAndCompute → MNN match；top_k=4096，detection_threshold=.05，min_cossim=-1；保存匹配描述子余弦相似度供展示选点，不增加阈值。完整原生配置、参数量、环境见各结果目录 model_metadata.json。', '',
        '版本、官方权重链接和缓存来源记录见 [model_versions.json](model_versions.json)。', '',
        '硬件与驱动记录见 [hardware.json](hardware.json)；全量缓存核验见 [verification.json](verification.json)。', '',
        '## 失败原因','',table(['方法','数据集','求解失败原因及数量'],[[name,mode,json.dumps(s.get('failure_reasons',{}),ensure_ascii=False)] for name,byset in summaries.items() for mode,s in byset.items()]),'',
        table(['方法','真实拼接生成失败原因及数量'],[[name,json.dumps(byset.get('real',{}).get('stitch_failure_reasons',{}),ensure_ascii=False)] for name,byset in summaries.items()]),'',
        '## 复现与续跑','', '```powershell',"& 'D:/Software/Miniconda/envs/Competitions/python.exe' tools/run_four_baselines_20261104.py",'```','',
        '每次只运行一个进程。自动先执行四模型各数据集前 2 对检查，再逐模型全量续跑。相同 settings 的完成 pair 跳过；不覆盖历史输出。单方法可加 --model EDM（或其他名称）；--report-only 只重建报告；--inventory-only 核对固定清单。结果 matches/*.npz 保存全部坐标、confidence、变换及 H/位姿；pairs.jsonl 保存逐 pair 耗时、显存、失败原因；真实 stitches 保存所有能生成的拼接；每套前 6 对保存约 24 条连线，按 confidence 和空间分散挑展示点。', '',
        '评分直接调用 agri_eval_common.py、tools/evaluate_grm_megadepth1500.py、tools/evaluate_grm_uav2025.py。原始数据、标注、模型源码不修改。没有训练、阈值搜索或 GRM 行。', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
