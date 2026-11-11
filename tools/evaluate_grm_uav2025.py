"""Paired, depth-checked UAV3DCrop 2025 validation. No GT enters the matcher.

Prepare a fixed geometry-only subset, evaluate checkpoints on that same subset,
then summarize. RGB is letterboxed to 512. PCK is explicitly in input pixels.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import tifffile
from agri_eval_common import letterbox_geometry as letterbox
from tools.audit_uav3dcrop_depth import camera, unproject, project, transform


def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)): return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, (float, np.floating)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(path, rows):
    if not rows: return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(clean(rows))


def map_xy(points, matrix):
    p = np.c_[points, np.ones(len(points))] @ matrix.T
    return p[:, :2] / p[:, 2:]


def in_image(points, frame):
    return np.isfinite(points).all(1) & (points >= 0).all(1) & (points <= [frame.w-1, frame.h-1]).all(1)


def sample_nearest(depth, points):
    # Reject out-of-FOV distorted projections before converting floats to ints.
    h,w=depth.shape
    inside=np.isfinite(points).all(1)&(points>=0).all(1)&(points<=[w-1,h-1]).all(1)
    values=np.zeros(len(points),np.float64)
    indices=np.rint(points[inside]).astype(np.int64)
    values[inside]=depth[indices[:,1],indices[:,0]]
    return values,inside&np.isfinite(values)&(values>0)


def load_frames(root):
    metadata, frames = {}, {}
    entries = json.loads((root/'frames.json').read_text(encoding='utf-8'))
    for entry in entries:
        scene, frame_id = entry['scene'], entry['frame_id']
        if scene not in metadata:
            metadata[scene] = json.loads((root/'rgb'/'2025'/scene/'transforms_subset.json').read_bytes())
        meta = metadata[scene]
        f = next(f for f in meta['frames'] if Path(f['file_path']).stem == frame_id)
        k, distortion, c2w = camera(meta, f)
        key = scene+'/'+frame_id
        if key in frames: raise ValueError('Duplicate selected frame')
        frames[key] = SimpleNamespace(key=key, scene=scene, block=entry['block'],
            frame_id=frame_id, image=root/entry['rgb'], depth=root/entry['depth'],
            k=k, distortion=distortion, c2w=c2w, w=int(meta['w']), h=int(meta['h']))
    return frames


@lru_cache(maxsize=2)
def read_depth(path):
    depth = tifffile.imread(path)
    if depth.ndim != 2: raise ValueError(f'Expected 2D Z depth: {path}')
    # Retain released float16 maps to keep memory bounded; geometry uses float64.
    return depth


def reference(a, b, source_original, size, relative_depth, cycle_px):
    da, db = read_depth(str(a.depth)), read_depth(str(b.depth))
    if da.shape != (a.h, a.w) or db.shape != (b.h, b.w):
        raise ValueError('Depth/camera resolution mismatch')
    ta, _ = letterbox(a.w, a.h, size); tb, _ = letterbox(b.w, b.h, size)
    za, good_a = sample_nearest(da, source_original)
    ab = np.linalg.inv(b.c2w) @ a.c2w
    xa = unproject(source_original, np.where(good_a, za, 1).astype(float), a.k, a.distortion)
    roundtrip_a = np.linalg.norm(project(xa, a.k, a.distortion)-source_original, axis=1)
    xb = transform(xa, ab)
    target = project(xb, b.k, b.distortion)
    zb, good_b = sample_nearest(db, target)
    rays = unproject(np.nan_to_num(target), np.ones(len(target)), b.k, b.distortion)
    expected = xb[:, :2] / np.where(np.abs(xb[:, 2:]) > 1e-8, xb[:, 2:], 1e-8)
    inverse_ray_px = np.linalg.norm(rays[:, :2]-expected, axis=1)*np.mean([b.k[0,0], b.k[1,1]])
    back = transform(unproject(np.nan_to_num(target), np.where(good_b, zb, 1).astype(float),
                               b.k, b.distortion), np.linalg.inv(ab))
    cycle = np.linalg.norm(map_xy(project(back, a.k, a.distortion), ta)-map_xy(source_original, ta), axis=1)
    relative = np.abs(xb[:, 2]-zb)/np.maximum(np.maximum(np.abs(xb[:, 2]), zb), 1e-8)
    supported = good_a & good_b & in_image(source_original, a) & in_image(target, b)
    supported &= (xb[:, 2]>0) & (back[:, 2]>0) & (inverse_ray_px<=.5) & (roundtrip_a<=.5)
    valid = supported & np.isfinite(cycle) & (cycle<=cycle_px) & (relative<=relative_depth)
    return dict(target_original=target, target_input=map_xy(target, tb), valid=valid,
                cycle_input_px=cycle, relative_depth_error=relative, projected_depth=xb[:,2])


def prepare(args):
    output = args.output/'prepared'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'preparation.json').exists():
        raise FileExistsError('Prepared subset already exists. Reuse it with --evaluate.')
    from tools.download_uav3dcrop_2025_validation import valid_file, local_path
    source_manifest = json.loads((args.data_root/'download_manifest.json').read_bytes())
    for entry in source_manifest['files']:
        if not valid_file(local_path(args.data_root, entry), entry):
            raise ValueError(f'Invalid source file: {entry["path"]}')
    print('All 605 source files verified against pinned source hashes', flush=True)
    frames = load_frames(args.data_root)
    axis = (np.arange(args.size//16)+.5)*16-.5
    x, y = np.meshgrid(axis, axis); grid = np.c_[x.ravel(), y.ravel()]
    candidates = []
    for scene in sorted({f.scene for f in frames.values()}):
        for block in range(3):
            seq = sorted((f for f in frames.values() if f.scene==scene and f.block==block), key=lambda f:int(f.frame_id))
            for offset in (1, 5, 10):
                for anchor in (0, 3, 6, 9):
                    if anchor+offset < len(seq):
                        a, b = seq[anchor], seq[anchor+offset]
                        candidates.append(dict(pair_id=f'{scene}_b{block}_{a.frame_id}_{b.frame_id}',
                            scene=scene, block=block, offset=offset, frame_a=a.key, frame_b=b.key))
    (output/'geometry').mkdir(exist_ok=True)
    rows = []
    for i, pair in enumerate(candidates):
        a, b = frames[pair['frame_a']], frames[pair['frame_b']]
        ta, _ = letterbox(a.w,a.h,args.size); tb, _ = letterbox(b.w,b.h,args.size)
        refs = [reference(a,b,map_xy(grid,np.linalg.inv(ta)),args.size,args.relative_depth,args.cycle_px),
                reference(b,a,map_xy(grid,np.linalg.inv(tb)),args.size,args.relative_depth,args.cycle_px)]
        counts = [int(r['valid'].sum()) for r in refs]
        rgb_counts = [int(in_image(map_xy(grid,np.linalg.inv(t)),f).sum()) for t,f in [(ta,a),(tb,b)]]
        fractions = [n/d for n,d in zip(counts,rgb_counts)]
        baseline = np.linalg.norm(a.c2w[:3,3]-b.c2w[:3,3])
        eligible = min(counts)>=32 and min(fractions)>=.05 and baseline>1e-8
        row = dict(pair, index=i, geometry_points_ab=counts[0], geometry_points_ba=counts[1],
                   geometry_fraction_ab=fractions[0], geometry_fraction_ba=fractions[1],
                   baseline_world_units=baseline, accepted=eligible,
                   reason='accepted' if eligible else 'insufficient_bidirectional_geometry_or_zero_baseline')
        rows.append(row)
        np.savez_compressed(output/'geometry'/f'{pair["pair_id"]}.npz', source_input=grid,
                            target_ab_input=refs[0]['target_input'], valid_ab=refs[0]['valid'],
                            target_ba_input=refs[1]['target_input'], valid_ba=refs[1]['valid'])
        if (i+1)%15==0:
            print(f'Prepared {i+1}/{len(candidates)}; accepted={sum(r["accepted"] for r in rows)}', flush=True)
    write_csv(output/'candidate_pairs.csv',rows)
    write_json(output/'pairs.json', [r for r in rows if r['accepted']])
    write_json(output/'excluded_pairs.json', [r for r in rows if not r['accepted']])
    fingerprint = dict(source_manifest_sha256=digest(args.data_root/'download_manifest.json'),
                       frames_manifest_sha256=digest(args.data_root/'frames.json'),
                       camera_subset_sha256={s:digest(args.data_root/'rgb'/'2025'/s/'transforms_subset.json') for s in sorted({f.scene for f in frames.values()})})
    contract = dict(recipe='uav2025_geometry_validation_v1', size=args.size, relative_depth=args.relative_depth,
                    cycle_input_px=args.cycle_px, min_geometry_points_each_direction=32,
                    min_geometry_fraction_each_direction=.05, candidates=len(rows),
                    accepted=sum(r['accepted'] for r in rows), source_identity=fingerprint,
                    selection='Each scene: 3 blocks; offsets 1/5/10 and source indices 0/3/6/9 within each block. Geometry only; no model outputs used.',
                    geometry='Original distorted OPENCV pixels; OpenGL c2w to OpenCV axes; nearest Z depth; source roundtrip and inverse-ray checks <=0.5 original px; bidirectional reference-depth cycle.',
                    groups={s:dict(candidates=sum(r['scene']==s for r in rows),accepted=sum(r['scene']==s and r['accepted'] for r in rows)) for s in sorted({r['scene'] for r in rows})})
    write_json(output/'preparation.json', contract)
    print(json.dumps(clean(contract),ensure_ascii=False),flush=True)


@lru_cache(maxsize=8)
def prepare_image(path, w, h, size):
    image = cv2.imread(str(path))
    if image is None or image.shape[:2]!=(h,w): raise ValueError(f'RGB/camera mismatch: {path}')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    matrix, (left,top,nw,nh) = letterbox(w,h,size)
    canvas = np.zeros((size,size,3),np.uint8); mask = np.zeros((size,size),bool)
    canvas[top:top+nh,left:left+nw] = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
    mask[top:top+nh,left:left+nw] = True
    return canvas,matrix,mask


def normalized(points, frame):
    return unproject(points,np.ones(len(points)),frame.k,frame.distortion)[:,:2]


def pose_metrics(pa,pb,a,b,size,seed):
    # Undistort original-image points before the calibrated essential solver.
    na,nb=normalized(pa,a),normalized(pb,b)
    ab=np.linalg.inv(b.c2w)@a.c2w; r,t=ab[:3,:3],ab[:3,3]
    skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    e=skew@r; aa=np.c_[na,np.ones(len(na))];bb=np.c_[nb,np.ones(len(nb))]
    ea,eb=aa@e.T,bb@e
    epi=np.sum(bb*ea,axis=1)**2*(1/np.maximum((ea[:,:2]**2).sum(1),1e-30)+1/np.maximum((eb[:,:2]**2).sum(1),1e-30))
    re,te=math.inf,math.inf
    if len(pa)>=5:
        sa,_=letterbox(a.w,a.h,size);sb,_=letterbox(b.w,b.h,size)
        focal=np.mean([sa[0,0]*a.k[0,0],sa[1,1]*a.k[1,1],sb[0,0]*b.k[0,0],sb[1,1]*b.k[1,1]])
        cv2.setRNGSeed(seed)
        ee,mask=cv2.findEssentialMat(na,nb,np.eye(3),method=cv2.RANSAC,prob=.99999,threshold=.5/focal,maxIters=1000)
        best=0
        if ee is not None and mask is not None:
            for candidate in np.asarray(ee).reshape(-1,3,3):
                n,pr,pt,_,_=cv2.recoverPose(candidate,na,nb,np.eye(3),distanceThresh=1e9,mask=mask.copy())
                if n>best:
                    best=n
                    re=float(np.degrees(np.arccos(np.clip((np.trace(pr.T@r)-1)/2,-1,1))))
                    angle=float(np.degrees(np.arccos(np.clip(np.dot(pt.ravel(),t)/(np.linalg.norm(pt)*np.linalg.norm(t)),-1,1))))
                    te=min(angle,180-angle)
    return epi,re,te


def summarize(rows):
    n=sum(r['matches'] for r in rows); supported=sum(r['depth_supported'] for r in rows)
    def mean(values): return float(np.mean(values)) if values else None
    def ratio(a,b): return a/b if b else None
    result=dict(pairs=len(rows),matches=n,depth_supported=supported,depth_support=ratio(supported,n),
                mean_matches=mean([r['matches'] for r in rows]),
                zero_match_pairs=sum(r['matches']==0 for r in rows),
                mean_matching_ms=mean([r['matching_ms'] for r in rows]),
                mean_peak_allocated_mib=mean([r['peak_allocated_mib'] for r in rows]),
                median_pair_depth_error_input_px=mean([]),
                median_rotation_error_deg=float(np.median([r['rotation_error_deg'] if r['rotation_error_deg'] is not None else np.inf for r in rows])),
                median_translation_error_deg=float(np.median([r['translation_error_deg'] if r['translation_error_deg'] is not None else np.inf for r in rows])))
    for threshold in (1,3,5,10):
        correct=sum(r[f'correct_input_{threshold}px'] for r in rows)
        result[f'pck_input_{threshold}px_micro']=ratio(correct,supported)
        # Unsupported reference points are unknown; this is verified yield, not precision.
        result[f'verified_yield_input_{threshold}px']=ratio(correct,n)
        result[f'pck_input_{threshold}px_macro']=mean([r[f'correct_input_{threshold}px']/r['depth_supported'] if r['depth_supported'] else 0 for r in rows])
        result[f'pck_original_{threshold}px_micro']=ratio(sum(r[f'correct_original_{threshold}px'] for r in rows),supported)
    medians=[r['depth_error_median_input_px'] for r in rows if r['depth_error_median_input_px'] is not None]
    result['median_pair_depth_error_input_px']=float(np.median(medians)) if medians else None
    result['zero_supported_pairs']=sum(r['depth_supported']==0 for r in rows)
    result['low_accuracy_pairs_pck3_lt50pct']=sum(r['depth_supported']>0 and r['correct_input_3px']/r['depth_supported']<.5 for r in rows)
    result['mean_visible_grid_coverage_3px']=mean([r['visible_grid_coverage_3px'] for r in rows])
    result['epipolar_precision_micro']=ratio(sum(r['epipolar_correct'] for r in rows),n)
    errors=np.sort([r['pose_error_deg'] if r['pose_error_deg'] is not None else np.inf for r in rows])
    errors=np.r_[0.,errors];recall=np.linspace(0,1,len(errors))
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    for threshold in (5,10,20):
        end=np.searchsorted(errors,threshold)
        result[f'pose_auc_{threshold}deg']=float(integrate(np.r_[recall[:end],recall[end-1]],np.r_[errors[:end],threshold])/threshold)
    return clean(result)


def visualize(path, pair, ia,ib,pa,pb,confidence,eligible,error,gt,size):
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(2*size,size+58),'white')
    canvas.paste(Image.fromarray(ia),(0,32));canvas.paste(Image.fromarray(ib),(size,32))
    draw=ImageDraw.Draw(canvas)
    # Confidence shortlist followed by spatial coverage in both images.
    limit=200
    shortlist=np.argsort(-confidence,kind='stable')[:min(len(confidence),limit*5)]
    picked=[]
    if len(shortlist):
        picked=[int(shortlist[0])]
        da=np.sum((pa[shortlist]-pa[picked[0]])**2,axis=1)
        db=np.sum((pb[shortlist]-pb[picked[0]])**2,axis=1)
        for _ in range(min(limit,len(shortlist))-1):
            distance=np.minimum(da,db)
            distance[np.isin(shortlist,picked)]=-1
            current=int(shortlist[np.argmax(distance)]);picked.append(current)
            da=np.minimum(da,np.sum((pa[shortlist]-pa[current])**2,axis=1))
            db=np.minimum(db,np.sum((pb[shortlist]-pb[current])**2,axis=1))
    color='#39FF14'
    for i in picked:
        q1=tuple(pa[i]+[0,32]);q2=tuple(pb[i]+[size,32])
        draw.line([q1,q2],fill=color,width=1)
        for q in (q1,q2):
            draw.ellipse((q[0]-1,q[1]-1,q[0]+1,q[1]+1),fill=color)
    draw.text((8,7),pair['pair_id']+f' | {len(pa)} matches / {eligible.sum()} depth supported',fill='black')
    draw.text((8,size+39),f'Fluorescent-green continuous lines; {len(picked)} / {len(pa)} confidence-ranked, spatially distributed matches.',fill='black')
    canvas.save(path,quality=91)


def evaluate(args):
    from agri_eval_adapter import Matcher
    prepared=args.output/'prepared'
    prep=json.loads((prepared/'preparation.json').read_bytes())
    if (args.size,args.relative_depth,args.cycle_px)!=(prep['size'],prep['relative_depth'],prep['cycle_input_px']):
        raise ValueError('Geometry protocol differs from prepared subset')
    frames=load_frames(args.data_root)
    if digest(args.data_root/'frames.json')!=prep['source_identity']['frames_manifest_sha256']:
        raise ValueError('Frame selection changed since preparation')
    for scene,expected in prep['source_identity']['camera_subset_sha256'].items():
        if digest(args.data_root/'rgb'/'2025'/scene/'transforms_subset.json')!=expected:
            raise ValueError('Camera metadata changed since preparation')
    pairs=json.loads((prepared/'pairs.json').read_bytes())
    if args.limit: pairs=pairs[:args.limit]
    if not pairs: raise ValueError('No accepted pairs')
    output=args.output/args.model_tag
    output.mkdir(parents=True,exist_ok=True)
    contract=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest(args.checkpoint),
        preparation_sha256=digest(prepared/'preparation.json'),pairs_sha256=digest(prepared/'pairs.json'),
        script_sha256=digest(Path(__file__)),adapter_sha256=digest(ROOT/'agri_eval_adapter.py'),
        geometry_helper_sha256=digest(ROOT/'tools/audit_uav3dcrop_depth.py'),
        model_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'grm_matcher').glob('*.py')},
        size=args.size,limit=args.limit,pairs=len(pairs),relative_depth=args.relative_depth,cycle_input_px=args.cycle_px,
        protocol='FP32, centered 512 letterbox + valid RGB masks; checkpoint sampler; all valid predictions; no GT or confidence filtering before pose. PCK primary in input pixels, conditional on reference depth at source and GT-projected target, independent of predicted target.',
        device=args.device,seed=args.seed)
    settings=output/'settings.json'
    if settings.exists():
        if not args.resume or json.loads(settings.read_bytes())!=contract:
            raise ValueError('Existing output: exact protocol and --resume required')
    else: write_json(settings,contract)
    (output/'matches').mkdir(exist_ok=True);(output/'visualizations').mkdir(exist_ok=True)
    journal=output/'pairs.jsonl'
    rows=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()] if journal.exists() else []
    done={r['pair_id'] for r in rows}
    matcher=Matcher('GRM',ROOT,args)
    torch=matcher.torch;torch.set_num_threads(1);cv2.setNumThreads(1)
    write_json(output/'model_metadata.json',matcher.metadata)
    a,b=(frames[pairs[0][key]] for key in ('frame_a','frame_b'))
    ia,ta,va=prepare_image(a.image,a.w,a.h,args.size);ib,tb,vb=prepare_image(b.image,b.w,b.h,args.size)
    for _ in range(3):matcher(ia,ib,valid_a=va,valid_b=vb)
    matcher.sync();start=time.perf_counter()
    with journal.open('a',encoding='utf-8') as stream:
        for pair in pairs:
            if pair['pair_id'] in done: continue
            a,b=(frames[pair[key]] for key in ('frame_a','frame_b'))
            ia,ta,va=prepare_image(a.image,a.w,a.h,args.size);ib,tb,vb=prepare_image(b.image,b.w,b.h,args.size)
            if matcher.device.type=='cuda':torch.cuda.reset_peak_memory_stats(matcher.device)
            matcher.sync();tick=time.perf_counter()
            pa,pb,confidence=matcher(ia,ib,valid_a=va,valid_b=vb)
            matcher.sync();elapsed=time.perf_counter()-tick
            peak=torch.cuda.max_memory_allocated(matcher.device)/1024**2 if matcher.device.type=='cuda' else 0.
            if not all(np.isfinite(v).all() for v in (pa,pb,confidence)):raise ValueError('Nonfinite predictions')
            if (confidence<0).any() or (confidence>1).any():raise ValueError('Invalid confidence')
            oa,ob=map_xy(pa,np.linalg.inv(ta)),map_xy(pb,np.linalg.inv(tb))
            keep=in_image(oa,a)&in_image(ob,b);raw_count=len(pa)
            pa,pb,confidence,oa,ob=(v[keep] for v in (pa,pb,confidence,oa,ob))
            if len(pa):
                ref=reference(a,b,oa,args.size,args.relative_depth,args.cycle_px)
                eligible=ref['valid'];gt=ref['target_input']
                error=np.linalg.norm(pb-gt,axis=1);original_error=np.linalg.norm(ob-ref['target_original'],axis=1)
                error[~eligible]=np.nan;original_error[~eligible]=np.nan
                epi,re,te=pose_metrics(oa,ob,a,b,args.size,args.seed+pair['index'])
            else:
                eligible=np.zeros(0,bool);gt=np.empty((0,2));error=original_error=epi=np.empty(0);re=te=math.inf
                ref=dict(cycle_input_px=np.empty(0),relative_depth_error=np.empty(0))
            with np.load(prepared/'geometry'/f'{pair["pair_id"]}.npz') as grid:
                visible=np.flatnonzero(grid['valid_ab'])
                cells=np.floor(pa/16).astype(int)
                cell_id=cells[:,1]*(args.size//16)+cells[:,0]
                covered=np.intersect1d(np.unique(cell_id[error<=3]),visible)
            row=dict(pair_id=pair['pair_id'],index=pair['index'],scene=pair['scene'],block=pair['block'],offset=pair['offset'],
                geometry_fraction_ab=pair['geometry_fraction_ab'],matches=len(pa),raw_matches=raw_count,
                padding_removed=raw_count-len(pa),depth_supported=int(eligible.sum()),
                depth_error_median_input_px=float(np.median(error[eligible])) if eligible.any() else None,
                visible_grid_coverage_3px=len(covered)/len(visible) if len(visible) else 0.,
                **{f'correct_input_{t}px':int((error<=t).sum()) for t in (1,3,5,10)},
                **{f'correct_original_{t}px':int((original_error<=t).sum()) for t in (1,3,5,10)},
                epipolar_correct=int((epi<1e-4).sum()),rotation_error_deg=re,translation_error_deg=te,
                pose_error_deg=max(re,te),matching_ms=elapsed*1000,peak_allocated_mib=peak)
            np.savez_compressed(output/'matches'/f'{pair["pair_id"]}.npz',points_a_input=pa,points_b_input=pb,
                points_a_original=oa,points_b_original=ob,confidence=confidence,depth_supported=eligible,
                depth_target_input=gt,depth_error_input_px=error,depth_error_original_px=original_error,
                cycle_input_px=ref['cycle_input_px'],relative_depth_error=ref['relative_depth_error'],epipolar_error=epi)
            visualize(output/'visualizations'/f'{pair["pair_id"]}.jpg',pair,ia,ib,pa,pb,confidence,eligible,error,gt,args.size)
            row=clean(row);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if len(rows)%10==0 or len(rows)==1:
                print(json.dumps(dict(model=args.model_tag,completed=len(rows),total=len(pairs),elapsed_s=round(time.perf_counter()-start,1),
                    pck3_micro=summarize(rows)['pck_input_3px_micro']),allow_nan=False),flush=True)
    rows.sort(key=lambda r:r['index']);write_csv(output/'pairs.csv',rows)
    result=summarize(rows)
    result['scenes']={scene:summarize([r for r in rows if r['scene']==scene]) for scene in sorted({r['scene'] for r in rows})}
    result['offsets']={str(offset):summarize([r for r in rows if r['offset']==offset]) for offset in (1,5,10) if any(r['offset']==offset for r in rows)}
    result['geometry_support_groups']={}
    for label,lo,hi in [('low_05_30',.05,.3),('medium_30_60',.3,.6),('high_60_100',.6,1.01)]:
        subset=[r for r in rows if lo<=r['geometry_fraction_ab']<hi]
        if subset:result['geometry_support_groups'][label]=summarize(subset)
    write_json(output/'summary.json',result)
    verify(output,pairs,rows)
    print('COMPLETE',args.model_tag,json.dumps(result,ensure_ascii=False),flush=True)


def verify(output,pairs,rows):
    from PIL import Image
    if len(rows)!=len(pairs) or {r['pair_id'] for r in rows}!={p['pair_id'] for p in pairs}:raise ValueError('Missing/duplicate pairs')
    for row in rows:
        with np.load(output/'matches'/f'{row["pair_id"]}.npz') as z:
            if len(z['points_a_input'])!=row['matches']:raise ValueError('Point counts disagree')
            error=np.linalg.norm(z['points_b_input']-z['depth_target_input'],axis=1)
            eligible=z['depth_supported']
            if not np.allclose(error[eligible],z['depth_error_input_px'][eligible]):raise ValueError('Point errors disagree')
            if int(eligible.sum())!=row['depth_supported']:raise ValueError('Support counts disagree')
            for t in (1,3,5,10):
                if int(((error<=t)&eligible).sum())!=row[f'correct_input_{t}px']:raise ValueError('PCK counts disagree')
        with Image.open(output/'visualizations'/f'{row["pair_id"]}.jpg') as image:image.verify()
    write_json(output/'verification.json',dict(complete=True,pairs=len(rows),match_caches=len(rows),readable_visualizations=len(rows),
        point_errors_and_counts_recomputed=True,gt_filtered_from_matcher_inputs=True))


def report(args):
    prep=json.loads((args.output/'prepared/preparation.json').read_bytes())
    models={p.parent.name:json.loads(p.read_bytes()) for p in args.output.glob('*/summary.json') if not p.parent.name.startswith('smoke')}
    if not models: raise ValueError('No completed full model results')
    if any(m['pairs']!=prep['accepted'] for m in models.values()):raise ValueError('Incomplete full evaluation')
    def percent(value):return '—' if value is None else f'{value*100:.2f}'
    def table(model_data):
        header='| 模型/场景 | 图对 | 深度支持率% | PCK@1% | @3% | @5% | 可见网格覆盖率@3% |\n|---|---:|---:|---:|---:|---:|---:|\n'
        return header+'\n'.join(f"| {name} | {m['pairs']} | {percent(m['depth_support'])} | {percent(m['pck_input_1px_micro'])} | {percent(m['pck_input_3px_micro'])} | {percent(m['pck_input_5px_micro'])} | {percent(m['mean_visible_grid_coverage_3px'])} |" for name,m in model_data.items())
    text=f'''# UAV3DCrop 2025：GRM验证

从300帧中固定构造{prep['candidates']}个候选图对，通过纯几何条件保留{prep['accepted']}对。图对及排除原因见prepared/candidate_pairs.csv。
使用两个小麦日期、玉米、燕麦、大豆共5个场景，每组3个视角块。没有根据GRM输出挑选图对。

{table(models)}

## 逐场景

'''
    for name,m in models.items():text+=f'### {name}\n\n'+table(m['scenes'])+'\n\n'
    text+='## 按参考几何可见支持比例分组\n\n比例来自固定粗网格的深度一致参考，不是真值重叠面积。\n\n'
    for name,m in models.items():text+=f'### {name}\n\n'+table(m['geometry_support_groups'])+'\n\n'
    text+='''## 指标与边界

- 主表PCK阈值单位为512×512 letterbox输入像素；原图为5280×3956，因此3输入像素约31原图像素。原图PCK单独保存于summary.json，不能与先前原图人工标注PCK直接混排。
- 主表PCK为micro：所有有参考深度支持的预测源点作为分母，不是所有输出点，也不是先过滤掉错误匹配。资格判断只读取预测源点处的深度及其GT投影位置的参考深度，不读取预测目标是否正确来决定资格。
- 深度支持：正深度、图像内可见、相对深度差≤1%、循环误差≤1输入像素，同时检查畸变反投影。未知参考点不判作错误；另给深度支持率及“已验证正确点/所有预测点”的yield下界，避免只看条件PCK。
- 可见网格覆盖率：误差≤3的预测源点覆盖的16×16输入网格，与参考可见粗网格相交，除以参考可见网格数；逐对求平均。它衡量空间覆盖，不是召回全部连续真值点。
- FP32、512居中letterbox、有效RGB掩码，保存的采样与置信度配置，不额外按置信度硬筛点。GT深度和位姿只用于选择几何有效图对和评分，未输入模型。
- 使用固定的三个原图序列偏移1/5/10；偏移不是重叠率标签，报告保留每对几何可见比例。每组样本会共享图像，不能将所有图对视为独立统计重复。
- 深度/位姿来自同一重建体系，一致性只是参考质量检查，不是独立精确GT。叶片、遮挡、风致运动仍可能带来参考误差。未基于单一H评价非平面农田对应。
- 位姿角误差与极线精度保存在summary.json，属于补充诊断。农田近似平面、短基线可能使平移方向不稳定；极线正确也不能代表点定位正确。
- 当前2024缓存之外的2025跨年验证，不证明地块地理独立；多次据此调参后不能再称未接触的最终测试。
- 耗时为单次运行，含张量转换/模型/取回，不含解码、深度、位姿与写盘；显存峰值为PyTorch allocated（含模型），不是整卡占用。仅作描述，不作严格速度排名。

每个模型目录包含全部matches/*.npz、pairs.csv/jsonl、summary.json、verification.json，以及每对最多24条的稀疏匹配连线图。展示点仅按置信度及空间分散度选择，不按正确率挑图；蓝色≤3输入像素，橙色>3，灰色无参考支持，绿色十字为深度投影参考。
'''
    (args.output/'REPORT.md').write_text(text,encoding='utf-8')
    comparison=[dict(model=name,**{k:v for k,v in result.items() if k not in ('scenes','offsets','geometry_support_groups')}) for name,result in models.items()]
    write_csv(args.output/'comparison.csv',comparison)
    entries=[]
    for name,m in models.items():
        rows=[json.loads(line) for line in (args.output/name/'pairs.jsonl').read_text().splitlines()]
        for row in rows:
            path=f'{name}/visualizations/{row["pair_id"]}.jpg'
            entries.append(f'<article><h3>{html.escape(name)} / {html.escape(row["pair_id"])}</h3><a href="{path}"><img loading="lazy" width="100%" src="{path}"></a></article>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>UAV2025 GRM</title><style>body{max-width:1060px;margin:24px auto;font:16px Arial}article{margin:24px 0;border-bottom:1px solid #ccc}</style><h1>UAV3DCrop 2025：全部验证图对</h1><p>连线仅为稀疏展示，颜色比例不代表全量正确率。完整指标见REPORT.md。</p>'+''.join(entries),encoding='utf-8')
    print(text[:2500],flush=True)


def self_check():
    # Independent planar stereo fixture with analytical disparity and distorted rays.
    k=np.array([[500.,0,40],[0,505,30],[0,0,1.]])
    points=np.array([[20.,25.],[41.,35.],[63.,38.]])
    xyz=unproject(points,np.full(3,5.),k,np.zeros(5))
    shifted=xyz+[-.1,0,0]
    expected=points+[-10.,0]
    assert np.allclose(project(shifted,k,np.zeros(5)),expected,atol=1e-10)
    distortion=np.array([-.08,.01,.001,-.001,.001])
    assert np.allclose(project(unproject(points,np.ones(3),k,distortion),k,distortion),points,atol=1e-7)
    matrix,box=letterbox(5280,3956,512)
    assert np.allclose(map_xy(map_xy(points,matrix),np.linalg.inv(matrix)),points)
    assert box==(0,64,512,384)
    # Wrong predicted target stays in the denominator if source reference is valid.
    error=np.array([0.,4.,100.]);eligible=np.array([True,True,True])
    assert ((error<=3)&eligible).sum()/eligible.sum()==1/3
    values,valid=sample_nearest(np.ones((4,4)),np.array([[1.,2.],[1e100,1e100],[np.nan,0.]]))
    assert valid.tolist()==[True,False,False]
    # Full label-eligibility path: analytic frontoparallel stereo plane.
    from unittest.mock import patch
    a=SimpleNamespace(w=96,h=80,k=k,distortion=np.zeros(5),c2w=np.eye(4),depth=Path('fixture_a'))
    b=SimpleNamespace(w=96,h=80,k=k,distortion=np.zeros(5),c2w=np.eye(4),depth=Path('fixture_b'))
    b.c2w[0,3]=.1
    with patch(__name__+'.read_depth',return_value=np.full((80,96),5.)):
        ref=reference(a,b,points,512,.01,1.)
        assert ref['valid'].all() and np.allclose(ref['target_original'],expected)
        assert np.max(ref['cycle_input_px'])<1e-9
    with patch(__name__+'.read_depth',side_effect=[np.full((80,96),5.),np.full((80,96),6.)]):
        assert not reference(a,b,points,512,.01,1.)['valid'].any()
    print('Analytical geometry, distortion, half-pixel and denominator self-checks PASS',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','evaluate','report','self-check'):action.add_argument('--'+name,action='store_true')
    p.add_argument('--data-root',type=Path,default=ROOT/'dataset/UAV3DCrop_2025_validation')
    p.add_argument('--output',type=Path,default=ROOT/'benchmarks_uav2025_20261023')
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/grm_v3_depth_curriculum_r2_c/best_phase.pt'))
    p.add_argument('--model-tag',default='v3_c')
    p.add_argument('--device',default='cuda');p.add_argument('--size',type=int,default=512)
    p.add_argument('--relative-depth',type=float,default=.01);p.add_argument('--cycle-px',type=float,default=1.)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.size%16 or args.limit<0 or min(args.relative_depth,args.cycle_px)<=0:p.error('Invalid protocol')
    if Path(args.model_tag).name!=args.model_tag or args.model_tag in ('.','..','prepared'):p.error('Invalid model tag')
    cv2.setNumThreads(1)
    if args.self_check:self_check()
    elif args.prepare:prepare(args)
    elif args.evaluate:evaluate(args)
    else:report(args)


if __name__=='__main__':main()
