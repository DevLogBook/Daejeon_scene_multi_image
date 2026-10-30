"""Prepare relocatable compact geometry supervision; run on CPU in an allocated job."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grm_matcher.depth_geometry import (discover_disk,discover_uav,group_splits,read_depth,
    candidate_pairs,direction,read_rgb,letterbox_geometry)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--megadepth-scenes',type=Path,required=True)
    p.add_argument('--megadepth-json',type=Path)
    p.add_argument('--uav-rgb',type=Path,required=True)
    p.add_argument('--uav-depth',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--year',default='2024')
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--max-pairs-per-scene',type=int,default=300)
    p.add_argument('--max-attempts-per-scene',type=int,default=6000)
    p.add_argument('--min-points',type=int,default=32)
    p.add_argument('--min-overlap',type=float,default=.05)
    p.add_argument('--max-overlap',type=float,default=.95)
    p.add_argument('--relative-depth',type=float,default=.01)
    p.add_argument('--cycle-px',type=float,default=1.,help='Resized/padded training pixels')
    p.add_argument('--split-json',type=Path)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--inspect',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if args.size<64 or args.size%32 or min(args.max_pairs_per_scene,args.max_attempts_per_scene,args.min_points)<=0:
        p.error('Invalid canvas/limits')
    if not 0<=args.min_overlap<args.max_overlap<=1 or min(args.relative_depth,args.cycle_px)<=0:
        p.error('Invalid geometry thresholds')
    cv2.setNumThreads(1)
    md,missing_md=discover_disk(args.megadepth_scenes,args.megadepth_json)
    uav,missing_uav=discover_uav(args.uav_rgb,args.uav_depth,args.year)
    scenes=md+uav
    def source_signature(scene):
        digest=hashlib.sha256()
        for frame in scene.frames:
            for path in (frame.image,frame.depth):
                stat=path.stat(); digest.update(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
            digest.update(frame.k.tobytes()+frame.distortion.tobytes()+frame.c2w.tobytes())
        digest.update(json.dumps(scene.pairs).encode())
        return digest.hexdigest()
    inventory=[dict(domain=s.domain,scene=s.name,group=s.group,frames=len(s.frames),
                    official_pairs=len(s.pairs) if s.pairs else None,source_signature=source_signature(s)) for s in scenes]
    print(json.dumps({'inventory':inventory,'missing_source_files':{'megadepth':len(missing_md),'uav':len(missing_uav)}},indent=2),flush=True)
    if args.inspect:
        if not md or not uav: raise ValueError('One domain has no usable scene; inspect paths and HDF5 fields')
        for s in (md[0],uav[0]):
            f=s.frames[0]
            depth=read_depth(str(f.depth))
            good=np.isfinite(depth)&(depth>0)
            print(json.dumps(dict(domain=s.domain,image=str(f.image),depth=str(f.depth),
                image_size=[f.width,f.height],depth_shape=list(depth.shape),
                valid_depth_fraction=float(good.mean()),K=f.k.tolist(),c2w=f.c2w.tolist()),indent=2),flush=True)
        return
    override=json.loads(args.split_json.read_text()) if args.split_json else None
    splits=group_splits(scenes,args.seed,override)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','inspect','output')}
    import grm_matcher.depth_geometry as geometry_module
    settings.update(recipe='v3_depth_cache_v1',inventory=inventory,group_split=splits,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(geometry_module.__file__).read_bytes()).hexdigest())
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume: raise FileExistsError('Output is nonempty; use --resume for this exact preparation')
        if json.loads((args.output/'preparation.json').read_text())!=settings:
            raise ValueError('Preparation configuration changed; use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    for sub in ('images','pairs','train','val','previews'): (args.output/sub).mkdir(exist_ok=True)
    atomic_json(args.output/'preparation.json',settings)
    journal=args.output/'pairs.jsonl'
    rows={}
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():
                row=json.loads(line)
                if all((args.output/row[k]).is_file() for k in ('image_a','image_b','labels')):
                    rows[row['pair_id']]=row
    image_records={}
    provenance_path=args.output/'image_sources.json'
    if provenance_path.exists(): image_records=json.loads(provenance_path.read_text())
    def export(frame,scene):
        identity=f'{scene.domain}/{scene.name}/{frame.key}'
        key=hashlib.sha256(identity.encode()).hexdigest()[:24]
        rel=f'images/{key}.npz'
        if identity in image_records and (args.output/rel).is_file(): return image_records[identity]
        rgb,source_hash=read_rgb(frame.image)
        if rgb.shape[:2]!=(frame.height,frame.width): raise ValueError(f'RGB/calibration size differs: {frame.image}')
        _,(left,top,nw,nh)=letterbox_geometry(frame.width,frame.height,args.size)
        canvas=np.zeros((args.size,args.size,3),np.uint8)
        mask=np.zeros((args.size,args.size),bool)
        canvas[top:top+nh,left:left+nw]=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA)
        mask[top:top+nh,left:left+nw]=True
        temp=args.output/(rel+'.tmp')
        with temp.open('wb') as stream: np.savez_compressed(stream,rgb=canvas,valid=mask)
        temp.replace(args.output/rel)
        record=dict(path=rel,source_sha256=source_hash,identity=identity,source=str(frame.image))
        image_records[identity]=record
        atomic_json(provenance_path,image_records)
        return record
    scene_reports=[]
    for scene in scenes:
        accepted=sum(r['domain']==scene.domain and r['scene']==scene.name for r in rows.values())
        attempts=0; rejected={}; started=time.monotonic()
        for i,j in candidate_pairs(scene,args.seed):
            if accepted>=args.max_pairs_per_scene or attempts>=args.max_attempts_per_scene: break
            a,b=scene.frames[i],scene.frames[j]
            identity=f'{scene.domain}/{scene.name}/{a.key}|{b.key}'
            key=hashlib.sha256(identity.encode()).hexdigest()[:24]
            if key in rows: continue
            attempts+=1
            try:
                ab=direction(a,b,args.size,args.relative_depth,args.cycle_px)
                ba=direction(b,a,args.size,args.relative_depth,args.cycle_px)
                # Denominator is the valid source RGB grid, not the padded canvas.
                def overlap(frame,labels):
                    from grm_matcher.depth_geometry import coarse_grid,map_points
                    matrix,_=letterbox_geometry(frame.width,frame.height,args.size)
                    xy=map_points(coarse_grid(args.size),np.linalg.inv(matrix))
                    valid=(xy>=0).all(1)&(xy<=[frame.width-1,frame.height-1]).all(1)
                    return float(labels['valid'].sum()/max(1,valid.sum()))
                ov=[overlap(a,ab),overlap(b,ba)]
                if min(ab['valid'].sum(),ba['valid'].sum())<args.min_points:
                    reason='too_few_trusted_points'
                elif min(ov)<args.min_overlap or max(ov)>args.max_overlap:
                    reason='overlap_out_of_range'
                else: reason=None
                if reason:
                    rejected[reason]=rejected.get(reason,0)+1; continue
                ia,ib=export(a,scene),export(b,scene)
                rel=f'pairs/{key}.npz'; temporary=args.output/(rel+'.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,**{prefix+'_'+k:v for prefix,data in (('ab',ab),('ba',ba)) for k,v in data.items()})
                temporary.replace(args.output/rel)
                row=dict(pair_id=key,domain=scene.domain,scene=scene.name,group=scene.group,
                    split=splits[scene.group],image_a=ia['path'],image_b=ib['path'],labels=rel,
                    source_sha256_a=ia['source_sha256'],source_sha256_b=ib['source_sha256'],
                    trusted_ab=int(ab['valid'].sum()),trusted_ba=int(ba['valid'].sum()),overlap=ov)
                with journal.open('a',encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
                rows[key]=row; accepted+=1
                if accepted<=2:
                    from tools.audit_uav3dcrop_depth import overlay
                    from grm_matcher.depth_geometry import coarse_grid,map_points
                    ta,_=letterbox_geometry(a.width,a.height,args.size); tb,_=letterbox_geometry(b.width,b.height,args.size)
                    overlay(a.image,b.image,map_points(coarse_grid(args.size),np.linalg.inv(ta)),
                        map_points(ab['target'],np.linalg.inv(tb)),ab['valid'],args.output/'previews'/f'{scene.domain}_{scene.name}_{accepted}.jpg')
            except (OSError,ValueError,KeyError) as exc:
                # Every rejected/error pair is recorded, never silently replaced with GT=0.
                with (args.output/'preparation_errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(pair=identity,error=str(exc)))+'\n')
                rejected['source_or_geometry_error']=rejected.get('source_or_geometry_error',0)+1
            if attempts%50==0 or accepted%25==0:
                print(f'{scene.domain}/{scene.name}: accepted={accepted}, attempts={attempts}, rejected={rejected}',flush=True)
        scene_reports.append(dict(domain=scene.domain,scene=scene.name,split=splits[scene.group],
            accepted=accepted,attempts_this_run=attempts,rejected_this_run=rejected,seconds=time.monotonic()-started))
        print(json.dumps(scene_reports[-1]),flush=True)
    # Reject exact image content leaking across splits, including cross-domain copies.
    hashes={split:{r[key] for r in rows.values() if r['split']==split for key in ('source_sha256_a','source_sha256_b')} for split in ('train','val')}
    if hashes['train'] & hashes['val']: raise ValueError('RGB content overlaps train/val; revise group split')
    counts={}
    for split in ('train','val'):
        selected=sorted((r for r in rows.values() if r['split']==split),key=lambda r:r['pair_id'])
        for domain in ('megadepth','uav'):
            n=sum(r['domain']==domain for r in selected)
            counts[split+'/'+domain]=n
            if not n: raise ValueError(f'No accepted {split}/{domain} pairs; see diagnostics, do not weaken geometry blindly')
        (args.output/split/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    atomic_json(args.output/'report.json',dict(counts=counts,scenes=scene_reports,
        source_missing=dict(megadepth=missing_md,uav=missing_uav),
        note='Trusted MVS geometry, not independently exact GT; rejected depth regions remain unknown'))
    print('READY',json.dumps(counts),flush=True)


if __name__=='__main__': main()
