"""Serial UDIS label preparation with a persistent metadata-based index."""
from pathlib import Path
import json
import os
import pickle
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CACHE_VERSION = 2


def rgb_tensor(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected HWC uint8 RGB')
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()/255


def validate_payload(payload):
    if payload.get('metadata', {}).get('cache_version') != CACHE_VERSION:
        raise ValueError('Legacy cache: use a NEW save directory; old labels must be regenerated')
    for name in ('img_a', 'img_b'):
        image=payload[name]
        # Validation needs the schema, not a full-size float RGB allocation.
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Expected HWC uint8 RGB')
    for direction in ('AB', 'BA'):
        warp, prob, raw = (payload[key+direction] for key in ('warp_', 'confidence_', 'confidence_raw_'))
        if warp.ndim != 3 or warp.shape[-1] != 2 or prob.shape != warp.shape[:2]:
            raise ValueError(f'Bad {direction} warp/probability shape')
        if raw.shape[:2] != warp.shape[:2] or raw.ndim not in (2, 3):
            raise ValueError(f'Bad {direction} raw confidence shape')
        if not all(torch.isfinite(tensor).all() for tensor in (warp, prob, raw)):
            raise FloatingPointError(f'Non-finite {direction} Teacher output')
        if prob.min() < 0 or prob.max() > 1:
            raise ValueError('Overlap probability outside [0,1]')
    for name in ('feat_A', 'feat_B'):
        if name in payload and (payload[name].ndim != 2 or not torch.isfinite(payload[name]).all()):
            raise ValueError(f'Invalid {name}')


def save_one(path, payload):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def trusted_direction(payload, direction, size=512, stride=16, threshold=.8, cycle_px=2., budget=512):
    reverse = 'BA' if direction == 'AB' else 'AB'
    axis = (torch.arange(size // stride).float()+.5)*stride-.5
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    source = torch.stack((x, y), -1).reshape(-1, 2)
    grid = ((source+.5)*2/size-1)[None, :, None]
    def sample(value, at):
        value = value.float()
        value = value.permute(2,0,1)[None] if value.ndim == 3 else value[None,None]
        return F.grid_sample(value, at, align_corners=False)[0,:,:,0].T
    target_norm = sample(payload['warp_'+direction], grid)
    target = (target_norm+1)*size/2-.5
    reverse_grid = target_norm[None,:,None]
    returned = (sample(payload['warp_'+reverse], reverse_grid)+1)*size/2-.5
    conf = sample(payload['confidence_'+direction],grid)[:,0]
    conf_reverse = sample(payload['confidence_'+reverse],reverse_grid)[:,0]
    cycle = (returned-source).norm(dim=-1)
    keep = ((target>=0)&(target<=size-1)).all(-1)
    keep &= torch.isfinite(target).all(-1) & (conf>=threshold) & (conf_reverse>=threshold) & (cycle<=cycle_px)
    ids = torch.where(keep)[0]
    if len(ids)>budget:
        ids = ids[torch.linspace(0,len(ids)-1,budget).round().long()]
    return source[ids],target[ids],torch.minimum(conf,conf_reverse)[ids],cycle[ids]


def _file_stamp(path):
    stat=path.stat()
    return (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def _audit_one(path, before, root, compact_path, size, stride, threshold, cycle_px, budget, min_points):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    validate_payload(payload)
    meta=payload['metadata']
    if meta.get('warp')!='normalized_target_xy_align_corners_false':
        raise ValueError(f'Unknown coordinate convention: {path}')
    if meta.get('teacher')!='RoMaV2' or len(meta.get('source_sha256',[]))!=2:
        raise ValueError(f'Missing Teacher/source provenance: {path}')
    if meta.get('teacher_setting') not in ('base','precise'):
        raise ValueError(f'Missing Teacher setting: {path}')
    labels=[trusted_direction(payload,d,size,stride,threshold,cycle_px,budget) for d in ('AB','BA')]
    if _file_stamp(path)!=before:
        raise RuntimeError(f'Teacher file changed during preparation: {path}')
    entry=dict(source_stamp=before,source_hashes=meta['source_sha256'],
        teacher_contract=(meta.get('weights_sha256'),meta['teacher_setting'],meta.get('teacher_input')),
        compact=None,compact_stamp=None,labels=None)
    if min(len(labels[0][0]),len(labels[1][0]))>=min_points:
        compact_path.parent.mkdir(parents=True,exist_ok=True)
        save_one(compact_path,{'img_a':payload['img_a'],'img_b':payload['img_b'],'source_cache':str(path)})
        entry.update(compact=compact_path.relative_to(root).as_posix(),
                     compact_stamp=_file_stamp(compact_path),labels=labels)
    return entry


class UDISCacheDataset(Dataset):
    def __init__(self, root, config, size=512, threshold=.8, cycle_px=2., min_points=32, limit=0):
        self.root=Path(root);self.config=config;self.size=size
        self.threshold=threshold;self.cycle_px=cycle_px
        self.records=[];self.source_hashes=set();self.teacher_contracts=set();self.rejected=0
        files=sorted(self.root.glob('*.pt'))
        if not files:raise FileNotFoundError(f'No cache-v2 PT files: {root}')
        started=time.monotonic()
        parameters=dict(size=size,stride=config.coarse_stride,threshold=threshold,cycle_px=cycle_px,
                        budget=config.max_source_queries,min_points=min_points)
        key=f's{size}_f{config.coarse_stride}_c{threshold:g}_y{cycle_px:g}_q{config.max_source_queries}_n{min_points}'
        compact_root=self.root/'student_inputs'/key
        index_path=compact_root/'audit_index.pt';entries={}
        candidates=[index_path] if index_path.exists() else sorted((self.root/'student_inputs').glob('*/audit_index.pt'))
        for candidate in candidates:
            try:
                cached=torch.load(candidate,map_location='cpu',weights_only=False)
                if all(cached['contract'].get(k)==v for k,v in parameters.items()):
                    entries=cached['entries'];index_path=candidate;compact_root=candidate.parent;break
            except (OSError,RuntimeError,EOFError,ValueError,KeyError,TypeError,pickle.UnpicklingError):
                continue
        refresh=os.environ.get('GRM_TEACHER_AUDIT_REFRESH','0')=='1'
        updated={};reused=audited=0;dirty=not index_path.exists()
        active={p.name for p in files};inventory=[]
        def save_index():
            compact_root.mkdir(parents=True,exist_ok=True)
            combined={name:e for name,e in entries.items() if name in active};combined.update(updated)
            save_one(index_path,dict(contract=dict(version=2,**parameters),entries=combined))
        print(json.dumps(dict(phase='teacher_cache_audit_cache_check',split=str(root),total=len(files),
                              index=str(index_path),serial=True,checksum=False)),flush=True)
        for number,path in enumerate(files,1):
            before=_file_stamp(path)
            inventory.append(dict(name=path.name,size=before[0],mtime_ns=before[1],ctime_ns=before[2]))
            previous=entries.get(path.name)
            hit=not refresh and previous is not None and previous['source_stamp']==before
            if hit and previous['compact'] is not None:
                try:hit=_file_stamp(self.root/previous['compact'])==previous['compact_stamp']
                except FileNotFoundError:hit=False
            if hit:
                entry=previous;reused+=1
            else:
                if audited==0 or audited%100==0:
                    print(json.dumps(dict(phase='teacher_cache_audit',split=str(root),file=number,total=len(files),
                        accepted=len(self.records),rejected=self.rejected,reused=reused,
                        seconds=time.monotonic()-started)),flush=True)
                compact_path=(self.root/previous['compact'] if previous and previous['compact'] else compact_root/path.name)
                entry=_audit_one(path,before,self.root,compact_path,size,config.coarse_stride,
                                 threshold,cycle_px,config.max_source_queries,min_points)
                audited+=1;dirty=True
            updated[path.name]=entry
            self.teacher_contracts.add(tuple(entry['teacher_contract']))
            if len(self.teacher_contracts)>1:raise ValueError(f'Mixed Teacher versions/settings in {root}')
            self.source_hashes.update(entry['source_hashes'])
            if entry['compact'] is None:self.rejected+=1
            else:self.records.append((self.root/entry['compact'],entry['labels']))
            if dirty and number%1000==0:save_index();dirty=False
        if dirty or set(entries)!=set(updated):save_index()
        self.signature=inventory
        if limit:self.records=self.records[:limit]
        if not self.records:raise ValueError(f'No pairs passed quality gates in {root}')
        self.audit=dict(accepted_used=len(self.records),rejected=self.rejected,files_audited=len(files),
            mean_trusted_points_per_direction=sum(len(l[0]) for _,labels in self.records for l in labels)/(2*len(self.records)),
            mean_trusted_cycle_px=sum(float(l[3].mean()) for _,labels in self.records for l in labels)/(2*len(self.records)))
        self.audit_cache=dict(reused=reused,revalidated=audited,index=str(index_path),seconds=time.monotonic()-started,
                              serial=True,checksum=False)
        print(json.dumps(dict(phase='teacher_cache_audit_cache_hit' if not audited else 'teacher_cache_audit_finished',
                             split=str(root),**self.audit,**self.audit_cache)),flush=True)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        path,labels=self.records[index]
        payload=torch.load(path,map_location='cpu',weights_only=False)
        result={'pair_id':str(path),'domain':'real'}
        for side in ('a','b'):
            result['image_'+side]=F.interpolate(rgb_tensor(payload['img_'+side])[None],(self.size,self.size),mode='bilinear',align_corners=False)[0]
        for d,label in zip(('ab','ba'),labels):
            for key,value in zip(('source','target','confidence','cycle'),label):result[d+'_'+key]=value
        return result
