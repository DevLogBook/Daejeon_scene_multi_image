"""Split-first data contracts and GT-centered *training-only* search offsets."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset,ConcatDataset

from .coordinates import flat_indices_to_image_xy
from .ordinal_search import level_labels
from .scratch_transfer import ResizedHomographyDataset,check_homography_splits


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SearchDataset(Dataset):
    def __init__(self,base,split,groups=None):
        self.base,self.split=base,split
        self.records=base.records
        self.groups=groups
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        row=dict(self.base[index]);record=self.records[index]
        group=next((str(record[key]) for key in ('flight_id','scene_id','scene','source_image','group') if record.get(key)), 'unlabelled')
        row.update(scene_id=group,sample_index=index)
        return row


def make_source(root,kind,split,config,size=512,domain='uav'):
    root=Path(root)
    if kind=='homography':
        isolation=check_homography_splits(root)
        membership={}
        for part in ('train','val','test'):
            path=root/part/'manifest.jsonl'
            if not path.exists():continue
            records=[json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
            # Flight/scene metadata takes precedence if it actually exists.
            for row in records:
                key=next((k for k in ('flight_id','scene_id','scene','source_image') if row.get(k)),None)
                if key is None:raise ValueError('A source or scene identity is required before offset generation')
                identity=key+':'+str(row[key]).replace('\\','/').casefold()
                if identity in membership and membership[identity]!=part:raise ValueError('Flight/source overlaps splits: '+identity)
                membership[identity]=part
        base=ResizedHomographyDataset(root,split,config,size)
        fingerprint=sha(root/'train/manifest.jsonl')
        membership_sha=hashlib.sha256(json.dumps(membership,sort_keys=True).encode()).hexdigest()
    else:
        from .depth_curriculum import GeometryData
        from tools.audit_grm_curriculum_data import audit
        isolation=audit(root/'preparation.json',root)
        if isolation['errors']:raise ValueError(str(isolation['errors']))
        base=GeometryData(root,split,domain,config)
        if base.size!=size:raise ValueError('Depth cache/input size differs; regenerate references instead of interpolating')
        fingerprint=sha(root/'train/manifest.jsonl')
        membership_sha=sha(root/'preparation.json')
    dataset=SearchDataset(base,split)
    info=dict(kind=kind,domain=domain,size=size,coarse_stride=config.coarse_stride,
        train_manifest_sha256=fingerprint,split_membership_sha256=membership_sha,split=split,
        manifest_sha256=sha(root/split/'manifest.jsonl'),isolation=isolation)
    return dataset,info


class FrozenUAVSearchDataset(Dataset):
    """Evaluation-only: frozen depth-visible 2025 references at original centers."""
    def __init__(self,prepared,rgb_root,config,*,image_cache=None):
        from tools.evaluate_grm_uav2025 import load_frames
        if config.coarse_stride!=16:raise ValueError('Frozen UAV reference grid is stride16; no interpolation is allowed')
        self.root=Path(prepared);self.frames=load_frames(Path(rgb_root));self.config=config
        self.records=json.loads((self.root/'pairs.json').read_text(encoding='utf-8-sig'))
        self.image_cache=Path(image_cache) if image_cache else None
        if self.image_cache:self.image_cache.mkdir(parents=True,exist_ok=True)
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        from tools.evaluate_grm_uav2025 import prepare_image
        row=self.records[index];result=dict(pair_id=row['pair_id'],scene_id=row['scene'],sample_index=index)
        for side,key in (('a','frame_a'),('b','frame_b')):
            frame=self.frames[row[key]]
            cached=None
            if self.image_cache:
                stat=frame.image.stat()
                key=hashlib.sha256(f'uav_rgb_letterbox512_v1:{frame.image.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{frame.w}:{frame.h}'.encode()).hexdigest()
                cached=self.image_cache/(key+'.npz')
            if cached is not None and cached.exists():
                with np.load(cached,allow_pickle=False) as z:image,valid=z['rgb'].copy(),z['valid'].copy()
            else:
                image,_,valid=prepare_image(frame.image,frame.w,frame.h,512)
                if cached is not None:
                    temp=cached.with_suffix('.tmp')
                    with temp.open('wb') as stream:np.savez_compressed(stream,rgb=image,valid=valid)
                    temp.replace(cached)
            value=torch.from_numpy(image).permute(2,0,1).float()/255
            if self.config.input_range=='imagenet_normalized':
                value=(value-value.new_tensor(self.config.normalization_mean)[:,None,None])/value.new_tensor(self.config.normalization_std)[:,None,None]
            result['image_'+side]=value;result['valid_'+side]=torch.from_numpy(valid)[None]
        with np.load(self.root/'geometry'/f"{row['pair_id']}.npz") as z:
            expected=flat_indices_to_image_xy(torch.arange(1024)[None],32,16)[0].numpy()
            if not np.allclose(z['source_input'],expected,atol=1e-5):raise ValueError('Frozen UAV source grid differs')
            for prefix in ('ab','ba'):
                result[prefix+'_target']=torch.from_numpy(z['target_'+prefix+'_input'].copy()).float()
                result[prefix+'_valid']=torch.from_numpy(z['valid_'+prefix].copy()).bool()
        return result


class PartitionedUAVSearchDataset(FrozenUAVSearchDataset):
    """Explicitly repurposed diagnostic data: split by scene before augmentation."""
    def __init__(self,prepared,rgb_root,config,split_file,split,*,image_cache=None):
        super().__init__(prepared,rgb_root,config,image_cache=image_cache)
        plan=json.loads(Path(split_file).read_text(encoding='utf-8-sig'))
        if plan.get('recipe')!='uav2025_repurposed_search_v1':raise ValueError('Explicit UAV repurposing split required')
        if plan['prepared_manifest_sha256']!=sha(self.root/'pairs.json'):raise ValueError('UAV pair manifest changed')
        mapping=plan['scene_split'];records=self.records
        if set(mapping)!={r['scene'] for r in records} or set(mapping.values())!={'train','val','test'}:
            raise ValueError('All scenes must be assigned to nonempty train/val/test splits')
        membership={};pair_ids=set()
        for row in records:
            if row['pair_id'] in pair_ids:raise ValueError('Duplicate UAV pair ID')
            pair_ids.add(row['pair_id']);part=mapping[row['scene']]
            for key in ('frame_a','frame_b'):
                frame=row[key]
                if frame in membership and membership[frame]!=part:raise ValueError('UAV frame crosses splits')
                membership[frame]=part
        self.records=[r for r in records if mapping[r['scene']]==split];self.split=split
        if not self.records:raise ValueError('Empty UAV partition')
        digest=lambda rows:hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        self.source_info=dict(kind='uav2025_repurposed',domain='uav',size=512,coarse_stride=16,split=split,
            train_manifest_sha256=digest([r for r in records if mapping[r['scene']]=='train']),
            manifest_sha256=digest(self.records),split_membership_sha256=sha(split_file),
            prepared_manifest_sha256=plan['prepared_manifest_sha256'],
            isolation=dict(repurposed_test=True,scene_split=mapping,frame_overlap=False,
                counts={p:sum(mapping[r['scene']]==p for r in records) for p in ('train','val','test')}))


class OffsetSampler:
    def __init__(self,*,split,distribution='mixed',empirical=None,mix=.5,max_radius=6.,
                 snap=True,max_attempts=24):
        if split!='train':raise ValueError('Artificial offsets are forbidden on validation/test')
        if distribution not in ('uniform','empirical','mixed'):raise ValueError('Unknown offset distribution')
        if not 0<=mix<=1 or max_radius<=4:raise ValueError('Invalid offset mixture/range')
        if distribution!='uniform' and empirical is None:raise ValueError('Empirical/Mixed requires a TRAIN-only reservoir')
        self.distribution,self.pool,self.mix=distribution,empirical,mix
        self.max_radius,self.snap,self.max_attempts=max_radius,snap,max_attempts

    def draw(self,truth,desired,known,config,shape,generator):
        b,n,_=truth.shape;device=truth.device
        accepted=torch.zeros_like(known);centers=torch.zeros_like(truth)
        origins=torch.zeros_like(known)
        bounds=((0.,2.),(2.,3.),(3.,4.),(4.,self.max_radius))
        for _ in range(self.max_attempts):
            delta=torch.zeros_like(truth);empirical_origin=torch.zeros_like(known)
            for category,(lo,hi) in enumerate(bounds):
                radius=(lo*lo+torch.rand((b,n),device=device,generator=generator)*(hi*hi-lo*lo)).sqrt()
                face=torch.randint(0,4,(b,n),device=device,generator=generator)
                along=(torch.rand((b,n),device=device,generator=generator)*2-1)*radius
                normal=torch.where(face<2,-radius,radius)
                proposed=torch.stack((torch.where(face%2==0,normal,along),torch.where(face%2==0,along,normal)),-1)
                if self.distribution!='uniform':
                    pool=self.pool.get(str(category),[])
                    if not pool and self.distribution=='empirical' and ((desired==category)&known).any():
                        raise ValueError(f'No empirical offsets in category {category}; use Mixed or collect more TRAIN samples')
                    if pool:
                        vectors=torch.as_tensor(pool,device=device,dtype=truth.dtype)
                        sampled=vectors[torch.randint(len(vectors),(b,n),device=device,generator=generator)]
                        use=torch.ones((b,n),device=device,dtype=torch.bool) if self.distribution=='empirical' else torch.rand((b,n),device=device,generator=generator)<self.mix
                        proposed=torch.where(use[...,None],sampled,proposed)
                        empirical_origin |= (desired==category)&use
                delta=torch.where((desired==category)[...,None],proposed,delta)
            proposed=truth+delta*config.mid_stride
            if self.snap:
                proposed=(torch.round((proposed+.5)/config.coarse_stride-.5)+.5)*config.coarse_stride-.5
            actual,_=level_labels(proposed,truth,known,config.mid_stride)
            inside=(proposed>=0).all(-1)&(proposed<=proposed.new_tensor([shape[1]-1,shape[0]-1])).all(-1)
            take=known&~accepted&inside&(actual==desired)
            centers=torch.where(take[...,None],proposed,centers);accepted |= take
            origins=torch.where(take,empirical_origin,origins)
            if (accepted|~known).all():break
        return centers,accepted,origins
