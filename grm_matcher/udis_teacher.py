"""Cache-v2 UDIS labels: independent directions, overlap and round-trip checks."""
from pathlib import Path
import hashlib
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dense_match.teacher_cache_generation import validate_payload, rgb_tensor, save_one, file_hash


def trusted_direction(payload, direction, size=512, stride=16, threshold=.8, cycle_px=2., budget=512):
    reverse = 'BA' if direction == 'AB' else 'AB'
    axis = (torch.arange(size // stride).float()+.5)*stride-.5
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    source = torch.stack((x, y), -1).reshape(-1, 2)
    grid = ((source+.5)*2/size-1)[None, :, None]
    def sample(value, at):
        value = value.float()
        value = value.permute(2,0,1)[None] if value.ndim == 3 else value[None,None]
        return F.grid_sample(value, at, align_corners=False) [0,:,:,0].T
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
    # Cap computation using spread-out indices; never fill with rejected points.
    if len(ids)>budget:
        ids = ids[torch.linspace(0,len(ids)-1,budget).round().long()]
    return source[ids],target[ids],torch.minimum(conf,conf_reverse)[ids],cycle[ids]


class UDISCacheDataset(Dataset):
    def __init__(self, root, config, size=512, threshold=.8, cycle_px=2., min_points=32, limit=0):
        self.root=Path(root); self.config=config; self.size=size
        self.threshold=threshold; self.cycle_px=cycle_px; self.records=[]; self.source_hashes=set()
        compact_key=hashlib.sha256(str((size,config.coarse_stride,threshold,cycle_px,config.max_source_queries,min_points)).encode()).hexdigest()[:16]
        compact_root=self.root/'student_inputs'/compact_key
        self.rejected=0; signature=hashlib.sha256()
        files=sorted(self.root.glob('*.pt'))
        if not files: raise FileNotFoundError(f'No cache-v2 PT files: {root}')
        self.teacher_contracts=set(); started=time.monotonic()
        for number,path in enumerate(files,1):
            if number==1 or number%100==0:
                print(json.dumps({'phase':'teacher_cache_audit','split':str(root),'file':number,'total':len(files),
                                  'accepted':len(self.records),'rejected':self.rejected,'seconds':time.monotonic()-started}),flush=True)
            payload=torch.load(path,map_location='cpu',weights_only=False); validate_payload(payload)
            meta=payload['metadata']
            if meta.get('warp')!='normalized_target_xy_align_corners_false':
                raise ValueError(f'Unknown coordinate convention: {path}')
            if meta.get('teacher')!='RoMaV2' or len(meta.get('source_sha256',[]))!=2:
                raise ValueError(f'Missing Teacher/source provenance: {path}')
            if len(meta.get('weights_sha256',''))!=64 or meta.get('teacher_setting') not in ('base','precise'):
                raise ValueError(f'Missing Teacher weights/setting provenance: {path}')
            self.teacher_contracts.add((meta['weights_sha256'],meta['teacher_setting'],meta.get('teacher_input')))
            if len(self.teacher_contracts)>1: raise ValueError(f'Mixed Teacher versions/settings in {root}')
            self.source_hashes.update(meta['source_sha256'])
            labels=[trusted_direction(payload,d,size,config.coarse_stride,threshold,cycle_px,config.max_source_queries) for d in ('AB','BA')]
            digest=file_hash(path)
            signature.update(path.name.encode()); signature.update(digest.encode())
            if min(len(labels[0][0]),len(labels[1][0]))<min_points:
                self.rejected+=1; continue
            # Dense Teacher warp/logit tensors are large and unused after auditing.
            # Read only compact RGB + trusted coordinates during all training epochs.
            compact_path=compact_root/(path.stem+'_'+digest[:16]+'.pt')
            if not compact_path.exists():
                compact_root.mkdir(parents=True,exist_ok=True)
                save_one(compact_path,{'img_a':payload['img_a'],'img_b':payload['img_b'],'source_cache':str(path)})
            self.records.append((compact_path,labels))
        self.signature=signature.hexdigest()
        if limit: self.records=self.records[:limit]
        if not self.records: raise ValueError(f'No pairs passed quality gates in {root}; inspect cache rather than silently relaxing gates')
        self.audit={'accepted_used':len(self.records),'rejected':self.rejected,'files_audited':len(files),
                    'mean_trusted_points_per_direction':sum(len(label[0]) for _,labels in self.records for label in labels)/(2*len(self.records)),
                    'mean_trusted_cycle_px':sum(float(label[3].mean()) for _,labels in self.records for label in labels)/(2*len(self.records))}
        print(json.dumps({'phase':'teacher_cache_audit_finished','split':str(root),**self.audit}),flush=True)
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        path,labels=self.records[index]
        payload=torch.load(path,map_location='cpu',weights_only=False)
        result={'pair_id':str(path),'domain':'real'}
        for side in ('a','b'):
            result['image_'+side]=F.interpolate(rgb_tensor(payload['img_'+side])[None],(self.size,self.size),mode='bilinear',align_corners=False)[0]
        for d,label in zip(('ab','ba'),labels):
            for key,value in zip(('source','target','confidence','cycle'),label): result[d+'_'+key]=value
        return result
