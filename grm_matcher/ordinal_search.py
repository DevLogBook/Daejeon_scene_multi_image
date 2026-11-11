"""Local-evidence ordinal routing. No GT/true displacement is a head input."""
from dataclasses import dataclass, fields
import hashlib
import json

import torch
from torch import nn
import torch.nn.functional as F

from .adaptive import ground_truth_at
from .architecture_v3 import target_support, select_queries
from .coordinates import flat_indices_to_image_xy
from .local_correlation import GRMRefinementOutput


class SearchLevelHead(nn.Module):
    def __init__(self, monotone=True):
        super().__init__();self.monotone=monotone
        self.net=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,32),nn.GELU(),nn.Linear(32,3))
        nn.init.normal_(self.net[-1].weight,std=1e-3);nn.init.zeros_(self.net[-1].bias)

    def forward(self,features):
        raw=self.net(features.float())
        logit3=raw[...,0]-F.softplus(raw[...,1]) if self.monotone else raw[...,1]
        return torch.stack((raw[...,0],logit3),-1),raw[...,2]


@torch.no_grad()
def level_labels(initial,truth,known,stride=8):
    radius=(initial.float()-truth.float()).abs().amax(-1)/stride
    known=known & torch.isfinite(truth).all(-1) & torch.isfinite(initial).all(-1)
    category=torch.full_like(known,4,dtype=torch.long)
    for boundary,label in ((float('inf'),3),(4,2),(3,1),(2,0)):
        category=torch.where(known & (radius<=boundary),label,category)
    return category,radius


def level_loss(logits,recoverable_logits,category,use_recoverability=True):
    supervised=category<4
    if supervised.any():
        targets=torch.stack((category>0,category>1),-1).float()
        effective=marginal_logits(logits,recoverable_logits) if use_recoverability else logits
        ordinal=F.binary_cross_entropy_with_logits(effective[supervised],targets[supervised])
    else:ordinal=logits.sum()*0
    known=category<4
    recovery=F.binary_cross_entropy_with_logits(recoverable_logits[known],(category[known]<3).float()) if known.any() and use_recoverability else recoverable_logits.sum()*0
    return ordinal,recovery


def marginal_logits(logits,recovery):
    # P(r>t) = P(r>4) + P(r<=4)*P(r>t | r<=4), for known visible GT.
    q=recovery.sigmoid()[...,None]
    probability=(1-q)+q*logits.sigmoid()
    return torch.logit(probability.clamp(1e-6,1-1e-6))


def selected_context(model,batch):
    fa,fb,ta,tb=model.encode_pair(batch['image_a'],batch['image_b'],batch.get('valid_a'),batch.get('valid_b'))
    ma,mb=fa['valid_coarse'].flatten(1),fb['valid_coarse'].flatten(1)
    corr=model.correlation(ta,tb,ma,mb)
    quality,target,support=target_support(corr,ma,mb,model.config)
    width=fa['F_coarse'].shape[-1]
    query,valid=select_queries(quality,support,width,model.config.max_source_queries,
        model.config.v3_spatial_bonus,model.config.v3_spatial_sigma_cells,underfill=model.config.v3_revision>=1)
    target=target.gather(1,query)
    return dict(fa=fa,fb=fb,correlation=corr,query=query,target=target,valid=valid,
        source=flat_indices_to_image_xy(query,width,model.config.coarse_stride),
        initial=flat_indices_to_image_xy(target,width,model.config.coarse_stride))


@torch.no_grad()
def truth_for_queries(batch,source,query,stride):
    if 'ab_target' not in batch:
        return ground_truth_at(batch,source)
    # Coarse-only depth references must NEVER be interpolated to arbitrary points.
    width=batch['image_a'].shape[-1]//stride
    expected=flat_indices_to_image_xy(query,width,stride)
    if not torch.allclose(source,expected,atol=1e-4,rtol=0):
        raise ValueError('Sparse depth-cache GT is only valid at its original source centers')
    truth=batch['ab_target'].gather(1,query[...,None].expand(-1,-1,2)).float()
    known=batch['ab_valid'].gather(1,query).bool()
    return truth,known


def probe_local(model,fa,fb,source,initial,valid):
    projected=(model.mid_refiner.correlation.descriptor_projection(fa['F_mid']),
               model.mid_refiner.correlation.descriptor_projection(fb['F_mid']))
    probe=model.mid_refiner(fa['F_mid'],fb['F_mid'],source,initial,valid,
        source_valid_map=fa.get('valid_mid'),target_valid_map=fb.get('valid_mid'),target_window_size=5,
        projected_features=projected)
    p=probe.local_probability.float();top=p.flatten(-2).topk(2,-1).values
    edge=p[...,0,:].sum(-1)+p[...,-1,:].sum(-1)+p[...,1:-1,0].sum(-1)+p[...,1:-1,-1].sum(-1)
    cosine=(probe.source_descriptor*probe.target_descriptor).sum(-1)
    features=torch.stack((probe.peak_probability,probe.entropy/torch.log(p.new_tensor(25.)),
        top[...,0]-top[...,1],edge,probe.delta_feature_xy.abs().amax(-1)/2,
        probe.covariance_feature.diagonal(dim1=-2,dim2=-1).sum(-1)/8,
        cosine,(p>0).float().mean((-2,-1))),-1)
    return features.detach(),probe,projected


@torch.no_grad()
def cap_levels(level,score,valid,budget):
    requested=(level>0)&valid
    # Upper bound only. Never force a minimum fraction or fill with easy queries.
    order=score.masked_fill(~requested,-torch.inf).argsort(dim=-1,descending=True,stable=True)
    cap=(valid.sum(-1)*budget).floor().long()
    keep=torch.zeros_like(valid).scatter(1,order,torch.arange(valid.shape[1],device=valid.device)[None]<cap[:,None])
    return torch.where(requested&keep,level,0)


@torch.no_grad()
def route_levels(logits,recovery,valid,config):
    p=logits.sigmoid();level=(p[...,0]>=config.ordinal_threshold).long()
    level=torch.where(p[...,1]>=config.ordinal_threshold,torch.full_like(level,2),level)
    if config.ordinal_recoverability:level=torch.where(recovery.sigmoid()>=config.ordinal_threshold,level,0)
    return cap_levels(level,p[...,0]+p[...,1],valid,config.adaptive_budget)


def refine_with_levels(model,fa,fb,source,initial,valid,levels,evidence=None):
    if evidence is None:evidence=probe_local(model,fa,fb,source,initial,valid)
    features,base,projected=evidence
    merged={f.name:getattr(base,f.name) for f in fields(GRMRefinementOutput)}
    merged['local_probability']=F.pad(base.local_probability,(2,2,2,2))
    b,n=valid.shape
    for level,size in ((1,7),(2,9)):
        route=(levels==level)&valid;count=int(route.sum(-1).max()) if n else 0
        if not count:continue
        index=route.long().argsort(dim=-1,descending=True,stable=True)[:,:count];active=route.gather(1,index)
        gather=lambda x:x.gather(1,index[...,None].expand(-1,-1,x.shape[-1]))
        part=model.mid_refiner(fa['F_mid'],fb['F_mid'],gather(source),gather(initial),active,
            source_valid_map=fa.get('valid_mid'),target_valid_map=fb.get('valid_mid'),
            target_window_size=size,projected_features=projected)
        for field in fields(GRMRefinementOutput):
            name=field.name;value=getattr(part,name)
            if name=='local_probability' and size==7:value=F.pad(value,(1,1,1,1))
            mask=active.reshape(b,count,*([1]*(value.ndim-2)))
            target=index.reshape(b,count,*([1]*(value.ndim-2))).expand_as(value)
            replacement=(value*mask).long() if value.dtype==torch.bool else value*mask
            scattered=replacement.new_zeros((b,n,*value.shape[2:])).scatter_add(1,target,replacement)
            old=merged[name];old_mask=route.reshape(b,n,*([1]*(old.ndim-2)))
            merged[name]=torch.where(old_mask,scattered.bool() if old.dtype==torch.bool else scattered,old)
    mid=GRMRefinementOutput(**merged)
    fine=model.fine_refiner(fa['F_fine'],fb['F_fine'],source,mid.dst_xy,mid.valid_mask,
        source_valid_map=fa.get('valid_fine'),target_valid_map=fb.get('valid_fine'))
    fine.valid_mask &= model.points_valid(fine.dst_xy,fb)
    return mid,fine


def refine_ordinal(model,fa,fb,source,initial,valid,truth=None,known=None,mode=None):
    mode=mode or model.config.adaptive_mode
    if mode.startswith('fixed_'):
        mid=model.mid_refiner(fa['F_mid'],fb['F_mid'],source,initial,valid,
            source_valid_map=fa.get('valid_mid'),target_valid_map=fb.get('valid_mid'),target_window_size=int(mode[-1]))
        fine=model.fine_refiner(fa['F_fine'],fb['F_fine'],source,mid.dst_xy,mid.valid_mask,
            source_valid_map=fa.get('valid_fine'),target_valid_map=fb.get('valid_fine'))
        fine.valid_mask &= model.points_valid(fine.dst_xy,fb)
        return mid,fine,None,None,None
    evidence=probe_local(model,fa,fb,source,initial,valid)
    logits,recovery=model.search_level_head(evidence[0])
    if mode=='oracle_5_7_9':
        if truth is None or known is None:raise ValueError('Ordinal Oracle requires explicit GT')
        category,radius=level_labels(initial,truth,valid&known,model.config.mid_stride)
        levels=torch.where(category<3,category,0)
        # Same expansion cap as Learned. Prioritize larger GT error if overloaded.
        levels=cap_levels(levels,radius,valid,model.config.adaptive_budget)
    else:levels=route_levels(logits,recovery,valid,model.config)
    mid,fine=refine_with_levels(model,fa,fb,source,initial,valid,levels,evidence)
    return mid,fine,logits,recovery,levels


def body_fingerprint(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith(('search_risk_head.','search_level_head.','variance_calibrator.')):continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    keys=('coarse_stride','mid_stride','fine_stride','v3_revision','v3_coarse_mode','temperature',
          'max_source_queries','v3_min_matchability','v3_relative_reliability','v3_spatial_bonus','v3_spatial_sigma_cells')
    digest.update(json.dumps({k:getattr(model.config,k) for k in keys},sort_keys=True).encode())
    return digest.hexdigest()
