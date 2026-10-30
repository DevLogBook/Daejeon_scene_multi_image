"""Dataset and stage-specific losses for geometry-reference V3 fine-tuning.

Old training entry points are unchanged. Unknown geometry is NEVER a negative.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .coordinates import flat_indices_to_image_xy,image_xy_to_normalized_grid
from .imc_dataset import sparse_bidirectional_labels
from .architecture_v3 import matching_probability
from .stage25 import GRMStage25CoarseOutput,_local_probability_nll
from .scratch_transfer import ScratchTransferObjective

FINE_PREFIXES=('fine_refiner.','fpn.fine_lateral.','fpn.mid_to_fine.',
               'fpn.fine_output.','fpn._fine_alpha_logit')


class DepthCacheDataset(Dataset):
    def __init__(self,root,split,domain,config,limit=0):
        self.root=Path(root); self.config=config; self.domain=domain
        preparation=json.loads((self.root/'preparation.json').read_text())
        if preparation['recipe']!='v3_depth_cache_v1': raise ValueError('Unsupported geometry cache')
        self.size=preparation['size']
        manifest=self.root/split/'manifest.jsonl'
        self.records=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        self.records=[r for r in self.records if r['domain']==domain]
        # Deterministic scene-interleaving avoids validating only the first scene.
        groups={}
        for r in self.records: groups.setdefault(r['scene'],[]).append(r)
        ordered=[]
        while any(groups.values()):
            for name in sorted(groups):
                if groups[name]: ordered.append(groups[name].pop(0))
        self.records=ordered[:limit] if limit else ordered
        if not self.records: raise ValueError(f'Empty geometry {split}/{domain}')
        self.groups={r['group'] for r in self.records}
        self.source_hashes={r[k] for r in self.records for k in ('source_sha256_a','source_sha256_b')}
        digest=hashlib.sha256((manifest.read_bytes()+(self.root/'preparation.json').read_bytes())
            +json.dumps(self.records,sort_keys=True).encode())
        seen=set()
        for row in self.records:
            for key in ('image_a','image_b','labels'):
                path=(self.root/row[key]).resolve()
                if not path.is_relative_to(self.root.resolve()) or not path.is_file():
                    raise ValueError(f'Missing/invalid cache path {row[key]}')
                if path not in seen:
                    stat=path.stat(); digest.update(f'{row[key]}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
                    seen.add(path)
        self.signature=digest.hexdigest()

    def __len__(self): return len(self.records)

    def __getitem__(self,index):
        row=self.records[index]; result={'domain':self.domain,'pair_id':row['pair_id']}
        for side in ('a','b'):
            with np.load(self.root/row['image_'+side],allow_pickle=False) as data:
                value=torch.from_numpy(data['rgb'].copy()).permute(2,0,1).float()/255
                valid=torch.from_numpy(data['valid'].copy()).bool()[None]
            if value.shape!=(3,self.size,self.size): raise ValueError('Cache image size changed')
            if self.config.input_range=='imagenet_normalized':
                value=(value-value.new_tensor(self.config.normalization_mean)[:,None,None])/value.new_tensor(self.config.normalization_std)[:,None,None]
            result['image_'+side]=value; result['valid_'+side]=valid
        with np.load(self.root/row['labels'],allow_pickle=False) as data:
            for direction in ('ab','ba'):
                for key in ('target','valid','weight'):
                    result[direction+'_'+key]=torch.from_numpy(data[direction+'_'+key].copy())
                expected=(self.size//16)**2
                if result[direction+'_target'].shape!=(expected,2) or result[direction+'_valid'].shape!=(expected,):
                    raise ValueError('Invalid correspondence grid in cache')
        return result


class MixedDepthReplay(Dataset):
    """Per block: MegaDepth, synthetic, UAV, synthetic (equal real domains)."""
    def __init__(self,md,uav,synthetic,pairs_per_domain,seed):
        self.sources=[md,synthetic,uav,synthetic]
        self.count=min(pairs_per_domain,len(md),len(uav))
        if self.count<1: raise ValueError('No balanced training samples')
        self.seed=seed; self.set_epoch(0)
    def set_epoch(self,epoch):
        generator=torch.Generator().manual_seed(self.seed+epoch)
        self.indices=[]
        for source in self.sources:
            order=[]
            while len(order)<self.count: order.extend(torch.randperm(len(source),generator=generator).tolist())
            self.indices.append(order[:self.count])
    def __len__(self): return 4*self.count
    def __getitem__(self,index):
        slot=index%4; row=dict(self.sources[slot][self.indices[slot][index//4]])
        if slot in (1,3): row['domain']='synthetic'
        return row


def masked_mean(value,mask,weight):
    selected=torch.where(mask,weight,torch.zeros_like(weight))
    return (value*selected).sum()/selected.sum().clamp_min(1e-8)


def refinement_terms(mid,fine,target,initial,known,weight,config,fine_weight=1.,nll_weight=.1,
                     mid_coordinate_weight=.5,fine_coordinate_weight=1.):
    """32px structural budget; valid Mid target is projected to its 24px box.

    Fine NLL is independently guarded by actual Mid->GT <=8px. The final
    coordinate loss retains gradient through Mid when Fine is not yet centered.
    """
    mr=(config.mid_window_size//2)*config.mid_stride
    fr=(config.fine_window_size//2)*config.fine_stride
    budget=((target-initial).abs()<=mr+fr).all(-1)
    mask=known & budget & mid.valid_mask & fine.valid_mask
    mid_target=initial+(target-initial).clamp(-mr,mr)
    mid_error=F.smooth_l1_loss(mid.dst_xy,mid_target,reduction='none').mean(-1)
    fine_error=F.smooth_l1_loss(fine.dst_xy,target,reduction='none').mean(-1)
    mid_coord=masked_mean(mid_error,mask,weight)
    fine_coord=masked_mean(fine_error,mask,weight)
    mid_nll=_local_probability_nll(mid.local_probability,initial,mid_target,
                                  config.mid_stride,mask,weight,1e-8)
    fine_mask=mask & ((target-mid.dst_xy.detach()).abs()<=fr).all(-1)
    fine_nll=_local_probability_nll(fine.local_probability,mid.dst_xy.detach(),target,
                                   config.fine_stride,fine_mask,weight,1e-8)
    total=mid_coordinate_weight*mid_coord+nll_weight*mid_nll+fine_weight*(fine_coordinate_weight*fine_coord+nll_weight*fine_nll)
    return total,dict(mid_coordinate=mid_coord.detach(),fine_coordinate=fine_coord.detach(),
        mid_nll=mid_nll.detach(),fine_nll=fine_nll.detach(),local_supervised_count=float(mask.sum()),
        fine_nll_supervised_count=float(fine_mask.sum()))


class DepthObjective:
    def __init__(self,config,args):
        self.config=config; self.args=args
        self.base=ScratchTransferObjective(config,epipolar_weight=0.,anchor_weight=args.anchor_weight,exact_dense=True)
        # No duplicate positive matchability hidden inside coarse CE.
        self.base.adaptation.matchability_distillation_weight=0.

    def freeze(self,model,real,train,epoch):
        backbone=train and epoch>=self.args.freeze_backbone_epochs
        fine=train and epoch>=self.args.freeze_fine_epochs and (not real or self.args.real_fine_weight>0)
        for name,parameter in model.named_parameters():
            if name.startswith('backbone.'): parameter.requires_grad_(backbone)
            elif name.startswith(FINE_PREFIXES): parameter.requires_grad_(fine)
        model.backbone.train(backbone)
        for name,module in model.named_modules():
            if name.startswith(FINE_PREFIXES): module.train(fine)

    def agri(self,model,batch,train,epoch):
        domain=batch['domain'][0]; real=domain!='synthetic'; self.freeze(model,real,train,epoch)
        a,b=batch['image_a'],batch['image_b']; size=a.shape[-1]
        h=size//self.config.coarse_stride
        grid=flat_indices_to_image_xy(torch.arange(h*h,device=a.device)[None],h,self.config.coarse_stride)
        directions={}
        for name,side in (('ab','a'),('ba','b')):
            if real:
                target=batch[name+'_target'].float(); valid=batch[name+'_valid'].bool(); weight=batch[name+'_weight'].float()
            else:
                sampling=image_xy_to_normalized_grid(grid,size,size).unsqueeze(2)
                field=batch.get('gt_dense_mapping' if name=='ab' else 'gt_dense_mapping_ba')
                if field is None:
                    from .stage25 import transform_points
                    target=transform_points(grid,torch.linalg.inv(batch['H_ab']))
                else:
                    target=F.grid_sample(field.float(),sampling,align_corners=False)[:,:,:,0].transpose(1,2)
                valid=F.grid_sample(batch['gt_overlap_'+side].float(),sampling,align_corners=False)[:,0,:,0]>.999
                valid &= torch.isfinite(target).all(-1)&(target>=0).all(-1)&(target<=size-1).all(-1)
                weight=valid.float()
            directions[name]=(target,valid,weight)
        def labels(name):
            target,valid,weight=directions[name]
            return sparse_bidirectional_labels(dict(image_a=a,sparse_a=grid,sparse_b=target,
                sparse_valid=valid,sparse_confidence=weight),a.device,self.config,self.base.adaptation).forward
        forward,backward=labels('ab'),labels('ba')
        if not forward.valid_mask.any() or not backward.valid_mask.any(): return None,{'valid_count':0.}
        if train:
            def color(image):
                if self.config.input_range!='zero_one': return image
                return (image*(.9+.2*torch.rand((1,3,1,1),device=image.device))
                        +(torch.rand((1,1,1,1),device=image.device)-.5)*.06).clamp(0,1)
            a,b=color(a),color(b)
        output,aux=model(a,b,return_auxiliary=True,valid_a=batch.get('valid_a'),valid_b=batch.get('valid_b'))
        corr=aux.correlation
        coarse_output=GRMStage25CoarseOutput(corr.similarity,corr.matchability_a,corr.matchability_b,h,h)
        coarse=self.base.coarse(coarse_output,forward,backward)['total']
        zero=coarse*0
        # Correct-GT anchors maintain refinement gradients even when learned queries underfill.
        ids=torch.where(forward.valid_mask[0])[0]
        if len(ids)>self.args.anchor_points:
            ids=ids[torch.linspace(0,len(ids)-1,self.args.anchor_points,device=ids.device).round().long()]
        am,af=model.refine_points(aux.features_a,aux.features_b,grid[:,ids],forward.coarse_dst_xy[:,ids],forward.valid_mask[:,ids])
        fw=self.args.real_fine_weight if real else 1.
        anchor,anchor_metrics=refinement_terms(am,af,forward.fine_dst_xy[:,ids],forward.coarse_dst_xy[:,ids],
            forward.valid_mask[:,ids],forward.confidence[:,ids],self.config,fw,self.args.local_nll_weight,
            self.args.mid_coordinate_weight,self.args.fine_coordinate_weight)
        predicted=confidence=overlap=cycle_loss=zero
        scored=correct=matches=0.; pred_metrics={}
        gt,gt_valid,gt_weight=directions['ab']
        if output.src_xy.shape[1]:
            q=aux.query_indices; target=gt.gather(1,q[...,None].expand(-1,-1,2))
            known=gt_valid.gather(1,q)&output.valid_mask
            weight=gt_weight.gather(1,q)
            initial=flat_indices_to_image_xy(aux.target_indices,h,self.config.coarse_stride)
            predicted,pred_metrics=refinement_terms(aux.mid,aux.fine,target,initial,known,weight,
                self.config,fw,self.args.local_nll_weight,self.args.mid_coordinate_weight,self.args.fine_coordinate_weight)
            error=(output.dst_xy-target).norm(dim=-1)
            if aux.confidence_logits is not None:
                confidence=masked_mean(F.binary_cross_entropy_with_logits(aux.confidence_logits,
                    (error.detach()<=3).float(),reduction='none'),known,weight)
            if self.args.cycle_weight>0:
                cycle=model.direct_cycle(aux.features_a,aux.features_b,matching_probability(corr,self.config),
                                         output.src_xy,output.dst_xy,known)
                cmask=known&(error.detach()<=3)&cycle.valid_mask
                cycle_loss=masked_mean(F.smooth_l1_loss(cycle.returned_src_xy,output.src_xy,reduction='none').mean(-1),cmask,weight)
            scored=float(known.sum()); correct=float(((error<=3)&known).sum()); matches=float(output.valid_mask.sum())
        if real:
            # Positives only. Missing/inconsistent depth is unknown, not an outlier.
            positive=self.base.coarse._positive_matchability_loss(coarse_output,forward,backward)
        else:
            positive=zero
            values=[]
            for side,pred,feat in (('a',corr.matchability_a,aux.features_a),('b',corr.matchability_b,aux.features_b)):
                truth=F.interpolate(batch['gt_overlap_'+side].float(),(h,h),mode='nearest-exact').flatten(1)
                mask=feat['valid_coarse'].flatten(1)
                if mask.any(): values.append(F.binary_cross_entropy(pred[mask].clamp(1e-6,1-1e-6),truth[mask]))
            if values: overlap=torch.stack(values).mean()
        total=coarse+self.args.anchor_weight*anchor+predicted+self.args.confidence_weight*confidence
        total+=self.args.positive_matchability_weight*positive+self.args.overlap_weight*overlap+self.args.cycle_weight*cycle_loss
        probability=matching_probability(corr,self.config)
        ranking=probability.topk(min(4,h*h),dim=-1).indices
        cells=torch.round((gt+.5)/self.config.coarse_stride-.5).long()
        coarse_known=gt_valid&(cells>=0).all(-1)&(cells<h).all(-1)
        target_cells=cells[...,1].clamp(0,h-1)*h+cells[...,0].clamp(0,h-1)
        initial_all=flat_indices_to_image_xy(ranking[...,0],h,self.config.coarse_stride)
        linf=(initial_all-gt).abs().amax(-1)
        metrics=dict(loss=total.detach(),coarse=coarse.detach(),anchor=anchor.detach(),refinement=predicted.detach(),
            confidence=confidence.detach(),overlap=overlap.detach(),matchability=positive.detach(),cycle=cycle_loss.detach(),
            valid_count=float(gt_valid.sum()),predicted_matches=scored,correct_at_3px=correct,
            output_matches=matches,reference_scored_matches=scored,
            coarse_reference_count=float(coarse_known.sum()),
            coarse_recall1_count=float(((ranking[...,0]==target_cells)&coarse_known).sum()),
            coarse_recall4_count=float(((ranking==target_cells[...,None]).any(-1)&coarse_known).sum()),
            initial_24_32_count=float(((linf>24)&(linf<=32)&coarse_known).sum()),
            coarse_initial_error_mean=float((initial_all-gt).norm(dim=-1)[coarse_known].mean()) if coarse_known.any() else 0.)
        metrics.update({'pred_'+k:float(v) for k,v in pred_metrics.items()})
        for name in ('megadepth','uav','synthetic'):
            metrics[name+'_batch']=float(domain==name)
            metrics[name+'_loss']=float(total.detach()) if domain==name else 0.
        return total,metrics
