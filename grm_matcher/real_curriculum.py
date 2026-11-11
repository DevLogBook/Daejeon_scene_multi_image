"""Full-coverage real-domain training data, without changing legacy recipes."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .depth_curriculum import CurriculumObjective, GeometryData, parameter_group
from .udis_teacher import UDISCacheDataset


class RealGeometryData(GeometryData):
    def __init__(self, *args, fine_min_weight=.8, fine_cycle_px=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.fine_min_weight, self.fine_cycle_px = fine_min_weight, fine_cycle_px

    def __getitem__(self, index):
        import numpy as np
        sample = super().__getitem__(index)
        with np.load(self.root/self.records[index]['labels'], allow_pickle=False) as labels:
            for d in ('ab', 'ba'):
                if d+'_cycle' not in labels:
                    raise ValueError('Fine quality mask requires per-direction cycle residuals; rebuild cache')
                cycle = torch.from_numpy(labels[d+'_cycle'].copy()).float()
                sample[d+'_fine_valid'] = (sample[d+'_valid'] & torch.isfinite(cycle)
                    & (cycle <= self.fine_cycle_px) & (sample[d+'_weight'] >= self.fine_min_weight))
        return sample


def teacher_grid(sample, config, size, fine_confidence=.95, fine_cycle_px=.5):
    """Scatter audited sparse Teacher samples onto their exact coarse cells.

    Absent cells remain unknown. The student query budget does not truncate
    Teacher supervision. No reverse direction is manufactured by swapping AB.
    """
    result = dict(domain='udis', pair_id=sample['pair_id'], scene_id='udis_training')
    n = (size//config.coarse_stride)**2
    for side in ('a', 'b'):
        image = sample['image_'+side]
        if config.input_range == 'imagenet_normalized':
            image = (image-image.new_tensor(config.normalization_mean)[:, None, None])/image.new_tensor(config.normalization_std)[:, None, None]
        result['image_'+side] = image
        result['valid_'+side] = torch.ones(1, size, size, dtype=torch.bool)
    for d in ('ab', 'ba'):
        source, target, conf, cycle = [sample[d+'_'+k] for k in ('source', 'target', 'confidence', 'cycle')]
        cell = (source+.5)/config.coarse_stride-.5
        ij = cell.round().long()
        ids = ij[:, 1]*(size//config.coarse_stride)+ij[:, 0]
        if not torch.allclose(cell, ij.float(), atol=1e-5, rtol=0) or len(ids.unique()) != len(ids):
            raise ValueError('Teacher source samples must be unique coarse cell centers')
        if len(ids) and ((ij<0).any() or (ij>=size//config.coarse_stride).any()):
            raise ValueError('Teacher source grid outside canvas')
        if not all(torch.isfinite(v).all() for v in (target, conf, cycle)):
            raise ValueError('Nonfinite trusted Teacher samples')
        if ((target<0)|(target>size-1)).any() or ((conf<=0)|(conf>1)).any() or (cycle<0).any():
            raise ValueError('Invalid trusted Teacher coordinates/quality')
        result[d+'_target'] = torch.zeros(n, 2)
        result[d+'_weight'] = torch.zeros(n)
        result[d+'_valid'] = torch.zeros(n, dtype=torch.bool)
        result[d+'_fine_valid'] = torch.zeros(n, dtype=torch.bool)
        result[d+'_target'][ids] = target
        result[d+'_weight'][ids] = conf
        result[d+'_valid'][ids] = True
        result[d+'_fine_valid'][ids] = (conf>=fine_confidence)&(cycle<=fine_cycle_px)
    return result


class TeacherGeometryData(Dataset):
    def __init__(self, root, config, size, threshold=.8, cycle_px=2., min_points=32,
                 fine_confidence=.95, fine_cycle_px=.5):
        self.config, self.size = config, size
        # Extract ALL trusted coarse rows, not just the student's <=512 queries.
        label_config = replace(config, max_source_queries=(size//config.coarse_stride)**2)
        self.base = UDISCacheDataset(root, label_config, size, threshold, cycle_px, min_points)
        self.source_hashes = self.base.source_hashes
        self.teacher_contracts = self.base.teacher_contracts
        self.signature = self.base.signature
        self.audit = self.base.audit
        self.fine_confidence, self.fine_cycle_px = fine_confidence, fine_cycle_px
        self.records = [dict(pair_id=str(path), scene='udis_training') for path, _ in self.base.records]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return teacher_grid(self.base[index], self.config, self.size, self.fine_confidence, self.fine_cycle_px)


class FullCoverageEpoch(Dataset):
    """R1 visits every MD pair once; R2 visits every UAV pair once per epoch."""
    def __init__(self, sources, phase, seed, epoch, uav_fraction=.6, offset=0):
        if phase not in ('R1', 'R2') or not 0<uav_fraction<1:
            raise ValueError('Invalid real phase/mixture')
        self.sources = sources
        g = torch.Generator().manual_seed(seed+epoch*10000019)
        primary = 'megadepth' if phase=='R1' else 'uav'
        plan = [(primary, int(i)) for i in torch.randperm(len(sources[primary]), generator=g)]
        if phase=='R2':
            count = max(1, round(len(plan)*(1-uav_fraction)/uav_fraction))
            picked = []
            while len(picked)<count:
                picked.extend(torch.randperm(len(sources['udis']), generator=g).tolist())
            plan.extend(('udis', i) for i in picked[:count])
        order = torch.randperm(len(plan), generator=g).tolist()
        self.plan = [plan[i] for i in order]
        if not 0<=offset<=len(self.plan):
            raise ValueError('Invalid saved epoch cursor')
        self.offset = offset
        self.counts = dict(Counter(d for d, _ in self.plan))
        self.unique_counts = {d: len({i for domain, i in self.plan if domain==d}) for d in self.counts}

    def __len__(self):
        return len(self.plan)-self.offset

    def __getitem__(self, index):
        domain, row = self.plan[index+self.offset]
        return self.sources[domain][row]


class RealCurriculumObjective(CurriculumObjective):
    def __init__(self, config, *, local_freeze_updates=0, geometry_fine_weight=.25,
                 teacher_fine_weight=.1, fine_only=False, **kwargs):
        super().__init__(config, 'C', **kwargs)
        self.local_freeze_updates = local_freeze_updates
        self.geometry_fine_weight, self.teacher_fine_weight = geometry_fine_weight, teacher_fine_weight
        self.fine_only = fine_only
        self.domain = None

    def set_model_mode(self, model, train):
        model.train(train)
        for name, p in model.named_parameters():
            group = parameter_group(name)
            if self.fine_only:
                p.requires_grad_(group == 'fine')
            else:
                p.requires_grad_(not (train and self.update<self.local_freeze_updates
                                       and group in ('mid', 'fine', 'bridge', 'confidence')))
        if train:
            for module in model.modules():
                params = list(module.parameters())
                if params and not any(p.requires_grad for p in params):
                    module.eval()

    def fine_weight(self, real):
        if not real:
            return 1.
        return self.teacher_fine_weight if self.domain=='udis' else self.geometry_fine_weight

    def fine_reference_mask(self, batch, direction, weight, valid, real):
        if not real:
            return valid
        return valid & batch[direction+'_fine_valid'].bool()

    def agri(self, model, batch, train, epoch=0):
        self.domain = batch['domain'][0]
        return super().agri(model, batch, train, epoch)


def audit_geometry(root, domain, config, size):
    """Audit full manifests, including rows outside a limited validation subset."""
    root = Path(root)
    meta = json.loads((root/'preparation.json').read_text(encoding='utf-8-sig'))
    if meta.get('size') != size or meta.get('coarse_stride',16) != config.coarse_stride:
        raise ValueError('Geometry cache size/stride differs from model; rebuild from depth')
    subsets, hashes = {}, {}
    for split in ('train', 'val'):
        rows = [json.loads(s) for s in (root/split/'manifest.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        rows = [r for r in rows if r['domain']==domain]
        if not rows or len({r['pair_id'] for r in rows})!=len(rows):
            raise ValueError(f'Empty/duplicate {domain}/{split}')
        for r in rows:
            if r['split']!=split or meta['group_split'].get(r['group'])!=split:
                raise ValueError('Manifest/preparation split disagreement')
            if domain=='megadepth' and str(r['scene']).zfill(4) in ('0015','0022'):
                raise ValueError('MegaDepth1500 test scene in training/validation cache')
        subsets[split] = rows
        hashes[split] = {r[k] for r in rows for k in ('source_sha256_a','source_sha256_b')}
    if hashes['train'] & hashes['val']:
        raise ValueError('Geometry train/val shares source image bytes')
    for key in ('pair_id', 'group'):
        if {r[key] for r in subsets['train']} & {r[key] for r in subsets['val']}:
            raise ValueError('Geometry train/val shares '+key)
    return dict(domain=domain, counts={k:len(v) for k,v in subsets.items()},
                manifest_sha256={s:hashlib.sha256((root/s/'manifest.jsonl').read_bytes()).hexdigest() for s in subsets},
                geometry_correctness_independently_verified=False)
