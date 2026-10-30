"""Frozen-architecture C -> D coarse recovery -> E joint adaptation."""
from collections import Counter, defaultdict
import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .architecture_v3 import target_support
from .coordinates import flat_indices_to_image_xy, image_xy_to_normalized_grid
from .depth_curriculum import CurriculumObjective, parameter_group
from .stage25 import transform_points


PHASES = {
    'D': dict(steps=2000, warmup=200, lr=dict(backbone=1e-6, coarse=1e-5)),
    'E': dict(steps=1000, warmup=100, lr=dict(backbone=5e-7, coarse=3e-6, mid=3e-6,
                                          fine=3e-6, bridge=3e-6, confidence=1e-5)),
}
QUOTA = dict(megadepth=4, uav=4, synthetic=2)
STRATA = ('low', 'medium', 'high')


def visibility_bucket(row):
    """A->B trusted support/RGB coarse grid, matching the scoring diagnostic."""
    if row.get('domain') == 'synthetic':
        return 'synthetic'
    overlap = row.get('overlap')
    if not isinstance(overlap, (tuple, list)) or len(overlap) != 2:
        raise ValueError(f'Missing bidirectional geometry fractions: {row.get("pair_id")}')
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in overlap):
        raise ValueError(f'Invalid geometry fractions: {row.get("pair_id")}')
    x = float(overlap[0])
    return 'low' if x < .3 else 'medium' if x < .6 else 'high'


def inventory(sources):
    result = {}
    for domain, ds in sources.items():
        counts = Counter(visibility_bucket(dict(r, domain=domain)) for r in ds.records)
        by_scene = defaultdict(Counter)
        for row in ds.records:
            by_scene[row.get('scene', row.get('source_image'))][visibility_bucket(dict(row, domain=domain))] += 1
        missing = [b for b in STRATA if not counts[b]] if domain != 'synthetic' else []
        result[domain] = dict(pairs=len(ds), by_visibility=dict(counts), missing_strata=missing,
            by_scene={k: dict(v) for k, v in sorted(by_scene.items())},
            target='equal available visibility strata, equal scenes within each stratum' if domain != 'synthetic'
                   else 'equal source images',
            warning='Missing strata cannot be created by resampling; rebuild training pairs if needed' if missing else None)
    return result


class StratifiedMixture(Dataset):
    """Exact 4/4/2 quota; cycling strata, uniform scenes, uniform pairs.

    Plans depend only on seed, phase and update; worker prefetch cannot affect resume.
    Missing strata are listed in inventory; available strata receive equal frequency.
    """
    def __init__(self, sources, phase, seed, start_update, updates):
        self.sources = sources
        pools = {}
        for domain, ds in sources.items():
            bins = defaultdict(lambda: defaultdict(list))
            for index, row in enumerate(ds.records):
                bucket = visibility_bucket(dict(row, domain=domain))
                scene = row['scene'] if domain != 'synthetic' else row['source_image']
                bins[bucket][scene].append(index)
            pools[domain] = {b: [scenes[k] for k in sorted(scenes)] for b, scenes in sorted(bins.items())}
            if not pools[domain]:
                raise ValueError(f'Empty training domain: {domain}')
        self.plan = []
        for update in range(start_update, start_update+updates):
            rng = torch.Generator().manual_seed(seed+(ord(phase)-64)*1_000_000_000+update)
            block = []
            for domain, count in QUOTA.items():
                available = [b for b in STRATA if b in pools[domain]] if domain != 'synthetic' else ['synthetic']
                for slot in range(count):
                    bucket = available[(update*count+slot) % len(available)]
                    scenes = pools[domain][bucket]
                    ids = scenes[torch.randint(len(scenes), (), generator=rng).item()]
                    index = ids[torch.randint(len(ids), (), generator=rng).item()]
                    block.append((domain, index, bucket))
            self.plan.extend(block[k] for k in torch.randperm(10, generator=rng).tolist())

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        domain, i, bucket = self.plan[index]
        return dict(self.sources[domain][i], visibility_bucket=bucket)


def make_optimizer(model, phase, weight_decay):
    groups = defaultdict(list)
    for name, param in model.named_parameters():
        groups[parameter_group(name)].append(param)
    return torch.optim.AdamW([dict(name=g, params=groups[g], lr=lr, peak_lr=lr)
                              for g, lr in PHASES[phase]['lr'].items()], weight_decay=weight_decay)


def gradient_audit(model, phase):
    report = {g: dict(parameters=0, gradient_parameters=0, squared_norm=0.) for g in
              ('backbone', 'coarse', 'mid', 'fine', 'bridge', 'confidence')}
    for name, p in model.named_parameters():
        group = parameter_group(name); row = report[group]; row['parameters'] += p.numel()
        if p.grad is not None:
            if group not in PHASES[phase]['lr'] or not torch.isfinite(p.grad).all():
                raise FloatingPointError(f'Unexpected/nonfinite gradient: {name}')
            row['gradient_parameters'] += p.numel()
            row['squared_norm'] += float(p.grad.float().square().sum())
    if any(report[g]['squared_norm'] == 0 for g in PHASES[phase]['lr']):
        raise RuntimeError('An active parameter group received no gradient; inspect supervision')
    return report


@torch.no_grad()
def scoring_counts(corr, ma, mb, truth, known, stride, width, config, query=None):
    """Counts on the same source rows before all prediction-based filtering."""
    n = corr.similarity.shape[1]
    if truth.shape != (1, n, 2):
        raise ValueError('Expected fixed coarse-reference grid for scoring metrics')
    known = known & ma & torch.isfinite(truth).all(-1)
    cells = torch.round((truth+.5)/stride-.5).long()
    target_cell = cells[..., 1].clamp(0, width-1)*width+cells[..., 0].clamp(0, width-1)
    result = dict(scoring_reference_count=float(known.sum()))
    best, target, support = target_support(corr, ma, mb, config)
    selected = torch.zeros_like(known)
    if query is not None:
        selected.scatter_(1, query, True)
    result.update(support_reference_count=float((known & support).sum()),
        selected_reference_count=float((known & selected).sum()))
    for name, score in [('raw', corr.similarity), ('dual', corr.dual_probability),
                        ('weighted', corr.probability)]:
        score = score.masked_fill(~mb[:, None], -torch.inf)
        idx = score.argmax(-1)
        if name == 'weighted':
            assert torch.equal(idx[ma], target[ma])
        top4 = score.topk(min(4, n), -1).indices
        xy = flat_indices_to_image_xy(idx, width, stride)
        error = (xy-truth).abs().amax(-1)
        result[name+'_cell1_count'] = float(((idx == target_cell) & known).sum())
        result[name+'_cell4_count'] = float(((top4 == target_cell[..., None]).any(-1) & known).sum())
        for budget in (32, 35):
            reachable = error <= budget
            result[f'{name}_reachable{budget}_count'] = float((known & reachable).sum())
            if name == 'weighted':
                result[f'supported_reachable{budget}_count'] = float((known & reachable & support).sum())
                result[f'rejected_reachable{budget}_count'] = float((known & reachable & ~support).sum())
        if name == 'weighted':
            sp = corr.matchability_a > config.v3_min_matchability
            tp = corr.matchability_b.gather(1, target) > config.v3_min_matchability
            hard = ma & mb.gather(1, target) & sp & tp
            result.update(source_matchability_reject_count=float((known & ~sp).sum()),
                target_matchability_reject_count=float((known & ~tp).sum()),
                relative_reject_count=float((known & hard & ~support & (best > 0)).sum()))
    return result


def scoring_rates(counts):
    n = counts.get('scoring_reference_count', 0.)
    rates = dict(reference_cells=n)
    for name in ('raw', 'dual', 'weighted'):
        for metric in ('cell1', 'cell4', 'reachable32', 'reachable35'):
            rates[name+'_'+metric] = counts.get(name+'_'+metric+'_count', 0.)/max(n, 1.)
    for metric in ('support_reference', 'selected_reference', 'supported_reachable32',
                   'rejected_reachable32', 'relative_reject', 'source_matchability_reject', 'target_matchability_reject'):
        rates[metric+'_fraction'] = counts.get(metric+'_count', 0.)/max(n, 1.)
    return rates


class RecoveryObjective(CurriculumObjective):
    """Reuse established losses; capture correlation only to add diagnostics.

    D has exactly A's active groups/coarse-only path, at NEW learning rates.
    E has C's active groups and real Fine quality mask, at NEW learning rates.
    No shared phase definitions or original training/model source are mutated.
    """
    def __init__(self, config, phase, **kwargs):
        self.recovery_phase = phase
        super().__init__(config, 'A' if phase == 'D' else 'C', **kwargs)

    def agri(self, model, batch, train, epoch=0):
        captured = {}
        def correlation_hook(module, args, output):
            captured['corr'] = output
            captured['ma'], captured['mb'] = args[2], args[3]
        def output_hook(module, args, output):
            if isinstance(output, tuple): captured['query'] = output[1].query_indices
        handles = [model.correlation.register_forward_hook(correlation_hook), model.register_forward_hook(output_hook)]
        try:
            loss, metrics = super().agri(model, batch, train, epoch)
            if loss is None: return loss, metrics
            with torch.no_grad():
                size = batch['image_a'].shape[-1]; w = size//self.config.coarse_stride
                grid = flat_indices_to_image_xy(torch.arange(w*w, device=batch['image_a'].device)[None],w,self.config.coarse_stride)
                if batch['domain'][0] != 'synthetic':
                    truth, known = batch['ab_target'].float(), batch['ab_valid'].bool()
                else:
                    sample = image_xy_to_normalized_grid(grid, size, size).unsqueeze(2)
                    field = batch.get('gt_dense_mapping')
                    truth = (F.grid_sample(field.float(),sample,align_corners=False)[:,:,:,0].transpose(1,2)
                             if field is not None else transform_points(grid,batch['H_ab']))
                    known = F.grid_sample(batch['gt_overlap_a'].float(),sample,align_corners=False)[:,0,:,0]>.999
                known = known & torch.isfinite(truth).all(-1) & (truth>=0).all(-1) & (truth<=size-1).all(-1)
                metrics.update(scoring_counts(captured['corr'],captured['ma'],captured['mb'],truth,known,
                    self.config.coarse_stride,w,self.config,captured.get('query')))
            return loss, metrics
        finally:
            for handle in handles: handle.remove()
            captured.clear()
