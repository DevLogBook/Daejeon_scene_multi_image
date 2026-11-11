"""R3 four-domain replay, deterministic sampling and guarded selection."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import torch
from torch.utils.data import Dataset

from .depth_curriculum import group_indices, parameter_group
from .real_curriculum import RealCurriculumObjective

QUOTA = dict(uav=40, udis=16, megadepth=16, synthetic=8)


class ReplayPlan(Dataset):
    """80 visits per block; scene-balanced shuffled buckets, never test records.

    The full plan is regenerated from seed on resume. Only the completed-visit
    cursor matters, so worker prefetch cannot advance training state.
    """
    def __init__(self, sources, seed, total_updates, accumulation, offset=0):
        if set(sources) != set(QUOTA) or total_updates <= 0 or accumulation <= 0:
            raise ValueError('Replay requires four nonempty training domains and positive lengths')
        self.sources, self.offset, self.plan = sources, offset, []
        rng = torch.Generator().manual_seed(seed)
        buckets = {d: list(group_indices(ds.records, d != 'synthetic').values()) for d, ds in sources.items()}
        if any(not b for b in buckets.values()):
            raise ValueError('Empty replay domain')
        queues, cursors = {}, {}

        def draw(domain):
            group = int(torch.randint(len(buckets[domain]), (), generator=rng))
            key = domain, group
            if key not in queues or cursors[key] == len(queues[key]):
                ids = buckets[domain][group]
                queues[key] = [ids[i] for i in torch.randperm(len(ids), generator=rng).tolist()]
                cursors[key] = 0
            index = queues[key][cursors[key]]
            cursors[key] += 1
            return domain, index

        visits = total_updates * accumulation
        while len(self.plan) < visits:
            block = [draw(d) for d, n in QUOTA.items() for _ in range(n)]
            self.plan.extend(block[i] for i in torch.randperm(len(block), generator=rng).tolist())
        self.plan = self.plan[:visits]
        if not 0 <= offset <= visits or offset % accumulation:
            raise ValueError('Resume cursor must be a completed optimizer update')
        self.signature = (seed, total_updates, accumulation)

    def __len__(self):
        return len(self.plan) - self.offset

    def __getitem__(self, index):
        domain, row = self.plan[self.offset + index]
        return self.sources[domain][row]

    def statistics(self, end=None):
        rows = self.plan if end is None else self.plan[:end]
        counts = Counter(d for d, _ in rows)
        unique, source_groups = defaultdict(set), defaultdict(set)
        for d, i in rows:
            unique[d].add(i)
            record = self.sources[d].records[i]
            source_groups[d].add(str(record['source_image'] if d == 'synthetic' else record['scene']))
        return {d: dict(visits=counts[d], unique_pairs=len(unique[d]),
                        unique_source_groups=len(source_groups[d]),
                        repeat_visits=counts[d]-len(unique[d])) for d in QUOTA}


def schedule_factor(update, total, warmup, minimum=.1):
    """One-based optimizer update, endpoints match the documented 8k plan."""
    if not 1 <= update <= total:
        raise ValueError('Update outside complete LR plan')
    if warmup and update <= warmup:
        return .1 + .9 * (update-1)/max(1, warmup-1)
    return minimum + (1-minimum)*.5*(1+math.cos(math.pi*(update-warmup)/(total-warmup)))


def set_learning_rates(optimizer, args, update):
    factor = schedule_factor(update, args.total_updates, args.warmup_updates, args.min_lr_ratio)
    for group in optimizer.param_groups:
        if group['name'] == 'coarse':
            activation = 1.
        elif update <= args.local_freeze_updates:
            activation = 0.
        elif args.local_warmup_updates:
            activation = .1 + .9*min(1., (update-args.local_freeze_updates-1)/max(1, args.local_warmup_updates-1))
        else:
            activation = 1.
        group['lr'] = group['peak_lr'] * factor * activation


class ReplayObjective(RealCurriculumObjective):
    def set_model_mode(self, model, train):
        super().set_model_mode(model, train)
        for name, p in model.named_parameters():
            if parameter_group(name) == 'backbone':
                p.requires_grad_(False)
        model.backbone.eval()

    def agri(self, model, batch, train, epoch=0):
        self.phase = 'A' if train and self.update < self.local_freeze_updates else 'C'
        return super().agri(model, batch, train, epoch)


def guarded_scores(result, baseline, args):
    checks = (('megadepth', 'coarse_recall1', args.md_guard),
              ('uav', 'pck1', args.uav_guard), ('synthetic', 'pck1', args.synthetic_guard))
    failures = [d+'/'+k for d, k, allowed in checks
                if result[d]['macro'][k] < baseline[d]['macro'][k] - allowed]
    uav = result['uav']['macro']
    return failures, dict(balanced=(uav['reference_f1_3'], uav['pck1']),
                          localization=(uav['pck1'], uav['reference_f1_3']))
