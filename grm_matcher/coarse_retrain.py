"""UAV-focused coarse-stage retraining; architecture and established losses stay fixed."""
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from .coarse_recovery import inventory, visibility_bucket
from .depth_curriculum import parameter_group


RECIPE = 'v3_coarse_retrain_r2'
QUOTA = dict(megadepth=2, uav=6, synthetic=2)
# Support is a geometry proxy, not a measured physical overlap or texture difficulty.
SUPPORT_CYCLE = ('low',)*5 + ('medium',)*3 + ('high',)*2
ACTIVE_GROUPS = ('backbone', 'coarse')


class RetrainMixture(Dataset):
    """Exact domain quota; 5:3:2 support cycle; uniform scenes then pairs.

    Plans depend on the absolute update, so worker prefetch and resuming do not
    change the sampling sequence. No model prediction or test result is used.
    """
    def __init__(self, sources, seed, start_update, updates):
        self.sources = sources
        pools = {}
        for domain, ds in sources.items():
            bins = defaultdict(lambda: defaultdict(list))
            for i, row in enumerate(ds.records):
                bucket = visibility_bucket(dict(row, domain=domain))
                scene = row['source_image'] if domain == 'synthetic' else row['scene']
                bins[bucket][scene].append(i)
            pools[domain] = {b: [ss[k] for k in sorted(ss)] for b, ss in bins.items()}
            needed = {'synthetic'} if domain == 'synthetic' else set(SUPPORT_CYCLE)
            if needed - pools[domain].keys():
                raise ValueError(f'{domain}: missing support bins {sorted(needed-pools[domain].keys())}')
        self.plan = []
        for update in range(start_update, start_update+updates):
            rng = torch.Generator().manual_seed(seed+7_000_000_000+update)
            block = []
            for domain, count in QUOTA.items():
                for slot in range(count):
                    bucket = ('synthetic' if domain == 'synthetic' else
                              SUPPORT_CYCLE[(update*count+slot) % len(SUPPORT_CYCLE)])
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


def make_optimizer(model, args):
    rates = dict(backbone=args.backbone_lr, coarse=args.coarse_lr)
    groups = defaultdict(list)
    for name, param in model.named_parameters():
        group = parameter_group(name)
        param.requires_grad_(group in ACTIVE_GROUPS)
        if param.requires_grad:
            groups[group].append(param)
    return torch.optim.AdamW([dict(name=g, params=groups[g], lr=rates[g], peak_lr=rates[g])
                              for g in ACTIVE_GROUPS], weight_decay=args.weight_decay)


def selection_metrics(validation):
    """Equal-scene metrics inside the UAV low-support bin; fixed reference sources."""
    m = validation['uav']['by_visibility']['low']['macro']
    return dict(score=.5*(m['raw_cell1']+m['raw_reachable32']),
                uav_low_raw_cell1=m['raw_cell1'], uav_low_raw32=m['raw_reachable32'],
                uav_low_weighted32=m['weighted_reachable32'],
                uav_low_reference_recall3=m['reference_recall3'])


def guard_failures(validation, baseline, args):
    failures = []
    # A higher scalar score must not conceal damage to either target component,
    # the deployed weighted ranking, or the other domain/support bins.
    for domain in ('megadepth', 'uav'):
        for bucket in ('low', 'medium', 'high'):
            new = validation[domain]['by_visibility'][bucket]['macro']
            old = baseline[domain]['by_visibility'][bucket]['macro']
            gap = args.target_guard if (domain, bucket) == ('uav', 'low') else args.other_guard
            for key in ('raw_cell1', 'raw_reachable32', 'weighted_reachable32'):
                if new[key] < old[key]-gap:
                    failures.append(f'{domain}/{bucket}/{key}')
    # Frozen local heads may need later adaptation. Track final metrics but do
    # not require unchanged final PCK while their input representation is trained.
    for key in ('raw_cell1', 'raw_reachable32', 'weighted_reachable32'):
        if validation['synthetic']['macro'][key] < baseline['synthetic']['macro'][key]-args.other_guard:
            failures.append('synthetic/'+key)
    return failures


def parameter_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def relative_updates(model, before):
    totals = defaultdict(lambda: [0., 0.])
    for n, p in model.named_parameters():
        if n in before:
            row = totals[parameter_group(n)]
            row[0] += float((p.detach().float()-before[n].float()).square().sum())
            row[1] += float(before[n].float().square().sum())
    return {g: (a/max(b, 1e-30))**.5 for g, (a, b) in totals.items()}


def require_strata(sources, validation):
    from .train_coarse_recovery import SubsetRecords
    inventories = dict(train=inventory(sources),
                       val=inventory({d: SubsetRecords(ds) for d, ds in validation.items()}))
    for split, inv in inventories.items():
        for domain, row in inv.items():
            if row['missing_strata']:
                raise ValueError(f'{split}/{domain}: missing support bins {row["missing_strata"]}')
    return inventories
