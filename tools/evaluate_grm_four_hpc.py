"""Evaluate a GRM checkpoint with the shared JamMa/GRM 512 px protocol."""

import argparse
import json
import platform
from pathlib import Path
from types import SimpleNamespace

from tools.evaluate_jamma_four import benchmark, load_inventory


ROOT = Path(__file__).resolve().parents[1]


def check_reference(reference, benchmark_inputs):
    pairs, metadata, _, _ = benchmark_inputs
    for dataset, rows in pairs.items():
        path = reference / dataset / 'settings.json'
        settings = json.loads(path.read_bytes())
        if settings['pair_ids'] != [row['pair_id'] for row in rows]:
            raise ValueError(f'{dataset}: ordered pair list differs from {path}')
        for key, value in metadata[dataset].items():
            if key.endswith('_sha256') and settings['data'].get(key) != value:
                raise ValueError(f'{dataset}: {key} differs from {path}')
        print(f'{dataset}: matches JamMa pair list and dataset hashes', flush=True)


class GRMMatcher:
    def __init__(self, checkpoint):
        import torch
        from agri_eval_adapter import Matcher

        torch.set_num_threads(1)
        benchmark.cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.native = Matcher('GRM', ROOT, SimpleNamespace(checkpoint=checkpoint, device='cuda:0', size=512))
        self.torch = torch
        self.device = self.native.device
        self.metadata = dict(
            self.native.metadata,
            checkpoint_sha256=benchmark.ag.digest(checkpoint),
            parameters=sum(p.numel() for p in self.native.model.parameters()),
            cuda=torch.version.cuda,
            python_version=platform.python_version(),
            opencv=benchmark.cv2.__version__,
            numpy=benchmark.np.__version__,
            warmup_calls_per_dataset=3,
            tf32=False,
        )

    def sync(self):
        self.native.sync()

    def __call__(self, a, b, valid_a, valid_b):
        return self.native(a, b, valid_a=valid_a, valid_b=valid_b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=benchmark.SETS, default=benchmark.SETS)
    for name in ('real', 'synthetic', 'megadepth', 'uav'):
        parser.add_argument(f'--{name}-root', type=Path)
    parser.add_argument('--uav-prepared', type=Path, default=ROOT / 'benchmarks_uav2025_20261023/prepared')
    parser.add_argument('--synthetic-count', type=int, choices=(1605, 2420), default=1605)
    parser.add_argument('--jamma-reference', type=Path, help='JamMa directory containing dataset settings.json files')
    parser.add_argument('--limit', type=int, default=0, help='Pairs per dataset. Use 0 for all pairs.')
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or len(set(args.datasets)) != len(args.datasets):
        parser.error('Use a nonnegative --limit and list each dataset at most once')

    inputs = load_inventory(args, model='GRM', synthetic_count=args.synthetic_count)
    if args.jamma_reference is not None:
        check_reference(args.jamma_reference, inputs)
    if args.inventory_only:
        return

    checkpoint = args.checkpoint.resolve()
    matcher = GRMMatcher(checkpoint)
    benchmark.evaluate('GRM', args.output.resolve(), args.limit, benchmark=inputs,
                       matcher=matcher, datasets=args.datasets, checkpoint=checkpoint, visual_matches=200)


if __name__ == '__main__':
    main()
