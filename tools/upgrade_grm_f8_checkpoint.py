"""Initialize an F8/F8-HRA experiment from a native C checkpoint, without optimizer state."""
import argparse
import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grm_matcher.config import GRMMatcherConfig
from grm_matcher.matcher import GRMMatcher


def upgrade(raw, mode, seed=2026):
    if mode not in ('f8', 'f8_hra'):
        raise ValueError('Select f8 or f8_hra')
    original = GRMMatcherConfig.from_dict(raw['config'])
    if original.architecture_version != 'v3' or original.v3_revision != 1 or original.v3_coarse_mode != 'native16':
        raise ValueError('Initialization requires a native stride16 V3 revision1 checkpoint')
    if getattr(original,'adaptive_enabled',False):
        raise ValueError('F8/F8-HRA initialization requires the plain C checkpoint with Adaptive disabled; adaptive head weights are not silently discarded')
    values = original.to_dict() | dict(coarse_stride=8, v3_coarse_mode=mode, use_amp=False,
        v3_spatial_sigma_cells=original.v3_spatial_sigma_cells*2,adaptive_enabled=False,adaptive_variance=False)
    config = GRMMatcherConfig.from_dict(values)
    # Isolate initialization RNG from a caller's training/augmentation state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GRMMatcher(config)
    state = {}
    for name, tensor in raw['model'].items():
        mapped = name
        if mode == 'f8_hra' and name.startswith('coarse_transformer.'):
            mapped = 'coarse_transformer.context.' + name[len('coarse_transformer.'):]
        state[mapped] = tensor
    incompatible = model.load_state_dict(state, strict=False)
    allowed = ('fpn.coarse_f8_', 'coarse_transformer.pool16.', 'coarse_transformer.pool32.',
               'coarse_transformer.output_norm.', 'coarse_transformer.gain')
    if incompatible.unexpected_keys or any(not name.startswith(allowed) for name in incompatible.missing_keys):
        raise ValueError(f'Unexpected checkpoint mismatch: {incompatible}')
    report = dict(source_mode=original.v3_coarse_mode, target_mode=mode, seed=seed,
                  copied_tensors=len(state), initialized_tensors=incompatible.missing_keys,
                  source_coarse_stride=16, target_coarse_stride=8,adaptive_enabled=False,
                  status='initialization_only_requires_training_and_confidence_recalibration')
    return dict(config=config.to_dict(), model=model.state_dict(), training_recipe='grm_f8_initialization_v1',
                confidence_head_trained=False, geometry_heads_trained=False, inlier_head_trained=False,
                migration=report)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('f8', 'f8_hra'), required=True)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new initialization checkpoint path')
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    converted = upgrade(raw, args.mode, args.seed)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    converted['provenance'] = dict(initial_checkpoint=str(args.checkpoint), source_sha256=digest,
                                    migration=converted['migration'], parent=raw.get('provenance'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting another initialization.
    with args.output.open('xb') as stream:
        torch.save(converted, stream)
    import json
    print(json.dumps(converted['migration'], indent=2))


if __name__ == '__main__':
    main()
