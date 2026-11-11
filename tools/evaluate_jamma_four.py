"""Evaluate official JamMa weights with the shared 512 px benchmark protocol."""

import argparse
import json
import platform
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

from tools import run_four_baselines_20261104 as benchmark


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = dict(real=200, synthetic=2420, megadepth1500=1500, uav2025=111)


def load_inventory(args, *, model='JamMa', synthetic_count=2420):
    data, metadata, frames = {}, {}, {}
    for dataset in args.datasets:
        if dataset in ('real', 'synthetic'):
            root = getattr(args, f'{dataset}_root')
            if root is None:
                raise ValueError(f'--{dataset}-root is required for {dataset}')
            settings = benchmark.ag.parser(model, dataset).parse_args(['--data-root', str(root)])
            data[dataset], metadata[dataset] = benchmark.ag.load_pairs(settings, dataset)
            metadata[dataset]['root'] = str(root.resolve())
            if dataset == 'real' and (metadata[dataset]['labeled_pairs'],
                                      sum(len(pair['points_a']) for pair in data[dataset])) != (185, 802):
                raise ValueError('Expected 185 labeled real pairs with 802 landmarks')
        elif dataset == 'megadepth1500':
            if args.megadepth_root is None:
                raise ValueError('--megadepth-root is required for MegaDepth1500')
            root = args.megadepth_root.resolve()
            data[dataset], metadata[dataset] = benchmark.md.inventory(root)
            metadata[dataset]['root'] = str(root)
        else:
            if args.uav_root is None:
                raise ValueError('--uav-root is required for UAV3DCrop')
            root = args.uav_root.resolve()
            prepared = args.uav_prepared
            pairs_path = prepared / 'pairs.json'
            prep_path = prepared / 'preparation.json'
            data[dataset] = json.loads(pairs_path.read_bytes())
            preparation = json.loads(prep_path.read_bytes())
            source = preparation['source_identity']
            if benchmark.ag.digest(root / 'frames.json') != source['frames_manifest_sha256']:
                raise ValueError('UAV frames.json does not match the prepared pair list')
            for scene, expected_hash in source['camera_subset_sha256'].items():
                camera = root / 'rgb' / '2025' / scene / 'transforms_subset.json'
                if benchmark.ag.digest(camera) != expected_hash:
                    raise ValueError(f'UAV camera metadata differs from the prepared subset: {scene}')
            frames = benchmark.uv.load_frames(root)
            for pair in data[dataset]:
                for key in ('frame_a', 'frame_b'):
                    frame = frames[pair[key]]
                    for path in (frame.image, frame.depth):
                        if not path.is_file():
                            raise FileNotFoundError(path)
                geometry = prepared / 'geometry' / f"{pair['pair_id']}.npz"
                if not geometry.is_file():
                    raise FileNotFoundError(geometry)
            metadata[dataset] = dict(root=str(root), prepared=str(prepared.resolve()),
                                     preparation=preparation,
                                     pairs_sha256=benchmark.ag.digest(pairs_path))
        expected = synthetic_count if dataset == 'synthetic' else EXPECTED[dataset]
        if len(data[dataset]) != expected:
            raise ValueError(f'{dataset}: expected {expected} pairs, got {len(data[dataset])}')
        print(f'{dataset}: {len(data[dataset])} pairs', flush=True)
    return data, metadata, frames, args.uav_prepared


class JamMaMatcher:
    def __init__(self, repo, checkpoint):
        import torch
        import torch.nn.functional as F

        if not torch.cuda.is_available():
            raise RuntimeError('JamMa requires an allocated CUDA GPU')
        self.torch = torch
        self.device = torch.device('cuda:0')
        torch.cuda.set_device(self.device)
        torch.set_num_threads(1)
        benchmark.cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        sys.path.insert(0, str(repo))
        import mamba_ssm
        from src.jamma import matching_module
        from src.jamma.backbone import CovNextV2_nano
        from src.jamma.jamma import JamMa

        config = runpy.run_path(str(repo / 'demo' / 'utlis.py'))['cfg']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
        cnn_state = {key[len('backbone.cnn.'):]: value for key, value in state.items()
                     if key.startswith('backbone.cnn.')}
        # The official constructor fetches CNN weights even when the full checkpoint is local.
        # Supply the identical parameter namespace from the full checkpoint during construction.
        with patch.object(torch.hub, 'load_state_dict_from_url', return_value=cnn_state):
            backbone = CovNextV2_nano()
        self.model = torch.nn.ModuleDict({'backbone': backbone, 'matcher': JamMa(config)})
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).float().eval()

        def centered_mask_border(matches, border, value, mask_a, mask_b):
            if border <= 0:
                return

            def excluded(mask):
                invalid = F.pad((~mask.bool())[:, None].float(), (border,) * 4, value=1.)
                return F.max_pool2d(invalid, 2 * border + 1, stride=1)[:, 0].bool()

            matches.masked_fill_(excluded(mask_a)[:, :, :, None, None], value)
            matches.masked_fill_(excluded(mask_b)[:, None, None, :, :], value)

        # JamMa assumes valid masks are anchored at the top left. The shared protocol
        # centers them, so remove the same border width around the actual valid region.
        matching_module.mask_border_with_padding = centered_mask_border
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        self.metadata = dict(
            model='JamMa', source='https://github.com/leoluxxx/JamMa', repo=str(repo),
            checkpoint=str(checkpoint), checkpoint_sha256=benchmark.ag.digest(checkpoint),
            parameters=sum(p.numel() for p in self.model.parameters()), config=config,
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            mamba_ssm=getattr(mamba_ssm, '__version__', 'unknown'), python_version=platform.python_version(),
            opencv=benchmark.cv2.__version__, numpy=benchmark.np.__version__,
            precision='FP32', tf32=False, batch_size=1, size=512, warmup_calls_per_dataset=3,
            external_cap=0, external_confidence_threshold=None,
            mask_adapter='stride-8 valid masks with a two-cell border around the actual valid region',
        )

    def sync(self):
        self.torch.cuda.synchronize(self.device)

    def __call__(self, image_a, image_b, mask_a, mask_b):
        torch = self.torch
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            batch = {}
            for index, (image, mask) in enumerate(((image_a, mask_a), (image_b, mask_b))):
                rgb = torch.from_numpy(image).to(self.device, dtype=torch.float32)
                rgb = (rgb.permute(2, 0, 1)[None] / 255. - self.mean) / self.std
                valid = torch.from_numpy(mask.copy()).to(self.device, dtype=torch.bool)
                batch[f'imagec_{index}'] = rgb.masked_fill(~valid[None, None], 0.)
                batch[f'mask{index}'] = valid[None, ::8, ::8].contiguous()
            # Keep coordinates in the 512 px input canvas. The shared scorer maps them back.
            self.model['backbone'](batch)
            self.model['matcher'](batch, mode='test')
            return tuple(batch[key].detach().float().cpu().numpy()
                         for key in ('mkpts0_f', 'mkpts1_f', 'mconf_f'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jamma-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Defaults to JAMMA_ROOT/jamma.ckpt')
    parser.add_argument('--datasets', nargs='+', choices=benchmark.SETS, default=benchmark.SETS)
    for dataset in ('real', 'synthetic', 'megadepth', 'uav'):
        parser.add_argument(f'--{dataset}-root', type=Path)
    parser.add_argument('--uav-prepared', type=Path, default=ROOT / 'benchmarks_uav2025_20261023' / 'prepared')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Max pairs per dataset, 0 means all')
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or len(set(args.datasets)) != len(args.datasets):
        parser.error('Use a nonnegative --limit and list each dataset at most once')
    repo = args.jamma_root.resolve()
    checkpoint = (args.checkpoint or repo / 'jamma.ckpt').resolve()
    for path in (checkpoint, repo / 'demo' / 'utlis.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    inputs = load_inventory(args)
    if args.inventory_only:
        return
    if {'real', 'synthetic'}.intersection(args.datasets) and not hasattr(benchmark.cv2, 'USAC_MAGSAC'):
        raise RuntimeError('Use an OpenCV build exposing cv2.USAC_MAGSAC for homography scoring')
    matcher = JamMaMatcher(repo, checkpoint)
    benchmark.evaluate('JamMa', args.output.resolve(), args.limit, benchmark=inputs,
                       matcher=matcher, datasets=args.datasets, checkpoint=checkpoint, visual_matches=200)
    for dataset in args.datasets:
        print(args.output.resolve() / 'JamMa' / dataset / 'summary.json')


if __name__ == '__main__':
    main()
