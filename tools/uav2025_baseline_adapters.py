"""Local pretrained matchers for the frozen UAV2025 512-pixel protocol."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.cache/edm_eval_dependencies'))


def centered_efficient_transformer(self, feat0, feat1, mask0=None, mask1=None, data=None):
    """Keep upstream attention/layers; translate its valid-rectangle crop/scatter."""
    if mask0 is None or mask1 is None:
        return self._uav_original_forward(feat0, feat1, mask0, mask1, data)
    if feat0.shape[0] != 1:
        raise ValueError('Centered EfficientLoFTR adapter requires batch=1')
    def bounds(mask, step):
        y, x = mask[0].nonzero(as_tuple=True)
        top, left = int(y.min()), int(x.min())
        height = (int(y.max()) - top + 1) // step * step
        width = (int(x.max()) - left + 1) // step * step
        if min(height, width) < step:
            raise ValueError('Insufficient valid rectangle for aggregated attention')
        return top, left, height, width
    t0, l0, h0, w0 = bounds(mask0, self.agg_size0)
    t1, l1, h1, w1 = bounds(mask1, self.agg_size1)
    f0, f1 = feat0[:, :, t0:t0+h0, l0:l0+w0], feat1[:, :, t1:t1+h1, l1:l1+w1]
    for layer, name in zip(self.layers, self.layer_names):
        if name == 'self':
            f0, f1 = layer(f0, f0), layer(f1, f1)
        elif name == 'cross':
            f0 = layer(f0, f1)
            f1 = layer(f1, f0)  # Preserve upstream sequential bilateral update.
        else:
            raise KeyError(name)
    out0, out1 = feat0.new_zeros(feat0.shape), feat1.new_zeros(feat1.shape)
    out0[:, :, t0:t0+h0, l0:l0+w0] = f0
    out1[:, :, t1:t1+h1, l1:l1+w1] = f1
    return out0, out1


class BaselineMatcher:
    def __init__(self, args):
        import torch
        self.torch, self.name = torch, args.model
        self.device = torch.device(args.device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('This comparison requires the shared CUDA device')
        self.metadata = dict(model=args.model, torch=torch.__version__, device=str(self.device),
            gpu_name=torch.cuda.get_device_name(), precision='FP32', input_size=args.size,
            ground_truth_used_for_estimation=False)
        if args.model == 'RoMaV2':
            # Reuse the existing trusted local-weight loader and native fast setting.
            path = ROOT.parent/'RoMaV2/agri_eval_adapter.py'
            spec = importlib.util.spec_from_file_location('_uav_roma_adapter', path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            args.dinov3_dir = ROOT/'backbones/facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa'
            args.allow_download = False
            self.native = module.Matcher('RoMaV2', ROOT.parent/'RoMaV2', args)
            self.model = self.native.model
            self.metadata.update(self.native.metadata)
            self.metadata.update(roma_samples=args.roma_samples,
                mask_handling='No native attention mask; black RGB padding; both endpoints filtered to real RGB before scoring',
                sampling='Native balanced sample; fixed torch seed per pair',
                memory_management='Release unused CUDA allocator cache before forward and between dense forward/sample; included in latency',
                coordinate_conversion='Native normalized to pixel coordinates minus 0.5')
        elif args.model == 'XFeatStar':
            sys.path.insert(0, str(ROOT.parent/'accelerated_features'))
            from modules.xfeat import XFeat
            self.model = XFeat(weights=str(args.checkpoint), top_k=args.xfeat_top_k).eval()
            self.metadata.update(method='match_xfeat_star', top_k=args.xfeat_top_k,
                multiscale=True, fine_conf=0.25,
                confidence='Official API returns coordinates only; stored score=1 is a placeholder, not calibrated confidence',
                mask_handling='No native attention mask; black RGB padding; both endpoints filtered to real RGB before scoring')
        else:
            sys.path.insert(0, str(ROOT.parent/args.model))
            from tools.evaluate_baseline_megadepth1500 import centered_border
            import src.loftr.utils.coarse_matching as coarse
            coarse.mask_border_with_padding = centered_border
            if args.model == 'EfficientLoFTR':
                from src.loftr import LoFTR, full_default_cfg, reparameter
                from src.loftr.loftr_module.transformer import LocalFeatureTransformer
                config = deepcopy(full_default_cfg)
                self.model = LoFTR(config)
                raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                self.model.load_state_dict(raw['state_dict'], strict=True)
                self.model = reparameter(self.model).float().to(self.device).eval()
                transformer = self.model.loftr_coarse
                original = transformer.forward
                transformer._uav_original_forward = original
                from types import MethodType
                transformer.forward = MethodType(centered_efficient_transformer, transformer)
                # Compare against upstream on its native top-left padded layout.
                with torch.inference_mode():
                    f0 = torch.randn(1, 256, 16, 16, device=self.device)
                    f1 = torch.randn_like(f0)
                    mask = torch.zeros(1,16,16,dtype=torch.bool,device=self.device); mask[:,:12,:] = True
                    expected = original(f0, f1, mask, mask)
                    actual = transformer(f0, f1, mask, mask)
                    assert all(torch.allclose(a,b,atol=1e-5,rtol=1e-5) for a,b in zip(expected,actual))
                self.metadata.update(reparameterized=True, centered_crop_native_equivalence=True,
                    positional_encoding='Upstream full_default_cfg NPE retained (832,832,832,832)',
                    mask_handling='Same upstream valid crop and sequential cross updates, translated to centered rectangle; native 2-cell border follows RGB bounds')
            elif args.model == 'LoFTR':
                from src.loftr import LoFTR, default_cfg
                config = deepcopy(default_cfg)
                self.model = LoFTR(config)
                raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                self.model.load_state_dict(raw.get('state_dict',raw), strict=True)
                self.model = self.model.float().to(self.device).eval()
                self.metadata.update(mask_handling='Native attention masks; native 2-cell border follows actual centered RGB bounds')
            else:
                raise ValueError(args.model)
            self.metadata.update(config=config, selection='Native pretrained thresholds, all output matches, no additional point cap')
        self.metadata['parameters'] = sum(p.numel() for p in self.model.parameters())

    def sync(self):
        self.torch.cuda.synchronize(self.device)

    def __call__(self, a, b, valid_a, valid_b):
        torch = self.torch
        with torch.inference_mode():
            if self.name == 'RoMaV2':
                from PIL import Image
                torch.cuda.empty_cache()
                predictions = self.model.match(Image.fromarray(a), Image.fromarray(b))
                torch.cuda.empty_cache()
                matches, confidence, _, _ = self.model.sample(predictions, self.native.args.roma_samples)
                pa, pb = self.model.to_pixel_coordinates(matches, *a.shape[:2], *b.shape[:2])
                return tuple(x.detach().float().cpu().numpy() for x in (pa-.5,pb-.5,confidence))
            if self.name == 'XFeatStar':
                pa, pb = self.model.match_xfeat_star(a, b)
                return pa, pb, np.ones(len(pa), dtype=np.float32)
            import torch.nn.functional as functional
            batch = {}
            for i,(rgb,valid) in enumerate(((a,valid_a),(b,valid_b))):
                batch[f'image{i}'] = torch.from_numpy(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)).to(self.device,dtype=torch.float32)[None,None]/255
                mask = torch.from_numpy(np.ascontiguousarray(valid)).to(self.device)[None,None].float()
                batch[f'mask{i}'] = functional.interpolate(mask,size=(rgb.shape[0]//8,rgb.shape[1]//8),mode='nearest')[:,0].bool()
            with torch.autocast(self.device.type,enabled=False):
                self.model(batch)
            return tuple(batch[k].detach().float().cpu().numpy() for k in ('mkpts0_f','mkpts1_f','mconf'))
