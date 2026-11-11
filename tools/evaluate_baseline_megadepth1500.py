"""EDM/LoFTR inference plugged into the unchanged GRM MegaDepth1500 geometry."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'.cache/edm_eval_dependencies'))
sys.path.insert(0, str(ROOT/'tools'))
import evaluate_grm_megadepth1500 as evaluation


def centered_border(mask, border, value, valid_a, valid_b):
    """Keep LoFTR's native border width for arbitrary translated RGB rectangles."""
    if border <= 0:
        return
    import torch.nn.functional as functional
    def interior(valid):
        outside = functional.pad((~valid.bool()).float()[:, None], [border]*4, value=1)
        return functional.max_pool2d(outside, 2*border+1, stride=1)[:, 0] == 0
    mask.masked_fill_(~interior(valid_a)[:, :, :, None, None], value)
    mask.masked_fill_(~interior(valid_b)[:, None, None, :, :], value)


def rectangle_attention(attention, query, key, value, q_mask=None, kv_mask=None):
    """Same EDM cosine attention, with crop/scatter at actual rectangle bounds."""
    def bounds(mask, tensor):
        if mask is None:
            return slice(0, tensor.shape[1]), slice(0, tensor.shape[2])
        y, x = mask[0].nonzero(as_tuple=True)
        return slice(int(y.min()), int(y.max())+1), slice(int(x.min()), int(x.max())+1)
    qa, qb = bounds(q_mask, query)
    ka, kb = bounds(kv_mask, key)
    batch, height, width, channels = query.shape
    cropped = [query[:, qa, qb], key[:, ka, kb], value[:, ka, kb]]
    q, k, v = [x.reshape(batch, -1, attention.nhead, channels//attention.nhead) for x in cropped]
    result = attention.attention(q, k, v, q_mask=None, kv_mask=None)
    if q_mask is None:
        return result
    output = query.new_zeros((batch, height, width, attention.nhead, channels//attention.nhead))
    output[:, qa, qb] = result.reshape(batch, qa.stop-qa.start, qb.stop-qb.start, attention.nhead, channels//attention.nhead)
    return output


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', required=True, choices=['EDM', 'LoFTR'])
    options, remaining = parser.parse_known_args()
    name = options.model
    upstream = ROOT.parent/name
    sys.path.insert(0, str(upstream))

    class BaselineMatcher:
        def __init__(self, ignored_name, ignored_root, args):
            import torch
            import torch.nn.functional as functional
            self.torch, self.functional = torch, functional
            self.device = torch.device(args.device)
            if self.device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            raw = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            state = {k.removeprefix('module.'):v for k,v in raw.get('state_dict', raw).items()}
            if name == 'EDM':
                from src.edm import EDM
                from src.config.default import get_cfg_defaults
                from src.utils.misc import lower_config
                from src.edm.neck.loftr_module import transformer
                # Preserve upstream cosine-attention math and positional features.
                # Only crop/scatter offsets differ for centered valid rectangles.
                probe = transformer.Attention(nhead=2, dim=4)
                q, k, v = [torch.randn(1, 8, 8, 8) for _ in range(3)]
                qm = torch.zeros(1, 8, 8, dtype=torch.bool)
                km = qm.clone()
                qm[:, :6, :] = True
                km[:, :, :4] = True
                expected = probe._forward(q, k, v, qm, km)
                actual = rectangle_attention(probe, q, k, v, qm, km)
                assert torch.allclose(expected, actual, atol=1e-6), 'Native EDM attention equivalence failed'
                transformer.Attention._forward = rectangle_attention
                cfg = get_cfg_defaults()
                cfg.merge_from_file(str(upstream/'configs/edm/outdoor/edm_base.py'))
                cfg.EDM.TEST_RES_H = cfg.EDM.TEST_RES_W = args.size
                cfg.EDM.NECK.NPE = [cfg.EDM.TRAIN_RES_H, cfg.EDM.TRAIN_RES_W, args.size, args.size]
                cfg.EDM.COARSE.TOPK = min(cfg.EDM.COARSE.TOPK, (args.size//8)**2)
                cfg.EDM.MP = cfg.EDM.HALF = False
                config = lower_config(cfg)['edm']
                self.model = EDM(config)
                border_note = 'native EDM border=0; same cosine attention; crop/scatter actual valid rectangle instead of top-left-only crop (native equivalence checked)'
            else:
                from src.loftr import LoFTR, default_cfg
                import src.loftr.utils.coarse_matching as coarse
                config = deepcopy(default_cfg)
                # Native helper assumes all RGB begins at (0,0). The shared
                # evaluator centers RGB, so apply the same margin to its real bounds.
                old = torch.ones((1, 8, 8, 8, 8), dtype=torch.bool)
                valid_a = torch.zeros((1, 8, 8), dtype=torch.bool)
                valid_b = valid_a.clone()
                valid_a[:, :6, :7] = True
                valid_b[:, :7, :5] = True
                equivalent = old.clone()
                coarse.mask_border_with_padding(old, 2, False, valid_a, valid_b)
                centered_border(equivalent, 2, False, valid_a, valid_b)
                assert torch.equal(old, equivalent), 'Native top-left border equivalence failed'
                shifted_a = torch.zeros_like(valid_a)
                shifted_b = torch.zeros_like(valid_b)
                shifted_a[:, 1:7, 1:8] = True
                shifted_b[:, 1:8, 2:7] = True
                shifted = torch.ones_like(old)
                centered_border(shifted, 2, False, shifted_a, shifted_b)
                assert torch.equal(old[:, :7, :7, :7, :5], shifted[:, 1:8, 1:8, 1:8, 2:7])
                coarse.mask_border_with_padding = centered_border
                self.model = LoFTR(config)
                border_note = 'same native 2-cell border around actual centered RGB bounds; top-left equivalence/translation checked'
            self.model.load_state_dict(state, strict=True)
            self.model = self.model.float().to(self.device).eval()
            self.metadata = dict(model=name, torch=torch.__version__, device=str(self.device),
                precision='FP32', config=config, checkpoint=str(args.checkpoint),
                gpu_name=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,
                input='grayscale /255; shared centered letterbox; actual RGB masks downsampled nearest to stride8',
                border_handling=border_note, selection='upstream native thresholds; all returned correspondences; no extra confidence threshold or point cap',
                ground_truth_used_for_estimation=False)

        def sync(self):
            if self.device.type == 'cuda':
                self.torch.cuda.synchronize(self.device)

        def __call__(self, a, b, *, valid_a=None, valid_b=None):
            import cv2
            import numpy as np
            torch = self.torch
            batch = {}
            for i, (rgb, valid) in enumerate(((a, valid_a), (b, valid_b))):
                tensor = torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).to(self.device, dtype=torch.float32)[None, None]/255
                batch[f'image{i}'] = tensor
                if valid is not None:
                    mask = torch.from_numpy(np.ascontiguousarray(valid)).to(self.device)[None, None].float()
                    batch[f'mask{i}'] = self.functional.interpolate(mask, size=(rgb.shape[0]//8, rgb.shape[1]//8), mode='nearest')[:, 0].bool()
            with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
                self.model(batch)
            return tuple(batch[k].detach().float().cpu().numpy() for k in ('mkpts0_f', 'mkpts1_f', 'mconf'))

    def write_report(output, result, rows, args):
        # Keep the original diagnostic-case CSV construction, then replace its
        # GRM-specific narrative and reproduction command with this model's facts.
        original_report(output, result, rows, args)
        lines = [f'# {name}：MegaDepth1500 512评估', '',
            f'完整处理{len(rows)}对；权重：{args.checkpoint}。512居中letterbox、FP32，与GRM R2使用相同预处理、位姿估计和评分函数。', '',
            '| AUC@5° % | AUC@10° % | AUC@20° % | 位姿误差中位数/° | 求解失败 |',
            '|---:|---:|---:|---:|---:|',
            f'| {100*result["pose_auc_5deg"]:.2f} | {100*result["pose_auc_10deg"]:.2f} | {100*result["pose_auc_20deg"]:.2f} | {result["pose_error_median_deg"]:.2f} | {result["pose_failures"]} |', '',
            'E-RANSAC阈值0.5输入像素，confidence=0.99999，maxIters=1000，四轴平均焦距，seed2026+pair_index。GT只用于评分；求解失败保留在分母。', '',
            '匹配器使用本地outdoor权重，灰度输入、模型原生阈值与有效掩码，不额外截断匹配数。EDM的NPE设为832→512，TOPK仅截到物理网格上限，注意力按真实有效矩形裁剪/回填；LoFTR保持原生2格边界宽度，边界跟随实际居中RGB区域。两个区域适配均通过原生左上角布局等价检查，模型结构及权重不变。', '',
            '[稀疏连线诊断](index.html)；全部点、位姿、极线误差和深度支持保存在matches/；逐场景和重叠分桶见summary.json。', '',
            '该成绩是本地固定512协议，不等于原论文高分辨率成绩。不同模型原生匹配数、阈值和训练数据有差异。可核查代码、配置和权重哈希见baseline_provenance.json、settings.json及model_metadata.json。', '',
            '复现命令：', '', '```powershell',
            f'& "{sys.executable}" tools/evaluate_baseline_megadepth1500.py --model {name} --checkpoint "{args.checkpoint}" --output "{args.output}" --preprocess letterbox --size 512 --device cuda --limit {args.limit} --resume', '```', '']
        (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        sources = {p.relative_to(upstream).as_posix():evaluation.digest(p) for p in (upstream/'src').rglob('*.py')}
        sources.update({p.relative_to(upstream).as_posix():evaluation.digest(p) for p in (upstream/'configs').rglob('*.py')})
        evaluation.write_json(output/'baseline_provenance.json', dict(model=name,
            checkpoint_sha256=evaluation.digest(args.checkpoint), wrapper_sha256=evaluation.digest(Path(__file__)),
            core_evaluator_sha256=evaluation.digest(Path(evaluation.__file__)), upstream_root=str(upstream), upstream_source_sha256=sources))

    original_report = evaluation.write_report
    evaluation.Matcher = BaselineMatcher
    evaluation.write_report = write_report
    sys.argv = [sys.argv[0], *remaining]
    evaluation.main()


if __name__ == '__main__':
    main()
