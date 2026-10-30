"""Generate cache-v2 RoMaV2 pseudo correspondences, with independent AB/BA.

Old caches must be regenerated into a NEW directory. This fixes generator
contracts, but does not establish that Teacher predictions are ground truth.
Example: python -m dense_match.precompute_teacher --help
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import albumentations as A
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
ROMA_SRC = REPO_ROOT / "RoMaV2" / "src"
sys.path.insert(0, str(REPO_ROOT.parent))
if str(ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(ROMA_SRC))

from dense_match.teacher_cache_generation import (
    CACHE_VERSION, rgb_tensor, make_teacher_inputs, extract_teacher_features_ds,
    teacher_overlap_map, read_pairs, save_one, file_hash, validate_payload,
)
import json


def setup_distributed() -> Tuple[int, int]:
    """
    初始化分布式环境（torchrun 启动时有效，单进程时退化为 rank=0, world=1）
    返回 (local_rank, world_size)
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_geo_transform(th: int, tw: int) -> A.Compose:
    """低分辨率图像依然使用随机缩放裁剪，并保持相同的仿射变换"""
    return A.ReplayCompose(
        [A.RandomResizedCrop(size=(th, tw), scale=(0.75, 1.0), p=1.0)],
        additional_targets={"image_b": "image"},
    )


def get_jittered_crop(img_a: np.ndarray, img_b: np.ndarray, th: int, tw: int, return_geometry=False):
    """
    高分辨率图像专属：锚点抖动裁剪法
    为 A 图随机寻找一个锚点 Crop 框，然后在 B 图该锚点坐标的 ±20% 范围内进行抖动 Crop。
    此方法不能保证不同视角的真实重叠率，需另行审计Teacher标签。
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # 防止目标尺寸大于原图尺寸
    crop_h_a, crop_w_a = min(th, h_a), min(tw, w_a)
    crop_h_b, crop_w_b = min(th, h_b), min(tw, w_b)

    # 1. 为 A 图随机选择锚点坐标 (左上角)
    y_a = random.randint(0, h_a - crop_h_a) if h_a > crop_h_a else 0
    x_a = random.randint(0, w_a - crop_w_a) if w_a > crop_w_a else 0

    # 2. 计算 B 图的抖动范围 (±20% 的目标尺寸)
    jitter_h = int(th * 0.2)
    jitter_w = int(tw * 0.2)

    # 3. 限制 B 图的合法随机裁剪范围，防止越界
    anchor_y_b = min(y_a, h_b - crop_h_b)
    y_b_min = max(0, anchor_y_b - jitter_h)
    y_b_max = min(h_b - crop_h_b, anchor_y_b + jitter_h)

    anchor_x_b = min(x_a, w_b - crop_w_b)
    x_b_min = max(0, anchor_x_b - jitter_w)
    x_b_max = min(w_b - crop_w_b, anchor_x_b + jitter_w)

    # 4. 随机生成 B 图的裁剪坐标
    y_b = random.randint(y_b_min, y_b_max) if y_b_max > y_b_min else y_b_min
    x_b = random.randint(x_b_min, x_b_max) if x_b_max > x_b_min else x_b_min

    # 5. 执行 numpy 切片裁剪
    crop_a = img_a[y_a: y_a + crop_h_a, x_a: x_a + crop_w_a]
    crop_b = img_b[y_b: y_b + crop_h_b, x_b: x_b + crop_w_b]

    if return_geometry:
        return crop_a, crop_b, {
            'crop_boxes_xywh': [[x_a, y_a, crop_w_a, crop_h_a], [x_b, y_b, crop_w_b, crop_h_b]],
            'source_to_saved': [[[1, 0, -x_a], [0, 1, -y_a], [0, 0, 1]],
                                [[1, 0, -x_b], [0, 1, -y_b], [0, 0, 1]]],
        }
    return crop_a, crop_b

POOL_SIZES = [(384, 384), (512, 512), (384, 512), (512, 384)]

# RoMaV2 internally normalizes. Feed RGB float [0,1], not ImageNet tensors.


def process_one_pair(
    idx: int,
    path_a: str,
    path_b: str,
    save_dir: Path,
    teacher: RoMaV2,
    teacher_grid_size: int,
    variations: int,
    device: torch.device,
    executor: ThreadPoolExecutor,
    seed: int = 42,
    provenance: dict = None,
    skip_features: bool = False,
    crop_mode: str = 'legacy',
    preserve_source_resolution: bool = False,
):
    """
    处理单个图像对的所有 variations，使用 batch 推理。
    返回：实际执行推理的 variation 数量（用于进度统计）
    """
    img_a_bgr = cv2.imread(path_a)
    img_b_bgr = cv2.imread(path_b)
    if img_a_bgr is None or img_b_bgr is None:
        raise ValueError(f"无法读取pair={idx}: {path_a}, {path_b}")

    img_a_rgb = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
    img_b_rgb = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_a_rgb.shape[:2]
    is_high_res = (orig_h >= 1080 or orig_w >= 1080)

    selected_sizes = random.Random(seed + idx).sample(POOL_SIZES, variations)
    futures = []
    geometries = {}
    pair_metadata = dict(provenance or {}, cache_version=CACHE_VERSION,
                         pair_id=f"{idx:06d}", source_paths=[path_a, path_b],
                         source_sha256=[file_hash(path_a), file_hash(path_b)],
                         source_hw=[list(img_a_rgb.shape[:2]), list(img_b_rgb.shape[:2])])
    pending: List[Tuple[int, Tuple[int, int], np.ndarray, np.ndarray]] = []

    for i, size in enumerate(selected_sizes):
        key = f"{idx:06d}_v{i}"
        save_path = save_dir / f"{key}.pt"
        if save_path.exists():
            existing = torch.load(save_path, map_location='cpu', weights_only=False)
            validate_payload(existing)
            for name, value in dict(pair_metadata, variation=i, requested_hw=list(size)).items():
                if existing['metadata'].get(name) != value:
                    raise ValueError(f"{save_path}: metadata differs: {name}; use a new directory")
            continue  # 断点续传：跳过已存在的

        th, tw = size
        variation_seed = seed + idx * 1009 + i * 9176
        random.seed(variation_seed)
        np.random.seed(variation_seed % (2**32))

        if crop_mode == 'full':
            img_a_geo = img_a_rgb.copy() if preserve_source_resolution else cv2.resize(img_a_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
            img_b_geo = img_b_rgb.copy() if preserve_source_resolution else cv2.resize(img_b_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
            transforms = []
            for image, saved in ((img_a_rgb, img_a_geo), (img_b_rgb, img_b_geo)):
                h, w = image.shape[:2]
                sh, sw = saved.shape[:2]
                sx, sy = sw/w, sh/h
                transforms.append([[sx, 0, .5*sx-.5], [0, sy, .5*sy-.5], [0, 0, 1]])
            geometry = {'crop_mode': 'full', 'source_to_saved': transforms}
        elif is_high_res:
            img_a_geo, img_b_geo, geometry = get_jittered_crop(img_a_rgb, img_b_rgb, th, tw, True)
        else:
            if img_a_rgb.shape[:2] != img_b_rgb.shape[:2]:
                raise ValueError(f'pair={idx}: shared RandomResizedCrop requires equal source sizes; '
                                 'use resized source images or high-resolution crop preprocessing')
            geo_transform = build_geo_transform(th, tw)
            if hasattr(geo_transform, 'set_random_seed'):
                geo_transform.set_random_seed(variation_seed)
            geo_out = geo_transform(image=img_a_rgb, image_b=img_b_rgb)
            img_a_geo, img_b_geo = geo_out["image"], geo_out["image_b"]
            geometry = {'albumentations_replay': geo_out['replay']}

        geometries[i] = geometry

        pending.append((i, size, img_a_geo, img_b_geo))

    if not pending:
        return 0  # 全部已缓存

    # 同一 pair 的不同 variations 可能尺寸不同，需要按尺寸分组 batch
    # 使用字典：size → list of (i, img_a, img_b)
    size_groups: dict = {}
    for i, size, img_a_geo, img_b_geo in pending:
        actual_shapes = (img_a_geo.shape, img_b_geo.shape)
        if actual_shapes not in size_groups:
            size_groups[actual_shapes] = []
        size_groups[actual_shapes].append((i, img_a_geo, img_b_geo))

    inferred_count = 0

    for size, items in size_groups.items():
        inference_start = time.perf_counter()
        # 构建这个尺寸组的 batch
        batch_a_list, batch_b_list = [], []
        for _, img_a_geo, img_b_geo in items:
            batch_a_list.append(rgb_tensor(img_a_geo))
            batch_b_list.append(rgb_tensor(img_b_geo))

        batch_a = torch.stack(batch_a_list).to(device)  # (B, 3, H, W)
        batch_b = torch.stack(batch_b_list).to(device)

        with torch.inference_mode():
            # make_teacher_inputs 支持 batch，直接传入
            img_a_lr, img_b_lr, img_a_hr, img_b_hr = make_teacher_inputs(batch_a, batch_b, teacher)
            t_out = teacher(img_a_lr, img_b_lr, img_a_hr, img_b_hr)
            if t_out.get('warp_BA') is None or t_out.get('confidence_BA') is None:
                reverse = teacher(img_b_lr, img_a_lr, img_b_hr, img_a_hr)
                t_out['warp_BA'], t_out['confidence_BA'] = reverse['warp_AB'], reverse['confidence_AB']
            t_feat_a, t_feat_b = extract_teacher_features_ds(
                teacher, img_a_lr, img_b_lr, teacher_grid_size
            ) if not skip_features else (None, None)
            # t_out['warp_AB']:      (B, H, W, 2)
            # teacher_overlap_map:   (B, H, W)
            # t_feat_a/b:            (B, N, D)
            warp_batch    = t_out["warp_AB"].float().cpu()
            conf_batch    = teacher_overlap_map(t_out["confidence_AB"]).float().cpu()
            feat_a_batch  = t_feat_a.half().cpu() if t_feat_a is not None else None
            feat_b_batch  = t_feat_b.half().cpu() if t_feat_b is not None else None

        for batch_idx, (var_i, img_a_geo, img_b_geo) in enumerate(items):
            key = f"{idx:06d}_v{var_i}"
            save_path = save_dir / f"{key}.pt"
            payload = {
                "img_a":          img_a_geo,
                "img_b":          img_b_geo,
                "warp_AB":        warp_batch[batch_idx],
                "confidence_AB":  conf_batch[batch_idx],
                "warp_BA": t_out['warp_BA'][batch_idx].float().cpu(),
                "confidence_BA": teacher_overlap_map(t_out['confidence_BA'])[batch_idx].float().cpu(),
                "confidence_raw_AB": t_out['confidence_AB'][batch_idx].float().cpu(),
                "confidence_raw_BA": t_out['confidence_BA'][batch_idx].float().cpu(),
                "metadata": dict(pair_metadata, variation=var_i,
                                 variation_seed=seed + idx*1009 + var_i*9176,
                                 requested_hw=list(selected_sizes[var_i]),
                                 saved_hw=[list(img_a_geo.shape[:2]), list(img_b_geo.shape[:2])]),
            }
            payload['metadata']['teacher_batch_seconds'] = time.perf_counter() - inference_start
            payload['metadata']['teacher_batch_size'] = len(items)
            if feat_a_batch is not None:
                payload.update(feat_A=feat_a_batch[batch_idx], feat_B=feat_b_batch[batch_idx])
            validate_payload(payload)
            payload['metadata']['geometry'] = geometries[var_i]
            payload['metadata']['feature_dim'] = payload['feat_A'].shape[-1] if 'feat_A' in payload else None
            quality = {}
            for direction in ('AB', 'BA'):
                warp, probability = payload['warp_'+direction], payload['confidence_'+direction]
                quality[direction] = {
                    'target_in_bounds_fraction': (warp.abs() <= 1).all(-1).float().mean().item(),
                    'mean_overlap_probability': probability.mean().item(),
                }
            payload['quality'] = quality
            futures.append(executor.submit(save_one, save_path, payload))
            print(json.dumps({'pair': idx, 'variation': var_i, 'quality': quality,
                              'feature_dim': payload['metadata']['feature_dim'],
                              'teacher_batch_seconds': payload['metadata']['teacher_batch_seconds']}), flush=True)
            inferred_count += 1

    for future in futures:
        future.result()  # Do not silently report success when disk writes fail.
    return inferred_count


def main():
    parser = argparse.ArgumentParser("RoMa v2 缓存预计算")
    parser.add_argument("--pairs-file",       type=Path, required=True)
    parser.add_argument("--save-dir",         type=Path, required=True)
    parser.add_argument("--teacher-setting",  type=str,  default="precise",
                        choices=["base", "precise"])
    parser.add_argument("--teacher-grid-size", type=int, default=32)
    parser.add_argument("--variations",       type=int,  default=2)
    parser.add_argument("--io-workers",       type=int,  default=4,
                        help="异步保存的线程数，建议设为磁盘通道数")
    parser.add_argument("--no-compile",       action="store_true",
                        help="禁用 torch.compile（调试用）")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-pairs', type=int, default=None, help='Only process first N pairs for auditing')
    parser.add_argument('--skip-features', action='store_true', help='Correspondence supervision only')
    parser.add_argument('--image-root', type=Path, help='Root for relative paths; default pairs-file parent')
    parser.add_argument('--crop-mode', choices=['legacy', 'full'], default='legacy',
                        help='full preserves the whole pair; legacy uses seeded crop variants')
    parser.add_argument('--preserve-source-resolution', action='store_true',
                        help='full mode only: avoid low-resolution pre-resizing before RoMa internal resize')
    args = parser.parse_args()
    if args.preserve_source_resolution and (args.crop_mode != 'full' or args.variations != 1):
        parser.error('--preserve-source-resolution requires --crop-mode full --variations 1')
    if not 1 <= args.variations <= 4 or args.teacher_grid_size < 1 or args.io_workers < 1:
        parser.error('variations must be 1..4; grid size/IO workers must be positive')
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error('--max-pairs must be positive')
    args.save_dir, args.pairs_file = args.save_dir.resolve(), args.pairs_file.resolve()
    if args.image_root:
        args.image_root = args.image_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: request a GPU and check the PyTorch environment')

    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = (global_rank == 0)
    # Import after selecting LOCAL_RANK: RoMa has a module-level CUDA device.
    from romav2 import RoMaV2

    args.save_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = read_pairs(args.pairs_file, args.image_root)
    if args.max_pairs:
        all_pairs = all_pairs[:args.max_pairs]
    # 每张卡处理自己的子集：rank 0 处理 [0, W, 2W, ...]，rank 1 处理 [1, W+1, ...]
    my_pairs = all_pairs[global_rank::world_size]

    if is_main:
        print(f"[Config] world_size={world_size}, total_pairs={len(all_pairs)}, "
              f"variations={args.variations}, teacher={args.teacher_setting}")
        print(f"[Config] 每卡处理约 {len(my_pairs)} 对图像")

    # Bundled RoMa resolves its local backbones against cwd. Caller paths were
    # made absolute first; no online checkpoint download is introduced here.
    os.chdir(REPO_ROOT.parent)
    provenance = dict(cache_version=CACHE_VERSION, generator_sha256=file_hash(__file__),
                      helper_sha256=file_hash(REPO_ROOT / 'teacher_cache_generation.py'),
                      pairs_sha256=file_hash(args.pairs_file),
                      teacher='RoMaV2', teacher_setting=args.teacher_setting,
                      weights_sha256=file_hash(Path('backbones/romav2.pt')),
                      seed=args.seed, variations=args.variations,
                      crop_mode=args.crop_mode,
                      feature_grid_size=args.teacher_grid_size, skip_features=args.skip_features,
                      image_color='RGB', teacher_input='float_0_1_internal_imagenet_normalization',
                      warp='normalized_target_xy_align_corners_false',
                      pixel_conversion='(normalized_xy+1)*[W,H]/2-0.5',
                      confidence='sigmoid(raw[...,0]); overlap probability, not correctness')
    if args.preserve_source_resolution:
        provenance['preserve_source_resolution'] = True
    # Check run contract once; validate each resumed file in its own shard.
    # Avoid reading the entire dense cache once PER GPU on every restart.
    settings_path = args.save_dir / 'settings.json'
    if settings_path.exists():
        cached_settings = json.loads(settings_path.read_text(encoding='utf-8'))
        for name, value in provenance.items():
            if cached_settings.get(name) != value:
                raise ValueError(f'{settings_path}: incompatible {name}; use a NEW save directory')
    elif next(args.save_dir.glob('*.pt'), None) is not None:
        raise ValueError('Cache directory has PT files without cache-v2 settings; use a NEW save directory')
    teacher = RoMaV2(RoMaV2.Cfg(setting=args.teacher_setting, compile=False)).to(device).eval()
    provenance['teacher_lr_hw'] = [teacher.H_lr, teacher.W_lr]
    provenance['teacher_hr_hw'] = [teacher.H_hr, teacher.W_hr]
    if is_main:
        settings_tmp = args.save_dir / 'settings.json.tmp'
        settings_tmp.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        os.replace(settings_tmp, settings_path)

    if not args.no_compile:
        # 使用 default 模式而不是 reduce-overhead：
        # - reduce-overhead 专为高频小 kernel 设计，Teacher 的大矩阵运算反而不适合
        # - default 模式对 ViT/Transformer 类模型更友好
        teacher = torch.compile(teacher, mode="default")

        if is_main:
            print("[Warmup] 预热 torch.compile JIT 图（约需 30~60 秒）...")
        dummy = torch.zeros(1, 3, 512, 512, device=device)
        with torch.inference_mode():
            for _ in range(3):
                try:
                    warm_inputs = make_teacher_inputs(dummy, dummy, teacher)
                    _ = teacher(*warm_inputs)
                except Exception as exc:
                    raise RuntimeError('RoMa compile warmup failed; retry with --no-compile') from exc
        if is_main:
            print("[Warmup] 完成。")

    executor = ThreadPoolExecutor(max_workers=args.io_workers)

    total_inferred = 0
    t_start = time.perf_counter()

    desc = f"[GPU {local_rank}] Precompute"
    pbar = tqdm(
        enumerate(my_pairs),
        total=len(my_pairs),
        desc=desc,
        position=local_rank,   # 多卡时每行一个进度条
        leave=True,
    )

    for idx_in_shard, (path_a, path_b) in pbar:
        # 全局 pair 索引：保证各卡保存的文件名不冲突
        global_pair_idx = global_rank + idx_in_shard * world_size

        count = process_one_pair(
            idx=global_pair_idx,
            path_a=path_a,
            path_b=path_b,
            save_dir=args.save_dir,
            teacher=teacher,
            teacher_grid_size=args.teacher_grid_size,
            variations=args.variations,
            device=device,
            executor=executor,
            seed=args.seed,
            provenance=provenance,
            skip_features=args.skip_features,
            crop_mode=args.crop_mode,
            preserve_source_resolution=args.preserve_source_resolution,
        )
        total_inferred += count

        # 动态速度显示
        elapsed = time.perf_counter() - t_start
        speed = total_inferred / elapsed if elapsed > 0 else 0.0
        pbar.set_postfix({"inferred": total_inferred, "var/s": f"{speed:.1f}"})

    if is_main:
        print("\n[IO] 等待所有文件写入完成...")
    executor.shutdown(wait=True)

    if is_main:
        elapsed = time.perf_counter() - t_start
        print(f"\n✅ 完成！共生成 {total_inferred} 个缓存文件（本卡），"
              f"耗时 {elapsed/60:.1f} 分钟，平均 {total_inferred/elapsed:.1f} var/s")

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
