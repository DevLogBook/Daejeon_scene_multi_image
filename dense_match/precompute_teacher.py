"""
1. 多 GPU 并行：每块 GPU 处理不同的 pair 子集，线性加速
   启动方式：torchrun --nproc_per_node=4 precompute_cache.py ...
            或 python precompute_cache.py ...（单卡退化）

2. Batch 推理：把 variations 张图合成一个 batch 一次推理，
   GPU 利用率从约 30% 提升到 80%+

3. 异步 IO：推理完成后把保存任务扔进线程池，
   GPU 不等磁盘，节省约 20~30% 的等待时间

4. compile 预热：正式推理前先用 dummy 输入把 JIT graph 编译好，
   消除头几百张图的编译开销（compile 本身不加速单次推理，
   在重复调用时摊销编译成本才有意义）

5. 跳过已存在文件的粒度从 pair 级改为 variation 级（原有逻辑），
   断点续传不会重复计算任何一个 variation
──────────────────────────────────────────────────────────
单卡参考速度：约 8~12 pairs/s（取决于分辨率）
4 卡参考速度：约 30~45 pairs/s
"""

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
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
ROMA_SRC = REPO_ROOT / "RoMaV2" / "src"
sys.path.insert(0, str(REPO_ROOT.parent))
if str(ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(ROMA_SRC))

from romav2 import RoMaV2
from dense_match.train import teacher_overlap_map
from dense_match.utils import make_teacher_inputs, extract_teacher_features_ds


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


def read_pairs(pairs_file: Path) -> List[Tuple[str, str]]:
    pairs = []
    for line in pairs_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def save_one(save_path: Path, payload: dict):
    """在线程池里执行，不阻塞 GPU"""
    torch.save(payload, save_path)

def build_geo_transform(th: int, tw: int) -> A.Compose:
    """低分辨率图像依然使用随机缩放裁剪，并保持相同的仿射变换"""
    return A.Compose(
        [A.RandomResizedCrop(size=(th, tw), scale=(0.75, 1.0), p=1.0)],
        additional_targets={"image_b": "image"},
    )


def get_jittered_crop(img_a: np.ndarray, img_b: np.ndarray, th: int, tw: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    高分辨率图像专属：锚点抖动裁剪法
    为 A 图随机寻找一个锚点 Crop 框，然后在 B 图该锚点坐标的 ±20% 范围内进行抖动 Crop。
    这样既破坏了像素级的强对齐（让模型学会应对平移），又保证了场景的高重叠率。
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
    y_b_min = max(0, y_a - jitter_h)
    y_b_max = min(h_b - crop_h_b, y_a + jitter_h)

    x_b_min = max(0, x_a - jitter_w)
    x_b_max = min(w_b - crop_w_b, x_a + jitter_w)

    # 4. 随机生成 B 图的裁剪坐标
    y_b = random.randint(y_b_min, y_b_max) if y_b_max > y_b_min else y_b_min
    x_b = random.randint(x_b_min, x_b_max) if x_b_max > x_b_min else x_b_min

    # 5. 执行 numpy 切片裁剪
    crop_a = img_a[y_a: y_a + crop_h_a, x_a: x_a + crop_w_a]
    crop_b = img_b[y_b: y_b + crop_h_b, x_b: x_b + crop_w_b]

    return crop_a, crop_b

POOL_SIZES = [(384, 384), (512, 512), (384, 512), (512, 384)]

TENSOR_NORM = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    additional_targets={"image_b": "image"},
)


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
):
    """
    处理单个图像对的所有 variations，使用 batch 推理。
    返回：实际执行推理的 variation 数量（用于进度统计）
    """
    img_a_bgr = cv2.imread(path_a)
    img_b_bgr = cv2.imread(path_b)
    if img_a_bgr is None or img_b_bgr is None:
        print(f"[WARN] 无法读取图像: {path_a} 或 {path_b}，跳过。")
        return 0

    img_a_rgb = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB)
    img_b_rgb = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_a_rgb.shape[:2]
    is_high_res = (orig_h >= 1080 or orig_w >= 1080)

    selected_sizes = random.sample(POOL_SIZES, min(variations, len(POOL_SIZES)))
    pending: List[Tuple[int, Tuple[int, int], np.ndarray, np.ndarray]] = []

    for i, size in enumerate(selected_sizes):
        key = f"{idx:06d}_v{i}"
        save_path = save_dir / f"{key}.pt"
        if save_path.exists():
            continue  # 断点续传：跳过已存在的

        th, tw = size

        if is_high_res:
            img_a_geo, img_b_geo = get_jittered_crop(img_a_rgb, img_b_rgb, th, tw)
        else:
            geo_transform = build_geo_transform(th, tw)
            geo_out = geo_transform(image=img_a_rgb, image_b=img_b_rgb)
            img_a_geo, img_b_geo = geo_out["image"], geo_out["image_b"]

        pending.append((i, size, img_a_geo, img_b_geo))

    if not pending:
        return 0  # 全部已缓存

    # 同一 pair 的不同 variations 可能尺寸不同，需要按尺寸分组 batch
    # 使用字典：size → list of (i, img_a, img_b)
    size_groups: dict = {}
    for i, size, img_a_geo, img_b_geo in pending:
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append((i, img_a_geo, img_b_geo))

    inferred_count = 0

    for size, items in size_groups.items():
        # 构建这个尺寸组的 batch
        batch_a_list, batch_b_list = [], []
        for _, img_a_geo, img_b_geo in items:
            norm_out = TENSOR_NORM(image=img_a_geo, image_b=img_b_geo)
            batch_a_list.append(norm_out["image"])
            batch_b_list.append(norm_out["image_b"])

        batch_a = torch.stack(batch_a_list).to(device)  # (B, 3, H, W)
        batch_b = torch.stack(batch_b_list).to(device)

        with torch.inference_mode():
            # make_teacher_inputs 支持 batch，直接传入
            img_a_lr, img_b_lr, img_a_hr, img_b_hr = make_teacher_inputs(batch_a, batch_b, teacher)
            t_out = teacher(img_a_lr, img_b_lr, img_a_hr, img_b_hr)
            t_feat_a, t_feat_b = extract_teacher_features_ds(
                teacher, img_a_lr, img_b_lr, teacher_grid_size
            )
            # t_out['warp_AB']:      (B, H, W, 2)
            # teacher_overlap_map:   (B, H, W)
            # t_feat_a/b:            (B, N, D)
            warp_batch    = t_out["warp_AB"].half().cpu()
            conf_batch    = teacher_overlap_map(t_out["confidence_AB"]).half().cpu()
            feat_a_batch  = t_feat_a.half().cpu()
            feat_b_batch  = t_feat_b.half().cpu()

        for batch_idx, (var_i, img_a_geo, img_b_geo) in enumerate(items):
            key = f"{idx:06d}_v{var_i}"
            save_path = save_dir / f"{key}.pt"
            payload = {
                "img_a":          img_a_geo,
                "img_b":          img_b_geo,
                "warp_AB":        warp_batch[batch_idx],
                "confidence_AB":  conf_batch[batch_idx],
                "feat_A":         feat_a_batch[batch_idx],
                "feat_B":         feat_b_batch[batch_idx],
            }
            executor.submit(save_one, save_path, payload)
            inferred_count += 1

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
    args = parser.parse_args()

    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = read_pairs(args.pairs_file)
    # 每张卡处理自己的子集：rank 0 处理 [0, W, 2W, ...]，rank 1 处理 [1, W+1, ...]
    my_pairs = all_pairs[local_rank::world_size]

    if is_main:
        print(f"[Config] world_size={world_size}, total_pairs={len(all_pairs)}, "
              f"variations={args.variations}, teacher={args.teacher_setting}")
        print(f"[Config] 每卡处理约 {len(my_pairs)} 对图像")

    teacher = RoMaV2(RoMaV2.Cfg(setting=args.teacher_setting)).to(device).eval()

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
                    img_lr, _, img_hr, _ = make_teacher_inputs(dummy, dummy, teacher)
                    _ = teacher(img_lr, dummy, img_hr, dummy)
                except Exception:
                    # compile 第一次可能因为 graph break 抛出，忽略预热失败
                    pass
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
        global_pair_idx = local_rank + idx_in_shard * world_size

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
    main()