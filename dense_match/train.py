import argparse
import io
import pathlib
import platform
import random
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')
REPO_ROOT = Path(__file__).resolve().parent
ROMA_SRC = REPO_ROOT / "RoMaV2" / "src"

project_root = REPO_ROOT.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dense_match.network import (
    AgriTPSStitcher,
    DistillationLoss,
    LocalGeometricConsistency,
    make_grid,
)
from dense_match.refine import (
    compute_photometric_loss,
    compute_cycle_consistency_loss,
)

from dataset.dataset import (
    MultiScaleDataset,
    CachedTeacherDataset,
    BucketedBatchSampler,
    BucketedH5TeacherDataset,
)

if str(ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(ROMA_SRC))
from romav2 import RoMaV2  # pyright: ignore[reportMissingImports]

torch.backends.cuda.enable_flash_sdp(True)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _finite_stats(x: torch.Tensor) -> str:
    x_det = x.detach()
    finite = torch.isfinite(x_det)
    if finite.any():
        vals = x_det[finite]
        mn = float(vals.min().item())
        mx = float(vals.max().item())
        mean = float(vals.mean().item())
    else:
        mn, mx, mean = float("nan"), float("nan"), float("nan")
    return f"shape={tuple(x_det.shape)} dtype={x_det.dtype} min={mn:.6g} max={mx:.6g} mean={mean:.6g}"


def _check_finite(name: str, x: torch.Tensor) -> bool:
    ok = torch.isfinite(x).all().item()
    if not ok:
        bad = (~torch.isfinite(x)).sum().item()
        print(f"[NaN/Inf] {name}: bad={bad} {_finite_stats(x)}")
    return bool(ok)


def plot_grid_to_image(dense_grid, img_shape):
    """
    像素级对齐绘图：确保 [-1, 1] 完美对应到 [0, H/W]
    """
    H, W = img_shape
    # grid 形状: [H, W, 2], 范围 [-1, 1]
    grid = dense_grid[0].detach().cpu().numpy()

    # 核心映射：将 [-1, 1] 线性映射到 [0, W-1] 和 [0, H-1]
    # x 坐标映射到宽度，y 坐标映射到高度
    grid_px = (grid + 1.0) / 2.0 * np.array([W - 1, H - 1])

    # 创建纯黑背景（之后会与原图叠加，这里用 0 填充）
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # 设置网格密度（例如每隔 32 像素画一根线）
    step_h = max(H // 16, 1)
    step_w = max(W // 16, 1)

    color = (0, 0, 255)  # 红色 (BGR 格式)
    thickness = 2

    # 绘制水平线 (Horizontal lines)
    for i in range(0, H, step_h):
        pts = grid_px[i, :, :].astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)

    # 绘制垂直线 (Vertical lines)
    for j in range(0, W, step_w):
        pts = grid_px[:, j, :].astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)

    # 绘制最外层边界线，确保“四个角贴死”
    border_idx = [0, -1]
    for b in border_idx:
        cv2.polylines(canvas, [grid_px[b, :, :].astype(np.int32)], False, color, thickness)
        cv2.polylines(canvas, [grid_px[:, b, :].astype(np.int32)], False, color, thickness)

    return torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0


def plot_grid_to_tensorboard(dense_grid, img_shape, step, writer, tag='Train/TPS_Grid'):
    B, H, W, _ = dense_grid.shape
    grid = dense_grid[0].detach().cpu().numpy()
    step_size = max(H // 32, 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(0, H, step_size):
        ax.plot(grid[i, :, 0], grid[i, :, 1], color='blue', linewidth=0.5)
    for j in range(0, W, step_size):
        ax.plot(grid[:, j, 0], grid[:, j, 1], color='blue', linewidth=0.5)
    ax.set_xlim(-1, 1);
    ax.set_ylim(1, -1);
    ax.set_aspect('equal');
    ax.axis('off')
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def create_checkerboard(img1, img2, num_squares=8):
    """
    创建棋盘格对比图，用于检查边缘缝合是否平滑连续
    """
    B, C, H, W = img1.shape
    img2 = img2.to(img1.device)
    grid_h = H // num_squares
    grid_w = W // num_squares

    # 创建掩码
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    mask = ((y // grid_h) % 2 == (x // grid_w) % 2).float().to(img1.device)

    return img1 * mask + img2 * (1 - mask)


def colorize_heatmap(tensor, cmap='jet'):
    x = tensor.detach().cpu().numpy().squeeze()

    # 归一化到 0-1 确保颜色准确
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    # 使用 plt.get_cmap 替代废弃的 cm.get_cmap
    color_map = plt.get_cmap(cmap)

    colorized = color_map(x)[..., :3]

    return torch.from_numpy(colorized).permute(2, 0, 1).float()


def visualize_results(img_a, img_b, stu_out, step, writer, phase="Train"):
    """
    第一行：A 原图 | B 原图 | Warped B (黑边填充) | 棋盘格拼接 (看对齐)
    第二行：彩色置信度 (Red=High) | 误差图 (L1) | TPS 密集网格 | 最终融合图+网格
    """
    matcher_out = stu_out['matcher_out']
    dense_grid = stu_out['dense_grid']  # [B, H, W, 2]
    device = img_a.device
    H_target, W_target = dense_grid.shape[1:3]

    def denorm(x):
        return (x[0:1].detach().cpu() * 0.225 + 0.45).clamp(0, 1)

    def process_for_vis(x, target_hw=None):
        if target_hw and x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        # 取 batch 第一个样本，去归一化，转 CPU
        return (x[0:1].detach() * 0.225 + 0.45).clamp(0, 1).cpu()

    with torch.no_grad():
        # 采样并生成所有图像
        warped_b = F.grid_sample(img_b, dense_grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # 生成所有可视化张量，确保都在 CPU 且尺寸一致 (H_target, W_target)
        img_a_vis = process_for_vis(img_a, (H_target, W_target))
        img_b_vis = process_for_vis(img_b, (H_target, W_target))
        warped_b_vis = process_for_vis(warped_b)  # 已经是目标尺寸

        # 缝合效果与掩码
        ones = torch.ones((1, 1, H_target, W_target), device=device)
        mask = F.grid_sample(ones, dense_grid[0:1], mode='nearest', padding_mode='zeros', align_corners=False).cpu()
        mask = (mask > 0.9).float()

        checker_vis = create_checkerboard(img_a_vis, warped_b_vis, num_squares=12)
        diff_vis = (img_a_vis - warped_b_vis).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        blended_vis = img_a_vis * (1 - mask * 0.5) + warped_b_vis * (mask * 0.5)

        # 几何信号处理
        conf = matcher_out['confidence_AB'][0:1].detach()
        if conf.shape[-2:] != (H_target, W_target):
            conf = F.interpolate(conf.unsqueeze(1), size=(H_target, W_target), mode='bilinear', align_corners=False)[0]
        else:
            conf = conf[0]
        conf_color = colorize_heatmap(conf.cpu().unsqueeze(0), cmap='jet')

        # 网格图绘制
        grid_vis = plot_grid_to_image(dense_grid, (H_target, W_target))
        overlay_vis = blended_vis[0] * 0.7 + grid_vis * 0.3

        geo_scores = matcher_out.get('geo_scores')
        top1_prob = matcher_out.get('top1_prob')
        if geo_scores is not None and top1_prob is not None:
            Hc, Wc = matcher_out['coarse_hw']
            geo_map = geo_scores.mean(dim=-1).reshape(-1, Hc, Wc)[0:1]
            writer.add_image('Debug/geo_scores', colorize_heatmap(geo_map), step)

            top1_map = top1_prob.reshape(-1, Hc, Wc)[0:1]
            writer.add_image('Debug/top1_prob', colorize_heatmap(top1_map), step)

        # 组装面板 (全部都在 CPU，尺寸均为 H_target x W_target)
        row1 = torch.cat([img_a_vis[0], img_b_vis[0], warped_b_vis[0], checker_vis[0]], dim=2)
        row2 = torch.cat([conf_color, diff_vis[0], grid_vis, overlay_vis], dim=2)
        dashboard = torch.cat([row1, row2], dim=1)

        writer.add_image(f"{phase}/Full_Dashboard", dashboard, step)


# EMA Loss Tracker
class EMALossTracker:
    def __init__(self, alpha: float = 0.1, window_size: int = 100):
        self.alpha = alpha
        self.ema_values: Dict[str, float] = {}
        self.windows: Dict[str, deque] = {}

    def update(self, losses: Dict[str, float]) -> Dict[str, float]:
        smoothed = {}
        for name, value in losses.items():
            if not isinstance(value, (int, float)):
                continue
            if name not in self.ema_values:
                self.ema_values[name] = value
                self.windows[name] = deque(maxlen=100)
            self.ema_values[name] = self.alpha * value + (1 - self.alpha) * self.ema_values[name]
            self.windows[name].append(value)
            smoothed[f"{name}_ema"] = self.ema_values[name]
        return smoothed

    def get_ema(self, name: str) -> float:
        return self.ema_values.get(name, float('inf'))


def multi_scale_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    if len(batch) == 0:
        return {}

    return {
        'img_a': [item['img_a'] for item in batch],
        'img_b': [item['img_b'] for item in batch],
    }


def cached_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(batch) == 0:
        return {}

    # 初始化列表（关键缺失）
    img_a_list, img_b_list = [], []
    t_warp_AB_list, t_conf_AB_list = [], []
    t_feat_a_list, t_feat_b_list = [], []

    target_h, target_w = batch[0]['img_a'].shape[-2:]

    for item in batch:
        a, b = item['img_a'], item['img_b']
        if a.shape[-2:] != (target_h, target_w):
            a = F.interpolate(a.unsqueeze(0), size=(target_h, target_w),
                              mode='bilinear', align_corners=False).squeeze(0)
            b = F.interpolate(b.unsqueeze(0), size=(target_h, target_w),
                              mode='bilinear', align_corners=False).squeeze(0)
        img_a_list.append(a)
        img_b_list.append(b)
        t_warp_AB_list.append(item['t_warp_AB'])
        t_conf_AB_list.append(item['t_conf_AB'])
        t_feat_a_list.append(item['t_feat_a'])
        t_feat_b_list.append(item['t_feat_b'])

    return {
        'img_a': torch.stack(img_a_list),
        'img_b': torch.stack(img_b_list),
        't_warp_AB': torch.stack(t_warp_AB_list),
        't_conf_AB': torch.stack(t_conf_AB_list),
        't_feat_a': torch.stack(t_feat_a_list),
        't_feat_b': torch.stack(t_feat_b_list),
    }


# 辅助函数
def teacher_overlap_map(confidence: torch.Tensor) -> torch.Tensor:
    if confidence.ndim == 4:
        return torch.sigmoid(confidence[..., 0])
    if confidence.ndim == 3:
        return torch.sigmoid(confidence)
    raise ValueError(f"Unexpected confidence shape: {confidence.shape}")


def make_teacher_inputs(
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        teacher: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    img_a_lr = F.interpolate(
        img_a, size=(teacher.H_lr, teacher.W_lr),
        mode="bicubic", align_corners=False, antialias=True,
    )
    img_b_lr = F.interpolate(
        img_b, size=(teacher.H_lr, teacher.W_lr),
        mode="bicubic", align_corners=False, antialias=True,
    )

    if teacher.H_hr is None or teacher.W_hr is None:
        return img_a_lr, img_b_lr, None, None

    img_a_hr = F.interpolate(
        img_a, size=(teacher.H_hr, teacher.W_hr),
        mode="bicubic", align_corners=False, antialias=True,
    )
    img_b_hr = F.interpolate(
        img_b, size=(teacher.H_hr, teacher.W_hr),
        mode="bicubic", align_corners=False, antialias=True,
    )
    return img_a_lr, img_b_lr, img_a_hr, img_b_hr


@torch.no_grad()
def extract_teacher_features_ds(
        teacher: Any,
        img_a_lr: torch.Tensor,
        img_b_lr: torch.Tensor,
        grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """提取 teacher 特征并下采样到指定尺寸"""
    f_list_a = teacher.f(img_a_lr)
    f_list_b = teacher.f(img_b_lr)

    if f_list_a[0].shape[1] < f_list_a[0].shape[-1]:
        feat_a = torch.cat([x.float() for x in f_list_a], dim=1)
        feat_b = torch.cat([x.float() for x in f_list_b], dim=1)
        feat_a_ds = F.interpolate(
            feat_a, size=(grid_size, grid_size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        feat_b_ds = F.interpolate(
            feat_b, size=(grid_size, grid_size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
    else:
        feat_a = torch.cat([x.float() for x in f_list_a], dim=-1)
        feat_b = torch.cat([x.float() for x in f_list_b], dim=-1)
        feat_a_ds = F.interpolate(
            feat_a.permute(0, 3, 1, 2), size=(grid_size, grid_size),
            mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        feat_b_ds = F.interpolate(
            feat_b.permute(0, 3, 1, 2), size=(grid_size, grid_size),
            mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

    bsz = feat_a_ds.shape[0]
    teacher_dim = feat_a_ds.shape[-1]
    feat_a_flat = feat_a_ds.reshape(bsz, grid_size * grid_size, teacher_dim)
    feat_b_flat = feat_b_ds.reshape(bsz, grid_size * grid_size, teacher_dim)
    return feat_a_flat, feat_b_flat


# Checkpoint
def save_checkpoint(
        save_dir: Path,
        epoch: int,
        step: int,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: Optional[torch.amp.GradScaler],
        args: argparse.Namespace,
        train_loss_ema: float,
        val_loss: float,
        best_val_loss: float,
        is_best: bool = False,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "step": step,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "args": vars(args),
        "train_loss_ema": train_loss_ema,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    last_path = save_dir / "last.pt"
    torch.save(payload, last_path)
    print(f"The last weight file update ! val loss: {val_loss:.6f} best val loss: {best_val_loss:.6f}")

    if is_best:
        shutil.copyfile(last_path, save_dir / "best.pt")
        print(f"✅ New best val_loss: {val_loss:.6f}")

    return last_path


def load_checkpoint(
        ckpt_path: Path,
        student: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler,
        scaler: Optional[torch.amp.GradScaler],
        device: torch.device,
) -> Tuple[int, int, float, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 加载模型（处理可能的 key 不匹配）
    try:
        student.load_state_dict(ckpt["student"], strict=True)
        print("[load] ✅ Model loaded (strict)")
    except RuntimeError as e:
        print(f"[load] ⚠️ Strict load failed: {e}")
        # 尝试非严格加载
        missing, unexpected = student.load_state_dict(ckpt["student"], strict=False)
        print(f"[load] Loaded with missing={len(missing)}, unexpected={len(unexpected)}")

    # 尝试加载 optimizer
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[load] ✅ Optimizer loaded")
        except Exception as e:
            print(f"[load] ⚠️ Optimizer load failed: {e}")

    # 尝试加载 scheduler
    if scheduler is not None and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("[load] ✅ Scheduler loaded")
        except Exception as e:
            print(f"[load] ⚠️ Scheduler load failed: {e}")

    # 尝试加载 scaler
    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[load] ⚠️ Scaler load failed: {e}")

    best_val_loss = ckpt.get("best_val_loss", ckpt.get("best_loss", float("inf")))
    train_loss_ema = ckpt.get("train_loss_ema", float("inf"))

    return (
        int(ckpt.get("epoch", 0)),
        int(ckpt.get("step", 0)),
        float(best_val_loss),
        float(train_loss_ema),
    )


# @torch.no_grad()
# def legacy_validate(student, val_loader, loss_fn, device, teacher_grid_size, max_batches=20):
#     student.eval()
#     total_fold = 0
#     total_distill_loss = 0
#     count = 0
#
#     for i, batch in enumerate(val_loader):
#         if i >= max_batches:
#             break
#
#         # 兼容两种 DataLoader：CachedTeacherDataset（tensor）和 MultiScaleDataset（list）
#         if isinstance(batch['img_a'], torch.Tensor):
#             img_a = batch['img_a'].to(device)
#             img_b = batch['img_b'].to(device)
#         else:
#             img_a = torch.stack(batch['img_a']).to(device)
#             img_b = torch.stack(batch['img_b']).to(device)
#
#         # 从缓存或实时推理获取 teacher 信号
#         if 't_warp_AB' in batch:
#             t_warp_AB = batch['t_warp_AB'].to(device)
#             t_conf_AB = batch['t_conf_AB'].to(device)
#             t_feat_a = batch['t_feat_a'].to(device)
#             t_feat_b = batch['t_feat_b'].to(device)
#             t_out = {
#                 'warp_AB': t_warp_AB,
#                 'confidence_AB': t_conf_AB,
#                 'confidence_is_prob': True,
#             }
#         # else:
#         #     with torch.inference_mode():
#         #         img_a_lr, img_b_lr, img_a_hr, img_b_hr = make_teacher_inputs(img_a, img_b, teacher)
#         #         t_out = teacher(img_a_lr, img_b_lr, img_a_hr, img_b_hr)
#         #         t_feat_a, t_feat_b = extract_teacher_features_ds(
#         #             teacher, img_a_lr, img_b_lr, teacher_grid_size
#         #         )
#         #         t_out['confidence_is_prob'] = False
#
#         stu_out = student(img_a, img_b)
#
#         d_loss = loss_fn(
#             stu_output=stu_out['matcher_out'],
#             teacher_warp=t_out['warp_AB'],
#             teacher_conf=(
#                 t_out['confidence_AB']
#                 if t_out.get('confidence_is_prob', False)
#                 else teacher_overlap_map(t_out['confidence_AB'])
#             ),
#             teacher_feat_A=t_feat_a,
#             teacher_feat_B=t_feat_b,
#         )
#         total_distill_loss += d_loss['total'].item()
#         total_fold += stu_out['tps_out'].get('fold_ratio', 0)
#         count += 1
#
#     student.train()
#     return {
#         'total': total_distill_loss / count,
#         'fold_ratio': total_fold / count,
#     }


def _prepare_teacher_targets(
        batch: Dict[str, Any],
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if isinstance(batch['img_a'], torch.Tensor):
        img_a = batch['img_a'].to(device, non_blocking=True)
        img_b = batch['img_b'].to(device, non_blocking=True)
    else:
        img_a = torch.stack(batch['img_a']).to(device, non_blocking=True)
        img_b = torch.stack(batch['img_b']).to(device, non_blocking=True)

    if 't_warp_AB' not in batch:
        raise RuntimeError("Online teacher inference is currently disabled. Please use --cache-dir with cached teacher supervision.")

    t_warp_AB = batch['t_warp_AB'].to(device, non_blocking=True)
    t_conf_AB = batch['t_conf_AB'].to(device, non_blocking=True)
    t_feat_a = batch['t_feat_a'].to(device, non_blocking=True)
    t_feat_b = batch['t_feat_b'].to(device, non_blocking=True)
    t_out = {
        'warp_AB': t_warp_AB,
        'confidence_AB': t_conf_AB,
        'confidence_is_prob': True,
    }
    return img_a, img_b, t_out, t_feat_a, t_feat_b


def _compute_loss_bundle(
        student: nn.Module,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        t_out: Dict[str, torch.Tensor],
        t_feat_a: torch.Tensor,
        t_feat_b: torch.Tensor,
        distill_loss_fn: DistillationLoss,
        geo_loss_fn: LocalGeometricConsistency,
        cfg: Dict[str, Any],
        device: torch.device,
        amp_enabled: bool,
        current_epoch: int = 1,
        ssim_warmup_epochs: int = 10,
) -> Dict[str, Any]:
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        stu_out = student(img_a, img_b)
        stu_output = stu_out["matcher_out"]

        final_grid = stu_out["dense_grid"]
        conf_refine = stu_output['confidence_AB']
        batch_size, height, width, _ = final_grid.shape
        grid_full = make_grid(batch_size, height, width, final_grid.device, final_grid.dtype)
        current_flow_final = final_grid - grid_full

        if conf_refine.shape[-2:] != (height, width):
            conf_for_geo = F.interpolate(
                conf_refine.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False
            ).squeeze(1)
        else:
            conf_for_geo = conf_refine

        geo_loss = geo_loss_fn(current_flow_final, conf_for_geo)
        d_loss = distill_loss_fn(
            stu_output=stu_output,
            teacher_warp=t_out["warp_AB"],
            teacher_conf=(
                t_out["confidence_AB"]
                if t_out.get("confidence_is_prob", False)
                else teacher_overlap_map(t_out["confidence_AB"])
            ),
            teacher_feat_A=t_feat_a,
            teacher_feat_B=t_feat_b,
        )

        f_loss = stu_out["tps_out"].get("fold_loss", torch.zeros((), device=device))
        if not torch.is_tensor(f_loss):
            f_loss = torch.tensor(f_loss, device=device)

        if "SELF_SUPERVISED" in cfg["phase"]:
            noise_std = 0.02
            noise_b = torch.randn_like(img_b) * noise_std
            noise_a = torch.randn_like(img_a) * noise_std
            b_min, b_max = img_b.min(), img_b.max()
            a_min, a_max = img_a.min(), img_a.max()
            img_b_noisy = (img_b + noise_b).clamp(b_min, b_max)
            img_a_noisy = (img_a + noise_a).clamp(a_min, a_max)

            stu_out_ba = student(img_b_noisy, img_a_noisy)
            c_loss = compute_cycle_consistency_loss(stu_out, stu_out_ba)
        else:
            c_loss = torch.zeros((), device=device)

        w = cfg["weights"]
        total_loss_no_photo = (
                w["distill"] * d_loss["total"]
                + w["fold"] * f_loss
                + w["cycle"] * c_loss
                + w["geo"] * geo_loss
        )
    use_ssim_now = (current_epoch > ssim_warmup_epochs)
    with torch.amp.autocast("cuda", enabled=False):
        p_loss = compute_photometric_loss(
            img_a.float(),
            img_b.float(),
            stu_out["dense_grid"].float(),
            stu_out["matcher_out"]["confidence_AB"].float(),
            use_ssim=use_ssim_now,
            alpha=0.85,
        )

    total_loss = total_loss_no_photo + w["photo"] * p_loss

    fold_ratio = stu_out["tps_out"].get("fold_ratio", 0.0)
    if torch.is_tensor(fold_ratio):
        fold_ratio = float(fold_ratio.detach().mean().item())
    else:
        fold_ratio = float(fold_ratio)

    return {
        'stu_out': stu_out,
        'distill': d_loss["total"],
        'fold': f_loss,
        'photo': p_loss,
        'cycle': c_loss,
        'geo': geo_loss,
        'total': total_loss,
        'fold_ratio': fold_ratio,
    }


@torch.no_grad()
def validate(
        student: nn.Module,
        val_loader: DataLoader,
        distill_loss_fn: DistillationLoss,
        geo_loss_fn: LocalGeometricConsistency,
        cfg: Dict[str, Any],
        device: torch.device,
        use_amp: bool,
        max_batches: int = 20,
        epoch: int = 0,
) -> Dict[str, float]:
    was_training = student.training
    student.eval()
    totals = {
        'distill': 0.0,
        'fold': 0.0,
        'photo': 0.0,
        'cycle': 0.0,
        'geo': 0.0,
        'total': 0.0,
        'fold_ratio': 0.0,
    }
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        img_a, img_b, t_out, t_feat_a, t_feat_b = _prepare_teacher_targets(batch, device)
        loss_bundle = _compute_loss_bundle(
            student=student,
            img_a=img_a,
            img_b=img_b,
            t_out=t_out,
            t_feat_a=t_feat_a,
            t_feat_b=t_feat_b,
            distill_loss_fn=distill_loss_fn,
            geo_loss_fn=geo_loss_fn,
            cfg=cfg,
            device=device,
            amp_enabled=use_amp,
            current_epoch=epoch,
            ssim_warmup_epochs=10,
        )
        totals['distill'] += float(loss_bundle['distill'].detach().item())
        totals['fold'] += float(loss_bundle['fold'].detach().item())
        totals['photo'] += float(loss_bundle['photo'].detach().item())
        totals['cycle'] += float(loss_bundle['cycle'].detach().item())
        totals['geo'] += float(loss_bundle['geo'].detach().item())
        totals['total'] += float(loss_bundle['total'].detach().item())
        totals['fold_ratio'] += float(loss_bundle['fold_ratio'])
        count += 1

    if was_training:
        student.train()
    if count == 0:
        return totals
    return {key: value / count for key, value in totals.items()}


# 参数解析
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("AgriMatcher Multi-Scale Training")

    p.add_argument("--pairs-file", type=Path, default=None,
                   help="原始图像对文件（不使用 --cache-dir 时必填）")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="预计算 teacher 缓存目录（与 --pairs-file 二选一，优先使用缓存）")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints_multiscale"))
    p.add_argument("--log-dir", type=Path, default=Path("runs/agrimatch_multiscale"))
    p.add_argument("--resume", type=Path, default=None)

    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--teacher-grid-size", type=int, default=32, help="Teacher 特征提取的 grid size")

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--force-amp", action="store_true", help="Force AMP even if auto-disabled for stability")
    p.add_argument("--accum-steps", type=int, default=4)

    p.add_argument("--teacher-setting", type=str, default="precise")
    p.add_argument("--teacher-compile", action="store_true")

    # Loss weights
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta-coarse", type=float, default=1.0)
    p.add_argument("--beta-refine", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--eta-coarse", type=float, default=0.5)
    p.add_argument("--eta-refine", type=float, default=1.0)
    p.add_argument("--lambda-tv-coarse", type=float, default=0.01)
    p.add_argument("--lambda-tv-refine", type=float, default=0.05)
    p.add_argument("--conf-thresh-kl", type=float, default=0.1)

    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--val-interval", type=int, default=200)

    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    return p


def get_phase_config(epoch):
    keyframes = {
        1: {"phase": "DISTILL", "w": {"distill": 1.0, "fold": 0.0, "photo": 0.5, "cycle": 0.0, "geo": 0.1},
            "train_aggregator": False, "train_backbone": True},
        8: {"phase": "DISTILL", "w": {"distill": 1.0, "fold": 0.0, "photo": 0.5, "cycle": 0.0, "geo": 0.1},
            "train_aggregator": False, "train_backbone": True},
        10: {"phase": "TPS_REFINE",
             "w": {"distill": 0.5, "fold": 0.5, "photo": 1.0, "cycle": 0.0, "geo": 0.5},
             "train_aggregator": True, "train_backbone": False},
        15: {"phase": "TPS_REFINE",
             "w": {"distill": 0.5, "fold": 1.0, "photo": 1.5, "cycle": 0.0, "geo": 0.2},
             "train_aggregator": True, "train_backbone": False},
        17: {"phase": "SELF_SUPERVISED",
             "w": {"distill": 0.3, "fold": 1.5, "photo": 2.0, "cycle": 1.0, "geo": 0.2},
             "train_aggregator": True, "train_backbone": False},
        100: {"phase": "SELF_SUPERVISED",
              "w": {"distill": 0.1, "fold": 1.5, "photo": 2.0, "cycle": 1.0, "geo": 0.2},
              "train_aggregator": True, "train_backbone": False},
    }

    # 定位当前 Epoch 所在的插值区间 [e1, e2]
    epochs = sorted(keyframes.keys())
    e1, e2 = epochs[0], epochs[-1]
    for i in range(len(epochs) - 1):
        if epochs[i] <= epoch <= epochs[i + 1]:
            e1, e2 = epochs[i], epochs[i + 1]
            break

    if epoch >= epochs[-1]:
        e1 = e2 = epochs[-1]

    kf1, kf2 = keyframes[e1], keyframes[e2]

    # 计算插值系数 alpha (0.0 到 1.0)
    alpha = 0.0 if e1 == e2 else (epoch - e1) / (e2 - e1)

    # 线性平滑混合权重
    blended_weights = {
        k: (1 - alpha) * kf1["w"][k] + alpha * kf2["w"][k]
        for k in kf1["w"]
    }

    # 状态平滑切换
    # 只要开始向新阶段过渡(alpha > 0)，就立刻解冻对应的网络层，使其提前开始适应微小的梯度
    train_aggregator = kf2["train_aggregator"] if alpha > 0 else kf1["train_aggregator"]
    train_backbone = kf2["train_backbone"] if alpha > 0 else kf1["train_backbone"]
    # 动态生成 Phase 监控名称
    if alpha == 0:
        phase_name = kf1["phase"]
    elif alpha == 1:
        phase_name = kf2["phase"]
    else:
        phase_name = f"TRANSITION ({kf1['phase']} -> {kf2['phase']})"

    return {
        "phase": phase_name,
        "weights": blended_weights,
        "train_aggregator": train_aggregator,
        "train_matcher": True,
        "train_backbone": train_backbone,
    }


# 主函数
def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp
    if device.type == "cuda" and use_amp and not args.force_amp:
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            print(
                f"[AMP] Auto-disabled AMP for stability on this GPU (capability={major}.{minor}). Use --force-amp to override.")
            use_amp = False
    writer = SummaryWriter(log_dir=str(args.log_dir))

    use_cache = args.cache_dir is not None and args.cache_dir.exists()
    cache_loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": cached_collate_fn,
        "pin_memory": True,
    }
    image_loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": multi_scale_collate_fn,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        cache_loader_kwargs["persistent_workers"] = True
        cache_loader_kwargs["prefetch_factor"] = 2
        image_loader_kwargs["persistent_workers"] = True
        image_loader_kwargs["prefetch_factor"] = 2

    if use_cache:
        print(f"[Dataset] 使用预计算缓存模式: {args.cache_dir}")
        cache_suffix = args.cache_dir.suffix.lower() if args.cache_dir.is_file() else ""
        if cache_suffix in {".h5", ".hdf5"}:
            train_dataset = BucketedH5TeacherDataset(
                str(args.cache_dir), val_ratio=args.val_ratio, split_seed=args.seed, return_split='train'
            )
            val_dataset = BucketedH5TeacherDataset(
                str(args.cache_dir), val_ratio=args.val_ratio, split_seed=args.seed, return_split='val'
            )
            train_batch_sampler = BucketedBatchSampler(
                train_dataset.bucket_to_indices,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                seed=args.seed,
            )
            val_batch_sampler = BucketedBatchSampler(
                val_dataset.bucket_to_indices,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                seed=args.seed,
            )
            train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, **cache_loader_kwargs)
            val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, **cache_loader_kwargs)
        else:
            train_dataset = CachedTeacherDataset(
                args.cache_dir, val_ratio=args.val_ratio, return_split='train'
            )
            val_dataset = CachedTeacherDataset(
                args.cache_dir, val_ratio=args.val_ratio, return_split='val'
            )
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, **cache_loader_kwargs
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, **cache_loader_kwargs
            )
    else:
        if args.pairs_file is None:
            raise ValueError("必须提供 --pairs-file 或有效的 --cache-dir")
        print(f"[Dataset] 使用原始图像对模式: {args.pairs_file}")
        train_dataset = MultiScaleDataset(args.pairs_file, val_ratio=args.val_ratio, return_split='train')
        val_dataset = MultiScaleDataset(args.pairs_file, val_ratio=args.val_ratio, return_split='val')
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **image_loader_kwargs
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, **image_loader_kwargs
        )

    # 缓存模式下 teacher 仅用于探测 feature dim，不参与训练循环推理
    # teacher = RoMaV2(RoMaV2.Cfg(setting=args.teacher_setting)).to(device).eval()

    # with torch.inference_mode():
    #     dummy_img = torch.zeros(1, 3, 512, 512, device=device)
    #     img_a_lr, _, _, _ = make_teacher_inputs(dummy_img, dummy_img, teacher)
    #     feat_a_flat, _ = extract_teacher_features_ds(teacher, img_a_lr, img_a_lr, args.teacher_grid_size)
    #     actual_teacher_dim = feat_a_flat.shape[-1]
    #     print(f"Detected Teacher Feature Dimension: {actual_teacher_dim}")

    # 缓存模式下释放 teacher 显存（训练循环里不再需要它）
    # if use_cache:
    #     teacher.cpu()
    #     torch.cuda.empty_cache()
    #     print("[Cache] Teacher moved to CPU to free GPU memory.")
    actual_teacher_dim_list = {
        'base': 80,
        'precise': 100
    }
    actual_teacher_dim =actual_teacher_dim_list[args.teacher_setting]
    student = AgriTPSStitcher(
        matcher_config={
            'd_model': args.d_model,
            'teacher_dim': actual_teacher_dim,
            'grid_size': args.teacher_grid_size
        },
        tps_config={
            'grid_size': 8,
            'feat_channels': args.d_model
        }
    ).to(device)

    backbone_params = []
    other_params = []

    for name, param in student.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    # 骨干网络使用 1/10 的学习率进行微调，防止预训练权重崩塌
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    total_optimizer_steps = (len(train_loader) * args.epochs) // args.accum_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_optimizer_steps, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_tracker = EMALossTracker(alpha=args.ema_alpha)

    # 损失函数类
    distill_loss_fn = DistillationLoss(
        alpha=args.alpha,
        beta_coarse=args.beta_coarse,
        beta_refine=args.beta_refine,
        gamma=args.gamma,
        eta_coarse=args.eta_coarse,
        eta_refine=args.eta_refine,
        lambda_tv_coarse=args.lambda_tv_coarse,
        lambda_tv_refine=args.lambda_tv_refine,
        conf_thresh_kl=args.conf_thresh_kl,
    ).to(device)
    geo_loss_fn = LocalGeometricConsistency()

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, global_step, best_val_loss, train_loss_ema = load_checkpoint(
            ckpt_path=args.resume,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_amp else None,
            device=device,
        )
        loss_tracker = EMALossTracker(alpha=args.ema_alpha)
        loss_tracker.ema_values['total'] = train_loss_ema

        print(f"Resumed from epoch={start_epoch}, step={global_step}, "
              f"best_val={best_val_loss:.4f}")

    prev_grad_states = {}

    for epoch in range(start_epoch, args.epochs + 1):
        cfg = get_phase_config(epoch)
        if use_cache and isinstance(getattr(train_loader, 'batch_sampler', None), BucketedBatchSampler):
            train_loader.batch_sampler.set_epoch(epoch)
        print(f"\n🚀 Entering Phase: {cfg['phase']} (Epoch {epoch})")

        reset_count = 0
        for name, param in student.named_parameters():
            # 确定当前参数应该的状态
            if "backbone" in name:
                target_req_grad = cfg['train_backbone']
            elif "tps_estimator.aggregator" in name:
                target_req_grad = cfg['train_aggregator']
            elif "matcher" in name:
                target_req_grad = cfg['train_matcher']
            else:
                target_req_grad = True
            # 状态翻转检测：如果由 False 变为 True
            if prev_grad_states.get(name, target_req_grad) is False and target_req_grad is True:
                # 彻底清空该参数的历史动量
                if param in optimizer.state:
                    optimizer.state[param] = {}
                    reset_count += 1

            param.requires_grad = target_req_grad
            prev_grad_states[name] = target_req_grad

        if reset_count > 0:
            print(f"🧹 Optimizer states reset for {reset_count} unfrozen parameter tensors!")
        student.train()
        if cfg['train_backbone']:
            pass
        else:
            student.matcher.backbone.eval()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", dynamic_ncols=True)
        for i, batch in pbar:
            if use_cache:
                img_a = batch['img_a'].to(device, non_blocking=True)
                img_b = batch['img_b'].to(device, non_blocking=True)
            else:
                img_a_list = [x.to(device, non_blocking=True) for x in batch['img_a']]
                img_b_list = [x.to(device, non_blocking=True) for x in batch['img_b']]

                target_h, target_w = img_a_list[0].shape[-2:]
                B = len(img_a_list)

                img_a = torch.empty((B, 3, target_h, target_w), device=device, dtype=img_a_list[0].dtype)
                img_b = torch.empty((B, 3, target_h, target_w), device=device, dtype=img_b_list[0].dtype)

                for b_idx, (a, b) in enumerate(zip(img_a_list, img_b_list)):
                    if a.shape[-2:] != (target_h, target_w):
                        img_a[b_idx] = F.interpolate(a.unsqueeze(0), size=(target_h, target_w),
                                                     mode='bilinear', align_corners=False).squeeze(0)
                        img_b[b_idx] = F.interpolate(b.unsqueeze(0), size=(target_h, target_w),
                                                     mode='bilinear', align_corners=False).squeeze(0)
                    else:
                        img_a[b_idx] = a
                        img_b[b_idx] = b

            if use_cache:
                t_warp_AB = batch['t_warp_AB'].to(device, non_blocking=True)
                t_conf_AB = batch['t_conf_AB'].to(device, non_blocking=True)
                t_feat_a = batch['t_feat_a'].to(device, non_blocking=True)
                t_feat_b = batch['t_feat_b'].to(device, non_blocking=True)
                t_out = {
                    'warp_AB': t_warp_AB,
                    'confidence_AB': t_conf_AB,
                    'confidence_is_prob': True,
                }
            # else:
            #     with torch.inference_mode():
            #         img_a_lr, img_b_lr, img_a_hr, img_b_hr = make_teacher_inputs(img_a, img_b, teacher)
            #         t_out = teacher(img_a_lr, img_b_lr, img_a_hr, img_b_hr)
            #         t_feat_a, t_feat_b = extract_teacher_features_ds(
            #             teacher, img_a_lr, img_b_lr, args.teacher_grid_size
            #         )
            #         t_out['confidence_is_prob'] = False

            def _forward_losses(amp_enabled: bool):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    stu_out = student(img_a, img_b)
                    stu_output = stu_out["matcher_out"]

                    final_grid = stu_out["dense_grid"]
                    conf_refine = stu_output['confidence_AB']

                    B_img, H_img, W_img, _ = final_grid.shape
                    grid_full = make_grid(B_img, H_img, W_img, final_grid.device, final_grid.dtype)
                    current_flow_final = final_grid - grid_full

                    if conf_refine.shape[-2:] != (H_img, W_img):
                        conf_for_geo = F.interpolate(
                            conf_refine.unsqueeze(1), size=(H_img, W_img),
                            mode='bilinear', align_corners=False
                        ).squeeze(1)
                    else:
                        conf_for_geo = conf_refine

                    geo_loss = geo_loss_fn(current_flow_final, conf_for_geo)
                    d_loss = distill_loss_fn(
                        stu_output=stu_output,
                        teacher_warp=t_out["warp_AB"],
                        teacher_conf=(
                            t_out["confidence_AB"]
                            if t_out.get("confidence_is_prob", False)
                            else teacher_overlap_map(t_out["confidence_AB"])
                        ),
                        teacher_feat_A=t_feat_a,
                        teacher_feat_B=t_feat_b,
                    )
                    f_loss = stu_out["tps_out"].get("fold_loss", torch.tensor(0.0, device=device))

                    if "SELF_SUPERVISED" in cfg["phase"]:
                        noise_std = 0.02
                        noise_b = torch.randn_like(img_b) * noise_std
                        noise_a = torch.randn_like(img_a) * noise_std
                        img_b_noisy = (img_b + noise_b).clamp(img_b.min(), img_b.max())
                        img_a_noisy = (img_a + noise_a).clamp(img_a.min(), img_a.max())
                        stu_out_ba = student(img_b_noisy, img_a_noisy)
                        c_loss = compute_cycle_consistency_loss(stu_out, stu_out_ba)
                    else:
                        c_loss = torch.tensor(0.0, device=device)

                # ── Photo loss 移到 autocast 块外，强制在 FP32 下计算 ──────────────
                # 即使 amp_enabled=False，放在外面也没有副作用
                use_ssim_now = (epoch > 10)
                with torch.amp.autocast("cuda", enabled=False):  # 显式禁用 autocast
                    p_loss = compute_photometric_loss(
                        img_a.float(),
                        img_b.float(),
                        stu_out["dense_grid"].float(),
                        stu_out["matcher_out"]["confidence_AB"].float(),
                        use_ssim=use_ssim_now,
                        alpha=0.85,
                    )

                w = cfg["weights"]
                total_loss = (
                        w["distill"] * d_loss["total"]
                        + w["fold"] * f_loss
                        + w["photo"] * p_loss
                        + w["cycle"] * c_loss
                        + w["geo"] * geo_loss
                )
                return stu_out, d_loss, f_loss, p_loss, c_loss, geo_loss, total_loss

            stu_out, d_loss, f_loss, p_loss, c_loss, geo_loss, total_loss = _forward_losses(use_amp)

            loss_dict = {
                "distill": d_loss["total"],
                "fold": f_loss if torch.is_tensor(f_loss) else torch.tensor(f_loss, device=device),
                "photo": p_loss,
                "cycle": c_loss if torch.is_tensor(c_loss) else torch.tensor(c_loss, device=device),
                "geo": geo_loss,
                "total": total_loss
            }

            ok = True
            for k, v in loss_dict.items():
                if not torch.isfinite(v).all().item():
                    pbar.write(f"🚨 [NaN/Inf] Loss '{k}' is invalid! Value: {v.item() if v.numel() == 1 else 'Tensor'}")
                    ok = False

            if not ok:
                pbar.write(f"⏭️ Skipping batch at step {global_step} due to NaN loss.")
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda": torch.cuda.empty_cache()
                continue

            loss_for_backward = total_loss / args.accum_steps
            total_loss_scalar = float(total_loss.detach().item())
            distill_scalar = float(d_loss['total'].detach().item())
            fold_scalar = float(f_loss.detach().item()) if torch.is_tensor(f_loss) else float(f_loss)
            photo_scalar = float(p_loss.detach().item())
            cycle_scalar = float(c_loss.detach().item()) if torch.is_tensor(c_loss) else float(c_loss)

            # 反向传播
            scaler.scale(loss_for_backward).backward()

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

                has_nan_grad = False
                for p in student.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all().item():
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    pbar.write(f"🛡️ [Gradient Shield] NaN/Inf gradient at step {global_step}! Skipping optimizer step.")
                else:
                    scaler.step(optimizer)

                # scheduler 无论是否 NaN 都步进，保持 LR 衰减与 epoch 进度同步
                scheduler.step()

                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                loss_metrics = {
                    'total': total_loss_scalar,
                    'distill': distill_scalar,
                    'fold': fold_scalar,
                    'photo': photo_scalar,
                    'cycle': cycle_scalar,
                    'geo': geo_loss.item()
                }
                smoothed = loss_tracker.update(loss_metrics)
                pbar.set_postfix({
                    'Loss': f"{smoothed['total_ema']:.4f}",
                    'Photo': f"{smoothed['photo_ema']:.4f}",
                    'Fold': f"{smoothed['fold_ema']:.4f}",
                    'Geo': f"{smoothed['geo_ema']:.4f}"
                })

                if global_step % args.log_interval == 0:
                    for k, v in smoothed.items():
                        writer.add_scalar(f"Train/{k}", v, global_step)

                    with torch.no_grad():
                        visualize_results(img_a, img_b, stu_out, global_step, writer)

                if global_step % args.val_interval == 0:
                    val_losses = validate(
                        student=student,
                        val_loader=val_loader,
                        distill_loss_fn=distill_loss_fn,
                        geo_loss_fn=geo_loss_fn,
                        cfg=cfg,
                        device=device,
                        use_amp=use_amp,
                        epoch=epoch
                    )
                    val_total = val_losses['total']

                    pbar.write(f"📊 [val] step={global_step} | total={val_total:.4f}")
                    writer.add_scalar("Val/Total", val_total, global_step)
                    writer.add_scalar("Val/Distill", val_losses['distill'], global_step)
                    writer.add_scalar("Val/Photo", val_losses['photo'], global_step)
                    writer.add_scalar("Val/Fold", val_losses['fold'], global_step)
                    writer.add_scalar("Val/Cycle", val_losses['cycle'], global_step)
                    writer.add_scalar("Val/Geo", val_losses['geo'], global_step)
                    writer.add_scalar("Val/FoldRatio", val_losses['fold_ratio'], global_step)

                    is_best = val_total < best_val_loss
                    if is_best:
                        best_val_loss = val_total

                    save_checkpoint(
                        args.save_dir, epoch, global_step,
                        student, optimizer, scheduler,
                        scaler if use_amp else None, args,
                        train_loss_ema=loss_tracker.get_ema('total'),
                        val_loss=val_total,
                        best_val_loss=best_val_loss,
                        is_best=is_best,
                    )

        pbar.write(f"[epoch {epoch}] Done. EMA={loss_tracker.get_ema('total'):.4f}")

    writer.close()

if __name__ == "__main__":
    main()
