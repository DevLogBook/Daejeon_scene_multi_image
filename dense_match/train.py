import argparse
import pathlib
import platform
import random
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
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
from dense_match.utils import (STAGE_DEFS, get_stage, interpolate_loss_weights, _get_module_param_groups,
                               apply_freeze_state, build_optimizer_and_scheduler, update_optimizer_and_scheduler)
if str(ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(ROMA_SRC))

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
        mn, mx, mean = float(vals.min()), float(vals.max()), float(vals.mean())
    else:
        mn, mx, mean = float("nan"), float("nan"), float("nan")
    return f"shape={tuple(x_det.shape)} dtype={x_det.dtype} min={mn:.6g} max={mx:.6g} mean={mean:.6g}"


def _check_finite(name: str, x: torch.Tensor) -> bool:
    ok = torch.isfinite(x).all().item()
    if not ok:
        bad = (~torch.isfinite(x)).sum().item()
        print(f"[NaN/Inf] {name}: bad={bad} {_finite_stats(x)}")
    return bool(ok)


def plot_warped_grid(dense_grid, img_shape):
    """
    通过物理重采样来可视化形变网格。
    展示图 B 的均匀网格在图 A 视角下被扭曲的真实形态。
    """
    B, H_target, W_target, _ = dense_grid.shape
    device = dense_grid.device

    # 1. 生成 Batch Size 为 1 的网格图片
    grid_img = torch.zeros((1, 3, H_target, W_target), device=device)
    step_h = max(H_target // 16, 1)
    step_w = max(W_target // 16, 1)

    for i in range(0, H_target, step_h):
        grid_img[0, 2, i:i+2, :] = 1.0
    for j in range(0, W_target, step_w):
        grid_img[0, 2, :, j:j+2] = 1.0

    grid_img[0, 2, 0:2, :] = 1.0
    grid_img[0, 2, -2:, :] = 1.0
    grid_img[0, 2, :, 0:2] = 1.0
    grid_img[0, 2, :, -2:] = 1.0

    warped_grid = F.grid_sample(
        grid_img, dense_grid[0:1],
        mode='bilinear', padding_mode='zeros', align_corners=False
    )

    return warped_grid[0].cpu()  # 返回 [3, H, W] 的 RGB 张量

def plot_grid_to_tensorboard(dense_grid, img_shape, step, writer, tag='Train/TPS_Grid'):
    B, H, W, _ = dense_grid.shape
    grid = dense_grid[0].detach().cpu().numpy()
    step_size = max(H // 32, 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(0, H, step_size):
        ax.plot(grid[i, :, 0], grid[i, :, 1], color='blue', linewidth=0.5)
    for j in range(0, W, step_size):
        ax.plot(grid[:, j, 0], grid[:, j, 1], color='blue', linewidth=0.5)
    ax.set_xlim(-1, 1); ax.set_ylim(1, -1); ax.set_aspect('equal'); ax.axis('off')
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def create_checkerboard(img1, img2, num_squares=8):
    B, C, H, W = img1.shape
    img2 = img2.to(img1.device)
    grid_h = H // num_squares
    grid_w = W // num_squares
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    mask = ((y // grid_h) % 2 == (x // grid_w) % 2).float().to(img1.device)
    return img1 * mask + img2 * (1 - mask)


def colorize_heatmap(tensor, cmap='jet'):
    x = tensor.detach().cpu().numpy().squeeze()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return torch.from_numpy(plt.get_cmap(cmap)(x)[..., :3]).permute(2, 0, 1).float()


def visualize_results(img_a, img_b, stu_out, step, writer, phase="Train"):
    matcher_out = stu_out['matcher_out']
    dense_grid = stu_out['dense_grid']
    device = img_a.device
    H_target, W_target = dense_grid.shape[1:3]

    def process_for_vis(x, target_hw=None):
        if target_hw and x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return (x[0:1].detach() * 0.225 + 0.45).clamp(0, 1).cpu()

    with torch.no_grad():
        warped_b = F.grid_sample(img_b, dense_grid, mode='bilinear',
                                  padding_mode='zeros', align_corners=False)
        img_a_vis    = process_for_vis(img_a, (H_target, W_target))
        img_b_vis    = process_for_vis(img_b, (H_target, W_target))
        warped_b_vis = process_for_vis(warped_b)
        ones  = torch.ones((1, 1, H_target, W_target), device=device)
        mask  = F.grid_sample(ones, dense_grid[0:1], mode='nearest',
                               padding_mode='zeros', align_corners=False).cpu()
        mask  = (mask > 0.9).float()
        checker_vis = create_checkerboard(img_a_vis, warped_b_vis, num_squares=12)
        diff_vis    = (img_a_vis - warped_b_vis).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        blended_vis = img_a_vis * (1 - mask * 0.5) + warped_b_vis * (mask * 0.5)
        conf        = matcher_out['confidence_AB'][0:1].detach()
        if conf.shape[-2:] != (H_target, W_target):
            conf = F.interpolate(conf.unsqueeze(1), size=(H_target, W_target),
                                  mode='bilinear', align_corners=False)[0]
        else:
            conf = conf[0]
        conf_color  = colorize_heatmap(conf.cpu().unsqueeze(0), cmap='jet')
        grid_vis = plot_warped_grid(dense_grid, (H_target, W_target))

        # 提取蓝色网格线的 Mask，只在有网格线的地方进行半透明叠加
        grid_mask = (grid_vis[2:3, :, :] > 0.1).float()
        overlay_vis = blended_vis[0] * (1 - grid_mask * 0.8) + grid_vis * (grid_mask * 0.8)
        geo_scores  = matcher_out.get('geo_scores')
        top1_prob   = matcher_out.get('top1_prob')
        if geo_scores is not None and top1_prob is not None:
            Hc, Wc   = matcher_out['coarse_hw']
            geo_map  = geo_scores.mean(dim=-1).reshape(-1, Hc, Wc)[0:1]
            writer.add_image('Debug/geo_scores', colorize_heatmap(geo_map), step)
            top1_map = top1_prob.reshape(-1, Hc, Wc)[0:1]
            writer.add_image('Debug/top1_prob', colorize_heatmap(top1_map), step)
        row1      = torch.cat([img_a_vis[0], img_b_vis[0], warped_b_vis[0], checker_vis[0]], dim=2)
        row2      = torch.cat([conf_color, diff_vis[0], grid_vis, overlay_vis], dim=2)
        dashboard = torch.cat([row1, row2], dim=1)
        writer.add_image(f"{phase}/Full_Dashboard", dashboard, step)


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
            if name not in self.windows:
                self.windows[name] = deque(maxlen=100)

            self.ema_values[name] = self.alpha * value + (1 - self.alpha) * self.ema_values[name]
            self.windows[name].append(value)
            smoothed[f"{name}_ema"] = self.ema_values[name]

        return smoothed

    def get_ema(self, name: str) -> float:
        return self.ema_values.get(name, float('inf'))

def multi_scale_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    if not batch:
        return {}
    return {'img_a': [item['img_a'] for item in batch],
            'img_b': [item['img_b'] for item in batch]}


def cached_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        return {}
    img_a_list, img_b_list = [], []
    t_warp_AB_list, t_conf_AB_list, t_feat_a_list, t_feat_b_list = [], [], [], []
    target_h, target_w = batch[0]['img_a'].shape[-2:]
    for item in batch:
        a, b = item['img_a'], item['img_b']
        if a.shape[-2:] != (target_h, target_w):
            a = F.interpolate(a.unsqueeze(0), size=(target_h, target_w),
                              mode='bilinear', align_corners=False).squeeze(0)
            b = F.interpolate(b.unsqueeze(0), size=(target_h, target_w),
                              mode='bilinear', align_corners=False).squeeze(0)
        img_a_list.append(a); img_b_list.append(b)
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


def teacher_overlap_map(confidence: torch.Tensor) -> torch.Tensor:
    if confidence.ndim == 4:
        return torch.sigmoid(confidence[..., 0])
    if confidence.ndim == 3:
        return torch.sigmoid(confidence)
    raise ValueError(f"Unexpected confidence shape: {confidence.shape}")


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
        raise RuntimeError("Online teacher inference is disabled. Use --cache-dir.")

    t_out = {
        'warp_AB': batch['t_warp_AB'].to(device, non_blocking=True),
        'confidence_AB': batch['t_conf_AB'].to(device, non_blocking=True),
        'confidence_is_prob': True,
    }
    t_feat_a = batch['t_feat_a'].to(device, non_blocking=True)
    t_feat_b = batch['t_feat_b'].to(device, non_blocking=True)
    return img_a, img_b, t_out, t_feat_a, t_feat_b


def tps_pixel_smoothness_loss(tps_residual: torch.Tensor) -> torch.Tensor:
    """
    惩罚 TPS 局部形变场（相对于 H 基准）的空间非平滑性。
    tps_residual: (B, H, W, 2)  tps_local_grid - base_grid
    """
    flow = tps_residual.permute(0, 3, 1, 2).float()  # [B, 2, H, W]

    # 二阶差分（Laplacian），对线性形变免疫
    lap_x = flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
    lap_y = flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]

    return lap_x.pow(2).mean() + lap_y.pow(2).mean()

def compute_loss_bundle(
    student: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    t_out: Dict[str, torch.Tensor],
    t_feat_a: torch.Tensor,
    t_feat_b: torch.Tensor,
    distill_loss_fn: DistillationLoss,
    geo_loss_fn: LocalGeometricConsistency,
    w: Dict[str, float],
    device: torch.device,
    amp_enabled: bool,
    use_cycle: bool = False,
) -> Dict[str, Any]:
    """
    统一的 Loss 计算函数。
    w 由调用方根据当前 epoch 动态插值传入，不在此函数内查询 epoch。
    """
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        stu_out = student(img_a, img_b)
        stu_output = stu_out["matcher_out"]

        final_grid = stu_out["dense_grid"]
        conf_refine = stu_output['confidence_AB']
        B, H, W, _ = final_grid.shape

        grid_full = make_grid(B, H, W, final_grid.device, final_grid.dtype)

        if conf_refine.shape[-2:] != (H, W):
            conf_for_geo = F.interpolate(
                conf_refine.unsqueeze(1), size=(H, W),
                mode='bilinear', align_corners=False
            ).squeeze(1)
        else:
            conf_for_geo = conf_refine

        tps_residual = stu_out.get("tps_residual_flow")
        if tps_residual is not None:
            geo_loss = geo_loss_fn(tps_residual, conf_for_geo)
        else:
            current_flow = final_grid - grid_full
            geo_loss = geo_loss_fn(current_flow, conf_for_geo)

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
        f_loss = f_loss.clamp(max=3.0)

        # 当 TPS 没有任何局部变形时（delta_cp_local=0），该 Loss 完美为 0
        tps_residual = stu_out.get("tps_residual_flow")
        if tps_residual is not None and w.get("tps_smooth", 0.0) > 0:
            smooth_loss = tps_pixel_smoothness_loss(tps_residual.float())
        else:
            smooth_loss = torch.zeros((), device=device)

        # Cycle Loss
        if use_cycle and w.get("cycle", 0.0) > 0:
            noise_std = 0.02
            img_b_noisy = (img_b + torch.randn_like(img_b) * noise_std).clamp(
                img_b.min(), img_b.max())
            img_a_noisy = (img_a + torch.randn_like(img_a) * noise_std).clamp(
                img_a.min(), img_a.max())
            stu_out_ba = student(img_b_noisy, img_a_noisy)
            c_loss = compute_cycle_consistency_loss(stu_out, stu_out_ba)
        else:
            c_loss = torch.zeros((), device=device)

        # Photo Loss 在 fp32 计算（SSIM 对精度敏感）
        total_no_photo = (
            w["distill"]    * d_loss["total"]
            + w["fold"]     * f_loss
            + w["cycle"]    * c_loss
            + w["geo"]      * geo_loss
            + w.get("tps_smooth", 0.0) * smooth_loss
        )

    photo_w = w.get("photo", 0.0)
    if photo_w > 0:
        with torch.amp.autocast("cuda", enabled=False):
            use_ssim_now = w.get("ssim", 0.0) > 0.01
            p_loss = compute_photometric_loss(
                img_a.float(), img_b.float(),
                stu_out["dense_grid"].float(),
                stu_out["matcher_out"]["confidence_AB"].float(),
                use_ssim=use_ssim_now,
                alpha=0.85,
            )
    else:
        p_loss = torch.zeros((), device=device)

    total_loss = total_no_photo + photo_w * p_loss

    fold_ratio = stu_out["tps_out"].get("fold_ratio", 0.0)
    if torch.is_tensor(fold_ratio):
        fold_ratio = float(fold_ratio.detach().mean().item())

    return {
        'stu_out':   stu_out,
        'distill':   d_loss["total"],
        'fold':      f_loss,
        'photo':     p_loss,
        'cycle':     c_loss,
        'geo':       geo_loss,
        'tps_smooth': smooth_loss,
        'total':     total_loss,
        'fold_ratio': fold_ratio,
    }


@torch.no_grad()
def validate(
    student: nn.Module,
    val_loader: DataLoader,
    distill_loss_fn: DistillationLoss,
    geo_loss_fn: LocalGeometricConsistency,
    w: Dict[str, float],
    device: torch.device,
    use_amp: bool,
    max_batches: int = 20,
) -> Dict[str, float]:
    was_training = student.training
    student.eval()
    totals = {k: 0.0 for k in ('distill', 'fold', 'photo', 'cycle', 'geo', 'tps_smooth', 'total', 'fold_ratio')}
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        img_a, img_b, t_out, t_feat_a, t_feat_b = _prepare_teacher_targets(batch, device)
        bundle = compute_loss_bundle(
            student=student, img_a=img_a, img_b=img_b,
            t_out=t_out, t_feat_a=t_feat_a, t_feat_b=t_feat_b,
            distill_loss_fn=distill_loss_fn, geo_loss_fn=geo_loss_fn,
            w=w, device=device, amp_enabled=use_amp, use_cycle=False,
        )
        for k in totals:
            v = bundle[k]
            totals[k] += float(v.detach().item()) if torch.is_tensor(v) else float(v)
        count += 1
    if was_training:
        student.train()
    if count == 0:
        return totals
    return {k: v / count for k, v in totals.items()}



def save_checkpoint(
    save_dir: Path, epoch: int, step: int,
    student: nn.Module, optimizer: torch.optim.Optimizer, scheduler,
    scaler: Optional[torch.amp.GradScaler], args: argparse.Namespace,
    train_loss_ema: float, val_loss: float, best_val_loss: float, is_best: bool = False,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch, "step": step,
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
    print(f"Checkpoint saved. val={val_loss:.6f}  best={best_val_loss:.6f}")
    if is_best:
        shutil.copyfile(last_path, save_dir / "best.pt")
        print(f"✅ New best: {val_loss:.6f}")
    return last_path


def load_checkpoint(
    ckpt_path: Path, student: nn.Module,
    optimizer: Optional[torch.optim.Optimizer], scheduler,
    scaler: Optional[torch.amp.GradScaler], device: torch.device,
) -> Tuple[int, int, float, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        student.load_state_dict(ckpt["student"], strict=True)
        print("[load] ✅ Model loaded (strict)")
    except RuntimeError as e:
        print(f"[load] ⚠️ Strict load failed: {e}")
        missing, unexpected = student.load_state_dict(ckpt["student"], strict=False)
        print(f"[load] Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[load] ⚠️ Optimizer load failed: {e}")
    if scheduler and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[load] ⚠️ Scheduler load failed: {e}")
    if scaler and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[load] ⚠️ Scaler load failed: {e}")
    best_val_loss = ckpt.get("best_val_loss", ckpt.get("best_loss", float("inf")))
    return (
        int(ckpt.get("epoch", 0)),
        int(ckpt.get("step", 0)),
        float(best_val_loss),
        float(ckpt.get("train_loss_ema", float("inf"))),
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("AgriMatcher Training v2")
    p.add_argument("--pairs-file", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints_v2"))
    p.add_argument("--log-dir", type=Path, default=Path("runs/agrimatch_v2"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--teacher-grid-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=65)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--force-amp", action="store_true")
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--teacher-setting", type=str, default="precise")
    # DistillationLoss 参数
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta-coarse", type=float, default=1.0)
    p.add_argument("--beta-refine", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--eta-coarse", type=float, default=0.5)
    p.add_argument("--eta-refine", type=float, default=1.0)
    p.add_argument("--lambda-tv-coarse", type=float, default=0.01)
    p.add_argument("--lambda-tv-refine", type=float, default=0.05)
    p.add_argument("--conf-thresh-kl", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--val-interval", type=int, default=200)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp
    if device.type == "cuda" and use_amp and not args.force_amp:
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            print(f"[AMP] Auto-disabled (capability={major}.{minor}). Use --force-amp to override.")
            use_amp = False

    writer = SummaryWriter(log_dir=str(args.log_dir))

    cache_loader_kwargs = {
        "num_workers": args.num_workers, "collate_fn": cached_collate_fn, "pin_memory": True,
    }
    image_loader_kwargs = {
        "num_workers": args.num_workers, "collate_fn": multi_scale_collate_fn, "pin_memory": True,
    }
    if args.num_workers > 0:
        for kw in (cache_loader_kwargs, image_loader_kwargs):
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 2

    use_cache = args.cache_dir is not None and args.cache_dir.exists()
    if use_cache:
        print(f"[Dataset] 缓存模式: {args.cache_dir}")
        suffix = args.cache_dir.suffix.lower() if args.cache_dir.is_file() else ""
        if suffix in {".h5", ".hdf5"}:
            train_ds = BucketedH5TeacherDataset(str(args.cache_dir), args.val_ratio, args.seed, 'train')
            val_ds   = BucketedH5TeacherDataset(str(args.cache_dir), args.val_ratio, args.seed, 'val')
            train_sampler = BucketedBatchSampler(train_ds.bucket_to_indices, args.batch_size, shuffle=True,  seed=args.seed)
            val_sampler   = BucketedBatchSampler(val_ds.bucket_to_indices,   args.batch_size, shuffle=False, seed=args.seed)
            train_loader  = DataLoader(train_ds, batch_sampler=train_sampler, **cache_loader_kwargs)
            val_loader    = DataLoader(val_ds,   batch_sampler=val_sampler,   **cache_loader_kwargs)
        else:
            train_ds = CachedTeacherDataset(args.cache_dir, args.val_ratio, return_split='train')
            val_ds   = CachedTeacherDataset(args.cache_dir, args.val_ratio, return_split='val')
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **cache_loader_kwargs)
            val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **cache_loader_kwargs)
    else:
        if args.pairs_file is None:
            raise ValueError("必须提供 --pairs-file 或有效的 --cache-dir")
        print(f"[Dataset] 原始图像对模式: {args.pairs_file}")
        train_ds = MultiScaleDataset(args.pairs_file, args.val_ratio, return_split='train')
        val_ds   = MultiScaleDataset(args.pairs_file, args.val_ratio, return_split='val')
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **image_loader_kwargs)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **image_loader_kwargs)

    actual_teacher_dim = {'base': 80, 'precise': 100}[args.teacher_setting]
    student = AgriTPSStitcher(
        matcher_config={'d_model': args.d_model, 'teacher_dim': actual_teacher_dim,
                        'grid_size': args.teacher_grid_size},
        tps_config={'grid_size': 8, 'feat_channels': args.d_model}
    ).to(device)

    distill_loss_fn = DistillationLoss(
        alpha=args.alpha, beta_coarse=args.beta_coarse, beta_refine=args.beta_refine,
        gamma=args.gamma, eta_coarse=args.eta_coarse, eta_refine=args.eta_refine,
        lambda_tv_coarse=args.lambda_tv_coarse, lambda_tv_refine=args.lambda_tv_refine,
        conf_thresh_kl=args.conf_thresh_kl,
    ).to(device)
    geo_loss_fn = LocalGeometricConsistency()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_tracker = EMALossTracker(alpha=args.ema_alpha)

    start_epoch   = 1
    global_step   = 0
    best_val_loss = float("inf")
    current_stage: Optional[Dict] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler = None

    steps_per_epoch = (len(train_loader) + args.accum_steps - 1) // args.accum_steps
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"], strict=False)
        start_epoch = int(ckpt.get("epoch", 1)) + 1  # 从下一个 epoch 继续
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        _ema = float(ckpt.get("train_loss_ema", float("inf")))
        loss_tracker.ema_values['total'] = _ema

        resumed_stage = get_stage(start_epoch)
        apply_freeze_state(student, resumed_stage)
        optimizer, scheduler = build_optimizer_and_scheduler(
            student, resumed_stage, steps_per_epoch, args.weight_decay)

        if optimizer and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[Resume] Optimizer state incompatible, starting fresh: {e}")
        if scheduler and ckpt.get("scheduler"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[Resume] Scheduler state incompatible: {e}")
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        current_stage = resumed_stage
        print(f"Resumed from epoch={start_epoch}, step={global_step}, stage={resumed_stage['name']}")

    for epoch in range(start_epoch, args.epochs + 1):

        new_stage = get_stage(epoch)
        if new_stage is not current_stage:
            print(f"\n{'═'*60}")
            print(f"[Stage Transition] Epoch {epoch} → Stage: {new_stage['name']}")
            print(f"{'═'*60}")
            current_stage = new_stage

            # 更新冻结/解冻状态
            apply_freeze_state(student, current_stage)

            # 重建优化器和调度器
            optimizer, scheduler = update_optimizer_and_scheduler(
                student, current_stage, steps_per_epoch, args.weight_decay,
                optimizer=optimizer
            )

            # 打印当前参数组信息
            for g in optimizer.param_groups:
                n_params = sum(p.numel() for p in g['params'])
                print(f"  Param group '{g.get('name','?')}': lr={g['lr']:.2e}, params={n_params:,}")

        w = interpolate_loss_weights(current_stage, epoch)
        use_cycle_this_epoch = w.get("cycle", 0.0) > 0.01

        # 记录到 TensorBoard
        for k, v in w.items():
            writer.add_scalar(f"LossWeights/{k}", v, epoch)

        if use_cache and isinstance(getattr(train_loader, 'batch_sampler', None), BucketedBatchSampler):
            train_loader.batch_sampler.set_epoch(epoch)

        print(f"\n🚀 Epoch {epoch}/{args.epochs} | Stage: {current_stage['name']}")
        print(f"   Weights: {', '.join(f'{k}={v:.3f}' for k, v in w.items() if v > 0)}")

        student.train()
        # 保持冻结模块处于 eval 状态（防止 BN 被小 batch 破坏）
        if "backbone" in current_stage.get("freeze", []) or not current_stage.get("unfreeze_backbone", True):
            student.matcher.backbone.eval()
        if "aggregator" in current_stage.get("freeze", []):
            student.tps_estimator.eval()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}", dynamic_ncols=True)

        for i, batch in pbar:
            if use_cache:
                img_a = batch['img_a'].to(device, non_blocking=True)
                img_b = batch['img_b'].to(device, non_blocking=True)
            else:
                img_a_list = [x.to(device, non_blocking=True) for x in batch['img_a']]
                img_b_list = [x.to(device, non_blocking=True) for x in batch['img_b']]
                tgt_h, tgt_w = img_a_list[0].shape[-2:]
                B = len(img_a_list)
                img_a = torch.empty((B, 3, tgt_h, tgt_w), device=device, dtype=img_a_list[0].dtype)
                img_b = torch.empty((B, 3, tgt_h, tgt_w), device=device, dtype=img_b_list[0].dtype)
                for b_idx, (a, b) in enumerate(zip(img_a_list, img_b_list)):
                    if a.shape[-2:] != (tgt_h, tgt_w):
                        img_a[b_idx] = F.interpolate(a.unsqueeze(0), size=(tgt_h, tgt_w),
                                                      mode='bilinear', align_corners=False).squeeze(0)
                        img_b[b_idx] = F.interpolate(b.unsqueeze(0), size=(tgt_h, tgt_w),
                                                      mode='bilinear', align_corners=False).squeeze(0)
                    else:
                        img_a[b_idx], img_b[b_idx] = a, b

            if use_cache:
                t_out = {
                    'warp_AB': batch['t_warp_AB'].to(device, non_blocking=True),
                    'confidence_AB': batch['t_conf_AB'].to(device, non_blocking=True),
                    'confidence_is_prob': True,
                }
                t_feat_a = batch['t_feat_a'].to(device, non_blocking=True)
                t_feat_b = batch['t_feat_b'].to(device, non_blocking=True)
            else:
                raise RuntimeError("Online teacher inference is disabled. Use --cache-dir.")

            bundle = compute_loss_bundle(
                student=student, img_a=img_a, img_b=img_b,
                t_out=t_out, t_feat_a=t_feat_a, t_feat_b=t_feat_b,
                distill_loss_fn=distill_loss_fn, geo_loss_fn=geo_loss_fn,
                w=w, device=device, amp_enabled=use_amp, use_cycle=use_cycle_this_epoch,
            )
            stu_out    = bundle['stu_out']
            total_loss = bundle['total']

            loss_dict = {k: bundle[k] for k in ('distill', 'fold', 'photo', 'cycle', 'geo', 'total')}
            has_nan = False
            for k, v in loss_dict.items():
                if torch.is_tensor(v) and not torch.isfinite(v).all():
                    pbar.write(f"🚨 [NaN/Inf] Loss '{k}' invalid!")
                    has_nan = True
            if has_nan:
                pbar.write(f"⏭️ Skipping batch at step {global_step}")
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            loss_for_backward = total_loss / args.accum_steps
            scaler.scale(loss_for_backward).backward()

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

                # NaN 梯度守卫
                has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in student.parameters()
                )
                if has_nan_grad:
                    pbar.write(f"🛡️ [GradShield] NaN gradient at step {global_step}! Skipping.")
                else:
                    scaler.step(optimizer)
                    scheduler.step()

                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                loss_metrics = {
                    'total':    float(total_loss.detach()),
                    'distill':  float(bundle['distill'].detach()),
                    'fold':     float(bundle['fold'].detach()),
                    'photo':    float(bundle['photo'].detach() if torch.is_tensor(bundle['photo']) else bundle['photo']),
                    'cycle':    float(bundle['cycle'].detach() if torch.is_tensor(bundle['cycle']) else bundle['cycle']),
                    'geo':      float(bundle['geo'].detach()),
                    'tps_smooth': float(bundle['tps_smooth'].detach()),
                }
                smoothed = loss_tracker.update(loss_metrics)
                fold_ratio = float(bundle['fold_ratio'])
                pbar.set_postfix({
                    'Loss':   f"{smoothed['total_ema']:.4f}",
                    'Photo':  f"{smoothed['photo_ema']:.4f}",
                    'Fold':   f"{smoothed['fold_ema']:.4f}",
                    'FoldR':  f"{fold_ratio:.3f}",
                    'Smooth': f"{smoothed['tps_smooth_ema']:.4f}",
                })

                if global_step % args.log_interval == 0:
                    for k, v in smoothed.items():
                        writer.add_scalar(f"Train/{k}", v, global_step)
                    for j, g in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"LR/group_{j}_{g.get('name','')}", g['lr'], global_step)
                    with torch.no_grad():
                        visualize_results(img_a, img_b, stu_out, global_step, writer)

                if global_step % args.val_interval == 0:
                    val_losses = validate(
                        student=student, val_loader=val_loader,
                        distill_loss_fn=distill_loss_fn, geo_loss_fn=geo_loss_fn,
                        w=w, device=device, use_amp=use_amp,
                    )
                    val_total = val_losses['total']
                    pbar.write(f"📊 [val] step={global_step} | total={val_total:.4f} | "
                               f"photo={val_losses['photo']:.4f} | fold={val_losses['fold']:.4f}")
                    for k, v in val_losses.items():
                        writer.add_scalar(f"Val/{k}", v, global_step)
                    is_best = val_total < best_val_loss
                    if is_best:
                        best_val_loss = val_total
                    save_checkpoint(
                        args.save_dir, epoch, global_step, student, optimizer, scheduler,
                        scaler if use_amp else None, args,
                        loss_tracker.get_ema('total'), val_total, best_val_loss, is_best,
                    )

        pbar.write(f"[epoch {epoch}] Done. EMA={loss_tracker.get_ema('total'):.4f}")

    writer.close()


if __name__ == "__main__":
    main()