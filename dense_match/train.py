import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
import shutil
import platform
import pathlib
import io

# 修复跨平台反序列化问题 (Linux -> Windows)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A  # pyright: ignore[reportMissingImports]
from albumentations.pytorch import ToTensorV2  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
ROMA_SRC = REPO_ROOT / "RoMaV2" / "src"

project_root = REPO_ROOT.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dense_match.network import (
    AgriTPSStitcher, 
    DistillationLoss, 
    LocalGeometricConsistency,
    compute_photometric_loss, 
    compute_cycle_consistency_loss,
    safe_grid_sample, 
    ssim_map,
    make_grid,
    )

if str(ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(ROMA_SRC))
from romav2 import RoMaV2  # pyright: ignore[reportMissingImports]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ssim_loss(img1, img2, window_size=11):
    """计算 SSIM Loss Map"""
    C = img1.shape[1]
    window = torch.ones((C, 1, window_size, window_size), device=img1.device) / (window_size**2)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)
    
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2
    
    C1, C2 = 0.01**2, 0.03**2
    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return 1 - ssim_n / (ssim_d + 1e-8)

def compute_photometric_loss(img_a, img_b, dense_grid, confidence, alpha=0.85):
    """SSIM + L1 混合光度损失"""
    # 将 img_b 变换到 img_a 的坐标系
    warped_b = F.grid_sample(img_b, dense_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    l1_map = torch.abs(img_a - warped_b).mean(dim=1, keepdim=True)
    ssim_map = ssim_loss(img_a, warped_b)
    
    photo_loss = alpha * ssim_map + (1 - alpha) * l1_map
    # 使用置信度加权：confidence 可能来自 matcher 的 coarse/fine 分辨率，需要对齐到图像分辨率
    if isinstance(confidence, torch.Tensor):
        if confidence.ndim == 4 and confidence.shape[-1] == 1:
            # (B,H,W,1) -> (B,H,W)
            confidence = confidence[..., 0]
        if confidence.ndim == 4 and confidence.shape[1] == 1:
            # (B,1,H,W) -> (B,H,W)
            confidence = confidence[:, 0]
        if confidence.ndim != 3:
            raise ValueError(f"confidence must be (B,H,W) (or broadcastable variants), got {confidence.shape}")
        if confidence.shape[-2:] != photo_loss.shape[-2:]:
            confidence = F.interpolate(
                confidence.unsqueeze(1),
                size=photo_loss.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
    # (B,C,H,W) * (B,1,H,W) broadcast
    return (photo_loss * confidence.unsqueeze(1)).sum() / (confidence.sum() + 1e-6)


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
    
    color = (0, 0, 255) # 红色 (BGR 格式)
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

def plot_grid_to_tensorboard(dense_grid, img_shape, step, writer, tag="Train/TPS_Grid"):
    """
    将 TPS 产生的密集网格绘制成线图，反映变形程度
    dense_grid: (1, H, W, 2) 范围 [-1, 1]
    """
    B, H, W, _ = dense_grid.shape
    grid = dense_grid[0].detach().cpu().numpy() # 取第一个样本 (H, W, 2)
    
    # 创建 matplotlib 画布
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 为了清晰，每隔 16 个像素画一根线
    step_size = max(H // 32, 1)
    
    # 绘制水平线
    for i in range(0, H, step_size):
        ax.plot(grid[i, :, 0], grid[i, :, 1], color='blue', linewidth=0.5)
    # 绘制垂直线
    for j in range(0, W, step_size):
        ax.plot(grid[:, j, 0], grid[:, j, 1], color='blue', linewidth=0.5)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1) # 反转 Y 轴匹配图片坐标系
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 将 plot 转换为 Tensor
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    image = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    image = torch.from_numpy(image).permute(2, 0, 1) # (3, H, W)
    
    writer.add_image(tag, image, step)

def create_checkerboard(img1, img2, num_squares=8):
    """
    创建棋盘格对比图，用于检查边缘缝合是否平滑连续
    """
    B, C, H, W = img1.shape
    grid_h = H // num_squares
    grid_w = W // num_squares
    
    # 创建掩码
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    mask = ((y // grid_h) % 2 == (x // grid_w) % 2).float().to(img1.device)
    
    return img1 * mask + img2 * (1 - mask)

def colorize_heatmap(tensor, cmap='jet'):
    """将 [1, H, W] 转换为彩色热力图 [3, H, W]"""
    x = tensor.detach().cpu().numpy()[0]
    color_map = cm.get_cmap(cmap)
    colorized = color_map(x)[:, :, :3] # 取 RGB
    return torch.from_numpy(colorized).permute(2, 0, 1).float()

def visualize_results(img_a, img_b, stu_out, step, writer, phase="Train"):
    """
    全方位监控面板：对比原图、变换图、缝合效果及几何稳定性
    """
    matcher_out = stu_out['matcher_out']
    dense_grid = stu_out['dense_grid']
    H, W = img_a.shape[-2:]

    def denorm(x): 
        # 恢复归一化图像用于显示 (假设使用了 ImageNet 标准差)
        return (x[0:1].detach().cpu() * 0.225 + 0.45).clamp(0, 1)

    with torch.no_grad():
        # 1. 基础图像准备
        img_a_dn = denorm(img_a)      # [1, 3, H, W]
        img_b_dn = denorm(img_b)      # [1, 3, H, W]
        # 对 img_b 进行 TPS 变换
        warped_b = F.grid_sample(img_b, dense_grid, padding_mode='zeros', align_corners=False)
        warped_b_dn = denorm(warped_b)

        # 2. 效果评估图
        # A. 混合缝合图 (Alpha Blending) - 观察是否有重影 (Ghosting)
        blended_vis = (img_a_dn * 0.5 + warped_b_dn * 0.5)

        # B. 棋盘格拼接 (Checkerboard) - 检查边缘纹理连续性
        grid_h, grid_w = H // 8, W // 8
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        m = ((yy // grid_h) % 2 == (xx // grid_w) % 2).float().to(img_a_dn.device)
        checker_vis = img_a_dn * m + warped_b_dn * (1 - m)

        # C. 误差图 (L1 Difference) - 越黑代表对齐越准
        diff_vis = (img_a_dn - warped_b_dn).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # 3. 几何信号图
        # 彩色置信度 (Jet: 红色=极高，蓝色=极低)
        conf = matcher_out['confidence_AB'][0:1].detach().cpu()
        conf_up = F.interpolate(conf.unsqueeze(1), size=(H, W), mode='bilinear')[0]
        conf_color = colorize_heatmap(conf_up, cmap='jet')

        # TPS 网格图
        grid_vis = plot_grid_to_image(dense_grid, (H, W)) # 之前写的绘图函数
        # 网格叠加在混合图上，看形变逻辑是否符合图像特征
        overlay_vis = blended_vis[0] * 0.7 + grid_vis * 0.3

        # 4. 拼接看板 (2 行 x 4 列)
        # 每张图都是 [3, H, W]
        row1 = torch.cat([img_a_dn[0], img_b_dn[0], warped_b_dn[0], blended_vis[0]], dim=2)
        row2 = torch.cat([checker_vis[0], diff_vis[0], conf_color, overlay_vis], dim=2)
        
        dashboard = torch.cat([row1, row2], dim=1) # [3, H*2, W*4]
        
        writer.add_image(f"{phase}/Full_Stitching_Dashboard", dashboard, step)

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


# 多尺度数据集
class MultiScaleDataset(Dataset):
    """
    多尺度数据集，支持 train/val 划分
    """
    
    def __init__(
        self,
        pairs_file: Path,
        is_train: bool = True,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        return_split: str = 'train',
    ):
        self.is_train = is_train
        self.return_split = return_split
        
        all_items: List[Tuple[Path, Path]] = []
        base_dir = pairs_file.parent
        
        for line in pairs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 2:
                continue
            a = Path(parts[0]) if Path(parts[0]).is_absolute() else (base_dir / parts[0]).resolve()
            b = Path(parts[1]) if Path(parts[1]).is_absolute() else (base_dir / parts[1]).resolve()
            all_items.append((a, b))
        
        rng = random.Random(split_seed)
        indices = list(range(len(all_items)))
        rng.shuffle(indices)
        
        val_size = int(len(all_items) * val_ratio)
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])
        
        if return_split == 'train':
            self.items = [all_items[i] for i in sorted(train_indices)]
        elif return_split == 'val':
            self.items = [all_items[i] for i in sorted(val_indices)]
        else:
            self.items = all_items
        
        print(f"[Dataset] split={return_split}, total={len(all_items)}, using={len(self.items)}")
        
        # 尺寸池
        self.pool_lowres = [(256, 256), (384, 384), (512, 512), (384, 512), (512, 384)]
        self.pool_highres = [(512, 768), (768, 512), (512, 512), (640, 480), (480, 640)]
        
        # 预构建 transforms
        self.transforms_lowres = {
            (h, w): self._build_transform(h, w, scale=(0.6, 1.0))
            for h, w in self.pool_lowres
        }
        self.transforms_highres = {
            (h, w): self._build_transform(h, w, scale=(0.4, 1.0))
            for h, w in self.pool_highres
        }
        
        # 验证集固定尺寸
        self.val_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'image_b': 'image'})
    
    def _build_transform(self, target_h: int, target_w: int, scale: Tuple[float, float]) -> A.Compose:
        if not self.is_train:
            return A.Compose([
                A.Resize(target_h, target_w),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'image_b': 'image'})
        
        return A.Compose([
            A.RandomResizedCrop(
                size=(target_h, target_w),
                scale=scale,
                ratio=(0.75, 1.33),
                p=1.0
            ),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'image_b': 'image'})
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path_a, path_b = self.items[idx]
        
        img_a = cv2.cvtColor(cv2.imread(str(path_a)), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(str(path_b)), cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img_a.shape[:2]
        
        # 验证集固定尺寸
        if self.return_split == 'val':
            transformed = self.val_transform(image=img_a, image_b=img_b)
            return {"img_a": transformed["image"], "img_b": transformed["image_b"]}
        
        # 训练集动态选择
        if max(orig_h, orig_w) > 1000:
            target_h, target_w = random.choice(self.pool_highres)
            transform = self.transforms_highres[(target_h, target_w)]
        else:
            target_h, target_w = random.choice(self.pool_lowres)
            transform = self.transforms_lowres[(target_h, target_w)]
        
        transformed = transform(image=img_a, image_b=img_b)
        return {"img_a": transformed["image"], "img_b": transformed["image_b"]}


def multi_scale_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """多尺度 collate：同一 batch 内 resize 到第一个样本的尺寸"""
    if len(batch) == 0:
        return {}
    
    first_img = batch[0]['img_a']
    target_h, target_w = first_img.shape[1], first_img.shape[2]
    
    img_a_list = []
    img_b_list = []
    
    for item in batch:
        img_a = item['img_a']
        img_b = item['img_b']
        
        _, h, w = img_a.shape
        
        if h != target_h or w != target_w:
            img_a = F.interpolate(
                img_a.unsqueeze(0), size=(target_h, target_w),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            img_b = F.interpolate(
                img_b.unsqueeze(0), size=(target_h, target_w),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        
        img_a_list.append(img_a)
        img_b_list.append(img_b)
    
    return {
        'img_a': torch.stack(img_a_list, dim=0),
        'img_b': torch.stack(img_b_list, dim=0),
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
    print(f"The last weight file update ! val loss: {val_loss:.6f}")
    
    if is_best:
        shutil.copyfile(last_path, save_dir / "best.pt")
        print(f" ✅ New best val_loss: {val_loss:.6f}")
    
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


@torch.no_grad()
def validate(student, teacher, val_loader, loss_fn, device, teacher_grid_size, max_batches=20):
    student.eval()
    total_ssim = 0
    total_fold = 0
    total_distill_loss = 0
    count = 0
    
    for i, batch in enumerate(val_loader):
        if i >= max_batches: break
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)

        # 获取 Teacher 信号用于评估
        img_a_lr, img_b_lr, img_a_hr, img_b_hr = make_teacher_inputs(img_a, img_b, teacher)
        t_out = teacher(img_a_lr, img_b_lr, img_a_hr, img_b_hr)
        t_feat_a, t_feat_b = extract_teacher_features_ds(teacher, img_a_lr, img_b_lr, teacher_grid_size)

        stu_out = student(img_a, img_b)
        
        # 计算蒸馏 Loss
        d_loss = loss_fn(
            stu_output=stu_out['matcher_out'],
            teacher_warp=t_out["warp_AB"],
            teacher_conf=teacher_overlap_map(t_out["confidence_AB"]),
            teacher_feat_A=t_feat_a,
            teacher_feat_B=t_feat_b
        )
        total_distill_loss += d_loss['total'].item()

        # 计算 SSIM
        warped_b = F.grid_sample(img_b, stu_out['dense_grid'], align_corners=False, padding_mode='zeros')
        s_map = ssim_map(img_a, warped_b) 
        total_ssim += s_map.mean().item()
        total_fold += stu_out['tps_out'].get('fold_ratio', 0)
        count += 1
    
    student.train()
    return {
        "total": total_distill_loss / count, # 这样 best_val_loss 才有意义
        "ssim": total_ssim / count,
        "fold_ratio": total_fold / count
    }


# 参数解析
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("AgriMatcher Multi-Scale Training")
    
    p.add_argument("--pairs-file", type=Path, required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints_multiscale"))
    p.add_argument("--log-dir", type=Path, default=Path("runs/agrimatch_multiscale"))
    p.add_argument("--resume", type=Path, default=None)
    
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--teacher-dim", type=int, default=None)
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
    p.add_argument("--alpha", type=float, default=0.2)
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
    if epoch <= 5:
        return {
            "phase": "DISTILL",
            "weights": {"distill": 1.0, "fold": 0.0, "photo": 0.0, "cycle": 0.0, "geo": 0.5},
            "train_aggregator": False,
            "train_matcher": True
        }
    elif epoch <= 15:
        return {
            "phase": "TPS_REFINE",
            # 开启 TPS，但折叠惩罚要给足，防止形变失控
            "weights": {"distill": 0.5, "fold": 1.5, "photo": 0.5, "cycle": 0.0, "geo": 1.0},
            "train_aggregator": True,
            "train_matcher": False
        }
    else:
        return {
            "phase": "SELF_SUPERVISED",
            # 自监督为主，降低蒸馏权重
            "weights": {"distill": 0.2, "fold": 3.0, "photo": 2.0, "cycle": 1.0, "geo": 0.8},
            "train_aggregator": True,
            "train_matcher": True
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
            print(f"[AMP] Auto-disabled AMP for stability on this GPU (capability={major}.{minor}). Use --force-amp to override.")
            use_amp = False
    writer = SummaryWriter(log_dir=str(args.log_dir))
    

    # 加载数据集
    train_dataset = MultiScaleDataset(args.pairs_file, is_train=True, val_ratio=args.val_ratio, return_split='train')
    val_dataset = MultiScaleDataset(args.pairs_file, is_train=False, val_ratio=args.val_ratio, return_split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=multi_scale_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=multi_scale_collate_fn)

    # 初始化 Teacher (RoMaV2)
    teacher = RoMaV2(RoMaV2.Cfg(setting=args.teacher_setting)).to(device).eval()
    
    # 初始化 Student (AgriTPSStitcher)
    with torch.no_grad():
        dummy_img = torch.zeros(1, 3, 512, 512, device=device)
        img_a_lr, _, _, _ = make_teacher_inputs(dummy_img, dummy_img, teacher)
        # 提取特征并检查 shape
        feat_a_flat, _ = extract_teacher_features_ds(teacher, img_a_lr, img_a_lr, args.teacher_grid_size)
        actual_teacher_dim = feat_a_flat.shape[-1]
        print(f"Detected Teacher Feature Dimension: {actual_teacher_dim}")

    student = AgriTPSStitcher(
        matcher_config={
            'd_model': args.d_model, 
            'teacher_dim': actual_teacher_dim,
            'grid_size': args.teacher_grid_size
        }, 
        tps_config={
            'grid_size': 10, 
            'feat_channels': args.d_model
        }
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
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

    for epoch in range(start_epoch, args.epochs + 1):
        cfg = get_phase_config(epoch)
        print(f"\n🚀 Entering Phase: {cfg['phase']} (Epoch {epoch})")
        
        for name, param in student.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
            elif "tps_estimator.aggregator" in name:
                param.requires_grad = cfg['train_aggregator']
            elif "matcher" in name:
                param.requires_grad = cfg['train_matcher']
            else:
                param.requires_grad = True

        student.train()
        student.matcher.backbone.eval()

        for i, batch in enumerate(train_loader):
            img_a = batch["img_a"].to(device, non_blocking=True)
            img_b = batch["img_b"].to(device, non_blocking=True)

            # input sanity
            if not torch.isfinite(img_a).all().item():
                print(f"[NaN/Inf] input/img_a: {_finite_stats(img_a)}")
                continue
            if not torch.isfinite(img_b).all().item():
                print(f"[NaN/Inf] input/img_b: {_finite_stats(img_b)}")
                continue

            with torch.no_grad():
                img_a_lr, img_b_lr, img_a_hr, img_b_hr = make_teacher_inputs(img_a, img_b, teacher)
                t_out = teacher(img_a_lr, img_b_lr, img_a_hr, img_b_hr)
                t_feat_a, t_feat_b = extract_teacher_features_ds(teacher, img_a_lr, img_b_lr, args.teacher_grid_size)

            # loss 计算
            def _forward_losses(amp_enabled: bool):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    stu_out = student(img_a, img_b)  # 返回 dense_grid, delta_cp, matcher_out, tps_out
                    stu_output=stu_out["matcher_out"]
                    warp_refine, conf_refine = stu_output['warp_AB'], stu_output['confidence_AB']
                    B, H, W, _ = warp_refine.shape
                    grid = make_grid(B, H, W, warp_refine.device, warp_refine.dtype)
                    current_flow = warp_refine - grid
                    d_loss = distill_loss_fn(
                        stu_output=stu_output,
                        teacher_warp=t_out["warp_AB"],
                        teacher_conf=teacher_overlap_map(t_out["confidence_AB"]),
                        teacher_feat_A=t_feat_a,
                        teacher_feat_B=t_feat_b,
                    )
                    geo_loss = geo_loss_fn(current_flow, conf_refine)
                    f_loss = stu_out["tps_out"].get("fold_loss", torch.tensor(0.0, device=device))
                    p_loss = compute_photometric_loss(
                        img_a, img_b, stu_out["dense_grid"], stu_out["matcher_out"]["confidence_AB"]
                    )

                    if cfg["phase"] == "SELF_SUPERVISED":
                        stu_out_ba = student(img_b, img_a)
                        c_loss = compute_cycle_consistency_loss(stu_out, stu_out_ba)
                    else:
                        c_loss = torch.tensor(0.0, device=device)

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

            # 反向传播
            loss_for_backward = total_loss / args.accum_steps
            total_loss_scalar = float(total_loss.detach().item())
            distill_scalar = float(d_loss['total'].detach().item())
            fold_scalar = float(f_loss.detach().item()) if torch.is_tensor(f_loss) else float(f_loss)
            photo_scalar = float(p_loss.detach().item())
            cycle_scalar = float(c_loss.detach().item()) if torch.is_tensor(c_loss) else float(c_loss)

            # NaN/Inf guard: locate which term first becomes non-finite
            ok = True
            ok &= _check_finite("loss/distill_total", d_loss["total"])
            ok &= _check_finite("loss/fold", f_loss if torch.is_tensor(f_loss) else torch.tensor(f_loss, device=device))
            ok &= _check_finite("loss/photo", p_loss)
            ok &= _check_finite("loss/cycle", c_loss if torch.is_tensor(c_loss) else torch.tensor(c_loss, device=device))
            ok &= _check_finite("loss/total", total_loss)
            if isinstance(stu_out, dict):
                if "dense_grid" in stu_out and torch.is_tensor(stu_out["dense_grid"]):
                    ok &= _check_finite("stu/dense_grid", stu_out["dense_grid"])
                mo = stu_out.get("matcher_out")
                if isinstance(mo, dict) and "confidence_AB" in mo and torch.is_tensor(mo["confidence_AB"]):
                    ok &= _check_finite("stu/confidence_AB", mo["confidence_AB"])
                to = stu_out.get("tps_out")
                if isinstance(to, dict) and "delta_cp" in to and torch.is_tensor(to["delta_cp"]):
                    ok &= _check_finite("stu/delta_cp", to["delta_cp"])

            if not ok:
                # AMP fallback: rerun this batch in fp32 once; if it becomes finite, disable AMP for stability.
                if use_amp:
                    print("[NaN/Inf] Detected under AMP. Retrying this batch with AMP disabled...")
                    stu_out, d_loss, f_loss, p_loss, c_loss, total_loss = _forward_losses(False)
                    ok2 = True
                    ok2 &= _check_finite("fp32/loss/total", total_loss)
                    if ok2:
                        print("[AMP] Disabling AMP due to instability (continuing in fp32).")
                        use_amp = False
                        scaler = torch.amp.GradScaler("cuda", enabled=False)
                        # continue to backward in fp32 below
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                else:
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            scaler.scale(loss_for_backward).backward()

            if (i + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # 记录 Log
                if global_step % args.log_interval == 0:
                    loss_metrics = {
                        'total': total_loss_scalar,
                        'distill': distill_scalar,
                        'fold': fold_scalar,
                        'photo': photo_scalar,
                        'cycle': cycle_scalar,
                        'geo': geo_loss.item()
                    }
                    smoothed = loss_tracker.update(loss_metrics)
                    print(f"""Epoch {epoch} | Step {global_step} | Loss: {smoothed['total_ema']:.4f} | Photo: {smoothed['photo_ema']:.4f} | Fold: {smoothed['fold_ema']:.4f} | Geo: {smoothed['geo_ema']}""")
                    for k, v in smoothed.items():
                        writer.add_scalar(f"Train/{k}", v, global_step)
                    
                    with torch.no_grad():
                        visualize_results(img_a, img_b, stu_out, global_step, writer)
                
                # Validate
                if global_step % args.val_interval == 0:
                    val_losses = validate(
                        student=student,
                        teacher=teacher,
                        val_loader=val_loader,
                        loss_fn=distill_loss_fn,
                        device=device,
                        teacher_grid_size=args.teacher_grid_size,
                    )
                    val_total = val_losses['total']
                    
                    print(f"📊 [val] step={global_step} | total={val_total:.4f}")
                    writer.add_scalar("Val/Total", val_total, global_step)
                    
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
        
        print(f"[epoch {epoch}] Done. EMA={loss_tracker.get_ema('total'):.4f}")

    writer.close()

def compute_cycle_consistency_loss(stu_out_ab, stu_out_ba):
    """计算 A->B->A 的坐标回环误差"""
    grid_ab = stu_out_ab['dense_grid']
    grid_ba = stu_out_ba['dense_grid']
    conf_a = stu_out_ab['matcher_out']['confidence_AB']
    
    B, H, W, _ = grid_ab.shape
    # 构建 Identity 网格
    identity = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, H, device=grid_ab.device),
        torch.linspace(-1, 1, W, device=grid_ab.device),
        indexing='ij'
    )[::-1], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    # 采样回环坐标
    back_to_a = F.grid_sample(grid_ba.permute(0, 3, 1, 2), grid_ab, align_corners=False, padding_mode='zeros').permute(0, 2, 3, 1)
    cycle_error = torch.norm(back_to_a - identity, dim=-1)
    
    return (cycle_error * conf_a).mean()


if __name__ == "__main__":
    main()