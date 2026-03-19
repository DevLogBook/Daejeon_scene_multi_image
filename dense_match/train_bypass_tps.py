"""
训练策略：
  Phase1 (epoch 1 ~ photo_start_epoch-1):
    只用 flow_consistency + smooth + fold 损失
    快速建立 TPS 估计能力，避免早期 warped_B 太差导致光度loss干扰
  Phase2 (epoch photo_start_epoch ~ end):
    加入光度损失，端到端优化视觉质量
"""

import argparse
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import platform
import pathlib

# 修复跨平台反序列化问题 (Linux -> Windows)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
# 路径配置
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
project_root = REPO_ROOT.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dense_match.network import AgriMatcher, TPSGridGenerator  # type: ignore
from flow_to_tps import BypassTPSEstimator    
from dense_match.train import ImagePairDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb_tensor(path: Path, size: int) -> torch.Tensor:
    """加载图片 → [0,1] float tensor，保持比例pad到正方形"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w >= h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_w, new_h = int(w * size / h), size
    img = TF.resize(img, [new_h, new_w], antialias=True)
    ten = TF.to_tensor(img)
    _, ch, cw = ten.shape
    return F.pad(ten, (0, size - cw, 0, size - ch), value=0.0)


# 损失函数

class BypassTPSLoss(nn.Module):
    """
    BypassTPSEstimator 训练损失。

    组成：
    1. flow_consistency：TPS密集场与AgriMatcher dense warp的Huber距离（主监督）
    2. photometric：TPS变换后img_B与img_A的Charbonnier光度误差（自监督）
    3. fold：折叠惩罚（来自BypassTPSEstimator输出）
    4. smooth：控制点偏移的二阶差分平滑
    5. coverage：低覆盖度控制点的正则惩罚

    两阶段权重：
    - Phase1（early）：flow主导，无photo
    - Phase2（late）：flow + photo
    """

    def __init__(
        self,
        lambda_flow:     float = 2.0,
        lambda_photo:    float = 1.0,
        lambda_fold:     float = 5.0,
        lambda_smooth:   float = 0.5,
        lambda_cov:      float = 0.1,
        photo_start_epoch: int = 5,
        huber_delta:     float = 0.1,
        charb_eps:       float = 1e-3,
    ):
        super().__init__()
        self.lambda_flow      = lambda_flow
        self.lambda_photo     = lambda_photo
        self.lambda_fold      = lambda_fold
        self.lambda_smooth    = lambda_smooth
        self.lambda_cov       = lambda_cov
        self.photo_start_epoch = photo_start_epoch
        self.huber_delta      = huber_delta
        self.charb_eps        = charb_eps


    def flow_consistency_loss(
        self,
        tps_field:   torch.Tensor,   # (B,H,W,2) TPS生成的密集场
        target_warp: torch.Tensor,   # (B,H,W,2) AgriMatcher warp（伪标签）
        confidence:  torch.Tensor,   # (B,H,W)  置信度权重
    ) -> torch.Tensor:
        """
        核心监督：TPS密集场应与AgriMatcher输出一致。
        置信度加权Huber loss：高置信区域贡献更大梯度。
        """
        # 尺寸对齐（tps_field通常是256x256，target_warp是64x64）
        H, W = target_warp.shape[1:3]
        if tps_field.shape[1:3] != (H, W):
            tps_aligned = F.interpolate(
                tps_field.permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            tps_aligned = tps_field

        err = F.huber_loss(
            tps_aligned, target_warp.detach(),
            reduction="none", delta=self.huber_delta,
        ).sum(-1)  # (B,H,W)

        w = confidence.detach().clamp(0.0, 1.0)
        return (err * w).sum() / (w.sum() + 1e-6)

    def photometric_loss(
        self,
        img_A:      torch.Tensor,   # (B, 3, H, W)   全图分辨率
        warped_B:   torch.Tensor,   # (B, 3, H, W)
        mask_B:     torch.Tensor,   # (B, 1, H, W)
        confidence: torch.Tensor,   # (B, Hf, Wf)    AgriMatcher输出，64×64
    ) -> torch.Tensor:
        """Charbonnier光度损失，置信度加权"""
        H, W = img_A.shape[-2:]
        eps  = self.charb_eps

        diff  = img_A - warped_B
        charb = torch.sqrt(diff ** 2 + eps ** 2).mean(1, keepdim=True)  # (B,1,H,W)

        conf_up = F.interpolate(
            confidence.unsqueeze(1).detach(),   # (B,1,Hf,Wf)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )                                        # (B,1,H,W)

        w = mask_B * conf_up                     # (B,1,H,W)
        return (charb * w).sum() / (w.sum() + 1e-6)

    def smoothness_loss(self, delta_cp: torch.Tensor) -> torch.Tensor:
        """
        控制点偏移的二阶差分平滑（鼓励线性变化，符合TPS物理特性）。
        比一阶TV更适合TPS的光滑先验。
        """
        d2x = delta_cp[:, :, :, 2:] - 2*delta_cp[:, :, :, 1:-1] + delta_cp[:, :, :, :-2]
        d2y = delta_cp[:, :, 2:, :] - 2*delta_cp[:, :, 1:-1, :] + delta_cp[:, :, :-2, :]
        return d2x.pow(2).mean() + d2y.pow(2).mean()

    def coverage_regularization(self, coverage: torch.Tensor) -> torch.Tensor:
        """
        惩罚低覆盖度控制点（这些点的投票结果不可信）。
        coverage < 0.3 的区域施加轻微惩罚。
        """
        return F.relu(0.3 - coverage).mean()


    def forward(
        self,
        tps_field:    torch.Tensor,   # (B,H,W,2)
        delta_cp:     torch.Tensor,   # (B,2,gs,gs)
        coverage:     torch.Tensor,   # (B,gs,gs)
        fold_loss:    torch.Tensor,   # scalar（来自BypassTPSEstimator）
        img_A:        torch.Tensor,   # (B,3,H,W)
        warped_B:     torch.Tensor,   # (B,3,H,W)
        mask_B:       torch.Tensor,   # (B,1,H,W)
        target_warp:  torch.Tensor,   # (B,Hf,Wf,2)
        confidence:   torch.Tensor,   # (B,Hf,Wf)
        current_epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:

        loss_flow   = self.flow_consistency_loss(tps_field, target_warp, confidence)
        loss_smooth = self.smoothness_loss(delta_cp)
        loss_cov    = self.coverage_regularization(coverage)

        # 光度损失从第 photo_start_epoch 轮开始加入
        if current_epoch >= self.photo_start_epoch:
            loss_photo = self.photometric_loss(img_A, warped_B, mask_B, confidence)
        else:
            loss_photo = torch.zeros_like(loss_flow)

        total = (
            self.lambda_flow   * loss_flow   +
            self.lambda_photo  * loss_photo  +
            self.lambda_fold   * fold_loss   +
            self.lambda_smooth * loss_smooth +
            self.lambda_cov    * loss_cov
        )

        return {
            "total":        total,
            "loss_flow":    loss_flow.detach().item(),
            "loss_photo":   loss_photo.detach().item(),
            "loss_fold":    fold_loss.detach().item(),
            "loss_smooth":  loss_smooth.detach().item(),
            "loss_cov":     loss_cov.detach().item(),
        }


# Checkpoint 工具

def save_checkpoint(
    save_dir:   Path,
    epoch:      int,
    step:       int,
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    scaler:     Optional[torch.amp.GradScaler],
    args:       argparse.Namespace,
    best_loss:  float,
    is_best:    bool = False,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":     epoch,
        "step":      step,
        "model":     model.state_dict(),   # key="model" 与distill脚本的"student"区分
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "args":      vars(args),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    last_path = save_dir / "last.pt"
    torch.save(payload, last_path)

    if is_best:
        best_path = save_dir / "best.pt"
        shutil.copyfile(last_path, best_path)

    return last_path


def load_checkpoint(
    ckpt_path:  Path,
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    scaler:     Optional[torch.amp.GradScaler],
    device:     torch.device,
) -> Tuple[int, int, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return (
        int(ckpt.get("epoch", 0))+1,
        int(ckpt.get("step", 0)),
        float(ckpt.get("best_loss", float("inf"))),
    )


# 可视化

def log_visualizations(
    writer:      SummaryWriter,
    global_step: int,
    img_a:       torch.Tensor,   # (B,3,H,W)
    img_b:       torch.Tensor,
    warped_b:    torch.Tensor,
    coverage:    torch.Tensor,   # (B,gs,gs)
    gate:        torch.Tensor,   # (B,1,gs,gs)
    delta_cp:    torch.Tensor,   # (B,2,gs,gs)
    max_display: int = 2,
) -> None:
    with torch.no_grad():
        bsz  = min(img_a.shape[0], max_display)
        cmap = plt.get_cmap("viridis")

        writer.add_images("TPS/1_ImgA",    img_a[:bsz],              global_step)
        writer.add_images("TPS/2_ImgB",    img_b[:bsz],              global_step)
        writer.add_images("TPS/3_WarpedB", warped_b[:bsz].clamp(0,1), global_step)

        # 差异图
        diff_rgb = (img_a[:bsz] - warped_b[:bsz].clamp(0,1)).abs()
        writer.add_images("TPS/4_PhotoDiff", diff_rgb, global_step)

        # 覆盖度图（伪彩色）
        cov_np  = coverage[:bsz].cpu().float().numpy()
        cov_rgb = torch.from_numpy(
            np.stack([cmap(c)[..., :3] for c in cov_np])
        ).permute(0, 3, 1, 2).float()
        writer.add_images("TPS/5_Coverage", cov_rgb, global_step)

        # 门控图（伪彩色）
        gate_np  = gate[:bsz, 0].cpu().float().numpy()
        gate_rgb = torch.from_numpy(
            np.stack([cmap(g)[..., :3] for g in gate_np])
        ).permute(0, 3, 1, 2).float()
        writer.add_images("TPS/6_Gate", gate_rgb, global_step)

        # 控制点偏移幅度
        mag      = delta_cp[:bsz].norm(dim=1, keepdim=True)
        mag_norm = mag / (mag.amax(dim=(1,2,3), keepdim=True) + 1e-6)
        writer.add_images("TPS/7_DeltaMag", mag_norm.expand(-1,3,-1,-1), global_step)


# AgriMatcher 推理辅助（冻结）

@torch.no_grad()
def run_matcher(
    matcher: nn.Module,
    img_a:   torch.Tensor,   # (B,3,H,W)
    img_b:   torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    对已冻结的 AgriMatcher 做前向推理。
    返回训练 BypassTPSEstimator 需要的所有字段：
      warp_AB, confidence_AB, warp_AB_coarse, sim_matrix, feat_A_64, feat_B_64
    """
    return matcher(img_a, img_b)


@torch.no_grad()
def warp_image(
    img_b:     torch.Tensor,   # (B,3,H,W)
    tps_field: torch.Tensor,   # (B,H,W,2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 img_b 应用 TPS 形变场。
    返回: warped_b (B,3,H,W), mask_b (B,1,H,W)
    """
    warped = F.grid_sample(
        img_b, tps_field,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    ones   = torch.ones_like(img_b[:, :1])
    mask   = F.grid_sample(
        ones, tps_field,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return warped, (mask > 0.5).float()


# 验证

@torch.no_grad()
def validate(
    val_loader:    DataLoader,
    matcher:       nn.Module,
    bypass_tps:    nn.Module,
    tps_gen:       nn.Module,
    loss_fn:       BypassTPSLoss,
    device:        torch.device,
    use_amp:       bool,
    current_epoch: int,
) -> Dict[str, float]:
    bypass_tps.eval()
    accum = {k: 0.0 for k in
             ["total", "loss_flow", "loss_photo", "loss_fold",
              "loss_smooth", "loss_cov", "fold_ratio"]}
    count = 0

    for batch in val_loader:
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)

        # AgriMatcher 推理
        m_out = run_matcher(matcher, img_a, img_b)

        with torch.autocast(device_type=device.type,
                            dtype=torch.float16, enabled=use_amp):
            tps_out = bypass_tps(
                warp_AB        = m_out["warp_AB"],
                confidence     = m_out["confidence_AB"],
                feat_A         = m_out["feat_A_64"],
                feat_B         = m_out["feat_B_64"],
                sim_matrix     = m_out.get("sim_matrix"),
                warp_AB_coarse = m_out.get("warp_AB_coarse"),
                dense_field    = None,   # 验证时不做密集折叠
            )
            delta_cp  = tps_out["delta_cp"]
            tps_field = tps_gen(delta_cp)

            warped_b, mask_b = warp_image(img_b, tps_field)

            loss_dict = loss_fn(
                tps_field    = tps_field,
                delta_cp     = delta_cp,
                coverage     = tps_out["coverage"],
                fold_loss    = tps_out["fold_loss"],
                img_A        = img_a,
                warped_B     = warped_b,
                mask_B       = mask_b,
                target_warp  = m_out["warp_AB"],
                confidence   = m_out["confidence_AB"],
                current_epoch = current_epoch,
            )

        for k in accum:
            if k == "fold_ratio":
                accum[k] += tps_out["fold_ratio"]
            elif k in loss_dict:
                v = loss_dict[k]
                accum[k] += v.item() if isinstance(v, torch.Tensor) else v

        count += 1

    bypass_tps.train()
    n = max(count, 1)
    return {k: v / n for k, v in accum.items()}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("BypassTPSEstimator Trainer")

    # 数据
    p.add_argument("--pairs-file",  type=Path, required=True,
                   help="图片对列表文件（每行: path_a path_b）")
    p.add_argument("--image-size",  type=int,  default=256)
    p.add_argument("--val-split",   type=float, default=0.05,
                   help="验证集比例")
    p.add_argument("--augment",     action="store_true", default=True)
    p.add_argument("--no-augment",  dest="augment", action="store_false")

    # 模型路径
    p.add_argument("--matcher-ckpt", type=Path, required=True,
                   help="已训练好的 AgriMatcher 权重（distill脚本保存的 best.pt）")
    p.add_argument("--save-dir",    type=Path, default=Path("checkpoints_bypass_tps"))
    p.add_argument("--log-dir",     type=Path, default=Path("runs/bypass_tps"))
    p.add_argument("--resume",      type=Path, default=None,
                   help="从 bypass_tps checkpoint 断点续训")

    # 训练超参
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--accum-steps",   type=int,   default=1,
                   help="梯度累积步数（显存不足时设为2/4）")

    # 模型结构
    p.add_argument("--grid-size",    type=int,   default=10,
                   help="TPS控制点网格边长（grid_size² 个控制点）")
    p.add_argument("--d-model",      type=int,   default=128,
                   help="AgriMatcher的d_model（决定feat_A/B通道数）")
    p.add_argument("--teacher-dim",  type=int,   default=256,
                   help="AgriMatcher teacher_dim（加载权重用）")
    p.add_argument("--hidden-ch",    type=int,   default=48)
    p.add_argument("--sigma-scale",  type=float, default=0.7)
    p.add_argument("--delta-scale",  type=float, default=0.2)
    p.add_argument("--no-entropy",   action="store_true",
                   help="消融：不使用 sim_matrix 熵信号")
    p.add_argument("--no-cf",        action="store_true",
                   help="消融：不使用 coarse-fine 一致性信号")

    # 损失权重
    p.add_argument("--lambda-flow",   type=float, default=2.0)
    p.add_argument("--lambda-photo",  type=float, default=1.0)
    p.add_argument("--lambda-fold",   type=float, default=5.0)
    p.add_argument("--lambda-smooth", type=float, default=0.5)
    p.add_argument("--lambda-cov",    type=float, default=0.1)
    p.add_argument("--photo-start-epoch", type=int, default=5,
                   help="从第N轮开始加入光度损失")

    # 日志
    p.add_argument("--log-interval",  type=int, default=10)
    p.add_argument("--vis-interval",  type=int, default=200)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--seed",          type=int, default=42)

    return p


def main() -> None:
    args   = build_argparser().parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    writer  = SummaryWriter(log_dir=str(args.log_dir))


    # distill 脚本保存时 key 为 "student"
    matcher_gs = args.image_size // 8   # grid_size for AgriMatcher = 32

    matcher = AgriMatcher(
        d_model     = args.d_model,
        teacher_dim = 80,
        grid_size   = matcher_gs,
    ).to(device)

    ckpt_m = torch.load(args.matcher_ckpt, map_location=device, weights_only=False)
    # 兼容两种 key（"student" 或 "model"）
    state_key = "student" if "student" in ckpt_m else "model"
    matcher.load_state_dict(ckpt_m[state_key], strict=True)
    matcher.eval()
    for param in matcher.parameters():
        param.requires_grad = False
    print(f"[Main] AgriMatcher loaded & frozen. (key='{state_key}')")

    # ── 验证 AgriMatcher 输出包含所需字段 ──
    with torch.no_grad():
        _dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
        _out   = matcher(_dummy, _dummy)
    required_keys = ["warp_AB", "confidence_AB", "warp_AB_coarse",
                     "sim_matrix", "feat_A_64", "feat_B_64"]
    missing = [k for k in required_keys if k not in _out]
    if missing:
        raise RuntimeError(
            f"AgriMatcher 输出缺少字段: {missing}\n"
            f"请确认 AgriMatcher.forward() 的 return dict 包含 feat_A_64 / feat_B_64。\n"
            f"只需在 return dict 中添加：'feat_A_64': feat_A_64, 'feat_B_64': feat_B_64"
        )
    print(f"[Main] AgriMatcher output fields OK: {list(_out.keys())}")
    del _dummy, _out

    # ── BypassTPSEstimator ──
    bypass_tps = BypassTPSEstimator(
        grid_size        = args.grid_size,
        feat_channels    = args.d_model,
        hidden_ch        = args.hidden_ch,
        flow_map_size    = 64,
        sigma_scale      = args.sigma_scale,
        delta_scale      = args.delta_scale,
        use_entropy      = not args.no_entropy,
        use_cf_consistency = not args.no_cf,
    ).to(device)

    # ── TPSGridGenerator（无参数，预计算TPS矩阵）──
    tps_gen = TPSGridGenerator(
        out_h      = args.image_size,
        out_w      = args.image_size,
        grid_size  = args.grid_size,
    ).to(device)

    # ── 损失函数 ──
    loss_fn = BypassTPSLoss(
        lambda_flow        = args.lambda_flow,
        lambda_photo       = args.lambda_photo,
        lambda_fold        = args.lambda_fold,
        lambda_smooth      = args.lambda_smooth,
        lambda_cov         = args.lambda_cov,
        photo_start_epoch  = args.photo_start_epoch,
    )

    # ── 数据集 ──
    full_dataset = ImagePairDataset(
        args.pairs_file, args.image_size
    )
    val_size   = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[Main] train={train_size}, val={val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
        drop_last   = False,
    )

    # ── 优化器（只优化 bypass_tps，matcher 完全冻结）──
    optimizer = torch.optim.AdamW(
        bypass_tps.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    # OneCycleLR（与 distill 训练脚本保持一致）
    accumulation_steps = max(1, args.accum_steps)
    steps_per_epoch    = math.ceil(len(train_loader) / accumulation_steps)
    total_steps        = max(1, steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr         = args.lr,
        total_steps    = total_steps,
        pct_start      = 0.05,
        anneal_strategy = "cos",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── 断点恢复 ──
    start_epoch = 1
    global_step = 0
    best_loss   = float("inf")

    if args.resume is not None and args.resume.exists():
        start_epoch, global_step, best_loss = load_checkpoint(
            args.resume, bypass_tps, optimizer, scheduler,
            scaler if use_amp else None, device,
        )
        start_epoch = max(1, start_epoch)
        print(f"[Resume] epoch={start_epoch} step={global_step} best={best_loss:.6f}")

    # ── 训练循环 ──
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs + 1):
        bypass_tps.train()

        for i, batch in enumerate(train_loader):
            img_a = batch["img_a"].to(device, non_blocking=True)
            img_b = batch["img_b"].to(device, non_blocking=True)

            # ── AgriMatcher 推理（冻结，无梯度）──
            with torch.no_grad():
                m_out = run_matcher(matcher, img_a, img_b)
                warp_fine   = m_out["warp_AB"]           # (B,64,64,2)
                conf_fine   = m_out["confidence_AB"]     # (B,64,64)
                warp_coarse = m_out.get("warp_AB_coarse")
                sim_mat     = m_out.get("sim_matrix")
                feat_a64    = m_out["feat_A_64"]         # (B,128,64,64)
                feat_b64    = m_out["feat_B_64"]

            # ── 前向传播（有梯度）──
            with torch.autocast(
                device_type = device.type,
                dtype       = torch.float16,
                enabled     = use_amp,
            ):
                # Step1: BypassTPSEstimator 估计控制点偏移
                tps_out = bypass_tps(
                    warp_AB        = warp_fine,
                    confidence     = conf_fine,
                    feat_A         = feat_a64,
                    feat_B         = feat_b64,
                    sim_matrix     = sim_mat,
                    warp_AB_coarse = warp_coarse,
                    dense_field    = None,   # 先不传，等TPS生成后再传
                )
                delta_cp = tps_out["delta_cp"]

                # Step2: TPS 生成密集形变场
                tps_field = tps_gen(delta_cp)   # (B,H,W,2)

                # Step3: 对 img_b 应用 TPS（用于光度loss和密集折叠）
                warped_b, mask_b = warp_image(img_b, tps_field)

                # Step4: 补充计算密集折叠惩罚
                # （在tps_field生成后，更新fold_loss）
                fold_out  = bypass_tps.folding(delta_cp, tps_field)
                fold_loss = fold_out["fold_loss"]

                # Step5: 计算总损失
                loss_dict = loss_fn(
                    tps_field     = tps_field,
                    delta_cp      = delta_cp,
                    coverage      = tps_out["coverage"],
                    fold_loss     = fold_loss,
                    img_A         = img_a,
                    warped_B      = warped_b,
                    mask_B        = mask_b,
                    target_warp   = warp_fine,
                    confidence    = conf_fine,
                    current_epoch = epoch,
                )
                loss = loss_dict["total"] / accumulation_steps

            # ── 反向传播 ──
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        bypass_tps.parameters(), args.grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            # 只在真正 optimizer step 后记录
            just_stepped = (
                (i + 1) % accumulation_steps == 0 or
                (i + 1) == len(train_loader)
            )
            if not just_stepped:
                continue

            # ── 日志 ──
            if global_step % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] ep={epoch} step={global_step} lr={current_lr:.2e} "
                    f"total={loss_dict['total'].item():.4f} "
                    f"flow={loss_dict['loss_flow']:.4f} "
                    f"photo={loss_dict['loss_photo']:.4f} "
                    f"fold={loss_dict['loss_fold']:.4f} "
                    f"smooth={loss_dict['loss_smooth']:.4f} "
                    f"cov={loss_dict['loss_cov']:.4f} "
                    f"fold_ratio={fold_out['fold_ratio']:.3f}"
                )
                writer.add_scalar("Loss/Total",       loss_dict["total"].item(), global_step)
                writer.add_scalar("Loss/Flow",        loss_dict["loss_flow"],    global_step)
                writer.add_scalar("Loss/Photo",       loss_dict["loss_photo"],   global_step)
                writer.add_scalar("Loss/Fold",        loss_dict["loss_fold"],    global_step)
                writer.add_scalar("Loss/Smooth",      loss_dict["loss_smooth"],  global_step)
                writer.add_scalar("Loss/Coverage",    loss_dict["loss_cov"],     global_step)
                writer.add_scalar("Diag/FoldRatio",   fold_out["fold_ratio"],    global_step)
                writer.add_scalar("Train/LR",         current_lr,                global_step)

            # ── 可视化 ──
            if global_step % args.vis_interval == 0:
                log_visualizations(
                    writer, global_step,
                    img_a, img_b, warped_b,
                    tps_out["coverage"],
                    tps_out["gate"],
                    delta_cp,
                )

            # ── Checkpoint ──
            if global_step % args.save_interval == 0:
                current_loss = loss_dict["total"].item()
                is_best      = current_loss < best_loss
                if is_best:
                    best_loss = current_loss
                save_checkpoint(
                    args.save_dir, epoch, global_step,
                    bypass_tps, optimizer, scheduler,
                    scaler if use_amp else None,
                    args, best_loss, is_best,
                )
                flag = "★ best" if is_best else ""
                print(
                    f"[ckpt] step={global_step} loss={current_loss:.4f} "
                    f"best={best_loss:.4f} {flag}"
                )

        # ── Epoch 结束：验证 + 保存 ──
        val_metrics = validate(
            val_loader, matcher, bypass_tps, tps_gen,
            loss_fn, device, use_amp, epoch,
        )
        print(
            f"[val] ep={epoch} "
            + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        )
        for k, v in val_metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

        is_best = val_metrics["total"] < best_loss
        if is_best:
            best_loss = val_metrics["total"]
        save_checkpoint(
            args.save_dir, epoch, global_step,
            bypass_tps, optimizer, scheduler,
            scaler if use_amp else None,
            args, best_loss, is_best,
        )
        print(
            f"[ckpt] End of epoch {epoch} → last.pt "
            f"{'(best)' if is_best else ''} | best={best_loss:.6f}"
        )

    writer.close()
    print(f"[Done] Training complete. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()