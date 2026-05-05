import argparse
import copy
import pathlib
import platform
import random
import shutil
import sys
import math
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
    AgriStitcher,
    make_grid,
)
from dense_match.refine import (
    grid_fold_loss,
    mesh_laplacian_loss,
    mesh_magnitude_loss,
)
from dense_match.losses import (
    DistillationLoss,
    LocalGeometricConsistency,
    build_h_inlier_pseudo_labels,
    build_inlier_pseudo_labels,
    compute_inlier_coverage_loss,
    compute_gradient_weight_map,
    compute_inlier_loss,
    compute_photometric_loss,
    compute_cycle_consistency_loss,
    residual_smoothness_loss,
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
    Visualize the deformation field by physically resampling a regular grid.

    The output shows how a uniform grid in image B is warped into image A's
    coordinate frame by the predicted dense grid.
    """
    B, H_target, W_target, _ = dense_grid.shape
    device = dense_grid.device

    # Build a single RGB grid image that can be warped by grid_sample.
    grid_img = torch.zeros((1, 3, H_target, W_target), device=device)
    step_h = max(H_target // 16, 1)
    step_w = max(W_target // 16, 1)

    for i in range(0, H_target, step_h):
        grid_img[0, 2, i:i + 2, :] = 1.0
    for j in range(0, W_target, step_w):
        grid_img[0, 2, :, j:j + 2] = 1.0

    grid_img[0, 2, 0:2, :] = 1.0
    grid_img[0, 2, -2:, :] = 1.0
    grid_img[0, 2, :, 0:2] = 1.0
    grid_img[0, 2, :, -2:] = 1.0

    warped_grid = F.grid_sample(
        grid_img, dense_grid[0:1],
        mode='bilinear', padding_mode='zeros', align_corners=False
    )

    return warped_grid[0].cpu()  # [3, H, W] RGB tensor.


def plot_grid_to_tensorboard(dense_grid, img_shape, step, writer, tag='Train/Stitch_Grid'):
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


def visualize_homography(H_mat: torch.Tensor, H_target: int, W_target: int) -> torch.Tensor:
    """
    Visualize the warped grid induced by the homography branch only.

    Decoder residuals are excluded so this view diagnoses the global transform.
    Returns a [3, H, W] RGB tensor with blue grid lines on a black background.
    """
    B = H_mat.shape[0]
    device = H_mat.device

    # Generate a normalized source grid at the target visualization resolution.
    grid_src = make_grid(1, H_target, W_target, device, torch.float32)  # [1, H, W, 2]
    src_pts = grid_src.reshape(1, -1, 2)
    ones = torch.ones(1, H_target * W_target, 1, device=device)
    src_homo = torch.cat([src_pts, ones], dim=-1)  # [1, HW, 3]
    H0 = H_mat[0:1].float()
    proj = torch.bmm(src_homo, H0.transpose(1, 2))
    z = proj[..., 2:3]
    sign = z.sign(); sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    denom = torch.where(z.abs() < 1e-3, 1e-3 * sign, z)
    dst_pts = (proj[..., :2] / denom).reshape(1, H_target, W_target, 2)  # [1, H, W, 2]
    dst_pts = dst_pts.clamp(-3.0, 3.0)

    # Draw grid lines in the projected destination coordinate system.
    vis = torch.zeros(3, H_target, W_target)
    step = max(H_target // 16, 1)
    grid_np = dst_pts[0].cpu().numpy()

    # Convert normalized coordinates to pixel coordinates.
    px = ((grid_np[..., 0] + 1.0) / 2.0 * W_target - 0.5).astype('int32').clip(0, W_target - 1)
    py = ((grid_np[..., 1] + 1.0) / 2.0 * H_target - 0.5).astype('int32').clip(0, H_target - 1)

    # Draw horizontal grid lines.
    for r in range(0, H_target, step):
        for c in range(W_target - 1):
            x0, y0 = px[r, c], py[r, c]
            x1, y1 = px[r, c + 1], py[r, c + 1]
            if abs(x1 - x0) < W_target // 2 and abs(y1 - y0) < H_target // 2:
                vis[2, y0, x0] = 1.0
    # Draw vertical grid lines.
    for c in range(0, W_target, step):
        for r in range(H_target - 1):
            x0, y0 = px[r, c], py[r, c]
            x1, y1 = px[r + 1, c], py[r + 1, c]
            if abs(x1 - x0) < W_target // 2 and abs(y1 - y0) < H_target // 2:
                vis[2, y0, x0] = 1.0

    return vis  # [3, H, W]


def decompose_homography_params(H_mat: torch.Tensor) -> Dict[str, float]:
    """
    Extract interpretable homography parameters for TensorBoard diagnostics.

    Convention: dst_homo = H @ src_homo for column-vector notation.
    """
    H = H_mat[0].float().cpu().numpy()
    params = {}

    # Translation in normalized coordinates: H[0,2], H[1,2].
    params['H_tx'] = float(H[0, 2])
    params['H_ty'] = float(H[1, 2])

    # scale: sqrt(H[0,0]^2 + H[1,0]^2)
    params['H_scale'] = float((H[0, 0] ** 2 + H[1, 0] ** 2) ** 0.5)

    # rotation (approx): atan2(H[1,0], H[0,0])
    import math as _math
    params['H_rot_deg'] = float(_math.degrees(_math.atan2(H[1, 0], H[0, 0])))

    # perspective distortion: magnitude of last row (h31, h32)
    params['H_persp'] = float((H[2, 0] ** 2 + H[2, 1] ** 2) ** 0.5)

    # det of top-left 2x2 (should be > 0 for valid H)
    params['H_det2x2'] = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])

    # identity residual: ||H - I||_F
    import numpy as np
    params['H_identity_residual'] = float(np.linalg.norm(H - np.eye(3)))

    return params
    


def colorize_heatmap(tensor, cmap='jet'):
    x = tensor.detach().cpu().numpy().squeeze()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return torch.from_numpy(plt.get_cmap(cmap)(x)[..., :3]).permute(2, 0, 1).float()


def _to_vis_image(x: torch.Tensor, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if target_hw and x.shape[-2:] != target_hw:
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
    return (x[0:1].detach() * 0.225 + 0.45).clamp(0, 1).cpu()


def _map_to_canvas(
        tensor: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
        cmap: str = 'jet',
) -> torch.Tensor:
    if tensor is None:
        return torch.zeros((3, target_hw[0], target_hw[1]), dtype=torch.float32)

    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            tensor = tensor[0, 0]
        elif tensor.shape[-1] == 1:
            tensor = tensor[0, ..., 0]
        else:
            tensor = tensor[0]
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        else:
            tensor = tensor[0]

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D map for visualization, got shape={tuple(tensor.shape)}")

    tensor = tensor.detach().float().unsqueeze(0).unsqueeze(0)
    if tensor.shape[-2:] != target_hw:
        tensor = F.interpolate(tensor, size=target_hw, mode='bilinear', align_corners=False)
    return colorize_heatmap(tensor[0], cmap=cmap)


def log_training_visuals(
        writer: SummaryWriter,
        phase: str,
        step: int,
        metrics: Dict[str, Optional[float]],
) -> None:
    for name, value in metrics.items():
        if value is None:
            continue
        writer.add_scalar(f"{phase}/{name}", value, step)

def create_checkerboard(img1, img2, num_squares=8):
    B, C, H, W = img1.shape
    img2 = img2.to(img1.device)
    grid_h = max(H // num_squares, 1)
    grid_w = max(W // num_squares, 1)
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    mask = ((y // grid_h) % 2 == (x // grid_w) % 2).float().to(img1.device)
    return img1 * mask + img2 * (1 - mask)

def visualize_results(img_a, img_b, stu_out, step, writer, phase="Train"):
    matcher_out = stu_out['matcher_out']
    dense_grid = stu_out['dense_grid']
    H_mat = stu_out.get('H_mat')          # [B, 3, 3]
    H_base_grid = stu_out.get('H_base_grid')  # [B, H, W, 2], H-only grid without decoder residual.
    device = img_a.device
    H_target, W_target = dense_grid.shape[1:3]

    with torch.no_grad():
        warped_b = F.grid_sample(img_b, dense_grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
        img_a_vis = _to_vis_image(img_a, (H_target, W_target))
        img_b_vis = _to_vis_image(img_b, (H_target, W_target))
        warped_b_vis = _to_vis_image(warped_b)
        ones = torch.ones((1, 1, H_target, W_target), device=device)
        mask = F.grid_sample(ones, dense_grid[0:1], mode='nearest',
                             padding_mode='zeros', align_corners=False).cpu()
        mask = (mask > 0.9).float()
        checker_vis = create_checkerboard(img_a_vis, warped_b_vis, num_squares=12)
        diff_vis = (img_a_vis - warped_b_vis).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        blended_vis = img_a_vis * (1 - mask * 0.5) + warped_b_vis * (mask * 0.5)
        conf_color = _map_to_canvas(matcher_out['confidence_AB'][0:1], (H_target, W_target), cmap='jet')

        # Full warp grid including decoder residuals.
        grid_vis = plot_warped_grid(dense_grid, (H_target, W_target))

        # H-only grid without decoder residuals; useful for diagnosing H quality.
        if H_mat is not None:
            H_grid_vis = visualize_homography(H_mat, H_target, W_target)
        else:
            H_grid_vis = torch.zeros(3, H_target, W_target)

        # H-only warped image for global-transform diagnostics.
        if H_base_grid is not None:
            warped_b_H = F.grid_sample(
                img_b.to(H_base_grid.dtype),
                H_base_grid,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            warped_b_H_vis = _to_vis_image(warped_b_H)
            diff_H_vis = (img_a_vis - warped_b_H_vis).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        else:
            warped_b_H_vis = torch.zeros_like(img_a_vis)
            diff_H_vis = torch.zeros_like(diff_vis)

        # Decoder residual magnitude: final_grid - H_base_grid.
        if H_base_grid is not None:
            residual_map = (dense_grid - H_base_grid).norm(dim=-1)[0:1]
            residual_vis = _map_to_canvas(residual_map, (H_target, W_target), cmap='magma')
        else:
            residual_vis = torch.zeros(3, H_target, W_target)

        stitch_mask_vis = _map_to_canvas(stu_out.get("stitch_mask"), (H_target, W_target), cmap='viridis')
        stitch_residual = stu_out.get("stitch_residual_flow")
        if stitch_residual is not None:
            stitch_residual_vis = _map_to_canvas(
                stitch_residual[0].detach().norm(dim=-1), (H_target, W_target), cmap='magma')
        else:
            stitch_residual_vis = torch.zeros_like(conf_color)
        valid_overlap_vis = _map_to_canvas(stu_out.get("valid_overlap_mask"), (H_target, W_target), cmap='gray')

        grid_mask = (grid_vis[2:3, :, :] > 0.1).float()
        overlay_vis = blended_vis[0] * (1 - grid_mask * 0.8) + grid_vis * (grid_mask * 0.8)

        # H grid overlay
        H_grid_mask = (H_grid_vis[2:3, :, :] > 0.1).float()
        H_overlay_vis = img_a_vis[0] * (1 - H_grid_mask * 0.8) + H_grid_vis * (H_grid_mask * 0.8)

        geo_scores = matcher_out.get('geo_scores')
        top1_prob = matcher_out.get('top1_prob')
        inlier_weights = matcher_out.get('inlier_weights')
        geo_canvas = torch.zeros_like(conf_color)
        top1_canvas = torch.zeros_like(conf_color)
        inlier_canvas = torch.zeros_like(conf_color)
        if geo_scores is not None and top1_prob is not None:
            Hc, Wc = matcher_out['coarse_hw']
            geo_map = geo_scores.mean(dim=-1).reshape(-1, Hc, Wc)[0:1]
            top1_map = top1_prob.reshape(-1, Hc, Wc)[0:1]
            geo_canvas = _map_to_canvas(geo_map, (H_target, W_target), cmap='plasma')
            top1_canvas = _map_to_canvas(top1_map, (H_target, W_target), cmap='cividis')
            writer.add_image(f'{phase}/Debug_GeoScores', geo_canvas, step)
            writer.add_image(f'{phase}/Debug_Top1Prob', top1_canvas, step)
        if inlier_weights is not None:
            iw_tensor = inlier_weights[0].detach()
            if iw_tensor.ndim == 2:
                iw_tensor = iw_tensor.squeeze(-1)
            fine_hw = matcher_out.get('fine_hw')
            can_show_inlier = fine_hw is not None and int(fine_hw[0]) * int(fine_hw[1]) == iw_tensor.numel()
            if can_show_inlier:
                Hf_iw, Wf_iw = int(fine_hw[0]), int(fine_hw[1])
                iw_map = iw_tensor.reshape(1, Hf_iw, Wf_iw)
                inlier_canvas = _map_to_canvas(iw_map, (H_target, W_target), cmap='magma')
                writer.add_image(f'{phase}/Debug_InlierWeights', inlier_canvas, step)

        # Log interpretable H parameters to TensorBoard.
        if H_mat is not None:
            h_params = decompose_homography_params(H_mat)
            for k, v in h_params.items():
                writer.add_scalar(f"{phase}/H_params/{k}", v, step)

        # Dashboard layout.
        # Row 1: img_a | img_b | warped_b(full) | checker
        # Row 2: conf | diff(full) | full_grid_overlay | H_grid_overlay
        # Row 3: warped_b_H_only | diff_H | residual(decoder_delta) | stitch_mask
        # Row 4: stitch_residual | valid_overlap | inlier | geo_scores
        row1 = torch.cat([img_a_vis[0], img_b_vis[0], warped_b_vis[0], checker_vis[0]], dim=2)
        row2 = torch.cat([conf_color, diff_vis[0], overlay_vis, H_overlay_vis], dim=2)
        row3 = torch.cat([warped_b_H_vis[0], diff_H_vis[0], residual_vis, stitch_mask_vis], dim=2)
        row4 = torch.cat([stitch_residual_vis, valid_overlap_vis, inlier_canvas, geo_canvas], dim=2)

        writer.add_image(f"{phase}/WarpGrid_Full", grid_vis, step)
        writer.add_image(f"{phase}/WarpGrid_H_only", H_grid_vis, step)
        writer.add_image(f"{phase}/Warped_H_only", warped_b_H_vis[0], step)
        writer.add_image(f"{phase}/Diff_H_only", diff_H_vis[0], step)
        writer.add_image(f"{phase}/Decoder_Delta_Magnitude", residual_vis, step)
        dashboard = torch.cat([row1, row2, row3, row4], dim=1)
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


@torch.no_grad()
def update_model_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        model_value = model_state[name].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


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
        img_a_list.append(a);
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


def teacher_overlap_map(confidence: torch.Tensor) -> torch.Tensor:
    if confidence.ndim == 4:
        confidence = confidence[..., 0]
    if confidence.ndim == 3:
        confidence = confidence
    else:
        raise ValueError(f"Unexpected confidence shape: {confidence.shape}")
    confidence = torch.nan_to_num(confidence.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    return torch.sigmoid(confidence).clamp(0.0, 1.0)


def teacher_confidence_prob(t_out: Dict[str, torch.Tensor]) -> torch.Tensor:
    conf = t_out["confidence_AB"]
    if t_out.get("confidence_is_prob", False):
        if conf.ndim == 4 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        conf = torch.nan_to_num(conf.float(), nan=0.0, posinf=1.0, neginf=0.0)
        return conf.clamp(0.0, 1.0)
    return teacher_overlap_map(conf)


def compute_h_only_losses(
        stu_out: Dict[str, Any],
        teacher_warp: torch.Tensor,
        teacher_conf_prob: torch.Tensor,
        weights: Dict[str, float],
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_distill_w = float(weights.get("h_distill", 0.0))
    h_match_w = float(weights.get("h_match", 0.0))
    h_budget_w = float(weights.get("h_residual_budget", 0.0))
    need_loss = h_distill_w > 0.0 or h_match_w > 0.0 or h_budget_w > 0.0
    zero = torch.zeros((), device=device, dtype=torch.float32)
    h_grid = stu_out.get("H_base_grid_train")
    if not need_loss or h_grid is None:
        return zero, zero, zero
    if torch.is_grad_enabled() and not h_grid.requires_grad:
        raise RuntimeError("H_base_grid_train must require grad when H-only losses are enabled.")

    with torch.amp.autocast("cuda", enabled=False):
        h_grid = torch.nan_to_num(h_grid.float(), nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
        B, H, W, _ = h_grid.shape

        if h_distill_w > 0.0 or h_budget_w > 0.0:
            t_warp_dense = F.interpolate(
                teacher_warp.detach().float().permute(0, 3, 1, 2),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            t_warp_dense = torch.nan_to_num(t_warp_dense, nan=0.0, posinf=1.5, neginf=-1.5).clamp(-1.5, 1.5)

            t_conf_dense = F.interpolate(
                teacher_conf_prob.detach().float().unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            t_conf_dense = torch.nan_to_num(t_conf_dense, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            t_weight = (t_conf_dense * (t_conf_dense > 0.5).float()).detach()
            conf_denom = t_weight.sum().clamp_min(1e-6)

            h_distill_err_map = F.huber_loss(
                h_grid,
                t_warp_dense,
                reduction="none",
                delta=0.5,
            ).sum(dim=-1)
            h_distill_err_map = torch.nan_to_num(h_distill_err_map, nan=0.0, posinf=10.0, neginf=0.0).clamp(0.0, 10.0)
            h_distill_loss = (h_distill_err_map * t_weight).sum() / conf_denom
        else:
            t_warp_dense = None
            t_conf_dense = None
            conf_denom = None
            h_distill_err_map = None
            h_distill_loss = zero

        matcher_out = stu_out.get("matcher_out", {})
        warp_ab = matcher_out.get("warp_AB")
        conf_ab = matcher_out.get("confidence_AB")
        if h_match_w > 0.0 and warp_ab is not None and conf_ab is not None:
            Hf, Wf = warp_ab.shape[1:3]
            h_grid_lowres = F.interpolate(
                h_grid.permute(0, 3, 1, 2),
                size=(Hf, Wf),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            h_grid_lowres = torch.nan_to_num(h_grid_lowres, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
            warp_ab_f = torch.nan_to_num(warp_ab.float(), nan=0.0, posinf=1.5, neginf=-1.5).clamp(-1.5, 1.5)
            conf_ab_f = torch.nan_to_num(conf_ab.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            h_match_weight = (conf_ab_f * (conf_ab_f > 0.5).float()).detach()
            h_match_err = F.huber_loss(h_grid_lowres, warp_ab_f.detach(), reduction="none", delta=0.25).sum(dim=-1)
            h_match_loss = (h_match_err * h_match_weight).sum() / h_match_weight.sum().clamp_min(1e-6)
        else:
            h_match_loss = zero

        stitch_residual = stu_out.get("stitch_residual_flow")
        stitch_mask = stu_out.get("stitch_mask")
        if h_budget_w > 0.0 and stitch_residual is not None:
            residual = torch.nan_to_num(stitch_residual.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            if stitch_mask is not None:
                mask = torch.nan_to_num(stitch_mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                residual = residual * mask
            residual_norm = residual.norm(dim=-1)
            if h_distill_err_map is None:
                h_error = (h_grid.detach() - t_warp_dense).norm(dim=-1)
            else:
                h_error = torch.sqrt((2.0 * h_distill_err_map.detach()).clamp_min(0.0))
            residual_budget = (0.01 + 0.35 * h_error).clamp(0.01, 0.25)
            over_budget = F.relu(residual_norm - residual_budget)
            h_residual_budget_loss = (over_budget.square() * t_weight).sum() / conf_denom
        else:
            h_residual_budget_loss = zero

    return h_distill_loss, h_match_loss, h_residual_budget_loss


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


def restore_stage_train_modes(student: nn.Module, stage: Optional[Dict]) -> None:
    if stage is None:
        return
    if "backbone" in stage.get("freeze", []) or not stage.get("unfreeze_backbone", True):
        student.matcher.backbone.eval()
    if "stitch_decoder" in stage.get("freeze", []) and student.stitch_decoder is not None:
        student.stitch_decoder.eval()
    if "inlier_predictor" in stage.get("freeze", []) or not stage.get("use_inlier_predictor", False):
        student.matcher.inlier_predictor.eval()


def _legacy_residual_smoothness_loss(residual_flow: torch.Tensor) -> torch.Tensor:
    """
    Penalize spatial curvature of the local residual field relative to the H base.

    The second-order difference is insensitive to purely linear residual fields,
    so it regularizes local bending rather than global affine motion.
    residual_flow: (B, H, W, 2)
    """
    flow = residual_flow.permute(0, 3, 1, 2).float()  # [B, 2, H, W]

    # Second-order finite differences approximate a Laplacian curvature penalty.
    lap_x = flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
    lap_y = flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]

    return lap_x.pow(2).mean() + lap_y.pow(2).mean()


def compute_valid_area_ratio(grid: torch.Tensor) -> torch.Tensor:
    grid_f = torch.nan_to_num(grid.float(), nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
    valid = (
            (grid_f[..., 0] >= -1.0) & (grid_f[..., 0] <= 1.0) &
            (grid_f[..., 1] >= -1.0) & (grid_f[..., 1] <= 1.0)
    )
    return valid.float().mean()


@torch.no_grad()
def _legacy_build_inlier_pseudo_labels(
        warp_fine: torch.Tensor,  # [B, Hf, Wf, 2]  student fine warp
        teacher_warp: torch.Tensor,  # [B, Ht, Wt, 2] teacher warp; size may differ.
        teacher_conf: torch.Tensor,  # [B, Ht, Wt]     teacher confidence
        sigma: float = 0.05,  # Error tolerance radius in normalized coordinates.
        conf_thresh: float = 0.1,  # Ignore teacher points below this confidence.
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build teacher-only soft pseudo labels for the legacy InlierPredictor path.

    Returns:
        pseudo_labels: [B, Hf*Wf] soft labels in [0, 1].
        valid_mask:    [B, Hf*Wf] bool mask for high-confidence teacher points.
    """
    B, Hf, Wf, _ = warp_fine.shape
    device = warp_fine.device

    # Interpolate teacher warp and confidence to the student's fine resolution.
    teacher_warp_fine = F.interpolate(
        teacher_warp.permute(0, 3, 1, 2).float(),  # [B, 2, Ht, Wt]
        size=(Hf, Wf),
        mode='bilinear',
        align_corners=False,
    ).permute(0, 2, 3, 1)  # [B, Hf, Wf, 2]

    teacher_conf_fine = F.interpolate(
        teacher_conf.unsqueeze(1).float(),  # [B, 1, Ht, Wt]
        size=(Hf, Wf),
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)  # [B, Hf, Wf]

    # Prediction error between student and teacher in normalized coordinates.
    error = (warp_fine.float() - teacher_warp_fine).norm(dim=-1)  # [B, Hf, Wf]

    # Gaussian soft label: smaller error produces a label closer to one.
    pseudo_labels = torch.exp(-(error ** 2) / (2 * sigma ** 2))  # [B, Hf, Wf]

    # Supervise only high-confidence teacher regions; repeated textures often
    # have unreliable teacher confidence and should not drive the inlier head.
    valid_mask = (teacher_conf_fine > conf_thresh)  # [B, Hf, Wf], bool

    return pseudo_labels.reshape(B, Hf * Wf), valid_mask.reshape(B, Hf * Wf)


def _legacy_compute_inlier_loss(
        inlier_weights: torch.Tensor,  # [B, N, 1] InlierPredictor output.
        pseudo_labels: torch.Tensor,  # [B, N] soft labels.
        valid_mask: torch.Tensor,  # [B, N]     bool
        teacher_conf: torch.Tensor,  # [B, N] teacher confidence used as weight.
        focal_gamma: float = 2.0,  # Focal-loss exponent for easy samples.
) -> torch.Tensor:
    """
    Focal BCE loss for inlier prediction.

    High-confidence teacher regions receive larger weight, while focal weighting
    emphasizes samples where the prediction is far from the pseudo label.
    """
    w = inlier_weights.squeeze(-1)  # [B, N]
    y = pseudo_labels.float().detach()  # [B, N]

    # Focal weight is small for well-predicted samples and large for hard cases.
    # pt = w * y + (1-w) * (1-y) is the probability of the pseudo target.
    pt = w * y + (1.0 - w) * (1.0 - y)
    focal_weight = (1.0 - pt.detach()).pow(focal_gamma)

    # Numerically stable binary cross entropy.
    bce = F.binary_cross_entropy(w.clamp(1e-6, 1 - 1e-6), y, reduction='none')  # [B, N]

    # Combine BCE with focal weights.
    focal_loss = focal_weight * bce  # [B, N]

    # Teacher-confidence weighting keeps unreliable pseudo labels weak.
    weight = teacher_conf.float().detach() * valid_mask.float().detach()  # [B, N]
    valid_count = weight.sum().clamp(min=1.0)

    return (focal_loss * weight).sum() / valid_count


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
    Unified loss computation for one training step.

    The caller provides stage-interpolated weights in `w`; this function does
    not query epoch state directly.
    """
    teacher_conf_prob = teacher_confidence_prob(t_out)
    teacher_warp = torch.nan_to_num(
        t_out["warp_AB"].float(), nan=0.0, posinf=1.5, neginf=-1.5
    ).clamp(-1.5, 1.5)
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        stu_out = student(img_a, img_b)
        stu_output = stu_out["matcher_out"]
        residual_mode = stu_out.get("residual_mode", "dense")

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

        stitch_residual = stu_out.get("stitch_residual_flow")
        if stitch_residual is not None:
            geo_loss = geo_loss_fn(stitch_residual, conf_for_geo)
        else:
            current_flow = final_grid - grid_full
            geo_loss = geo_loss_fn(current_flow, conf_for_geo)

        d_loss = distill_loss_fn(
            stu_output=stu_output,
            teacher_warp=teacher_warp,
            teacher_conf=teacher_conf_prob,
            teacher_feat_A=t_feat_a,
            teacher_feat_B=t_feat_b,
        )
        if w.get("residual_distill", 0.0) > 0:
            t_warp_dense = F.interpolate(
                teacher_warp.permute(0, 3, 1, 2),
                size=(H, W), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)

            t_conf_dense = F.interpolate(
                teacher_conf_prob.unsqueeze(1),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(1).clamp(0.0, 1.0)
        # Huber residual-distillation error gated by teacher confidence.
            teacher_gate = (t_conf_dense > 0.5).float()
            residual_distill_weight = (t_conf_dense * teacher_gate).detach()
            residual_distill_err = F.huber_loss(
                final_grid,
                t_warp_dense.detach(),
                reduction="none",
                delta=0.25,
            ).sum(dim=-1)
            residual_distill_loss = (
                    residual_distill_err * residual_distill_weight
            ).sum() / (residual_distill_weight.sum() + 1e-6)
        else:
            residual_distill_loss = torch.zeros((), device=device)
        h_distill_loss, h_match_loss, h_residual_budget_loss = compute_h_only_losses(
            stu_out=stu_out,
            teacher_warp=teacher_warp,
            teacher_conf_prob=teacher_conf_prob,
            weights=w,
            device=device,
        )

        f_loss = torch.zeros((), device=device)

        mesh_delta = stu_out.get("mesh_delta")
        mesh_mask_lowres = stu_out.get("mesh_mask_lowres")
        mesh_lap = torch.zeros((), device=device)
        mesh_mag = torch.zeros((), device=device)
        if stitch_residual is not None and w.get("stitch_residual", 0.0) > 0:
            if residual_mode == "mesh" and mesh_delta is not None:
                mesh_lap = mesh_laplacian_loss(mesh_delta)
                mesh_mag = mesh_magnitude_loss(mesh_delta)
                smooth_loss = mesh_lap + 0.1 * mesh_mag
            else:
                smooth_loss = residual_smoothness_loss(stitch_residual.float())
        else:
            smooth_loss = torch.zeros((), device=device)
        stitch_mask = stu_out.get("stitch_mask")
        if stitch_mask is not None and w.get("stitch_mask", 0.0) > 0:
            if residual_mode == "mesh" and mesh_mask_lowres is not None:
                mask_for_reg = mesh_mask_lowres
                mask_dense_for_tv = F.interpolate(
                    mesh_mask_lowres.float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            else:
                mask_for_reg = stitch_mask
                mask_dense_for_tv = stitch_mask
            mask_tv = residual_smoothness_loss(mask_dense_for_tv.expand(-1, -1, -1, 2))
            mask_mean = mask_for_reg.mean()
            mask_target = max(0.0, min(0.5, float(w.get("stitch_mask_target", 0.08))))
            mask_sparsity = F.relu(mask_mean - mask_target).square()
            mask_reg = mask_tv + 0.25 * mask_sparsity + 0.02 * mask_mean
        else:
            mask_reg = torch.zeros((), device=device)
            mask_mean = torch.zeros((), device=device)
        fold_loss = grid_fold_loss(final_grid.float()) if w.get("fold", 0.0) > 0 else torch.zeros((), device=device)

        # One-sided cycle loss: AB is the detached anchor and gradients update
        # the auxiliary BA prediction. See compute_cycle_consistency_loss.
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
        inlier_w = w.get("inlier", 0.0)
        inlier_coverage_loss = torch.zeros((), device=device)
        h_inlier_pos_ratio = torch.zeros((), device=device)
        h_inlier_valid_ratio = torch.zeros((), device=device)
        h_inlier_used = torch.zeros((), device=device)
        inlier_weights = stu_output.get("raw_inlier_weights", stu_output.get("inlier_weights"))
        if inlier_w > 0 and student.matcher.use_inlier_predictor and inlier_weights is not None:
            # Use raw inlier probabilities returned by the matcher; solver
            # weights are detached and are not used for the inlier BCE loss.
            if inlier_weights is not None:
                Hf, Wf = stu_output['fine_hw']
                h_grid = stu_out.get("H_base_grid_train", None)
                if h_grid is not None:
                    h_grid_fine = F.interpolate(
                        h_grid.permute(0, 3, 1, 2).float(),
                        size=(Hf, Wf), mode='bilinear', align_corners=False
                    ).permute(0, 2, 3, 1)
                    # H-aware labels are detached targets: they check whether a
                    # dense correspondence is both teacher-consistent and
                    # explainable by the current trainable H_final grid.
                    pseudo_labels, valid_mask, teacher_conf_fine_flat = build_h_inlier_pseudo_labels(
                        warp_fine=stu_output['warp_AB'].detach(),
                        teacher_warp=teacher_warp,
                        teacher_conf=teacher_conf_prob,
                        h_grid_fine=h_grid_fine.detach(),
                        sigma_teacher=w.get("inlier_sigma_teacher", 0.04),
                        sigma_h=w.get("inlier_sigma_h", 0.06),
                        conf_thresh=w.get("inlier_conf_thresh", 0.5),
                    )
                    h_inlier_used = torch.ones((), device=device)
                else:
                    pseudo_labels, valid_mask = build_inlier_pseudo_labels(
                        warp_fine=stu_output['warp_AB'].detach(),
                        teacher_warp=teacher_warp,
                        teacher_conf=teacher_conf_prob,
                        sigma=0.05,
                        conf_thresh=0.15,
                    )
                    # Flatten teacher confidence at the fine resolution for loss weighting.
                    teacher_conf_fine_flat = F.interpolate(
                        teacher_conf_prob.unsqueeze(1),
                        size=(Hf, Wf), mode='bilinear', align_corners=False
                    ).squeeze(1).reshape(inlier_weights.shape[0], -1).clamp(0.0, 1.0)
                    if not getattr(compute_loss_bundle, "_warned_h_inlier_fallback", False):
                        print("[Warning] H_base_grid_train missing; falling back to teacher-only inlier pseudo labels.")
                        compute_loss_bundle._warned_h_inlier_fallback = True
                valid_mask_f = valid_mask.float()
                h_inlier_valid_ratio = valid_mask_f.mean()
                h_inlier_pos_ratio = ((pseudo_labels > 0.5).float() * valid_mask_f).sum() / valid_mask_f.sum().clamp_min(1.0)

                inlier_loss = compute_inlier_loss(
                    inlier_weights=inlier_weights,
                    pseudo_labels=pseudo_labels,
                    valid_mask=valid_mask,
                    teacher_conf=teacher_conf_fine_flat,
                    focal_gamma=2.0,
                )
            else:
                inlier_loss = torch.zeros((), device=device)
        else:
            inlier_loss = torch.zeros((), device=device)

        if inlier_weights is not None and w.get("inlier_coverage", 0.0) > 0:
            Hf_cov, Wf_cov = stu_output['fine_hw']
            inlier_coverage_loss = compute_inlier_coverage_loss(
                inlier_weights,
                stu_output['confidence_AB'],
                hw=(Hf_cov, Wf_cov),
                bins=int(w.get("inlier_coverage_bins", 4)),
                min_mass=float(w.get("inlier_coverage_min_mass", 0.03)),
            )

        total_no_photo = (
                w["distill"] * d_loss["total"]
                + w["cycle"] * c_loss
                + w["geo"] * geo_loss
                + w.get("stitch_residual", 0.0) * smooth_loss
                + w.get("stitch_mask", 0.0) * mask_reg
                + w.get("residual_distill", 0.0) * residual_distill_loss
                + inlier_w * inlier_loss
                + w.get("h_distill", 0.0) * h_distill_loss
                + w.get("h_match", 0.0) * h_match_loss
                + w.get("h_residual_budget", 0.0) * h_residual_budget_loss
                + w.get("fold", 0.0) * fold_loss
                + w.get("inlier_coverage", 0.0) * inlier_coverage_loss
        )

    photo_w = w.get("photo", 0.0)
    if photo_w > 0:
        with torch.amp.autocast("cuda", enabled=False):
            use_ssim_now = w.get("ssim", 0.0) > 0.01

            # Sobel gradient weights emphasize stable high-frequency structures
            # such as field rows and stems during photometric supervision.
            grad_wmap = compute_gradient_weight_map(
                img_a.float(), amplify=2.0, blur_radius=1
            )  # [B, 1, H, W], detached from the input image.

            photo_conf = stu_out["matcher_out"]["confidence_AB"].float()
            conf_threshold = float(w.get("photo_conf_threshold", 0.0))
            if conf_threshold > 0.0:
                photo_conf = photo_conf * (photo_conf > conf_threshold).float()

            p_loss, area_ratio = compute_photometric_loss(
                img_a.float(), img_b.float(),
                stu_out["dense_grid"].float(),
                photo_conf,
                use_ssim=use_ssim_now,
                alpha=0.85,
                grad_weight_map=grad_wmap,
            )

            # Area penalty is handled below with detached area ratio so it does
            # not drive the decoder to fold the grid inward.
    else:
        p_loss = torch.zeros((), device=device)
        area_ratio = compute_valid_area_ratio(stu_out["dense_grid"].float()).detach()

    area_penalty_w = w.get("area_penalty", 0.0)
    if area_penalty_w > 0:
        area_ratio_for_penalty = compute_valid_area_ratio(stu_out["dense_grid"].float()).detach()
        area_penalty = F.relu(0.3 - area_ratio_for_penalty) * area_penalty_w
    else:
        area_penalty = torch.zeros((), device=device)

    total_loss = total_no_photo + photo_w * p_loss + area_penalty

    conf_mean = float(conf_for_geo.detach().mean())
    valid_overlap = stu_out.get("valid_overlap_mask")
    valid_overlap_ratio = float(valid_overlap.detach().float().mean()) if valid_overlap is not None else 0.0
    inlier_weights = stu_output.get("raw_inlier_weights", stu_output.get("inlier_weights"))
    inlier_mean = float(inlier_weights.detach().mean()) if inlier_weights is not None else None
    raw_inlier_mean = float(inlier_mean) if inlier_mean is not None else 0.0
    solver_weight_mean_t = stu_output.get("solver_weight_mean")
    solver_weight_nonzero_t = stu_output.get("solver_weight_nonzero_ratio")
    solver_weight_mean = float(solver_weight_mean_t.detach()) if torch.is_tensor(solver_weight_mean_t) else 0.0
    solver_weight_nonzero_ratio = float(solver_weight_nonzero_t.detach()) if torch.is_tensor(solver_weight_nonzero_t) else 0.0
    stitch_ratio = float(mask_mean.detach()) if torch.is_tensor(mask_mean) else float(mask_mean)
    residual_mean = (
        float(stitch_residual.detach().norm(dim=-1).mean())
        if stitch_residual is not None else 0.0
    )
    area_ratio_value = float(area_ratio.detach()) if photo_w > 0 and torch.is_tensor(area_ratio) else 0.0

    return {
        'stu_out': stu_out,
        'distill': d_loss["total"],
        'photo': p_loss,
        'area_penalty': area_penalty,
        'cycle': c_loss,
        'geo': geo_loss,
        'inlier': inlier_loss,
        'h_distill': h_distill_loss,
        'h_match': h_match_loss,
        'h_residual_budget': h_residual_budget_loss,
        'residual_distill': residual_distill_loss,
        'stitch_smooth': smooth_loss,
        'mask_reg': mask_reg,
        'inlier_coverage': inlier_coverage_loss,
        'fold': fold_loss,
        'mesh_lap': mesh_lap,
        'mesh_mag': mesh_mag,
        'total': total_loss,
        'residual_mode': residual_mode,
        'stitch_ratio': stitch_ratio,
        'conf_mean': conf_mean,
        'valid_overlap_ratio': valid_overlap_ratio,
        'inlier_mean': inlier_mean,
        'raw_inlier_mean': raw_inlier_mean,
        'solver_weight_mean': solver_weight_mean,
        'solver_weight_nonzero_ratio': solver_weight_nonzero_ratio,
        'h_inlier_pos_ratio': h_inlier_pos_ratio,
        'h_inlier_valid_ratio': h_inlier_valid_ratio,
        'h_inlier_used': float(h_inlier_used.detach()),
        'residual_mean': residual_mean,
        'area_ratio': area_ratio_value,
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
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = None,
        stage: Optional[Dict] = None,
        log_images: bool = True,
) -> Dict[str, float]:
    was_training = student.training
    student.eval()
    totals = {
        k: 0.0 for k in (
            'distill', 'photo', 'cycle', 'geo', 'inlier',
            'h_distill', 'h_match', 'h_residual_budget',
            'residual_distill', 'stitch_smooth', 'mask_reg',
            'inlier_coverage',
            'fold', 'mesh_lap', 'mesh_mag',
            'area_penalty', 'total', 'stitch_ratio',
            'conf_mean', 'valid_overlap_ratio', 'residual_mean', 'area_ratio',
            'solver_weight_mean', 'solver_weight_nonzero_ratio',
            'raw_inlier_mean', 'h_inlier_pos_ratio', 'h_inlier_valid_ratio',
            'h_inlier_used',
        )
    }
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
        if log_images and i == 0 and writer is not None and step is not None:
            visualize_results(img_a, img_b, bundle['stu_out'], step, writer, phase="Val")
        for k in totals:
            v = bundle[k]
            totals[k] += float(v.detach().item()) if torch.is_tensor(v) else float(v)
        count += 1
    if was_training:
        student.train()
        restore_stage_train_modes(student, stage)
    if count == 0:
        return totals
    return {k: v / count for k, v in totals.items()}


def save_checkpoint(
        save_dir: Path, epoch: int, step: int,
        student: nn.Module, optimizer: torch.optim.Optimizer, scheduler,
        scaler: Optional[torch.amp.GradScaler], args: argparse.Namespace,
        train_loss_ema: float, val_loss: float, best_val_loss: float, is_best: bool = False,
        student_ema: Optional[nn.Module] = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch, "step": step,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "args": vars(args),
        "stage_name": get_stage(epoch)["name"],
        "train_loss_ema": train_loss_ema,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "ema_decay": getattr(args, "ema_decay", None),
    }
    if student_ema is not None:
        payload["student_ema"] = student_ema.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    last_path = save_dir / "last.pt"
    torch.save(payload, last_path)
    print(f"Checkpoint saved. val={val_loss:.6f}  best={best_val_loss:.6f}")
    if is_best:
        shutil.copyfile(last_path, save_dir / "best.pt")
        print(f"[best] New best: {val_loss:.6f}")
    return last_path


def load_checkpoint(
        ckpt_path: Path, student: nn.Module,
        optimizer: Optional[torch.optim.Optimizer], scheduler,
        scaler: Optional[torch.amp.GradScaler], device: torch.device,
) -> Tuple[int, int, float, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        student.load_state_dict(ckpt["student"], strict=True)
        print("[load] Model loaded (strict)")
    except RuntimeError as e:
        print(f"[load] Strict load failed: {e}")
        missing, unexpected = student.load_state_dict(ckpt["student"], strict=False)
        print(f"[load] Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
    if optimizer and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[load] Optimizer load failed: {e}")
    if scheduler and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[load] Scheduler load failed: {e}")
    if scaler and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[load] Scaler load failed: {e}")
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
    p.add_argument("--residual-mode", choices=["dense", "mesh", "none"], default="mesh")
    p.add_argument("--mesh-size", type=int, default=12)
    p.add_argument("--max-residual-px", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=65)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--force-amp", action="store_true")
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--teacher-setting", type=str, default="precise")
    # DistillationLoss hyperparameters.
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
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vis-interval", type=int, default=400)
    p.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True)
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
        print(f"[Dataset] cache mode: {args.cache_dir}")
        suffix = args.cache_dir.suffix.lower() if args.cache_dir.is_file() else ""
        if suffix in {".h5", ".hdf5"}:
            train_ds = BucketedH5TeacherDataset(str(args.cache_dir), args.val_ratio, args.seed, 'train')
            val_ds = BucketedH5TeacherDataset(str(args.cache_dir), args.val_ratio, args.seed, 'val')
            train_sampler = BucketedBatchSampler(train_ds.bucket_to_indices, args.batch_size, shuffle=True,
                                                 seed=args.seed)
            val_sampler = BucketedBatchSampler(val_ds.bucket_to_indices, args.batch_size, shuffle=False, seed=args.seed)
            train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **cache_loader_kwargs)
            val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **cache_loader_kwargs)
        else:
            train_ds = CachedTeacherDataset(args.cache_dir, args.val_ratio, return_split='train')
            val_ds = CachedTeacherDataset(args.cache_dir, args.val_ratio, return_split='val')
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **cache_loader_kwargs)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **cache_loader_kwargs)
    else:
        if args.pairs_file is None:
            raise ValueError("Either --pairs-file or a valid --cache-dir must be provided.")
        print(f"[Dataset] raw image-pair mode: {args.pairs_file}")
        train_ds = MultiScaleDataset(args.pairs_file, args.val_ratio, return_split='train')
        val_ds = MultiScaleDataset(args.pairs_file, args.val_ratio, return_split='val')
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **image_loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **image_loader_kwargs)

    actual_teacher_dim = {'base': 80, 'precise': 100}[args.teacher_setting]
    student = AgriStitcher(
        matcher_config={'d_model': args.d_model, 'teacher_dim': actual_teacher_dim,
                        'grid_size': args.teacher_grid_size},
        decoder_config={
            'feat_channels': args.d_model,
            'residual_mode': args.residual_mode,
            'mesh_size': args.mesh_size,
            'max_residual_px': args.max_residual_px,
        }
    ).to(device)
    student_ema = copy.deepcopy(student).eval()
    for p in student_ema.parameters():
        p.requires_grad_(False)

    distill_loss_fn = DistillationLoss(
        alpha=args.alpha, beta_coarse=args.beta_coarse, beta_refine=args.beta_refine,
        gamma=args.gamma, eta_coarse=args.eta_coarse, eta_refine=args.eta_refine,
        lambda_tv_coarse=args.lambda_tv_coarse, lambda_tv_refine=args.lambda_tv_refine,
        conf_thresh_kl=args.conf_thresh_kl,
    ).to(device)
    geo_loss_fn = LocalGeometricConsistency()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_tracker = EMALossTracker(alpha=args.ema_alpha)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    last_val_loss = float("inf")
    current_stage: Optional[Dict] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler = None

    steps_per_epoch = (len(train_loader) + args.accum_steps - 1) // args.accum_steps
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"], strict=False)
        if "student_ema" in ckpt:
            missing, unexpected = student_ema.load_state_dict(ckpt["student_ema"], strict=False)
            if missing or unexpected:
                print(f"[Resume] EMA loaded with missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            student_ema.load_state_dict(student.state_dict(), strict=False)
        start_epoch = int(ckpt.get("epoch", 1)) + 1  # Resume from the next epoch.
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        last_val_loss = float(ckpt.get("val_loss", float("inf")))
        _ema = float(ckpt.get("train_loss_ema", float("inf")))
        loss_tracker.ema_values['total'] = _ema

        resumed_stage = get_stage(start_epoch)
        apply_freeze_state(student, resumed_stage)
        optimizer, scheduler = build_optimizer_and_scheduler(
            student, resumed_stage, steps_per_epoch, args.weight_decay)

        ckpt_stage_name = ckpt.get("stage_name")
        can_load_training_state = ckpt_stage_name is None or ckpt_stage_name == resumed_stage["name"]
        if not can_load_training_state:
            print(
                f"[Resume] Checkpoint stage={ckpt_stage_name}, resume stage={resumed_stage['name']}; "
                "starting optimizer/scheduler fresh for the new stage."
            )

        if optimizer and "optimizer" in ckpt and can_load_training_state:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[Resume] Optimizer state incompatible, starting fresh: {e}")
        if scheduler and ckpt.get("scheduler") and can_load_training_state:
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
            print(f"\n{'=' * 60}")
            print(f"[Stage Transition] Epoch {epoch} -> Stage: {new_stage['name']}")
            print(f"{'=' * 60}")
            current_stage = new_stage

            # Update frozen/unfrozen module state.
            apply_freeze_state(student, current_stage)

            # Rebuild optimizer and scheduler for the new stage.
            optimizer, scheduler = update_optimizer_and_scheduler(
                student, current_stage, steps_per_epoch, args.weight_decay,
                optimizer=optimizer
            )

            # Print active parameter-group diagnostics.
            for g in optimizer.param_groups:
                n_params = sum(p.numel() for p in g['params'])
                print(f"  Param group '{g.get('name', '?')}': lr={g['lr']:.2e}, params={n_params:,}")
            print(f"  Inlier predictor enabled: {student.matcher.use_inlier_predictor}")

        w = interpolate_loss_weights(current_stage, epoch)
        use_cycle_this_epoch = w.get("cycle", 0.0) > 0.01

        # Log stage loss weights to TensorBoard.
        for k, v in w.items():
            writer.add_scalar(f"LossWeights/{k}", v, epoch)

        if use_cache and isinstance(getattr(train_loader, 'batch_sampler', None), BucketedBatchSampler):
            train_loader.batch_sampler.set_epoch(epoch)

        print(f"\n[Epoch] {epoch}/{args.epochs} | Stage: {current_stage['name']}")
        print(f"   Weights: {', '.join(f'{k}={v:.3f}' for k, v in w.items() if v > 0)}")

        student.train()
        # Keep frozen modules in eval mode to avoid small-batch BN drift.
        if "backbone" in current_stage.get("freeze", []) or not current_stage.get("unfreeze_backbone", True):
            student.matcher.backbone.eval()
        if "stitch_decoder" in current_stage.get("freeze", []) and student.stitch_decoder is not None:
            student.stitch_decoder.eval()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}", dynamic_ncols=True)
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)

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
            stu_out = bundle['stu_out']
            total_loss = bundle['total']

            loss_dict = {
                k: bundle[k] for k in (
                    'distill', 'photo', 'cycle', 'geo',
                    'h_distill', 'h_match', 'h_residual_budget',
                    'residual_distill', 'stitch_smooth', 'mask_reg',
                    'inlier_coverage',
                    'fold', 'mesh_lap', 'mesh_mag',
                    'area_penalty', 'total',
                )
            }
            has_nan = False
            for k, v in loss_dict.items():
                if torch.is_tensor(v) and not torch.isfinite(v).all():
                    pbar.write(f"[NaN/Inf] Loss '{k}' invalid!")
                    has_nan = True
            if has_nan:
                pbar.write(f"[skip] Skipping batch at step {global_step}")
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            loss_for_backward = total_loss / args.accum_steps
            scaler.scale(loss_for_backward).backward()
            accum_count += 1

            if accum_count >= args.accum_steps or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)

                bad_grad_names = [
                    name for name, p in student.named_parameters()
                    if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                has_nan_grad = len(bad_grad_names) > 0
                if has_nan_grad:
                    shown = ", ".join(bad_grad_names[:8])
                    more = "" if len(bad_grad_names) <= 8 else f", ... +{len(bad_grad_names) - 8} more"
                    pbar.write(f"[GradShield] NaN gradient at step {global_step}! Bad params: {shown}{more}")
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                else:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    update_model_ema(student_ema, student, args.ema_decay)
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_step += 1

                if has_nan_grad:
                    if use_amp and scaler.is_enabled():
                        scaler.update(new_scale=max(float(scaler.get_scale()) * 0.5, 1.0))
                    else:
                        scaler.update()

                loss_metrics = {
                    'total': float(total_loss.detach()),
                    'distill': float(bundle['distill'].detach()),
                    'photo': float(bundle['photo'].detach() if torch.is_tensor(bundle['photo']) else bundle['photo']),
                    'cycle': float(bundle['cycle'].detach() if torch.is_tensor(bundle['cycle']) else bundle['cycle']),
                    'geo': float(bundle['geo'].detach()),
                    'inlier': float(bundle['inlier'].detach()),
                    'h_distill': float(bundle['h_distill'].detach()),
                    'h_match': float(bundle['h_match'].detach()),
                    'h_residual_budget': float(bundle['h_residual_budget'].detach()),
                    'residual_distill': float(bundle['residual_distill'].detach()),
                    'stitch_smooth': float(bundle['stitch_smooth'].detach()),
                    'mask_reg': float(bundle['mask_reg'].detach()),
                    'inlier_coverage': float(bundle['inlier_coverage'].detach()),
                    'fold': float(bundle['fold'].detach()),
                    'mesh_lap': float(bundle['mesh_lap'].detach()),
                    'mesh_mag': float(bundle['mesh_mag'].detach()),
                    'area_penalty': float(bundle['area_penalty'].detach()),
                    'h_inlier_used': float(bundle.get('h_inlier_used', 0.0)),
                }
                smoothed = loss_tracker.update(loss_metrics)
                stitch_ratio = float(bundle['stitch_ratio'])
                pbar.set_postfix({
                    'Loss': f"{smoothed['total_ema']:.4f}",
                    'Photo': f"{smoothed['photo_ema']:.4f}",
                    'Inlier': f"{smoothed['inlier_ema']:.4f}",
                    'H': f"{smoothed['h_distill_ema']:.4f}",
                    'StitchR': f"{stitch_ratio:.3f}",
                    'Smooth': f"{smoothed['stitch_smooth_ema']:.4f}",
                })

                should_log = global_step == 1 or (args.log_interval > 0 and global_step % args.log_interval == 0)
                if should_log:
                    for k, v in loss_metrics.items():
                        writer.add_scalar(f"TrainRaw/{k}", v, global_step)
                    for k, v in smoothed.items():
                        writer.add_scalar(f"TrainEMA/{k}", v, global_step)
                    log_training_visuals(writer, "TrainDiag", global_step, {
                        'mask_ratio': stitch_ratio,
                        'confidence_mean': bundle['conf_mean'],
                        'valid_overlap_ratio': bundle['valid_overlap_ratio'],
                        'residual_mean': bundle['residual_mean'],
                        'area_ratio': bundle['area_ratio'],
                        'inlier_mean': bundle['inlier_mean'],
                    })
                    for j, g in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"LR/group_{j}_{g.get('name', '')}", g['lr'], global_step)
                    writer.flush()

                log_images_this_step = args.vis_interval > 0 and global_step % args.vis_interval == 0
                if log_images_this_step:
                    with torch.no_grad():
                        visualize_results(img_a, img_b, stu_out, global_step, writer)

                if args.val_interval > 0 and global_step % args.val_interval == 0:
                    val_losses = validate(
                        student=student, val_loader=val_loader,
                        distill_loss_fn=distill_loss_fn, geo_loss_fn=geo_loss_fn,
                        w=w, device=device, use_amp=use_amp,
                        writer=writer, step=global_step,
                        stage=current_stage,
                        log_images=log_images_this_step,
                    )
                    val_total = val_losses['total']
                    last_val_loss = val_total
                    pbar.write(f"[val] step={global_step} | total={val_total:.4f} | "
                               f"photo={val_losses['photo']:.4f} | smooth={val_losses['stitch_smooth']:.4f}")
                    for k, v in val_losses.items():
                        writer.add_scalar(f"Val/{k}", v, global_step)
                    writer.flush()
                    is_best = val_total < best_val_loss
                    if is_best:
                        best_val_loss = val_total
                    save_checkpoint(
                        args.save_dir, epoch, global_step, student, optimizer, scheduler,
                        scaler if use_amp else None, args,
                        loss_tracker.get_ema('total'), val_total, best_val_loss, is_best,
                        student_ema=student_ema,
                    )

        pbar.write(f"[epoch {epoch}] Done. EMA={loss_tracker.get_ema('total'):.4f}")
        if args.save_every_epoch:
            save_checkpoint(
                args.save_dir, epoch, global_step, student, optimizer, scheduler,
                scaler if use_amp else None, args,
                loss_tracker.get_ema('total'), last_val_loss, best_val_loss, False,
                student_ema=student_ema,
            )
            writer.flush()

    writer.close()


if __name__ == "__main__":
    main()
