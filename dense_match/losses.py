import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sanitize_tensor(
    x: torch.Tensor,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
    clamp: Tuple[float, float] | None = None,
) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if clamp is not None:
        x = x.clamp(min=clamp[0], max=clamp[1])
    return x


def ssim_map(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    _, C, _, _ = img1.shape
    device = img1.device

    gauss = torch.exp(
        -(torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2) ** 2
        / (2 * sigma ** 2)
    )
    gauss = gauss / gauss.sum()
    window = gauss[:, None].mm(gauss[None, :]).view(1, 1, window_size, window_size)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq).clamp(min=0.0)
    sigma2_sq = (F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq).clamp(min=0.0)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return ssim_n / (ssim_d + 1e-8)


def compute_gradient_weight_map(
    img: torch.Tensor,
    amplify: float = 4.0,
    blur_radius: int = 1,
) -> torch.Tensor:
    assert img.ndim == 4, "img must be [B,C,H,W]"
    img_f = img.float()
    B, _, _, _ = img_f.shape
    device = img_f.device

    gray = img_f.mean(dim=1, keepdim=True)
    if blur_radius > 0:
        k = 2 * blur_radius + 1
        blur_k = torch.ones(1, 1, k, k, device=device, dtype=torch.float32) / (k * k)
        gray = F.conv2d(gray, blur_k, padding=blur_radius)

    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=device, dtype=torch.float32,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-2, -1).contiguous()

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = (gx ** 2 + gy ** 2).sqrt()
    max_val = grad_mag.flatten(1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-6)
    grad_norm = (grad_mag / max_val).clamp(0.0, 1.0)
    return (1.0 + (amplify - 1.0) * grad_norm).detach()


def compute_photometric_loss(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    dense_grid: torch.Tensor,
    confidence: torch.Tensor,
    use_ssim: bool = False,
    alpha: float = 0.85,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    grad_weight_map: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert img_a.dtype == torch.float32, f"img_a dtype={img_a.dtype}, expected float32"
    assert img_b.dtype == torch.float32, f"img_b dtype={img_b.dtype}, expected float32"
    B, _, _, _ = img_a.shape
    device = img_a.device

    dense_grid_f = dense_grid.float()
    mean = torch.tensor(list(img_mean), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(list(img_std), device=device, dtype=torch.float32).view(1, 3, 1, 1)

    img_a_raw = (img_a.float() * std + mean).clamp(0, 1)
    img_b_raw = (img_b.float() * std + mean).clamp(0, 1)
    warped_b = F.grid_sample(img_b_raw, dense_grid_f, mode="bilinear", padding_mode="zeros", align_corners=False)

    target_hw = warped_b.shape[-2:]
    if img_a_raw.shape[-2:] != target_hw:
        img_curr = F.interpolate(img_a_raw, size=target_hw, mode="bilinear", align_corners=False)
    else:
        img_curr = img_a_raw

    ones = torch.ones((B, 1, *target_hw), device=device, dtype=torch.float32)
    soft_mask = F.grid_sample(ones, dense_grid_f, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = (soft_mask > 0.9).float().detach()

    l1_map = torch.abs(img_curr - warped_b).mean(dim=1, keepdim=True)
    if use_ssim:
        s_map = ssim_map(img_curr, warped_b)
        s_map_masked = s_map * mask.expand_as(s_map)
        ssim_loss_map = 1.0 - s_map_masked.mean(dim=1, keepdim=True)
        photo_loss_map = alpha * ssim_loss_map + (1.0 - alpha) * l1_map
    else:
        photo_loss_map = l1_map

    conf = confidence.float()
    if conf.ndim == 3:
        conf = conf.unsqueeze(1)
    if conf.shape[-2:] != target_hw:
        conf = F.interpolate(conf, size=target_hw, mode="bilinear", align_corners=False)

    total_weight = (conf.clamp(0.0, 1.0) * mask).detach()
    if grad_weight_map is not None:
        gw = grad_weight_map.float()
        if gw.shape[-2:] != target_hw:
            gw = F.interpolate(gw, size=target_hw, mode="bilinear", align_corners=False)
        total_weight = total_weight * gw.detach()

    return (photo_loss_map * total_weight).sum() / (total_weight.sum() + 1e-6), soft_mask.mean()


def compute_cycle_consistency_loss(stu_out_ab: Dict, stu_out_ba: Dict) -> torch.Tensor:
    grid_ab = stu_out_ab["matcher_out"]["warp_AB"].detach()
    conf_a = stu_out_ab["matcher_out"]["confidence_AB"].detach()
    grid_ba = stu_out_ba["matcher_out"]["warp_AB"]

    B, H, W, _ = grid_ab.shape
    device = grid_ab.device
    valid_in_B = (grid_ab[..., 0].abs() <= 1.0) & (grid_ab[..., 1].abs() <= 1.0)

    back_to_a = F.grid_sample(
        grid_ba.permute(0, 3, 1, 2),
        grid_ab,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    ys = (torch.arange(H, device=device, dtype=grid_ab.dtype) + 0.5) * (2.0 / H) - 1.0
    xs = (torch.arange(W, device=device, dtype=grid_ab.dtype) + 0.5) * (2.0 / W) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    diff = back_to_a.float() - identity.float()
    cycle_error = (diff ** 2).sum(dim=-1).clamp(min=1e-8).sqrt()
    final_mask = (valid_in_B & (conf_a > 0.1)).float()
    valid_count = final_mask.sum().clamp(min=1.0)
    return (cycle_error * conf_a * final_mask).sum() / valid_count


def residual_smoothness_loss(residual_flow: torch.Tensor) -> torch.Tensor:
    flow = residual_flow.permute(0, 3, 1, 2).float()
    lap_x = flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
    lap_y = flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]
    return lap_x.pow(2).mean() + lap_y.pow(2).mean()


@torch.no_grad()
def build_inlier_pseudo_labels(
    warp_fine: torch.Tensor,
    teacher_warp: torch.Tensor,
    teacher_conf: torch.Tensor,
    sigma: float = 0.05,
    conf_thresh: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Hf, Wf, _ = warp_fine.shape
    teacher_warp_fine = F.interpolate(
        teacher_warp.permute(0, 3, 1, 2).float(),
        size=(Hf, Wf),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    teacher_conf_fine = F.interpolate(
        teacher_conf.unsqueeze(1).float(),
        size=(Hf, Wf),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    error = (warp_fine.float() - teacher_warp_fine).norm(dim=-1)
    pseudo_labels = torch.exp(-(error ** 2) / (2 * sigma ** 2))
    valid_mask = teacher_conf_fine > conf_thresh
    return pseudo_labels.reshape(B, Hf * Wf), valid_mask.reshape(B, Hf * Wf)


def compute_inlier_loss(
    inlier_weights: torch.Tensor,
    pseudo_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    teacher_conf: torch.Tensor,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    w = inlier_weights.squeeze(-1)
    y = pseudo_labels.float().detach()
    pt = w * y + (1.0 - w) * (1.0 - y)
    focal_weight = (1.0 - pt.detach()).pow(focal_gamma)
    bce = F.binary_cross_entropy(w.clamp(1e-6, 1 - 1e-6), y, reduction="none")
    weight = teacher_conf.float().detach() * valid_mask.float().detach()
    return (focal_weight * bce * weight).sum() / weight.sum().clamp(min=1.0)


class LocalGeometricConsistency(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("laplacian", lap.view(1, 1, 3, 3))

    def forward(self, flow: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = flow.shape
        f = _sanitize_tensor(
            flow.permute(0, 3, 1, 2).float(),
            nan=0.0,
            posinf=3.0,
            neginf=-3.0,
            clamp=(-3.0, 3.0),
        )
        lap = self.laplacian.to(device=f.device, dtype=f.dtype)

        if confidence.ndim == 3:
            conf = confidence.unsqueeze(1).float()
        elif confidence.shape[-1] == 1:
            conf = confidence.permute(0, 3, 1, 2).float()
        else:
            conf = confidence.unsqueeze(1).float()
        conf = _sanitize_tensor(conf, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        second_order = F.conv2d(f.reshape(B * 2, 1, H, W), lap, padding=0).abs()
        second_order = second_order.reshape(B, 2, H - 2, W - 2).mean(1, keepdim=True)
        conf_cropped = conf[:, :, 1:-1, 1:-1]
        denom = conf_cropped.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        return ((second_order * conf_cropped).sum(dim=(1, 2, 3)) / denom).mean()


class DistillationLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.4,
        beta_coarse: float = 1.0,
        beta_refine: float = 1.0,
        gamma: float = 0.1,
        eta_coarse: float = 0.5,
        eta_refine: float = 1.0,
        lambda_tv_coarse: float = 0.02,
        lambda_tv_refine: float = 0.05,
        conf_thresh_kl: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta_coarse = beta_coarse
        self.beta_refine = beta_refine
        self.gamma = gamma
        self.eta_coarse = eta_coarse
        self.eta_refine = eta_refine
        self.lambda_tv_coarse = lambda_tv_coarse
        self.lambda_tv_refine = lambda_tv_refine
        self.conf_thresh_kl = conf_thresh_kl

    @staticmethod
    def total_variation_loss(warp: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = warp.shape
        device = warp.device
        xs = (torch.arange(W, device=device, dtype=warp.dtype) + 0.5) * (2.0 / W) - 1.0
        ys = (torch.arange(H, device=device, dtype=warp.dtype) + 0.5) * (2.0 / H) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        identity_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        flow = warp - identity_grid
        dx2 = torch.abs(flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :])
        dy2 = torch.abs(flow[:, 2:, :, :] - 2 * flow[:, 1:-1, :, :] + flow[:, :-2, :, :])
        return dx2.mean() + dy2.mean()

    @staticmethod
    def _weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        pred = _sanitize_tensor(pred.float(), nan=0.0, posinf=1.5, neginf=-1.5, clamp=(-1.5, 1.5))
        target = _sanitize_tensor(target.float(), nan=0.0, posinf=1.0, neginf=-1.0, clamp=(-1.0, 1.0))
        err = F.huber_loss(pred, target, reduction="none", delta=delta).sum(dim=-1)
        weight = _sanitize_tensor(weight.float(), nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 2.0)).detach()
        return (err * weight).sum() / (weight.sum() + 1e-6)

    @staticmethod
    def _resize_warp(teacher_warp: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        teacher_warp_B = teacher_warp[..., -2:]
        teacher_warp_B = _sanitize_tensor(
            teacher_warp_B.float(), nan=0.0, posinf=1.5, neginf=-1.5, clamp=(-1.5, 1.5)
        )
        return F.interpolate(
            teacher_warp_B.permute(0, 3, 1, 2),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

    @staticmethod
    def _resize_conf(teacher_conf: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if teacher_conf.ndim == 4 and teacher_conf.shape[-1] == 1:
            teacher_conf = teacher_conf[..., 0]
        teacher_conf = _sanitize_tensor(
            teacher_conf.float(), nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0)
        )
        return F.interpolate(
            teacher_conf.unsqueeze(1),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    @staticmethod
    def _clamp_normalized(xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if H <= 1 or W <= 1:
            return xy
        x = xy[..., 0].clamp(min=-1.0 + 1.0 / W, max=1.0 - 1.0 / W)
        y = xy[..., 1].clamp(min=-1.0 + 1.0 / H, max=1.0 - 1.0 / H)
        return torch.stack([x, y], dim=-1)

    def _warp_to_prob(self, warp_B: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = warp_B.shape[0]
        N = H * W
        device = warp_B.device
        dtype = warp_B.dtype
        if H == 1 and W == 1:
            return torch.ones((B, 1, 1), device=device, dtype=dtype)

        warp_B_f = _sanitize_tensor(warp_B.float(), nan=0.0, posinf=1.0, neginf=-1.0, clamp=(-1.0, 1.0))
        xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * (2.0 / W) - 1.0
        ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * (2.0 / H) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        token_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        warp_flat = warp_B_f.reshape(B, N, 2)
        dist2 = ((warp_flat.unsqueeze(2) - token_coords.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)
        sigma = 0.5 * (2.0 / W + 2.0 / H)
        soft_prob = torch.exp((-dist2 / (2 * sigma ** 2 + 1e-8)).clamp(min=-50.0, max=0.0))
        soft_prob = torch.nan_to_num(soft_prob, nan=0.0, posinf=1.0, neginf=0.0)
        return (soft_prob / soft_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)).to(dtype)

    def forward(
        self,
        stu_output: Dict[str, torch.Tensor],
        teacher_warp: torch.Tensor,
        teacher_conf: torch.Tensor,
        teacher_feat_A: torch.Tensor,
        teacher_feat_B: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        Hc, Wc = stu_output.get("coarse_hw", (32, 32))
        Hf, Wf = stu_output.get("fine_hw", (64, 64))
        warp_coarse = _sanitize_tensor(stu_output["warp_AB_coarse"], nan=0.0, posinf=1.5, neginf=-1.5, clamp=(-1.5, 1.5))
        conf_logits_coarse = _sanitize_tensor(stu_output["conf_logits_coarse"].float(), nan=0.0, posinf=20.0, neginf=-20.0, clamp=(-20.0, 20.0))
        warp_refine = _sanitize_tensor(stu_output["warp_AB"], nan=0.0, posinf=1.5, neginf=-1.5, clamp=(-1.5, 1.5))
        conf_logits_refine = _sanitize_tensor(stu_output["conf_logits"].float(), nan=0.0, posinf=20.0, neginf=-20.0, clamp=(-20.0, 20.0))

        B = warp_coarse.shape[0]
        N_coarse = Hc * Wc
        stu_feat_A = stu_output["distill_feat_A"]
        stu_feat_B = stu_output["distill_feat_B"]
        N_teacher = teacher_feat_A.shape[1]

        if stu_feat_A.shape[1] != N_teacher:
            teacher_gs = int(math.sqrt(N_teacher))
            d = stu_feat_A.shape[-1]
            stu_feat_A_aligned = F.interpolate(
                stu_feat_A.reshape(B, Hc, Wc, d).permute(0, 3, 1, 2),
                size=(teacher_gs, teacher_gs),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).reshape(B, N_teacher, d)
            stu_feat_B_aligned = F.interpolate(
                stu_feat_B.reshape(B, Hc, Wc, d).permute(0, 3, 1, 2),
                size=(teacher_gs, teacher_gs),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).reshape(B, N_teacher, d)
        else:
            stu_feat_A_aligned = stu_feat_A
            stu_feat_B_aligned = stu_feat_B

        def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a_norm = F.normalize(a.float(), p=2, dim=-1)
            b_norm = F.normalize(b.float(), p=2, dim=-1)
            return (1.0 - (a_norm * b_norm).sum(dim=-1)).mean()

        loss_feat = _cosine_loss(stu_feat_A_aligned, teacher_feat_A) + _cosine_loss(stu_feat_B_aligned, teacher_feat_B)

        teacher_warp_c = self._clamp_normalized(self._resize_warp(teacher_warp, (Hc, Wc)), Hc, Wc)
        teacher_warp_r = self._clamp_normalized(self._resize_warp(teacher_warp, (Hf, Wf)), Hf, Wf)
        teacher_conf_c = _sanitize_tensor(self._resize_conf(teacher_conf, (Hc, Wc)), nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        teacher_conf_r = _sanitize_tensor(self._resize_conf(teacher_conf, (Hf, Wf)), nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))
        conf_mean = teacher_conf_c.mean(dim=(1, 2), keepdim=True).clamp(min=0.1)
        teacher_conf_c_normalized = (teacher_conf_c / conf_mean).clamp(0, 2.0)

        loss_warp_coarse = self._weighted_huber(warp_coarse, teacher_warp_c, teacher_conf_c_normalized)
        loss_warp_refine = self._weighted_huber(warp_refine, teacher_warp_r, teacher_conf_r)

        sim_matrix = stu_output.get("sim_matrix_kl")
        if sim_matrix is not None and sim_matrix.shape[1] == N_coarse:
            teacher_prob_c = self._warp_to_prob(teacher_warp_c, Hc, Wc)
            sim_for_kl = _sanitize_tensor(sim_matrix.float(), nan=0.0, posinf=50.0, neginf=-50.0, clamp=(-50.0, 50.0))
            log_prob = F.log_softmax(sim_for_kl, dim=-1)
            target_prob = torch.nan_to_num(teacher_prob_c.detach().float(), nan=0.0, posinf=1.0, neginf=0.0)
            target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            kl_map = F.kl_div(log_prob, target_prob, reduction="none").sum(dim=-1)
            valid_weight = teacher_conf_c.reshape(B, N_coarse).detach()
            valid_mask = (valid_weight > self.conf_thresh_kl).float()
            loss_kl = (kl_map * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        else:
            loss_kl = torch.zeros((), device=warp_coarse.device)

        loss_conf_coarse = F.binary_cross_entropy_with_logits(conf_logits_coarse, teacher_conf_c.detach())
        loss_conf_refine = F.binary_cross_entropy_with_logits(conf_logits_refine, teacher_conf_r.detach())
        loss_smooth_coarse = self.total_variation_loss(warp_coarse)
        loss_smooth_refine = self.total_variation_loss(warp_refine)

        total = (
            self.alpha * loss_feat
            + self.beta_coarse * loss_warp_coarse
            + self.beta_refine * loss_warp_refine
            + self.gamma * loss_kl
            + self.eta_coarse * loss_conf_coarse
            + self.eta_refine * loss_conf_refine
            + self.lambda_tv_coarse * loss_smooth_coarse
            + self.lambda_tv_refine * loss_smooth_refine
        )
        return {
            "total": total,
            "loss_feat": float(loss_feat.detach().item()),
            "loss_warp_coarse": float(loss_warp_coarse.detach().item()),
            "loss_warp_refine": float(loss_warp_refine.detach().item()),
            "loss_kl": float(loss_kl.detach().item()),
            "loss_conf_coarse": float(loss_conf_coarse.detach().item()),
            "loss_conf_refine": float(loss_conf_refine.detach().item()),
            "loss_smooth_coarse": float(loss_smooth_coarse.detach().item()),
            "loss_smooth_refine": float(loss_smooth_refine.detach().item()),
        }
