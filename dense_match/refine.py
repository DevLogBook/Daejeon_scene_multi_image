import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Utils
def make_grid(
    B: int,
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    生成 align_corners=False 风格的 normalized grid。
    返回: (B, H, W, 2), 最后维度为 (x, y)，范围约 [-1, 1]
    """
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (2.0 / W) - 1.0
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (2.0 / H) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()


def safe_grid_sample(
    feat: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    安全封装 F.grid_sample，保证 shape 正确。
    feat: (B, C, H, W)
    grid: (B, H, W, 2)  normalized coords (x,y)
    返回: (B, C, H, W)
    """
    assert feat.ndim == 4, f"feat must be (B,C,H,W), got {feat.shape}"
    assert grid.ndim == 4 and grid.shape[-1] == 2, f"grid must be (B,H,W,2), got {grid.shape}"
    B, C, H, W = feat.shape
    assert grid.shape[0] == B, "grid batch must match feat batch"
    assert grid.shape[1] == H and grid.shape[2] == W, "grid spatial must match feat spatial"

    # grid_sample 对 grid dtype 更敏感，显式对齐 dtype/device
    grid = grid.to(device=feat.device, dtype=feat.dtype)

    # padding_mode='border' 对拼接更稳：越界时取边界值而不是 0
    sampled = F.grid_sample(
        feat,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return sampled


def prob_to_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    p in [0,1] -> logit
    """
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def upsample_warp_and_overlap(
    coarse_warp: torch.Tensor,
    coarse_overlap: torch.Tensor,
    out_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 coarse 输出上采样到 refine 分辨率。
    coarse_warp: (B, Hc, Wc, 2)  normalized coords (x,y)
    coarse_overlap: (B, Hc, Wc) or (B, Hc, Wc, 1)
    返回:
      warp_up: (B, Hr, Wr, 2)
      overlap_up: (B, Hr, Wr, 1)
    """
    assert coarse_warp.ndim == 4 and coarse_warp.shape[-1] == 2, "warp must be (B,H,W,2)"
    H, W = out_hw

    # warp: (B,H,W,2) -> (B,2,H,W) -> upsample -> (B,H_out,W_out,2)
    warp_chw = coarse_warp.permute(0, 3, 1, 2).contiguous()
    warp_up = F.interpolate(warp_chw, size=(H, W), mode="bilinear", align_corners=False)
    warp_up = warp_up.permute(0, 2, 3, 1).contiguous()

    # overlap
    if coarse_overlap.ndim == 3:
        coarse_overlap = coarse_overlap.unsqueeze(-1)
    assert coarse_overlap.ndim == 4 and coarse_overlap.shape[-1] == 1, "overlap must end with 1 chan"
    
    overlap_chw = coarse_overlap.permute(0, 3, 1, 2).contiguous()
    overlap_up = F.interpolate(overlap_chw, size=(H, W), mode="bilinear", align_corners=False)
    overlap_up = overlap_up.permute(0, 2, 3, 1).contiguous()

    return warp_up, overlap_up


# Lightweight Conv Blocks
def _gn_groups(ch: int, max_groups: int = 8) -> int:
    # 让 GroupNorm 组数尽量整除且不太大
    for g in [max_groups, 4, 2, 1]:
        if ch % g == 0:
            return g
    return 1


def compute_local_correlation(
        feat_A: torch.Tensor,  # (B, C, H, W)
        sampled_B: torch.Tensor,  # (B, C, H, W)
        radius: int = 2,
    ) -> torch.Tensor:  # (B, (2*radius+1)^2, H, W)
    B, C, H, W = feat_A.shape
    ws = 2 * radius + 1

    # 对 sampled_B 做 replicate padding
    padded_B = F.pad(sampled_B, [radius] * 4, mode='replicate')  # (B,C,H+2r,W+2r)

    # F.unfold 一次性提取所有 ws×ws 邻域 patch
    # 输出形状: (B, C*ws^2, H*W)
    unfolded = F.unfold(padded_B, kernel_size=ws)

    # 整理为 (B, C, ws^2, H*W) 方便与 feat_A 做点积
    unfolded = unfolded.view(B, C, ws * ws, H * W)

    # feat_A 展平并增加 ws^2 维度以便广播
    feat_A_flat = feat_A.view(B, C, 1, H * W)  # (B, C, 1, H*W)

    # 沿通道维度求点积
    dot = (feat_A_flat * unfolded).sum(dim=1)  # (B, ws^2, H*W)

    # L2 归一化
    norm_A = feat_A_flat.norm(dim=1)  # (B, 1,    H*W)
    norm_B = unfolded.norm(dim=1)  # (B, ws^2, H*W)

    # 余弦相似度
    corr = dot / (norm_A * norm_B + 1e-8)  # (B, ws^2, H*W)
    return corr.view(B, ws * ws, H, W)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, dilation: int = 1):
        super().__init__()
        if p is None:
            p = (k // 2) * dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=p, dilation=dilation, bias=False)
        self.gn = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class DWConvGNAct(nn.Module):
    """
    Depthwise 3x3 + Pointwise 1x1，轻量且适合部署
    """
    def __init__(self, ch: int, expansion: int = 2):
        super().__init__()
        mid = ch * expansion
        self.pw1 = ConvGNAct(ch, mid, k=1, s=1, p=0)
        self.dw = nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, groups=mid, bias=False)
        self.gn = nn.GroupNorm(_gn_groups(mid), mid)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(mid, ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(_gn_groups(ch), ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.pw1(x)
        x = self.act(self.gn(self.dw(x)))
        x = self.act(self.gn2(self.pw2(x)))
        return x + r


class WarpRefiner(nn.Module):
    def __init__(
        self,
        C: int,
        hidden: int = 96,
        num_blocks: int = 2,
        max_pixel_delta: int = 4,   # ← 改为像素数上限，而非固定归一化值
        residual_overlap: bool = True,
        corr_radius: int = 2
    ):
        super().__init__()
        self.corr_radius = corr_radius
        corr_channels = (2 * corr_radius + 1) ** 2
        self.C = C
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.max_pixel_delta = max_pixel_delta   # 最大修正像素数
        self.residual_overlap = bool(residual_overlap)

        in_ch = 4 * C + 5 + corr_channels + 2
        self.stem = ConvGNAct(in_ch, hidden, k=1, s=1, p=0)
        self.blocks = nn.Sequential(
            DWConvGNAct(hidden, expansion=2),
            ConvGNAct(hidden, hidden, k=3, dilation=2),
            DWConvGNAct(hidden, expansion=2)
        )
        self.delta_head = nn.Conv2d(hidden, 2, kernel_size=3, stride=1, padding=1)
        self.ov_head = nn.Conv2d(hidden, 1, kernel_size=3, stride=1, padding=1)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.ov_head.weight)
        nn.init.zeros_(self.ov_head.bias)

    def forward(
        self,
        feat_A: torch.Tensor,
        feat_B: torch.Tensor,
        prev_warp: torch.Tensor,
        prev_overlap: torch.Tensor,
    ):
        B, C, H, W = feat_A.shape
        H_target, W_target = prev_warp.shape[1], prev_warp.shape[2]

        if (H_target, W_target) != (H, W):
            prev_warp, prev_overlap = upsample_warp_and_overlap(
                prev_warp, prev_overlap, out_hw=(H, W),
            )
            H_target, W_target = H, W

        if prev_overlap.ndim == 3:
            prev_overlap = prev_overlap.unsqueeze(-1)

        sampled_B = safe_grid_sample(feat_B, prev_warp, align_corners=False)
        src_grid = make_grid(B, H, W, device=feat_A.device, dtype=feat_A.dtype)
        corr = compute_local_correlation(feat_A, sampled_B, radius=self.corr_radius)

        prev_overlap_chw = prev_overlap.permute(0, 3, 1, 2).contiguous()
        prev_warp_chw = prev_warp.permute(0, 3, 1, 2).contiguous()
        src_grid_chw = src_grid.permute(0, 3, 1, 2).contiguous()

        absdiff = (feat_A - sampled_B).abs()
        prod = feat_A * sampled_B
        flow_chw = (prev_warp - src_grid).permute(0, 3, 1, 2).contiguous()

        x = torch.cat(
            [feat_A, sampled_B, absdiff, prod, corr,
             prev_overlap_chw, prev_warp_chw, src_grid_chw, flow_chw],
            dim=1,
        )
        x = self.stem(x)
        x = self.blocks(x)

        # 动态 delta_scale：根据当前特征图分辨率计算归一化修正上限
        # max_pixel_delta 个像素对应的归一化距离 = max_pixel_delta * (2/H)
        # 对 H 和 W 取较小值，保守约束
        delta_scale = self.max_pixel_delta * (2.0 / min(H, W))

        delta = self.delta_head(x)
        delta = delta_scale * torch.tanh(delta)  # 修正量被约束在 ±max_pixel_delta 像素以内

        if delta.shape[2] != H_target or delta.shape[3] != W_target:
            delta = F.interpolate(delta, size=(H_target, W_target),
                                  mode='bilinear', align_corners=False)
        delta = delta.permute(0, 2, 3, 1).contiguous()
        refined_warp = prev_warp + delta

        ov_delta = self.ov_head(x)
        if ov_delta.shape[2] != H_target or ov_delta.shape[3] != W_target:
            ov_delta = F.interpolate(ov_delta, size=(H_target, W_target),
                                     mode='bilinear', align_corners=False)
        ov_delta = ov_delta.permute(0, 2, 3, 1).contiguous()

        if self.residual_overlap:
            if prev_overlap.ndim == 3:
                prev_overlap = prev_overlap.unsqueeze(-1)
            prev_logit = prob_to_logit(prev_overlap)
            overlap_logits = prev_logit + ov_delta
        else:
            overlap_logits = ov_delta

        overlap = torch.sigmoid(overlap_logits)
        return refined_warp, overlap_logits, overlap


def ssim_map(img1, img2, mask, window_size=7, sigma=1.5):
    _, C, H, W = img1.shape
    device = img1.device

    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size).float() - window_size // 2) ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, sigma).unsqueeze(1).to(device=device, dtype=torch.float32)  # 显式 fp32
    window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    # 预计算 Mask 的卷积权重（用于归一化均值）
    # mask 形状假设为 [B, 1, H, W]
    mask_weight = F.conv2d(mask.expand(-1, C, -1, -1), window, padding=window_size // 2, groups=C)
    mask_weight = mask_weight.clamp(min=1e-6)  # 避免除零

    # 计算归一化均值 mu = E[x] = conv(img * mask) / conv(mask)
    mu1 = F.conv2d(img1 * mask, window, padding=window_size // 2, groups=C) / mask_weight
    mu2 = F.conv2d(img2 * mask, window, padding=window_size // 2, groups=C) / mask_weight

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算归一化方差 sigma^2 = E[x^2] - (E[x])^2
    # E[x^2] = conv(img^2 * mask) / conv(mask)
    sigma1_sq = F.conv2d((img1 * img1) * mask, window, padding=window_size // 2, groups=C) / mask_weight - mu1_sq
    sigma2_sq = F.conv2d((img2 * img2) * mask, window, padding=window_size // 2, groups=C) / mask_weight - mu2_sq
    sigma12 = F.conv2d((img1 * img2) * mask, window, padding=window_size // 2, groups=C) / mask_weight - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # SSIM 公式
    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return ssim_n / (ssim_d + 1e-8)

def compute_cycle_loss(warp_AB, warp_BA, conf_A):
    """
    warp_AB: (B, H, W, 2) - A到B的坐标
    warp_BA: (B, H, W, 2) - B到A的坐标
    conf_A:  (B, H, W)    - A点的置信度
    """
    B, H, W, _ = warp_AB.shape
    # 生成 A 的原始坐标网格
    grid_A = make_grid(B, H, W, warp_AB.device, warp_AB.dtype) 
    
    # 将 warp_BA 重采样，得到“如果在B点，它认为对应的A点在哪里”
    # 然后根据 warp_AB 的指向，取回对应的坐标
    identity_A_pred = safe_grid_sample(warp_BA.permute(0,3,1,2), warp_AB).permute(0,2,3,1)
    
    # 计算循环误差
    cycle_dist = torch.norm(identity_A_pred - grid_A, dim=-1)
    return (cycle_dist * conf_A).mean()


def compute_photometric_loss(img_a, img_b, dense_grid, confidence,
                              use_ssim: bool = False, alpha: float = 0.85):
    assert img_a.dtype == torch.float32, f"img_a dtype={img_a.dtype}, expected float32"
    assert img_b.dtype == torch.float32, f"img_b dtype={img_b.dtype}, expected float32"
    B, C, H, W = img_a.shape
    device = img_a.device

    img_a_f = img_a.float()
    img_b_f = img_b.float()
    dense_grid_f = dense_grid.float()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    img_a_raw = (img_a_f * std + mean).clamp(0, 1)
    img_b_raw = (img_b_f * std + mean).clamp(0, 1)

    warped_b = F.grid_sample(img_b_raw, dense_grid_f, mode='bilinear',
                              padding_mode='zeros', align_corners=False)

    target_hw = warped_b.shape[-2:]
    if img_a_raw.shape[-2:] != target_hw:
        img_curr = F.interpolate(img_a_raw, size=target_hw, mode='bilinear', align_corners=False)
    else:
        img_curr = img_a_raw

    ones = torch.ones((B, 1, *target_hw), device=device, dtype=torch.float32)
    mask = F.grid_sample(ones, dense_grid_f, mode='nearest',
                          padding_mode='zeros', align_corners=False)
    mask = (mask > 0.9).float().detach()

    l1_map = torch.abs(img_curr - warped_b).mean(dim=1, keepdim=True)

    if use_ssim:
        s_map = ssim_map(img_curr, warped_b, mask)
        ssim_loss_map = 1.0 - s_map.mean(dim=1, keepdim=True)
        photo_loss_map = alpha * ssim_loss_map + (1.0 - alpha) * l1_map
    else:
        photo_loss_map = l1_map

    # 置信度对齐
    conf = confidence.float()
    if conf.ndim == 3:
        conf = conf.unsqueeze(1)
    if conf.shape[-2:] != target_hw:
        conf = F.interpolate(conf, size=target_hw, mode='bilinear', align_corners=False)

    total_weight = (conf * mask).detach() if use_ssim else mask.detach()

    return (photo_loss_map * total_weight).sum() / (total_weight.sum() + 1e-6)


def compute_cycle_consistency_loss(stu_out_ab, stu_out_ba):
    """
    高鲁棒性 A->B->A 坐标回环误差计算 (防坍塌设计)
    """
    # 切断前向梯度的传播 (Stop-Gradient)，将 A->B 的场作为绝对的 "伪标签"，只允许梯度更新 B->A 的网络参数。
    # 这打破了前向和后向网络共同退化为恒等映射(Identity)的捷径。
    grid_ab = stu_out_ab['dense_grid'].detach()  # [B, H, W, 2] -> 必须 detach
    conf_a = stu_out_ab['matcher_out']['confidence_AB'].detach()  # [B, h_m, w_m] -> 必须 detach

    grid_ba = stu_out_ba['dense_grid']  # [B, H, W, 2] -> 保留梯度

    B, H, W, _ = grid_ab.shape
    device = grid_ab.device

    # 构建 Identity 网格 (全分辨率)
    # y 轴和 x 轴生成时必须与 make_grid 的 align_corners=False 规范严格对齐
    ys = (torch.arange(H, device=device, dtype=grid_ab.dtype) + 0.5) * (2.0 / H) - 1.0
    xs = (torch.arange(W, device=device, dtype=grid_ab.dtype) + 0.5) * (2.0 / W) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # 采样回环坐标
    # 拿着 A->B 的坐标，去 B->A 的形变场里查表，看看它指回 A 的哪里
    back_to_a = F.grid_sample(
        grid_ba.permute(0, 3, 1, 2),
        grid_ab,
        mode='bilinear',
        padding_mode='border',  # 使用 border 比 zeros 更安全，防止边缘点回环出界导致大幅误判
        align_corners=False
    ).permute(0, 2, 3, 1)  # [B, H, W, 2]

    # 计算欧氏距离误差 [B, H, W]
    cycle_error = torch.norm(back_to_a - identity, dim=-1)

    # 置信度尺寸对齐
    if conf_a.shape[-2:] != (H, W):
        conf_a = F.interpolate(
            conf_a.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    # 动态掩码阈值截断，极低置信度的区域往往是视野外(Out-of-FOV)或严重遮挡，强行计算回环会让网络混乱。
    # 丢弃掉 conf < 0.1 的死区，只在有效区域计算 loss
    valid_mask = (conf_a > 0.1).float()

    # 加权求平均：依靠 detach 后的 confidence 作为权重，防止网络通过降低置信度来逃避 Loss
    weighted_error = cycle_error * conf_a * valid_mask

    return weighted_error.sum() / (valid_mask.sum() + 1e-6)

if __name__ == "__main__":
    pass