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

def compute_local_correlation(feat_A, sampled_B, radius=2):
    """
    计算 feat_A 与 sampled_B 局部邻域的相关性
    feat_A: (B, C, H, W)
    sampled_B: (B, C, H, W) - 已经是根据 prev_warp 采样后的特征
    radius: 2 表示 5x5 窗口
    返回: (B, (2*radius+1)^2, H, W)
    """
    B, C, H, W = feat_A.shape
    window_size = 2 * radius + 1
    
    # 对 sampled_B 进行 padding，以便处理边缘
    padded_B = F.pad(sampled_B, (radius, radius, radius, radius), mode='replicate')
    
    # 使用 unfold 提取所有 5x5 的 patch
    # output shape: (B, C * window_size^2, H * W)
    patches_B = F.unfold(padded_B, kernel_size=window_size)
    
    # 重塑为 (B, C, window_size^2, H, W)
    patches_B = patches_B.view(B, C, window_size**2, H, W)
    
    # 计算 feat_A 与每个 patch 位置的点积 (Correlation)
    # feat_A: (B, C, 1, H, W) * patches_B: (B, C, window_size^2, H, W)
    # 在 C 维度求和 -> (B, window_size^2, H, W)
    corr = (feat_A.unsqueeze(2) * patches_B).sum(dim=1)
    corr = corr / (torch.norm(feat_A, dim=1, keepdim=True) * torch.norm(patches_B, dim=1) + 1e-8)
    
    return corr


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
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
        x = self.gn2(self.pw2(x))
        return self.act(x + r)


# WarpRefiner
class WarpRefiner(nn.Module):
    """
    轻量级两级匹配的 refine block
    输入:
      feat_A: (B, C, H, W)
      feat_B: (B, C, H, W)
      prev_warp: (B, H, W, 2)  normalized coords (x,y) in [-1,1] with align_corners=False convention
      prev_overlap: (B, H, W) or (B, H, W, 1)
    输出:
      warp: (B, H, W, 2)
      overlap_logits: (B, H, W, 1)
      overlap: (B, H, W, 1)
    """
    def __init__(
        self,
        C: int,
        hidden: int = 96,
        num_blocks: int = 2,
        delta_scale: float = 0.4,
        residual_overlap: bool = True,
        corr_radius: int = 2
    ):
        super().__init__()
        self.corr_radius = corr_radius
        corr_channels = (2 * corr_radius + 1) ** 2
        self.C = C
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.delta_scale = float(delta_scale)
        self.residual_overlap = bool(residual_overlap)
        
        in_ch = 4 * C + 5 + corr_channels
        self.stem = ConvGNAct(in_ch, hidden, k=1, s=1, p=0)
        # self.blocks = nn.Sequential(*[DWConvGNAct(hidden, expansion=2) for _ in range(num_blocks)])
        self.blocks = nn.Sequential(
                    DWConvGNAct(hidden, expansion=2),
                    ConvGNAct(hidden, hidden, k=3, p=2), # Dilated Conv
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B, C, H, W = feat_A.shape
        H_target, W_target = prev_warp.shape[1], prev_warp.shape[2]

        # Robustness: in some call paths `prev_warp/prev_overlap` may come from a different scale.
        # `grid_sample` and the concatenation below require matching spatial sizes.
        if (H_target, W_target) != (H, W):
            prev_warp, prev_overlap = upsample_warp_and_overlap(
                prev_warp,
                prev_overlap,
                out_hw=(H, W),
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

        x = torch.cat(
            [feat_A, sampled_B, absdiff, prod, corr, 
             prev_overlap_chw, prev_warp_chw, src_grid_chw],
            dim=1,
        )
        x = self.stem(x)
        x = self.blocks(x)

        delta = self.delta_head(x)
        delta = self.delta_scale * torch.tanh(delta)
        if delta.shape[2] != H_target or delta.shape[3] != W_target:
            delta = F.interpolate(
                delta, 
                size=(H_target, W_target), 
                mode='bilinear', 
                align_corners=False
            )
        delta = delta.permute(0, 2, 3, 1).contiguous()

        refined_warp = prev_warp + delta

        ov_delta = self.ov_head(x)
        if ov_delta.shape[2] != H_target or ov_delta.shape[3] != W_target:
            ov_delta = F.interpolate(
                ov_delta, 
                size=(H_target, W_target), 
                mode='bilinear', 
                align_corners=False
            )
    
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


def ssim_map(img1, img2, window_size=11, sigma=1.5):
    """
    计算两张图之间的像素级 SSIM Map
    """
    _, C, _, _ = img1.shape
    
    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size).float() - window_size//2)**2 / (2 * sigma**2))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, sigma).unsqueeze(1).to(img1.device, img1.dtype)
    window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return ssim_n / ssim_d

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

import torch.nn.functional as F

def compute_photometric_loss(img_A, img_B, warp_AB, confidence_AB, alpha=0.85):
    """
    修正后的光度损失：仅计算有效重叠区域
    """
    B, C, H, W = img_A.shape
    device = img_A.device

    warped_B = F.grid_sample(img_B, warp_AB, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # 创建 img_B 尺寸的全 1 矩阵，进行同样的 Warp
    ones = torch.ones((B, 1, H, W), device=device, dtype=img_A.dtype)
    mask_B = F.grid_sample(ones, warp_AB, mode='nearest', padding_mode='zeros', align_corners=False)
    
    # 只要采样值 > 0.99，说明是 img_B 映射过来的有效像素
    overlap_mask = (mask_B > 0.99).float() 

    # L1 Loss Map
    l1_map = torch.abs(img_A - warped_B).mean(dim=1, keepdim=True) # [B, 1, H, W]
    
    # SSIM Map (注意：ssim_map 原本返回 0~1，我们需要 1 - SSIM 作为 Loss)
    s_map = ssim_map(img_A, warped_B) # [B, C, H, W] 或 [B, 1, H, W]
    ssim_loss_map = 1 - s_map.mean(dim=1, keepdim=True)

    photo_loss_map = alpha * ssim_loss_map + (1 - alpha) * l1_map

    # 确保 confidence_AB 维度对齐 [B, 1, H, W]
    if confidence_AB.ndim == 3:
        confidence_AB = confidence_AB.unsqueeze(1)
    
    final_weight = overlap_mask * confidence_AB

    # 使用 1e-6 防止重叠面积为 0 时除以零
    loss = (photo_loss_map * final_weight).sum() / (final_weight.sum() + 1e-6)
    
    return loss


def compute_cycle_consistency_loss(stu_out_ab, stu_out_ba):
    """计算 A->B->A 的坐标回环误差"""
    grid_ab = stu_out_ab['dense_grid']  # [B, H, W, 2] -> 全分辨率
    grid_ba = stu_out_ba['dense_grid']  # [B, H, W, 2]
    conf_a = stu_out_ab['matcher_out']['confidence_AB']  # [B, h_m, w_m] -> 匹配器分辨率 (96)

    B, H, W, _ = grid_ab.shape
    device = grid_ab.device

    # 构建 Identity 网格 (全分辨率)
    identity = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )[::-1], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # 采样回环坐标
    back_to_a = F.grid_sample(
        grid_ba.permute(0, 3, 1, 2),
        grid_ab,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    ).permute(0, 2, 3, 1)  # [B, H, W, 2]

    # 计算欧氏距离误差 [B, H, W]
    cycle_error = torch.norm(back_to_a - identity, dim=-1)

    if conf_a.shape[-2:] != (H, W):
        # [B, 96, 96] -> [B, 1, 96, 96] -> 插值 -> [B, 384, 384]
        conf_a = F.interpolate(
            conf_a.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    # 4. 加权求平均：只在模型认为有匹配的地方计算回环损失
    return (cycle_error * conf_a).mean()

if __name__ == "__main__":
    pass