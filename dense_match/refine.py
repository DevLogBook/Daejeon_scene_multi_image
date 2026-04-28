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
    assert grid.ndim == 4 and grid.shape[-1] == 2, f"grid must be (B,H_out,W_out,2), got {grid.shape}"
    assert grid.shape[0] == feat.shape[0], "grid batch must match feat batch"

    # grid_sample 对 grid dtype 更敏感，显式对齐 dtype/device
    out_dtype = feat.dtype
    with torch.amp.autocast("cuda", enabled=False):
        feat_f = torch.nan_to_num(feat.float(), nan=0.0, posinf=0.0, neginf=0.0)
        grid_f = torch.nan_to_num(
            grid.to(device=feat.device, dtype=torch.float32),
            nan=0.0, posinf=1.5, neginf=-1.5,
        ).clamp(-1.5, 1.5)

    # padding_mode='border' 对拼接更稳：越界时取边界值而不是 0
        sampled = F.grid_sample(
            feat_f,
            grid_f,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
    return sampled.to(out_dtype)


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
    norm_A = (feat_A_flat ** 2).sum(dim=1).add(1e-8).sqrt()
    norm_B = (unfolded ** 2).sum(dim=1).add(1e-8).sqrt()

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
            max_pixel_delta: int = 4,  # ← 改为像素数上限，而非固定归一化值
            residual_overlap: bool = True,
            corr_radius: int = 4  # 从 2 扩大到 4：搜索窗口 9×9，覆盖更大位移
    ):
        super().__init__()
        self.corr_radius = corr_radius
        corr_channels = (2 * corr_radius + 1) ** 2
        self.C = C
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.max_pixel_delta = max_pixel_delta  # 最大修正像素数
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


class StitchingDecoder(nn.Module):
    """
    U-Net 风格的高效 Decoder。
    先将高分辨率 RGB 降采样到特征分辨率，完成轻量融合后，再上采样输出。
    """
    def __init__(
            self,
            feat_channels: int,
            hidden: int = 128,
            num_blocks: int = 3,
            residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        
        # RGB 降采样编码器 (H, W -> H/8, W/8) 提取高频细节
        # 输入 6 通道: img_A + img_B_warped
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, feat_channels, kernel_size=3, stride=2, padding=1), nn.GELU(),
        )

        # 特征融合层 (在 H/8, W/8 小尺度进行，极度省显存)
        # feat_A(C) + feat_B_warped(C) + rgb_encoded(C) + flow(2)
        in_ch = 3 * feat_channels + 2
        self.fuse = ConvGNAct(in_ch, hidden, k=3, s=1, p=1)

        # 在这个尺度下堆叠 DWConv，计算量和显存占用缩小了 64 倍！
        self.num_blocks = int(max(1, num_blocks))
        blocks = [DWConvGNAct(hidden, expansion=2) for _ in range(self.num_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # 渐进式上采样解码器 (H/8, W/8 -> H, W)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden, 64, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.GELU(),
        )

        self.mask_head = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.flow_head = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)

        nn.init.zeros_(self.mask_head.weight); nn.init.zeros_(self.mask_head.bias)
        nn.init.zeros_(self.flow_head.weight); nn.init.zeros_(self.flow_head.bias)

    def forward(
            self,
            img_A: torch.Tensor,
            img_B_warped: torch.Tensor,
            feat_A: torch.Tensor,
            feat_B_warped: torch.Tensor,
            base_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = img_A.shape
        
        # RGB 降采样
        rgb_concat = torch.cat([img_A, img_B_warped], dim=1)
        rgb_encoded = self.rgb_stem(rgb_concat) # [B, C, H/8, W/8]

        # 提取基准偏移流，并缩放到 H/8
        src_grid = make_grid(B, H, W, device=img_A.device, dtype=img_A.dtype)
        base_flow = (base_grid - src_grid).permute(0, 3, 1, 2).contiguous()
        base_flow_down = F.interpolate(base_flow, size=rgb_encoded.shape[-2:], mode="area")

        # 在小尺度下进行深层拼接与卷积
        x = torch.cat([feat_A, feat_B_warped, rgb_encoded, base_flow_down], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)

        # 上采样回全分辨率
        x = self.up(x) # [B, 16, H, W]

        # 输出
        mask_logits = self.mask_head(x).permute(0, 2, 3, 1).contiguous()
        stitch_mask = torch.sigmoid(mask_logits)

        flow = self.flow_head(x)
        flow = torch.tanh(flow) * self.residual_scale
        flow = flow.permute(0, 2, 3, 1).contiguous()

        return flow, mask_logits, stitch_mask


def ssim_map(img1, img2, window_size=7, sigma=1.5):
    _, C, H, W = img1.shape
    device = img1.device

    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size).float() - window_size // 2) ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, sigma).unsqueeze(1).to(device=device, dtype=torch.float32)
    window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    # 均值 E[x]
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 方差 E[x^2] - (E[x])^2
    sigma1_sq = (F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq).clamp(min=0.0)
    sigma2_sq = (F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq).clamp(min=0.0)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # SSIM 公式
    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return ssim_n / (ssim_d + 1e-8)


def compute_gradient_weight_map(
        img: torch.Tensor,
        amplify: float = 4.0,
        blur_radius: int = 1,
) -> torch.Tensor:
    """
    用 Sobel 算子提取图像梯度幅值，生成逐像素的损失权重图。
    梯度大（高频纹理/边缘）的区域权重被放大 amplify 倍，
    迫使网络优先对齐茎秆、田垄等结构性特征线。

    Args:
        img:         [B, C, H, W]  归一化后的图像张量（float32）
        amplify:     高梯度区域最大权重倍率，建议 3-5
        blur_radius: Sobel 前先用均值模糊降噪（0=不模糊）

    Returns:
        weight_map:  [B, 1, H, W]  权重范围 [1.0, amplify]，float32
    """
    assert img.ndim == 4, "img must be [B,C,H,W]"
    img_f = img.float()
    B, C, H, W = img_f.shape
    device = img_f.device

    # 转灰度
    gray = img_f.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # 可选：均值模糊降噪（防止孤立噪声点被当作边缘）
    if blur_radius > 0:
        k = 2 * blur_radius + 1
        blur_k = torch.ones(1, 1, k, k, device=device, dtype=torch.float32) / (k * k)
        gray = F.conv2d(gray, blur_k, padding=blur_radius)

    # Sobel 核
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=device, dtype=torch.float32
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-2, -1).contiguous()

    gx = F.conv2d(gray, sobel_x, padding=1)  # [B, 1, H, W]
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = (gx ** 2 + gy ** 2).sqrt()  # [B, 1, H, W]

    # 每张图独立归一化到 [0, 1]（避免批次间光照差异影响权重分布）
    max_val = grad_mag.flatten(1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-6)
    grad_norm = (grad_mag / max_val).clamp(0.0, 1.0)

    # 线性混合：weight = 1 + (amplify - 1) * grad_norm
    # → 低梯度区域权重=1.0，高梯度区域权重=amplify
    weight_map = 1.0 + (amplify - 1.0) * grad_norm  # [B, 1, H, W]
    return weight_map.detach()  # 不反传梯度到输入图像


def compute_photometric_loss(
        img_a,
        img_b,
        dense_grid,
        confidence,
        use_ssim: bool = False,
        alpha: float = 0.85,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        grad_weight_map: torch.Tensor = None,
):
    """
    光度损失，支持可选的梯度权重图（Seam-weighted Loss）。

    grad_weight_map: [B, 1, H, W]，由 compute_gradient_weight_map() 从 img_a 生成。
                     梯度高的区域（田垄/茎秆边缘）权重放大 3-5 倍，
                     迫使网络优先对齐高频结构线而非纹理均匀区域。
                     传 None 时退化为原始等权损失。
    """
    assert img_a.dtype == torch.float32, f"img_a dtype={img_a.dtype}, expected float32"
    assert img_b.dtype == torch.float32, f"img_b dtype={img_b.dtype}, expected float32"
    B, C, H, W = img_a.shape
    device = img_a.device

    img_a_f = img_a.float()
    img_b_f = img_b.float()
    dense_grid_f = dense_grid.float()

    mean = torch.tensor(list(img_mean), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(list(img_std), device=device, dtype=torch.float32).view(1, 3, 1, 1)

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
    soft_mask = F.grid_sample(ones, dense_grid_f, mode='bilinear',
                              padding_mode='zeros', align_corners=False)
    # soft_mask 保留梯度用于 area_penalty，hard mask 用于 loss 权重
    mask = (soft_mask > 0.9).float().detach()

    l1_map = torch.abs(img_curr - warped_b).mean(dim=1, keepdim=True)  # [B,1,H,W]

    if use_ssim:
        s_map = ssim_map(img_curr, warped_b)
        s_map_masked = s_map * mask.expand_as(s_map)
        ssim_loss_map = (1.0 - s_map_masked.mean(dim=1, keepdim=True))
        photo_loss_map = alpha * ssim_loss_map + (1.0 - alpha) * l1_map
    else:
        photo_loss_map = l1_map

    # 置信度权重
    conf = confidence.float()
    if conf.ndim == 3:
        conf = conf.unsqueeze(1)
    if conf.shape[-2:] != target_hw:
        conf = F.interpolate(conf, size=target_hw, mode='bilinear', align_corners=False)

    base_weight = (conf.clamp(0.0, 1.0) * mask).detach()

    # Seam-weighted：将梯度权重图叠加到 base_weight
    if grad_weight_map is not None:
        gw = grad_weight_map.float()
        if gw.shape[-2:] != target_hw:
            gw = F.interpolate(gw, size=target_hw, mode='bilinear', align_corners=False)
        # 梯度权重只作用于 valid 区域（mask>0），不扩展 valid 范围
        total_weight = base_weight * gw.detach()
    else:
        total_weight = base_weight

    return (photo_loss_map * total_weight).sum() / (total_weight.sum() + 1e-6), soft_mask.mean()


def compute_cycle_consistency_loss(stu_out_ab, stu_out_ba):
    # grid_ab 是 Detach 的物理基准，代表 A -> B
    grid_ab = stu_out_ab['matcher_out']['warp_AB'].detach()  # [B, Hf, Wf, 2]
    conf_a = stu_out_ab['matcher_out']['confidence_AB'].detach()  # [B, Hf, Wf]

    # grid_ba 是需要被优化训练的参数，代表 B -> A
    grid_ba = stu_out_ba['matcher_out']['warp_AB']  # [B, Hf, Wf, 2]

    B, H, W, _ = grid_ab.shape
    device = grid_ab.device

    # 因为 grid_ab 被 detach 了，所以这不会截断梯度，只作为客观物理条件的 Mask
    valid_in_B = (grid_ab[..., 0].abs() <= 1.0) & \
                 (grid_ab[..., 1].abs() <= 1.0)  # [B, H, W], bool

    # 使用 padding_mode='border' 防止边界采样产生指向图片中心的错误梯度
    back_to_a = F.grid_sample(
        grid_ba.permute(0, 3, 1, 2),  # [B, 2, Hf, Wf]
        grid_ab,
        mode='bilinear',
        padding_mode='border',  # 核心防线！
        align_corners=False,
    ).permute(0, 2, 3, 1)  # [B, Hf, Wf, 2]

    # 构建理论完美的 identity 坐标
    ys = (torch.arange(H, device=device, dtype=grid_ab.dtype) + 0.5) * (2.0 / H) - 1.0
    xs = (torch.arange(W, device=device, dtype=grid_ab.dtype) + 0.5) * (2.0 / W) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    diff = back_to_a.float() - identity.float()
    cycle_error = (diff ** 2).sum(dim=-1).clamp(min=1e-8).sqrt()  # [B, H, W]

    conf_mask = (conf_a > 0.1)  # [B, H, W], bool
    final_mask = (valid_in_B & conf_mask).float()

    # 防止所有点都失效导致的除以零 NaN
    valid_count = final_mask.sum().clamp(min=1.0)

    # 计算最终带有权重的平均误差
    weighted_error = cycle_error * conf_a * final_mask

    return weighted_error.sum() / valid_count


if __name__ == "__main__":
    pass
