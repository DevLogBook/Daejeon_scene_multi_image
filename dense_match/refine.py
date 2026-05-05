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
    Create an align_corners=False normalized grid.

    Returns:
        Tensor of shape [B, H, W, 2] with the last dimension ordered as (x, y).
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
    Numerically safe wrapper around F.grid_sample.

    feat: (B, C, H, W)
    grid: (B, H, W, 2)  normalized coords (x,y)
    Returns: (B, C, H_out, W_out)
    """
    assert feat.ndim == 4, f"feat must be (B,C,H,W), got {feat.shape}"
    assert grid.ndim == 4 and grid.shape[-1] == 2, f"grid must be (B,H_out,W_out,2), got {grid.shape}"
    assert grid.shape[0] == feat.shape[0], "grid batch must match feat batch"

    # grid_sample is sensitive to grid dtype; force FP32 coordinates under AMP.
    out_dtype = feat.dtype
    with torch.amp.autocast("cuda", enabled=False):
        feat_f = torch.nan_to_num(feat.float(), nan=0.0, posinf=0.0, neginf=0.0)
        grid_f = torch.nan_to_num(
            grid.to(device=feat.device, dtype=torch.float32),
            nan=0.0, posinf=1.5, neginf=-1.5,
        ).clamp(-1.5, 1.5)

        # Border padding is often more stable for stitching than zero padding.
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
    Upsample coarse correspondence and overlap maps to the refinement resolution.

    coarse_warp: (B, Hc, Wc, 2)  normalized coords (x,y)
    coarse_overlap: (B, Hc, Wc) or (B, Hc, Wc, 1)
    Returns:
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
    # Prefer the largest valid GroupNorm group count up to max_groups.
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

    # Replicate padding keeps local correlation well-defined at borders.
    padded_B = F.pad(sampled_B, [radius] * 4, mode='replicate')  # (B,C,H+2r,W+2r)

    # F.unfold extracts every ws x ws neighborhood in one vectorized operation.
    # Output shape: (B, C*ws^2, H*W)
    unfolded = F.unfold(padded_B, kernel_size=ws)

    # Reshape to (B, C, ws^2, H*W) for dot products against feat_A.
    unfolded = unfolded.view(B, C, ws * ws, H * W)

    # Flatten feat_A and add the neighborhood dimension for broadcasting.
    feat_A_flat = feat_A.view(B, C, 1, H * W)  # (B, C, 1, H*W)

    # Channel-wise dot product.
    dot = (feat_A_flat * unfolded).sum(dim=1)  # (B, ws^2, H*W)

    # L2 normalization.
    norm_A = (feat_A_flat ** 2).sum(dim=1).add(1e-8).sqrt()
    norm_B = (unfolded ** 2).sum(dim=1).add(1e-8).sqrt()

    # Cosine similarity.
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
    Depthwise 3x3 followed by pointwise 1x1 convolution.

    This block keeps the decoder lightweight while preserving local spatial
    context, which is useful for high-resolution stitching.
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
            max_pixel_delta: int = 4,  # Pixel-domain upper bound for the delta.
            residual_overlap: bool = True,
            corr_radius: int = 4  # Radius 4 gives a 9x9 local search window.
    ):
        super().__init__()
        self.corr_radius = corr_radius
        corr_channels = (2 * corr_radius + 1) ** 2
        self.C = C
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.max_pixel_delta = max_pixel_delta  # Maximum correction in pixels.
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

        # Convert the pixel-domain delta bound to normalized grid units for the
        # current feature resolution. The min(H, W) denominator is conservative.
        delta_scale = self.max_pixel_delta * (2.0 / min(H, W))

        delta = self.delta_head(x)
        delta = delta_scale * torch.tanh(delta)  # Bound correction by max_pixel_delta.

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
    Efficient U-Net-style dense residual decoder.

    RGB evidence is first downsampled to the feature scale, fused with aligned
    features and the base flow, then decoded back to full image resolution.
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
        
        # RGB encoder (H, W -> H/8, W/8) captures high-frequency alignment cues.
        # Input has 6 channels: img_A concatenated with img_B_warped.
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, feat_channels, kernel_size=3, stride=2, padding=1), nn.GELU(),
        )

        # Feature fusion is performed at H/8, W/8 to reduce memory use.
        # feat_A(C) + feat_B_warped(C) + rgb_encoded(C) + flow(2)
        in_ch = 3 * feat_channels + 2
        self.fuse = ConvGNAct(in_ch, hidden, k=3, s=1, p=1)

        # Depthwise blocks at this scale provide local context at low cost.
        self.num_blocks = int(max(1, num_blocks))
        blocks = [DWConvGNAct(hidden, expansion=2) for _ in range(self.num_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # Progressive upsampling decoder (H/8, W/8 -> H, W).
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
        
        # Encode RGB evidence at reduced resolution.
        rgb_concat = torch.cat([img_A, img_B_warped], dim=1)
        rgb_encoded = self.rgb_stem(rgb_concat) # [B, C, H/8, W/8]

        # Convert the base grid to a base-flow representation and downsample it.
        src_grid = make_grid(B, H, W, device=img_A.device, dtype=img_A.dtype)
        base_flow = (base_grid - src_grid).permute(0, 3, 1, 2).contiguous()
        base_flow_down = F.interpolate(base_flow, size=rgb_encoded.shape[-2:], mode="area")

        # Fuse semantic, photometric, and geometric evidence at low resolution.
        x = torch.cat([feat_A, feat_B_warped, rgb_encoded, base_flow_down], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)

        # Decode back to full resolution.
        x = self.up(x) # [B, 16, H, W]

        # Predict residual flow and mask in NHWC layout for grid arithmetic.
        mask_logits = self.mask_head(x).permute(0, 2, 3, 1).contiguous()
        stitch_mask = torch.sigmoid(mask_logits)

        flow = self.flow_head(x)
        flow = torch.tanh(flow) * self.residual_scale
        flow = flow.permute(0, 2, 3, 1).contiguous()

        return flow, mask_logits, stitch_mask


class MeshStitchingDecoder(nn.Module):
    """
    Low-DOF stitching decoder.
    Predicts a coarse residual-control mesh and upsamples it to a dense residual field.
    """

    def __init__(
            self,
            feat_channels: int = 128,
            hidden: int = 128,
            num_blocks: int = 3,
            mesh_size: int = 12,
            max_residual_px: float = 4.0,
            min_mask_bias: float = -3.0,
    ):
        super().__init__()
        self.mesh_size = int(mesh_size)
        self.max_residual_px = float(max_residual_px)
        self.min_mask_bias = float(min_mask_bias)

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, feat_channels, kernel_size=3, stride=2, padding=1), nn.GELU(),
        )

        in_ch = 3 * feat_channels + 2
        self.fuse = ConvGNAct(in_ch, hidden, k=3, s=1, p=1)
        self.blocks = nn.Sequential(*[
            DWConvGNAct(hidden, expansion=2)
            for _ in range(int(max(1, num_blocks)))
        ])
        self.mesh_pool = nn.AdaptiveAvgPool2d((self.mesh_size, self.mesh_size))
        self.delta_head = nn.Sequential(
            ConvGNAct(hidden, hidden, k=3, s=1, p=1),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
        )
        self.mask_head = nn.Sequential(
            ConvGNAct(hidden, hidden, k=3, s=1, p=1),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.zeros_(self.mask_head[-1].bias)

    def forward(
            self,
            img_A: torch.Tensor,
            img_B_warped: torch.Tensor,
            feat_A: torch.Tensor,
            feat_B_warped: torch.Tensor,
            base_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        B, _, H, W = img_A.shape

        rgb_concat = torch.cat([img_A, img_B_warped], dim=1)
        rgb_feat = self.rgb_stem(rgb_concat)  # [B, C, H/8, W/8]
        fuse_hw = rgb_feat.shape[-2:]

        feat_A_low = feat_A
        if feat_A_low.shape[-2:] != fuse_hw:
            feat_A_low = F.interpolate(feat_A_low, size=fuse_hw, mode="bilinear", align_corners=False)
        feat_A_low = feat_A_low.to(rgb_feat.dtype)
        feat_B_low = feat_B_warped
        if feat_B_low.shape[-2:] != fuse_hw:
            feat_B_low = F.interpolate(feat_B_low, size=fuse_hw, mode="bilinear", align_corners=False)
        feat_B_low = feat_B_low.to(rgb_feat.dtype)

        with torch.amp.autocast("cuda", enabled=False):
            src_grid = make_grid(B, H, W, device=img_A.device, dtype=torch.float32)
            base_grid_f = torch.nan_to_num(
                base_grid.float(), nan=0.0, posinf=3.0, neginf=-3.0
            ).clamp(-3.0, 3.0)
            base_flow = (base_grid_f - src_grid).permute(0, 3, 1, 2).contiguous()
            base_flow_low = F.interpolate(base_flow, size=fuse_hw, mode="bilinear", align_corners=False)
        base_flow_low = base_flow_low.to(rgb_feat.dtype)

        x = torch.cat([feat_A_low, feat_B_low, rgb_feat, base_flow_low], dim=1)
        x = self.blocks(self.fuse(x))
        mesh_feat = self.mesh_pool(x)  # [B, hidden, Gh, Gw]

        normalized_scale = 2.0 * self.max_residual_px / max(float(min(H, W)), 1.0)
        mesh_delta = torch.tanh(self.delta_head(mesh_feat).float()) * normalized_scale
        mesh_delta = torch.nan_to_num(
            mesh_delta, nan=0.0, posinf=normalized_scale, neginf=-normalized_scale
        ).clamp(-normalized_scale, normalized_scale)

        mask_logits_lowres = self.mask_head(mesh_feat).float()
        mesh_mask_lowres = torch.sigmoid(mask_logits_lowres + self.min_mask_bias)
        mesh_mask_lowres = torch.nan_to_num(
            mesh_mask_lowres, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)

        residual_chw = F.interpolate(
            mesh_delta, size=(H, W), mode="bicubic", align_corners=False
        )
        residual_chw = torch.nan_to_num(
            residual_chw, nan=0.0, posinf=normalized_scale, neginf=-normalized_scale
        ).clamp(-normalized_scale, normalized_scale)
        residual_flow = residual_chw.permute(0, 2, 3, 1).contiguous().to(img_A.dtype)  # [B,H,W,2]

        mask_logits = F.interpolate(
            mask_logits_lowres + self.min_mask_bias,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()
        stitch_mask = torch.sigmoid(mask_logits)
        stitch_mask = torch.nan_to_num(
            stitch_mask, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0).to(img_A.dtype)  # [B,H,W,1]

        aux = {
            "mesh_delta": mesh_delta.to(img_A.dtype),  # [B,2,Gh,Gw]
            "mesh_mask_lowres": mesh_mask_lowres.to(img_A.dtype),  # [B,1,Gh,Gw]
            "mesh_size": self.mesh_size,
            "max_residual_px": self.max_residual_px,
        }
        return residual_flow, mask_logits.to(img_A.dtype), stitch_mask, aux


def mesh_laplacian_loss(mesh_delta: torch.Tensor) -> torch.Tensor:
    """Second-order curvature penalty for mesh_delta [B,2,Gh,Gw]."""
    if mesh_delta is None:
        raise ValueError("mesh_delta must not be None")
    delta = torch.nan_to_num(mesh_delta.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    loss = delta.new_zeros(())
    if delta.shape[-1] >= 3:
        dx2 = delta[..., :, 2:] - 2.0 * delta[..., :, 1:-1] + delta[..., :, :-2]
        loss = loss + dx2.square().mean()
    if delta.shape[-2] >= 3:
        dy2 = delta[..., 2:, :] - 2.0 * delta[..., 1:-1, :] + delta[..., :-2, :]
        loss = loss + dy2.square().mean()
    return loss


def mesh_magnitude_loss(mesh_delta: torch.Tensor) -> torch.Tensor:
    """Mean L2 magnitude penalty for mesh_delta [B,2,Gh,Gw]."""
    if mesh_delta is None:
        raise ValueError("mesh_delta must not be None")
    delta = torch.nan_to_num(mesh_delta.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    return delta.square().sum(dim=1).clamp_min(1e-12).sqrt().mean()


def grid_fold_loss(grid: torch.Tensor) -> torch.Tensor:
    """Penalize local Jacobian determinant <= 0 for grid [B,H,W,2]."""
    g = torch.nan_to_num(grid.float(), nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
    if g.shape[1] < 2 or g.shape[2] < 2:
        return g.new_zeros(())
    dx = g[:, :-1, 1:, :] - g[:, :-1, :-1, :]
    dy = g[:, 1:, :-1, :] - g[:, :-1, :-1, :]
    det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    return F.relu(1e-6 - det).mean()


def ssim_map(img1, img2, window_size=7, sigma=1.5):
    _, C, H, W = img1.shape
    device = img1.device

    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size).float() - window_size // 2) ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, sigma).unsqueeze(1).to(device=device, dtype=torch.float32)
    window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    # Mean term E[x].
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Variance term E[x^2] - E[x]^2.
    sigma1_sq = (F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq).clamp(min=0.0)
    sigma2_sq = (F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq).clamp(min=0.0)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # Standard SSIM numerator and denominator.
    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return ssim_n / (ssim_d + 1e-8)


def compute_gradient_weight_map(
        img: torch.Tensor,
        amplify: float = 4.0,
        blur_radius: int = 1,
) -> torch.Tensor:
    """
    Build a per-pixel photometric weight map from Sobel gradient magnitude.

    High-gradient image structures receive larger weights, encouraging the
    photometric loss to prioritize stable structural cues such as field rows,
    stems, and object boundaries.

    Args:
        img:         [B, C, H, W] normalized image tensor.
        amplify:     Maximum weight multiplier for high-gradient regions.
        blur_radius: Optional mean blur radius before Sobel filtering.

    Returns:
        weight_map:  [B, 1, H, W] with values in [1.0, amplify].
    """
    assert img.ndim == 4, "img must be [B,C,H,W]"
    img_f = img.float()
    B, C, H, W = img_f.shape
    device = img_f.device

    # Convert to grayscale.
    gray = img_f.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # Optional mean blur suppresses isolated noise before edge extraction.
    if blur_radius > 0:
        k = 2 * blur_radius + 1
        blur_k = torch.ones(1, 1, k, k, device=device, dtype=torch.float32) / (k * k)
        gray = F.conv2d(gray, blur_k, padding=blur_radius)

    # Sobel kernels for horizontal and vertical image gradients.
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=device, dtype=torch.float32
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-2, -1).contiguous()

    gx = F.conv2d(gray, sobel_x, padding=1)  # [B, 1, H, W]
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = (gx ** 2 + gy ** 2).sqrt()  # [B, 1, H, W]

    # Normalize each image independently to reduce batch-level illumination bias.
    max_val = grad_mag.flatten(1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-6)
    grad_norm = (grad_mag / max_val).clamp(0.0, 1.0)

    # Linear blend: low-gradient regions keep weight 1.0, while strong
    # gradients approach the configured amplify factor.
    weight_map = 1.0 + (amplify - 1.0) * grad_norm  # [B, 1, H, W]
    return weight_map.detach()  # Do not backpropagate through the input image.


if __name__ == "__main__":
    pass
