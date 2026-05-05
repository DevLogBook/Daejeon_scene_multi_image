import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_match.backbone import MobileViTBackbone
from dense_match.refine import (WarpRefiner,
                                StitchingDecoder,
                                MeshStitchingDecoder,
                                upsample_warp_and_overlap,
                                make_grid, safe_grid_sample)
from dense_match.geometry import (
                                _sanitize_tensor,
                                _sanitize_homography,
                                _sanitize_base_transform,
                                project_grid_with_h,
                                solve_robust_base_transform_from_dense_flow)
from dense_match.heads import ContextAwareInlierPredictor, GlobalSimilarityHead


def _gn_groups(ch: int, max_groups: int = 8) -> int:
    """Compute a valid GroupNorm group count."""
    for g in [max_groups, 4, 2, 1]:
        if ch % g == 0:
            return g
    return 1


class MatchAttention(nn.Module):
    def __init__(self, d_model=128, nhead=4):
        super().__init__()
        self.self_attn_A = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.self_attn_B = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.cross_attn_A = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.cross_attn_B = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.norm1_A = nn.LayerNorm(d_model)
        self.norm1_B = nn.LayerNorm(d_model)
        self.norm2_A = nn.LayerNorm(d_model)
        self.norm2_B = nn.LayerNorm(d_model)
        self.ffn_A = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.ffn_B = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm3_A = nn.LayerNorm(d_model)
        self.norm3_B = nn.LayerNorm(d_model)

    def forward(self, feat_A, feat_B):
        feat_A = self.norm1_A(feat_A + self.self_attn_A(feat_A, feat_A, feat_A)[0])
        feat_B = self.norm1_B(feat_B + self.self_attn_B(feat_B, feat_B, feat_B)[0])

        feat_A_out = self.norm2_A(feat_A + self.cross_attn_A(feat_A, feat_B, feat_B)[0])
        feat_B_out = self.norm2_B(feat_B + self.cross_attn_B(feat_B, feat_A, feat_A)[0])

        # FFN
        feat_A_out = self.norm3_A(feat_A_out + self.ffn_A(feat_A_out))
        feat_B_out = self.norm3_B(feat_B_out + self.ffn_B(feat_B_out))

        return feat_A_out, feat_B_out


class MatchWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, window_size=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.mha_A = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.mha_B = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_A = nn.LayerNorm(d_model)
        self.norm_B = nn.LayerNorm(d_model)
        self.ffn_norm_A = nn.LayerNorm(d_model)
        self.ffn_norm_B = nn.LayerNorm(d_model)
        self.ffn_A = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.ffn_B = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )

        local_pe = self._generate_2d_sincos_pe(window_size, d_model)
        self.register_buffer('local_pe', local_pe)

    def _generate_2d_sincos_pe(self, window_size, d_model):
        """Generates 2D sine-cosine positional encoding for the local window."""
        pe = torch.zeros(window_size * window_size, d_model)
        d_half = d_model // 2

        # Create 2D grid coordinates for the window
        y_pos = torch.arange(window_size).unsqueeze(1).repeat(1, window_size).flatten().float()
        x_pos = torch.arange(window_size).unsqueeze(0).repeat(window_size, 1).flatten().float()

        # Calculate div term
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))

        # Apply sin/cos to y coordinates (first half of channels)
        pe[:, 0:d_half:2] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pe[:, 1:d_half:2] = torch.cos(y_pos.unsqueeze(1) * div_term)

        # Apply sin/cos to x coordinates (second half of channels)
        pe[:, d_half::2] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pe[:, d_half + 1::2] = torch.cos(x_pos.unsqueeze(1) * div_term)

        return pe.unsqueeze(0)  # Shape: [1, window_size*window_size, d_model]

    def window_partition(self, x, H, W):
        """Partition token features into local windows."""
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # Pad spatial dimensions so each feature map is divisible by window_size.
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows, (Hp, Wp)

    def window_reverse(self, windows, H, W, Hp, Wp):
        """Restore local windows back to the token feature map."""
        B = windows.shape[0] // ((Hp // self.window_size) * (Wp // self.window_size))
        x = windows.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        # Remove the padded region introduced by window partitioning.
        return x[:, :H, :W, :].reshape(B, H * W, -1)

    def forward(self, feat_A, feat_B, H, W):
        # Window Partition
        short_cut_A, short_cut_B = feat_A, feat_B
        win_A, (Hp, Wp) = self.window_partition(feat_A, H, W)
        win_B, _ = self.window_partition(feat_B, H, W)

        win_A = win_A + self.local_pe
        win_B = win_B + self.local_pe

        # Legacy same-window cross-attention. This assumes small local displacement
        # and is kept for checkpoint compatibility; the main AgriMatcher path no
        # longer uses it before coarse global matching.
        attn_A, _ = self.mha_A(win_A, win_B, win_B)
        attn_B, _ = self.mha_B(win_B, win_A, win_A)

        feat_A = self.window_reverse(attn_A, H, W, Hp, Wp)
        feat_B = self.window_reverse(attn_B, H, W, Hp, Wp)

        feat_A = self.norm_A(feat_A + short_cut_A)
        feat_A = self.ffn_norm_A(feat_A + self.ffn_A(feat_A))
        feat_B = self.norm_B(feat_B + short_cut_B)
        feat_B = self.ffn_norm_B(feat_B + self.ffn_B(feat_B))

        return feat_A, feat_B


class LocalGeometricValidator(nn.Module):
    """
    Estimate local geometric consistency for top-k correspondence candidates.

    Each candidate induces a displacement field. The validator measures the
    neighborhood variance of that field with a vectorized avg-pooling pass.
    `count_include_pad=False` avoids padding-induced variance bias near borders.
    """

    def __init__(self, neighbor_radius: int = 3):
        super().__init__()
        self.radius = neighbor_radius
        self.kernel_size = 2 * neighbor_radius + 1

    def forward(self, src_pos, candidate_positions, H, W):
        """
        src_pos:             [B, N, 2]
        candidate_positions: [B, N, K, 2]
        Returns:             [B, N, K] consistency scores; larger is better.
        """
        B, N, K, _ = candidate_positions.shape
        src_2d = src_pos.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # Merge K candidate fields into the batch dimension so one avg_pool2d
        # evaluates all candidate displacement fields.
        # candidate_positions: [B, N, K, 2] -> [B, K, N, 2] -> [B*K, H, W, 2]
        cand_all = candidate_positions.permute(0, 2, 1, 3).reshape(B * K, H, W, 2).permute(0, 3, 1, 2)
        src_expanded = src_2d.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, 2, H, W)
        disp_all = cand_all - src_expanded  # [B*K, 2, H, W]

        # Border pixels average over valid neighbors only; this removes the
        # systematic variance bias introduced by zero padding.
        mean_disp = F.avg_pool2d(
            disp_all, self.kernel_size, stride=1,
            padding=self.radius, count_include_pad=False
        )
        mean_sq = F.avg_pool2d(
            disp_all ** 2, self.kernel_size, stride=1,
            padding=self.radius, count_include_pad=False
        )
        var_disp = (mean_sq - mean_disp ** 2).clamp(min=0.0)  # [B*K, 2, H, W]

        total_var = var_disp.sum(dim=1).reshape(B, K, N)  # [B, K, N]
        score = 1.0 / (1.0 + total_var * 100.0)  # [B, K, N]
        return score.permute(0, 2, 1)  # [B, N, K]


class MultiPeakAwareAttention(nn.Module):
    def __init__(self, d_model=128, top_k=8, sinkhorn_iters=15):
        super().__init__()
        self.top_k = top_k
        self.sinkhorn_iters = sinkhorn_iters
        self.geometric_validator = LocalGeometricValidator(neighbor_radius=3)
        self.dustbin_score = nn.Parameter(torch.tensor(-1.0))
        self.geo_weight = nn.Parameter(torch.tensor(0.5))

    def optimal_transport(self, scores, current_temp):
        """
        scores: [B, M, N] similarity logits already scaled by current_temp.
        current_temp: temperature used to scale the dustbin score consistently.
        """
        # Force FP32 for log-domain optimal transport; AMP half precision can
        # overflow or underflow in logsumexp on low-overlap image pairs.
        scores_f32 = torch.nan_to_num(
            scores.float(), nan=0.0, posinf=50.0, neginf=-50.0
        ).clamp(-50.0, 50.0)
        B, M, N = scores_f32.shape
        device = scores_f32.device

        temp = torch.as_tensor(current_temp, device=device, dtype=torch.float32).clamp(min=0.20, max=10.0)
        ds = (self.dustbin_score.float() / temp).clamp(-50.0, 50.0)

        row_dust = ds.view(1, 1, 1).expand(B, M, 1)  # [B, M, 1]
        col_dust = ds.view(1, 1, 1).expand(B, 1, N)  # [B, 1, N]
        corner = torch.zeros(B, 1, 1, device=device)  # [B, 1, 1]
        top = torch.cat([scores_f32, row_dust], dim=2)  # [B, M, N+1]
        bot = torch.cat([col_dust, corner], dim=2)  # [B, 1, N+1]
        Z = torch.cat([top, bot], dim=1)

        norm = -math.log(M + N)
        log_mu = torch.empty(B, M + 1, device=device, dtype=torch.float32).fill_(norm)
        log_mu[:, M] = math.log(N) + norm

        log_nu = torch.empty(B, N + 1, device=device, dtype=torch.float32).fill_(norm)
        log_nu[:, N] = math.log(M) + norm

        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)

        # Run all but the final Sinkhorn iteration without gradients to reduce
        # memory while keeping the last normalization step differentiable.
        with torch.no_grad():
            for _ in range(self.sinkhorn_iters - 1):
                u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
                v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

        log_assign = Z + u.unsqueeze(2) + v.unsqueeze(1)
        log_assign = torch.nan_to_num(log_assign, nan=-50.0, posinf=0.0, neginf=-50.0)
        # Return the real assignment block; the caller controls downstream dtype.
        return log_assign[:, :M, :N]

    def forward(self, feat_A, feat_B, pos_A, pos_B, H, W, current_temp, pos_B_for_geo=None):
        B, N, C = feat_A.shape

        if pos_B_for_geo is None:
            pos_B_for_geo = pos_B

        norm_A = F.normalize(feat_A, p=2, dim=-1)
        norm_B = F.normalize(feat_B, p=2, dim=-1)
        raw_sim_matrix = torch.bmm(norm_A, norm_B.transpose(1, 2))

        scores = raw_sim_matrix / current_temp

        log_assign = self.optimal_transport(scores, current_temp)

        ot_prob_f32 = torch.exp(log_assign.float().clamp(min=-50.0, max=0.0))
        ot_prob_f32 = torch.nan_to_num(ot_prob_f32, nan=0.0, posinf=1.0, neginf=0.0)
        row_mass = ot_prob_f32.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        prob_dist = (ot_prob_f32 / row_mass).to(feat_A.dtype)
        valid_prob_sum = (row_mass * float(log_assign.shape[1] + log_assign.shape[2])).clamp(0.0, 1.0).to(feat_A.dtype)

        base_expected_pos_B = torch.bmm(prob_dist.to(pos_B.dtype), pos_B)

        # Select candidate correspondences once from assignment probability, then
        # gather every candidate-dependent quantity with the same indices. This
        # keeps probability mass, log-probability, position, and geometry aligned.
        topk_values, topk_indices = ot_prob_f32.topk(self.top_k, dim=-1)
        top1_mass = topk_values[..., 0].float()
        if self.top_k >= 2:
            top2_mass = topk_values[..., 1].float()
        else:
            top2_mass = torch.zeros_like(top1_mass)
        row_mass_scalar = row_mass.squeeze(-1).float().clamp_min(1e-8)
        valid_mass = valid_prob_sum.squeeze(-1).float().clamp(0.0, 1.0)
        peak_ratio = (top1_mass / row_mass_scalar).clamp(0.0, 1.0)
        peak_margin = ((top1_mass - top2_mass) / row_mass_scalar).clamp(0.0, 1.0)
        match_confidence = (0.6 * peak_ratio + 0.4 * peak_margin) * valid_mass
        match_confidence = torch.nan_to_num(
            match_confidence, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0).to(feat_A.dtype)

        topk_positions = torch.gather(
            pos_B.unsqueeze(1).expand(-1, N, -1, -1),
            dim=2,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        )
        topk_positions_for_geo = torch.gather(
            pos_B_for_geo.unsqueeze(1).expand(-1, N, -1, -1),
            dim=2,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        )

        geo_scores = self.geometric_validator(pos_A, topk_positions_for_geo, H, W)
        geo_weight = self.geo_weight.clamp(0.0, 2.0)

        topk_logp = torch.gather(log_assign.float(), dim=-1, index=topk_indices)
        combined_scores = topk_logp + geo_weight.float() * geo_scores.float()
        soft_weights = F.softmax(combined_scores, dim=-1).to(feat_A.dtype)
        geo_refined_offset = (topk_positions * soft_weights.unsqueeze(-1)).sum(dim=2) - base_expected_pos_B.detach()

        refined_warp = base_expected_pos_B + geo_refined_offset
        refined_warp = torch.nan_to_num(refined_warp, nan=0.0, posinf=1.5, neginf=-1.5).clamp(-1.5, 1.5)

        entropy = -(prob_dist.float() * prob_dist.float().clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy.to(feat_A.dtype) * valid_prob_sum.squeeze(-1)

        return refined_warp, entropy, raw_sim_matrix, geo_scores, match_confidence


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GeometryAwareFusion(nn.Module):
    """
    Fuse A features with B features sampled by the coarse correspondence field.

    The displacement and confidence channels expose the geometric route to the
    convolutional fusion block without assuming same-index spatial alignment.
    """
    def __init__(self, d_model: int):
        super().__init__()
        in_ch = d_model * 4 + 3
        self.fuse = nn.Sequential(
            ConvGNAct(in_ch, d_model, k=1, p=0),
            ConvGNAct(d_model, d_model, k=3),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(d_model), d_model),
            nn.GELU(),
        )

    def forward(
            self,
            feat_A_map: torch.Tensor,
            feat_B_map: torch.Tensor,
            coarse_warp: torch.Tensor,
            confidence: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = feat_A_map.shape
        out_dtype = feat_A_map.dtype

        warp = torch.nan_to_num(
            coarse_warp.float(), nan=0.0, posinf=1.5, neginf=-1.5
        ).clamp(-1.5, 1.5)

        feat_B_warped = safe_grid_sample(
            feat_B_map,
            warp,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        src_grid = make_grid(B, H, W, feat_A_map.device, torch.float32)
        disp = (warp - src_grid).permute(0, 3, 1, 2).to(out_dtype)

        if confidence.ndim == 4:
            conf = confidence.squeeze(-1)
        else:
            conf = confidence
        conf = torch.nan_to_num(
            conf.float(), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0).unsqueeze(1).to(out_dtype)

        fused_in = torch.cat([
            feat_A_map,
            feat_B_warped,
            (feat_A_map - feat_B_warped).abs(),
            feat_A_map * feat_B_warped,
            disp,
            conf,
        ], dim=1)
        return self.fuse(fused_in)


class WarpAwareLocalRefiner(nn.Module):
    """
    Refine dense correspondences with a small search window in the original B space.

    Sampling locations are routed by a detached coarse warp so refinement losses
    train the local delta predictor without pulling the coarse matcher through
    the grid-sampling path in the stable training setting.
    """

    def __init__(
            self,
            d_model: int,
            hidden: int = 64,
            max_delta_px: int = 4,
            corr_radius: int = 1,
    ):
        super().__init__()
        self.max_delta_px = float(max_delta_px)
        self.corr_radius = int(corr_radius)
        corr_ch = (2 * self.corr_radius + 1) ** 2
        in_ch = d_model * 3 + 2 + 1 + corr_ch

        self.encoder = nn.Sequential(
            ConvGNAct(in_ch, hidden, k=1, p=0),
            ConvGNAct(hidden, hidden, k=3),
            ConvGNAct(hidden, hidden, k=3),
        )
        self.delta_head = nn.Conv2d(hidden, 2, kernel_size=3, padding=1)
        self.conf_head = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.conf_head.weight)
        nn.init.zeros_(self.conf_head.bias)

    def _warp_aware_correlation(
            self,
            feat_A: torch.Tensor,
            feat_B: torch.Tensor,
            route_warp: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = feat_A.shape
        r = self.corr_radius
        feat_A_n = F.normalize(
            torch.nan_to_num(feat_A.float(), nan=0.0, posinf=0.0, neginf=0.0),
            p=2,
            dim=1,
        )
        feat_B_f = torch.nan_to_num(feat_B.float(), nan=0.0, posinf=0.0, neginf=0.0)

        corr_list = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offset = torch.tensor(
                    [2.0 * dx / max(float(W), 1.0), 2.0 * dy / max(float(H), 1.0)],
                    device=route_warp.device,
                    dtype=torch.float32,
                ).view(1, 1, 1, 2)
                grid = torch.nan_to_num(
                    route_warp.float() + offset, nan=0.0, posinf=1.5, neginf=-1.5
                ).clamp(-1.5, 1.5)
                sampled = safe_grid_sample(
                    feat_B_f,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                sampled_n = F.normalize(sampled.float(), p=2, dim=1)
                corr_list.append((feat_A_n * sampled_n).sum(dim=1, keepdim=True))

        return torch.cat(corr_list, dim=1).to(feat_A.dtype)

    def forward(
            self,
            feat_A_map: torch.Tensor,
            feat_B_map: torch.Tensor,
            coarse_warp: torch.Tensor,
            confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = feat_A_map.shape
        out_dtype = feat_A_map.dtype

        coarse_warp_safe = torch.nan_to_num(
            coarse_warp.float(), nan=0.0, posinf=1.5, neginf=-1.5
        ).clamp(-1.5, 1.5)
        route_warp = coarse_warp_safe.detach()
        warped_B = safe_grid_sample(
            feat_B_map,
            route_warp,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        src_grid = make_grid(B, H, W, feat_A_map.device, torch.float32)
        disp = (coarse_warp_safe - src_grid).permute(0, 3, 1, 2).to(out_dtype)

        if confidence.ndim == 4:
            conf = confidence.squeeze(-1)
        else:
            conf = confidence
        conf = torch.nan_to_num(
            conf.float(), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        conf_chw = conf.unsqueeze(1).to(out_dtype)

        corr = self._warp_aware_correlation(feat_A_map, feat_B_map, route_warp)
        x = torch.cat([
            feat_A_map,
            warped_B,
            (feat_A_map - warped_B).abs(),
            disp,
            conf_chw,
            corr,
        ], dim=1)

        hidden = self.encoder(x)
        delta_raw = torch.tanh(self.delta_head(hidden)).permute(0, 2, 3, 1)
        delta_scale = torch.tensor(
            [
                2.0 * self.max_delta_px / max(float(W), 1.0),
                2.0 * self.max_delta_px / max(float(H), 1.0),
            ],
            device=feat_A_map.device,
            dtype=torch.float32,
        ).view(1, 1, 1, 2)
        delta_flow = delta_raw.float() * delta_scale
        fine_warp = torch.nan_to_num(
            route_warp + delta_flow, nan=0.0, posinf=1.5, neginf=-1.5
        ).clamp(-1.5, 1.5).to(out_dtype)

        conf_delta_logits = self.conf_head(hidden).squeeze(1).float()
        base_logit = torch.logit(conf.clamp(1e-4, 1.0 - 1e-4))
        refined_conf_logits = base_logit + 0.5 * conf_delta_logits
        refined_conf = torch.sigmoid(refined_conf_logits).clamp(0.0, 1.0).to(out_dtype)
        return fine_warp, refined_conf, refined_conf_logits.to(out_dtype)


class AgriMatcher(nn.Module):
    def __init__(self, d_model=128, teacher_dim=100, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.refine_grid_size = grid_size * 2
        self._pe_cache = {}
        self.backbone = MobileViTBackbone(out_channels=d_model)
        self.use_inlier_predictor = False

        omega = 2 * math.pi * torch.randn(d_model // 2, 2)
        self.register_buffer('omega', omega)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

        self.global_encoder = nn.ModuleList([
            MatchAttention(d_model, nhead=4),
            MatchAttention(d_model, nhead=4),
        ])
        self.peak_aware_layer = MultiPeakAwareAttention(
            d_model=d_model,
            top_k=4,
            sinkhorn_iters=10,
        )
        self.geometry_fusion = GeometryAwareFusion(d_model)
        self.warp_aware_refiner = WarpAwareLocalRefiner(
            d_model=d_model,
            hidden=max(32, d_model // 2),
            max_delta_px=4,
        )
        self.warp_aware_refiner_fine = WarpAwareLocalRefiner(
            d_model=d_model,
            hidden=max(32, d_model // 2),
            max_delta_px=2,
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.feature_projector = nn.Linear(d_model, teacher_dim)

        self.coord_embed = nn.Linear(2, d_model)
        self.warp_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Tanh()
        )
        self.warp_delta_scale = 0.25
        self.conf_head = nn.Sequential(
            nn.Linear(d_model * 2 + 6, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # Deprecated: kept only for backward checkpoint compatibility. Not used in V3 forward.
        self.refiner_64 = WarpRefiner(
            C=d_model,
            hidden=max(32, d_model // 2),
            num_blocks=1,
            max_pixel_delta=4,
            residual_overlap=True,
            corr_radius=4,
        )

        self.inlier_predictor = ContextAwareInlierPredictor(
            feat_dim=d_model,
            feat_compress_dim=32,
            hidden_dim=128,
            num_layers=3,
        )
        self.h_proxy_head = GlobalSimilarityHead(feat_dim=d_model, hidden=128)

    def _forward_legacy_window(self, *args, **kwargs):
        raise RuntimeError(
            "_forward_legacy_window has been deprecated. Use AgriMatcher.forward "
            "with Global -> MultiPeak -> GeometryFusion -> WarpAwareRefiner."
        )

    def forward(self, img_A, img_B, H_prior=None):
        B = img_A.shape[0]
        device = img_A.device
        dtype = img_A.dtype

        feat_A_coarse, feat_A_fine = self.backbone(img_A)
        feat_B_coarse, feat_B_fine = self.backbone(img_B)
        _, _, Hc, Wc = feat_A_coarse.shape
        _, _, Hf, Wf = feat_A_fine.shape
        N_coarse = Hc * Wc

        feat_A_tokens_orig = feat_A_coarse.flatten(2).transpose(1, 2).contiguous()
        feat_B_tokens_orig = feat_B_coarse.flatten(2).transpose(1, 2).contiguous()
        feat_A = feat_A_tokens_orig
        feat_B = feat_B_tokens_orig
        pos_embed = self._get_pos_embed(Hc, Wc, device, torch.float32).to(feat_A.dtype)
        feat_A = feat_A + pos_embed
        feat_B = feat_B + pos_embed
        for encoder_layer in self.global_encoder:
            feat_A, feat_B = encoder_layer(feat_A, feat_B)

        pos_A = self._get_token_coords(B, Hc, Wc, device, feat_A.dtype)
        pos_B = self._get_token_coords(B, Hc, Wc, device, feat_B.dtype)
        temp = self.temperature.clamp(min=0.20, max=1.0).float()

        coarse_warp_tokens, match_entropy, raw_sim_matrix, geo_scores, match_conf_tokens = self.peak_aware_layer(
            feat_A, feat_B, pos_A, pos_B, Hc, Wc, temp
        )
        warp_AB_coarse_init = torch.nan_to_num(
            coarse_warp_tokens.reshape(B, Hc, Wc, 2).float(),
            nan=0.0,
            posinf=1.5,
            neginf=-1.5,
        ).clamp(-1.5, 1.5).to(feat_A.dtype)

        confidence_init = _sanitize_tensor(
            match_conf_tokens.reshape(B, Hc, Wc),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
            clamp=(0.0, 1.0),
        ).to(feat_A.dtype)

        feat_A_encoded_map = feat_A.transpose(1, 2).reshape(B, -1, Hc, Wc).contiguous()
        feat_B_encoded_map = feat_B.transpose(1, 2).reshape(B, -1, Hc, Wc).contiguous()
        fused_A_coarse = self.geometry_fusion(
            feat_A_encoded_map,
            feat_B_encoded_map,
            warp_AB_coarse_init,
            confidence_init,
        )
        warp_AB_coarse, confidence_AB_coarse, conf_logits_coarse = self.warp_aware_refiner(
            fused_A_coarse,
            feat_B_encoded_map,
            warp_AB_coarse_init,
            confidence_init,
        )
        confidence_AB_coarse = _sanitize_tensor(
            confidence_AB_coarse, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0)
        )

        warp_AB_fine, confidence_fine_ch = upsample_warp_and_overlap(
            warp_AB_coarse, confidence_AB_coarse, out_hw=(Hf, Wf)
        )
        warp_AB_fine = torch.nan_to_num(
            warp_AB_fine.float(), nan=0.0, posinf=1.5, neginf=-1.5
        ).clamp(-1.5, 1.5).to(feat_A_fine.dtype)
        confidence_AB_fine = _sanitize_tensor(
            confidence_fine_ch.squeeze(-1),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
            clamp=(0.0, 1.0),
        )
        warp_AB_fine, confidence_AB_fine, conf_logits_fine = self.warp_aware_refiner_fine(
            feat_A_fine,
            feat_B_fine,
            warp_AB_fine,
            confidence_AB_fine,
        )
        confidence_AB_fine = _sanitize_tensor(
            confidence_AB_fine, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0)
        )

        sim_matrix = raw_sim_matrix.float() / temp
        sim_matrix_kl = sim_matrix
        prob_A_to_B = F.softmax(sim_matrix, dim=-1)
        if N_coarse >= 2:
            top2_vals = torch.topk(prob_A_to_B, k=2, dim=-1).values
            top1_prob = top2_vals[..., 0:1].to(feat_A.dtype)
            top2_prob = top2_vals[..., 1:2].to(feat_A.dtype)
        else:
            top1_prob = prob_A_to_B.to(feat_A.dtype)
            top2_prob = torch.zeros_like(top1_prob)
        margin = top1_prob - top2_prob
        entropy = match_entropy.reshape(B, Hc, Wc)

        distill_feat_A = self.feature_projector(feat_A)
        distill_feat_B = self.feature_projector(feat_B_tokens_orig)

        valid_overlap_mask = (
            (warp_AB_fine[..., 0] >= -1.0) & (warp_AB_fine[..., 0] <= 1.0) &
            (warp_AB_fine[..., 1] >= -1.0) & (warp_AB_fine[..., 1] <= 1.0)
        ).float()

        warp_fine_flat = warp_AB_fine.reshape(B, Hf * Wf, 2)
        pos_A_fine = self._get_token_coords(B, Hf, Wf, device, dtype)
        feat_A_fine_flat = feat_A_fine.flatten(2).transpose(1, 2)
        feat_B_aligned_for_ransac = safe_grid_sample(
            feat_B_fine,
            warp_AB_fine,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
        feat_B_fine_flat = feat_B_aligned_for_ransac.flatten(2).transpose(1, 2)

        fused_A_fine_for_proxy = F.interpolate(
            fused_A_coarse.to(feat_A_fine.dtype),
            size=(Hf, Wf),
            mode='bilinear',
            align_corners=False,
        )
        H_proxy = self.h_proxy_head(
            fused_A_fine_for_proxy,
            feat_B_aligned_for_ransac,
            warp_AB_fine,
            confidence_AB_fine,
        )

        raw_inlier = None
        solver_used = torch.zeros(B, device=device, dtype=torch.bool)
        solver_weight_balanced = torch.zeros(B, Hf * Wf, device=device, dtype=torch.float32)
        if H_prior is not None:
            # External prior path: no robust dense-flow solver is run, so solver
            # diagnostics and solver weights must remain zero rather than
            # reporting confidence-derived values that never entered the solver.
            H_to_invert = _sanitize_homography(H_prior.detach().to(torch.float32))
            H_to_invert = _sanitize_base_transform(H_to_invert, max_shift=0.85).detach()
            base_stats = {
                "base_model_id": torch.full((B,), -2, device=device, dtype=torch.long),
                "base_p90": torch.zeros(B, device=device, dtype=torch.float32),
                "base_points": torch.zeros(B, device=device, dtype=torch.float32),
                "base_bins": torch.zeros(B, device=device, dtype=torch.float32),
                "base_quads": torch.zeros(B, device=device, dtype=torch.float32),
                "base_dst_span_x": torch.zeros(B, device=device, dtype=torch.float32),
                "base_dst_span_y": torch.zeros(B, device=device, dtype=torch.float32),
            }
        else:
            valid_overlap_flat = (
                    (warp_AB_fine[..., 0].abs() <= 1.0) &
                    (warp_AB_fine[..., 1].abs() <= 1.0)
            ).reshape(B, Hf * Wf).float().detach()
            if self.use_inlier_predictor:
                raw_inlier = self.inlier_predictor(
                    pos_A_fine.detach(),
                    warp_fine_flat.detach(),
                    feat_A_fine_flat.float(),
                    feat_B_fine_flat.float(),
                )
                solver_weight = (
                        raw_inlier.squeeze(-1).detach()
                        * confidence_AB_fine.reshape(B, Hf * Wf).detach()
                        * valid_overlap_flat
                )
            else:
                solver_weight = confidence_AB_fine.reshape(B, Hf * Wf).detach() * valid_overlap_flat

            valid_mask = (solver_weight > 0.03).float()
            solver_weight_raw = solver_weight * valid_mask
            solver_weight_2d = solver_weight_raw.reshape(B, 1, Hf, Wf)
            local_density = F.avg_pool2d(
                solver_weight_2d, kernel_size=9, stride=1, padding=4, count_include_pad=False
            )
            solver_weight_spread = solver_weight_2d / (local_density + 1e-4)
            original_energy = solver_weight_2d.view(B, -1).sum(dim=1, keepdim=True)
            new_energy = solver_weight_spread.view(B, -1).sum(dim=1, keepdim=True)
            scale_factor = original_energy / (new_energy + 1e-8)
            solver_weight_spread = (solver_weight_spread * scale_factor.view(B, 1, 1, 1)).clamp(0, 3.0)
            solver_weight_balanced = solver_weight_spread.reshape(B, Hf * Wf)

            with torch.no_grad():
                H_to_invert, base_stats = solve_robust_base_transform_from_dense_flow(
                    src_pts=pos_A_fine.float(),
                    dst_pts=warp_fine_flat.detach().float(),
                    weights=solver_weight_balanced.detach().float(),
                    H_grid=Hf,
                    W_grid=Wf,
                    bins_y=8,
                    bins_x=8,
                    topk_per_bin=8,
                    min_points=48,
                    min_bins=10,
                    min_quads=2,
                    min_dst_span=0.25,
                    max_shift=0.75,
                    allow_similarity=True,
                    allow_affine=True,
                    return_stats=True,
                )
            H_to_invert = H_to_invert.detach()
            solver_used = torch.ones(B, device=device, dtype=torch.bool)

        out = {
            'sim_matrix': sim_matrix,
            'sim_matrix_kl': sim_matrix_kl,
            'raw_sim_matrix': raw_sim_matrix,
            'geo_scores': geo_scores,
            'entropy': entropy,
            'match_confidence': match_conf_tokens.reshape(B, Hc, Wc),
            'top1_prob': top1_prob,
            'top2_prob': top2_prob,
            'margin': margin,
            'warp_AB_coarse': warp_AB_coarse,
            'warp_AB_coarse_init': warp_AB_coarse_init,
            'confidence_AB_coarse': confidence_AB_coarse,
            'confidence_AB_coarse_init': confidence_init,
            'conf_logits_coarse': conf_logits_coarse,
            'warp_AB': warp_AB_fine,
            'confidence_AB': confidence_AB_fine,
            'conf_logits': conf_logits_fine,
            'distill_feat_A': distill_feat_A,
            'distill_feat_B': distill_feat_B,
            'feat_A_64': feat_A_fine,
            'feat_B_64': feat_B_fine,
            'coarse_hw': (Hc, Wc),
            'fine_hw': (Hf, Wf),
            'H_base': H_to_invert,
            'H_proxy': H_proxy,
            'inlier_weights': raw_inlier,
            'raw_inlier_weights': raw_inlier,
            'solver_weights': solver_weight_balanced.detach(),
            'solver_weight_mean': solver_weight_balanced.detach().mean(),
            'solver_weight_nonzero_ratio': (solver_weight_balanced.detach() > 0.0).float().mean(),
            'solver_used': solver_used,
            'valid_overlap_mask': valid_overlap_mask,
            'base_model_id': base_stats['base_model_id'],
            'base_p90': base_stats['base_p90'],
            'base_points': base_stats['base_points'],
            'base_bins': base_stats['base_bins'],
            'base_quads': base_stats['base_quads'],
            'base_dst_span_x': base_stats['base_dst_span_x'],
            'base_dst_span_y': base_stats['base_dst_span_y'],
        }
        return out

    @staticmethod
    def _token_coords(B: int, gs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Match the normalized pixel-center convention used by RoMa-style grids
        # with align_corners=False.
        if gs == 1:
            coords = torch.zeros((1, 2), device=device, dtype=dtype)
            return coords.unsqueeze(0).expand(B, -1, -1)
        y = torch.linspace(-1.0 + 1.0 / gs, 1.0 - 1.0 / gs, gs, device=device, dtype=dtype)
        x = torch.linspace(-1.0 + 1.0 / gs, 1.0 - 1.0 / gs, gs, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (N,2)
        return coords.unsqueeze(0).expand(B, -1, -1)  # (B,N,2)

    @staticmethod
    def _clamp_normalized(xy: torch.Tensor, gs: int) -> torch.Tensor:
        if gs <= 1:
            return xy
        lo = -1.0 + 1.0 / gs
        hi = 1.0 - 1.0 / gs
        return xy.clamp(min=lo, max=hi)

    def _get_pos_embed(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate Fourier positional encodings for the current coarse feature map.

        Returns:
            pos_embed: (1, H*W, d_model)
        """
        key = (H, W, str(device), str(dtype))
        # IMPORTANT:
        # Do NOT cache tensors that depend on learnable params (self.pos_scale),
        # otherwise the cached tensor will keep the first forward graph and cause:
        # "Trying to backward through the graph a second time" on the next iteration.
        #
        # We only cache the coordinate grid (no grad), and recompute the projection
        # every forward so gradients flow correctly to pos_scale.
        if key not in self._pe_cache:
            y = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype)
            x = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([grid_x, grid_y], dim=-1).reshape(H * W, 2)
            self._pe_cache[key] = coords.detach()

        coords = self._pe_cache[key]
        proj = F.linear(coords, self.pos_scale.to(dtype) * self.omega.to(dtype))
        return torch.cat([proj.sin(), proj.cos()], dim=-1).unsqueeze(0)

    def _apply(self, fn):
        self._pe_cache.clear()
        return super()._apply(fn)

    def _get_token_coords(self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate normalized token coordinates for a dynamic H x W grid.

        Returns:
            (B, H*W, 2)
        """
        y = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        return coords.unsqueeze(0).expand(B, -1, -1)

    def _clamp_coords(self, xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Clamp normalized coordinates to the valid feature-grid range."""
        if H <= 1 or W <= 1:
            return xy
        x = xy[..., 0].clamp(min=-1.0 + 1.0 / W, max=1.0 - 1.0 / W)
        y = xy[..., 1].clamp(min=-1.0 + 1.0 / H, max=1.0 - 1.0 / H)
        return torch.stack([x, y], dim=-1)


class AgriStitcher(nn.Module):
    def __init__(
            self,
            matcher_config,
            decoder_config=None,
            residual_mode: str = "mesh",
            mesh_size: int = 12,
            max_residual_px: float = 4.0,
    ):
        super().__init__()
        self.matcher = AgriMatcher(**matcher_config)
        decoder_config = decoder_config or {}
        self.residual_mode = str(decoder_config.get("residual_mode", residual_mode)).lower()
        feat_channels = int(decoder_config.get("feat_channels", matcher_config.get("d_model", 128)))
        decoder_hidden = int(decoder_config.get("decoder_hidden", 128))
        decoder_blocks = int(decoder_config.get("decoder_blocks", 3))
        residual_scale = float(decoder_config.get("residual_scale", 0.08))
        mesh_size = int(decoder_config.get("mesh_size", mesh_size))
        max_residual_px = float(decoder_config.get("max_residual_px", max_residual_px))

        if self.residual_mode == "dense":
            self.stitch_decoder = StitchingDecoder(
                feat_channels=feat_channels,
                hidden=decoder_hidden,
                num_blocks=decoder_blocks,
                residual_scale=residual_scale,
            )
        elif self.residual_mode == "mesh":
            self.stitch_decoder = MeshStitchingDecoder(
                feat_channels=feat_channels,
                hidden=decoder_hidden,
                num_blocks=decoder_blocks,
                mesh_size=mesh_size,
                max_residual_px=max_residual_px,
                min_mask_bias=float(decoder_config.get("min_mask_bias", -3.0)),
            )
        elif self.residual_mode == "none":
            self.stitch_decoder = None
        else:
            raise ValueError(f"Unknown residual_mode={self.residual_mode!r}; expected dense, mesh, or none")

    def forward(self, img_A, img_B, H_prior=None):
        m_out = self.matcher(img_A, img_B, H_prior=H_prior)
        warp_AB = _sanitize_tensor(m_out['warp_AB'], nan=0.0, posinf=1.5, neginf=-1.5, clamp=(-1.5, 1.5))
        conf_AB = _sanitize_tensor(m_out['confidence_AB'], nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        B, H_feat, W_feat, _ = warp_AB.shape
        device = warp_AB.device
        dtype = warp_AB.dtype

        H_img, W_img = img_A.shape[2], img_A.shape[3]
        # Three-H formulation:
        # H_solver is the robust no-grad estimate from dense correspondences.
        # H_delta is the learnable correction predicted by the matcher head.
        # H_final is the trainable base transform used by the stitching path.
        H_solver = _sanitize_homography(m_out['H_base'].detach().to(torch.float32))
        H_solver = _sanitize_base_transform(H_solver, max_shift=0.85)

        H_delta = m_out.get('H_proxy')
        if H_delta is not None:
            H_delta = _sanitize_base_transform(H_delta.to(torch.float32), max_shift=0.85)
            H_final = torch.bmm(H_delta, H_solver.detach())
            H_final = _sanitize_homography(H_final)
            H_final = _sanitize_base_transform(H_final, max_shift=0.85)
        else:
            H_final = H_solver

        grid_solver_nograd = project_grid_with_h(H_solver, B, H_img, W_img, device).detach()
        grid_base_train = project_grid_with_h(H_final, B, H_img, W_img, device)
        if self.training and H_delta is not None and not grid_base_train.requires_grad:
            raise RuntimeError("H_base_grid_train must require grad in training when H_delta is available.")
        base_grid = grid_base_train

        with torch.amp.autocast("cuda", enabled=False):
            img_B_warped = F.grid_sample(
                torch.nan_to_num(img_B.float(), nan=0.0, posinf=0.0, neginf=0.0),
                base_grid.float(),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            ).to(img_B.dtype)
        feat_B_aligned = safe_grid_sample(
            m_out['feat_B_64'],
            warp_AB,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )

        aux = {}
        if self.residual_mode == "none":
            residual_flow = torch.zeros_like(base_grid, dtype=img_A.dtype)
            mask_logits = torch.full((B, H_img, W_img, 1), -20.0, device=device, dtype=img_A.dtype)
            stitch_mask = torch.zeros((B, H_img, W_img, 1), device=device, dtype=img_A.dtype)
        elif self.residual_mode == "dense":
            residual_flow, mask_logits, stitch_mask = self.stitch_decoder(
                img_A=img_A,
                img_B_warped=img_B_warped,
                feat_A=m_out['feat_A_64'],
                feat_B_warped=feat_B_aligned,
                base_grid=base_grid.to(img_A.dtype),
            )
        elif self.residual_mode == "mesh":
            residual_flow, mask_logits, stitch_mask, aux = self.stitch_decoder(
                img_A=img_A,
                img_B_warped=img_B_warped,
                feat_A=m_out['feat_A_64'],
                feat_B_warped=feat_B_aligned,
                base_grid=base_grid.to(img_A.dtype),
            )
        else:
            raise RuntimeError(f"Invalid residual_mode={self.residual_mode!r}")
        stitch_residual_flow = _sanitize_tensor(
            residual_flow, nan=0.0, posinf=1.0, neginf=-1.0, clamp=(-1.0, 1.0)
        )
        stitch_mask = _sanitize_tensor(
            stitch_mask, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0)
        )

        if self.residual_mode == "none":
            final_grid = _sanitize_tensor(base_grid, nan=0.0, posinf=3.0, neginf=-3.0, clamp=(-3.0, 3.0))
        else:
            final_grid = _sanitize_tensor(
                base_grid + stitch_mask * stitch_residual_flow,
                nan=0.0, posinf=3.0, neginf=-3.0, clamp=(-3.0, 3.0)
            )
        valid_overlap = (
            (final_grid[..., 0] >= -1.0) & (final_grid[..., 0] <= 1.0) &
            (final_grid[..., 1] >= -1.0) & (final_grid[..., 1] <= 1.0)
        ).to(conf_AB.dtype)

        out = {
            'dense_grid': final_grid,
            'stitch_residual_flow': stitch_residual_flow,
            'stitch_mask': stitch_mask,
            'stitch_mask_logits': mask_logits,
            'valid_overlap_mask': valid_overlap,
            'matcher_out': m_out,
            'H_mat': H_final,
            'H_solver_nograd': H_solver.detach(),
            'H_delta': H_delta,
            'H_final': H_final,
            'H_proxy': H_delta,
            'H_proxy_mat': H_delta,
            # Deprecated alias: this is H_final's train grid, not the old standalone H_proxy grid.
            'H_proxy_grid_train': grid_base_train,
            'grid_solver_nograd': grid_solver_nograd,
            'H_base_grid_train': grid_base_train,
            'H_base_grid_nograd': grid_solver_nograd,
            'residual_mode': self.residual_mode,
            'H_base_grid': grid_base_train.detach(),
        }
        if "mesh_delta" in aux:
            out["mesh_delta"] = aux["mesh_delta"]
        if "mesh_mask_lowres" in aux:
            out["mesh_mask_lowres"] = aux["mesh_mask_lowres"]
        if aux:
            out["mesh_regularization_inputs"] = aux
        return out


if __name__ == "__main__":
    pass
