import functools
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_match.backbone import MobileViTBackbone
from dense_match.refine import (WarpRefiner,
                                StitchingDecoder,
                                upsample_warp_and_overlap,
                                make_grid, safe_grid_sample)
from dense_match.geometry import (
    _sanitize_tensor,
    _sanitize_homography,
    _sanitize_base_transform_no_inplace,
    project_grid_with_h,
    solve_robust_base_transform_from_dense_flow,
)
from dense_match.heads import ContextAwareInlierPredictor, GlobalSimilarityHead




def _gn_groups(ch: int, max_groups: int = 8) -> int:
    """计算 GroupNorm 的组数"""
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
        """将特征图划分为窗口"""
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # 填充以确保能被 window_size 整除
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows, (Hp, Wp)

    def window_reverse(self, windows, H, W, Hp, Wp):
        """将窗口还原回特征图"""
        B = windows.shape[0] // ((Hp // self.window_size) * (Wp // self.window_size))
        x = windows.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        # 裁切掉填充部分
        return x[:, :H, :W, :].reshape(B, H * W, -1)

    def forward(self, feat_A, feat_B, H, W):
        # Window Partition
        short_cut_A, short_cut_B = feat_A, feat_B
        win_A, (Hp, Wp) = self.window_partition(feat_A, H, W)
        win_B, _ = self.window_partition(feat_B, H, W)

        win_A = win_A + self.local_pe
        win_B = win_B + self.local_pe

        # Local Cross Attention (A attends to B within the same window)
        # 这里的逻辑是：只在对应的空间窗口内寻找匹配，适合位移不大的精细化匹配
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
    验证候选匹配的局部几何一致性。
    向量化实现：将 K 个候选合并到 batch 维度，单次 avg_pool2d 完成全部计算。
    count_include_pad=False 修复了边缘区域方差系统性偏高的问题。
    """

    def __init__(self, neighbor_radius: int = 3):
        super().__init__()
        self.radius = neighbor_radius
        self.kernel_size = 2 * neighbor_radius + 1

    def forward(self, src_pos, candidate_positions, H, W):
        """
        src_pos:             [B, N, 2]
        candidate_positions: [B, N, K, 2]
        返回:                [B, N, K]  几何一致性得分，越大越好
        """
        B, N, K, _ = candidate_positions.shape
        src_2d = src_pos.reshape(B, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # 将 K 个候选合并到 batch 维度，一次完成全部 avg_pool2d
        # candidate_positions: [B, N, K, 2] → permute → [B, K, N, 2] → reshape → [B*K, H, W, 2]
        cand_all = candidate_positions.permute(0, 2, 1, 3).reshape(B * K, H, W, 2).permute(0, 3, 1, 2)
        src_expanded = src_2d.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, 2, H, W)
        disp_all = cand_all - src_expanded  # [B*K, 2, H, W]

        # count_include_pad=False：边缘像素只用实际邻域均值，消除零填充对方差的系统性拉偏
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
    def __init__(self, d_model=128, top_k=8, sinkhorn_iters=15):  # 修改迭代次数为 15
        super().__init__()
        self.top_k = top_k
        self.sinkhorn_iters = sinkhorn_iters
        self.geometric_validator = LocalGeometricValidator(neighbor_radius=3)
        self.dustbin_score = nn.Parameter(torch.tensor(-1.0))
        self.geo_weight = nn.Parameter(torch.tensor(0.5))

    def optimal_transport(self, scores, current_temp):
        """
        scores: [B, M, N] 已被 current_temp 除过的相似度矩阵
        current_temp: 当前温度参数，用于对齐 dustbin 分数
        """
        # 强制转为 FP32，免疫 AMP 导致的 logsumexp 溢出 NaN
        scores_f32 = torch.nan_to_num(
            scores.float(), nan=0.0, posinf=50.0, neginf=-50.0
        ).clamp(-50.0, 50.0)
        B, M, N = scores_f32.shape
        device = scores_f32.device

        # 将垃圾桶分数除以温度，使其与传入的 scores 尺度一致
        temp = torch.as_tensor(current_temp, device=device, dtype=torch.float32).clamp(min=0.03, max=10.0)
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

        # 前 sinkhorn_iters - 1 次不计算梯度以节省显存和算力
        with torch.no_grad():
            for _ in range(self.sinkhorn_iters - 1):
                u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
                v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

        log_assign = Z + u.unsqueeze(2) + v.unsqueeze(1)
        log_assign = torch.nan_to_num(log_assign, nan=-50.0, posinf=0.0, neginf=-50.0)
        # 转回原始的 dtype (FP16 或 FP32) 以保持外层计算图兼容
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
        ot_prob_matrix = ot_prob_f32.to(feat_A.dtype)

        base_expected_pos_B = torch.bmm(prob_dist.to(pos_B.dtype), pos_B)

        topk_values, topk_indices = ot_prob_matrix.topk(self.top_k, dim=-1)

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

        # 将几何得分转化为对 Top-K 坐标的局部微调
        topk_logp, topk_indices = log_assign.float().topk(self.top_k, dim=-1)
        combined_scores = topk_logp + geo_weight.float() * geo_scores.float()
        soft_weights = F.softmax(combined_scores, dim=-1).to(feat_A.dtype)
        geo_refined_offset = (topk_positions * soft_weights.unsqueeze(-1)).sum(dim=2) - base_expected_pos_B.detach()

        refined_warp = base_expected_pos_B + geo_refined_offset
        refined_warp = torch.nan_to_num(refined_warp, nan=0.0, posinf=1.5, neginf=-1.5).clamp(-1.5, 1.5)

        entropy = -(prob_dist.float() * prob_dist.float().clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy.to(feat_A.dtype) * valid_prob_sum.squeeze(-1)

        return refined_warp, entropy, raw_sim_matrix, geo_scores




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

        self.attn_layers = nn.ModuleList([
            MatchAttention(d_model, nhead=4),
            MatchWindowAttention(d_model, num_heads=4, window_size=8),
            MatchWindowAttention(d_model, num_heads=4, window_size=8),
            MatchAttention(d_model, nhead=4)
        ])
        self.peak_aware_layer = MultiPeakAwareAttention(d_model=d_model, top_k=4)

        self.temperature = nn.Parameter(torch.tensor(1.0))  # 从高温开始，由优化器逐步衰减
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
        self.refiner_64 = WarpRefiner(
            C=d_model,
            hidden=max(32, d_model // 2),
            num_blocks=1,
            max_pixel_delta=4,
            residual_overlap=True,
            corr_radius=4,  # 9×9 搜索窗口，覆盖更大位移
        )

        # ContextAwareInlierPredictor: NG-RANSAC 风格的语义引导权重头
        # 替代纯几何 margin 权重，解决重复纹理欺骗 IRLS 的漏洞
        self.inlier_predictor = ContextAwareInlierPredictor(
            feat_dim=d_model,
            feat_compress_dim=32,
            hidden_dim=128,
            num_layers=3,
        )
        self.h_proxy_head = GlobalSimilarityHead(feat_dim=d_model, hidden=128)

    def forward(self, img_A, img_B, H_prior=None):
        B = img_A.shape[0]

        device = img_A.device
        dtype = img_A.dtype

        # Backbone 提取特征
        feat_A_coarse, feat_A_fine = self.backbone(img_A)
        feat_B_coarse, feat_B_fine = self.backbone(img_B)

        # 获取实际的 coarse 特征图尺寸（动态）
        _, _, Hc, Wc = feat_A_coarse.shape
        _, _, Hf, Wf = feat_A_fine.shape
        N_coarse = Hc * Wc

        # 展平为 token 序列
        feat_A = feat_A_coarse.flatten(2).transpose(1, 2)  # (B, N, d)
        feat_B_orig = feat_B_coarse.flatten(2).transpose(1, 2)
        feat_B = feat_B_orig

        pos_embed = self._get_pos_embed(Hc, Wc, device, torch.float32).to(dtype)

        # 将位置编码注入到自注意力的 Q/K 计算，而不是永久改变 feat
        feat_A_pos = feat_A + pos_embed
        feat_B_pos = feat_B + pos_embed
        feat_A_pos, feat_B_pos = self.attn_layers[0](feat_A_pos, feat_B_pos)

        with torch.no_grad():
            norm_A_tmp = F.normalize(feat_A_pos.float(), p=2, dim=-1)
            norm_B_tmp = F.normalize(feat_B_pos.float(), p=2, dim=-1)
            # 粗略匹配：每个 A token 找 B 中最相似的位置
            sim_tmp = torch.bmm(norm_A_tmp, norm_B_tmp.transpose(1, 2))  # [B, N, N]
            prob_tmp = F.softmax(sim_tmp / 0.5, dim=-1)  # temperature=0.5
            pos_B_tmp = self._get_token_coords(B, Hc, Wc, device, feat_A_pos.dtype)
            # 期望坐标：[B, N, 2]
            expected_pos_B = torch.bmm(prob_tmp.to(feat_B_pos.dtype), pos_B_tmp)
            # 将 B 特征按预测的对应关系 warp 到 A 坐标系
            # expected_pos_B: [B, N, 2] → reshape → [B, Hc, Wc, 2]（作为 grid）
            warp_grid_tmp = expected_pos_B.reshape(B, Hc, Wc, 2)

        # 将 B 的特征 warp 到 A 的空间位置
        feat_B_warped_chw = F.grid_sample(
            feat_B.transpose(1, 2).reshape(B, -1, Hc, Wc).float(),  # ← 使用 feat_B（无 pos）
            warp_grid_tmp.float(),
            mode='bilinear', padding_mode='border', align_corners=False
        ).to(feat_B_pos.dtype)
        feat_B_warped = feat_B_warped_chw.reshape(B, -1, Hc * Wc).transpose(1, 2)

        # 在 warp 后为 B 特征重新注入位置编码（代表 A 坐标系中的位置）
        feat_B_warped = feat_B_warped + pos_embed

        # 两层窗口注意力：在已对齐的坐标系内做局部精细匹配
        feat_A_pos, feat_B_warped = self.attn_layers[1](feat_A_pos, feat_B_warped, Hc, Wc)
        feat_A_pos, feat_B_warped = self.attn_layers[2](feat_A_pos, feat_B_warped, Hc, Wc)

        feat_A_pos, _ = self.attn_layers[3](feat_A_pos, feat_B_warped)

        feat_A = feat_A_pos
        feat_B = feat_B_warped

        pos_A = self._get_token_coords(B, Hc, Wc, device, dtype)  # [B, N, 2]
        pos_B = warp_grid_tmp.reshape(B, Hc * Wc, 2).to(dtype)

        # 特征蒸馏
        distill_feat_A = self.feature_projector(feat_A)
        distill_feat_B = self.feature_projector(feat_B_orig)

        temp = self.temperature.clamp(min=0.07, max=1.0).float()
        norm_A = F.normalize(feat_A.float(), p=2, dim=-1)
        norm_B = F.normalize(feat_B.float(), p=2, dim=-1)
        sim_matrix = torch.bmm(norm_A, norm_B.transpose(1, 2)) / temp
        prob_A_to_B = F.softmax(sim_matrix, dim=-1)  # fp32
        norm_B_orig = F.normalize(feat_B_orig.float(), p=2, dim=-1)
        sim_matrix_kl = torch.bmm(norm_A, norm_B_orig.transpose(1, 2)) / temp

        # 分类特征（用于 conf_head 输入）
        top2_vals = torch.topk(prob_A_to_B, k=2, dim=-1).values
        top1_prob = top2_vals[..., 0:1]
        top2_prob = top2_vals[..., 1:2]
        margin = top1_prob - top2_prob
        prob_fp32 = prob_A_to_B.float()
        entropy = -(prob_fp32 * torch.log(prob_fp32.clamp_min(1e-6))).sum(
            dim=-1, keepdim=True
        ).to(prob_A_to_B.dtype)

        # 先用粗略期望坐标为 PeakAwareLayer 提供 H 先验去畸变
        # 注意：这里的 H 只用于 pos_B_for_geo（几何验证辅助），不是最终输出的 H_base
        pos_B_for_geo = pos_B
        with torch.no_grad():
            expected_pos_B_crude = torch.bmm(prob_A_to_B.to(dtype), pos_B)
            # 用粗略 margin 权重算一个仅供 LocalGeometricValidator 去畸变用的临时 H
            H_for_geo = solve_robust_base_transform_from_dense_flow(
                src_pts=pos_A.detach().float(),
                dst_pts=expected_pos_B_crude.detach().float(),
                weights=margin.detach().squeeze(-1).float(),
                H_grid=Hc,
                W_grid=Wc,
                bins_y=4,
                bins_x=4,
                topk_per_bin=4,
                min_points=24,
                min_bins=6,
                min_quads=2,
                min_dst_span=0.35,
                max_shift=0.75,
                allow_similarity=True,
                allow_affine=False,
                return_stats=False,
            )
            H_geo_inv = torch.linalg.pinv(H_for_geo.float())
            H_geo_inv = torch.nan_to_num(H_geo_inv, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            finite_inv = torch.isfinite(H_geo_inv).all(dim=(-2, -1))
            eye_geo = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
            H_geo_inv = torch.where(finite_inv.view(B, 1, 1), H_geo_inv, eye_geo)

            pb_homo = torch.cat([pos_B.float(), torch.ones(B, Hc * Wc, 1, device=device, dtype=torch.float32)], dim=-1)
            pb_in_A = torch.bmm(pb_homo, H_geo_inv.transpose(1, 2))
            z_A = pb_in_A[..., 2:3]
            sign_A = z_A.sign()
            sign_A = torch.where(sign_A == 0, torch.ones_like(sign_A), sign_A)
            safe_z = torch.where(z_A.abs() < 1e-3, 1e-3 * sign_A, z_A)
            pos_B_for_geo = (pb_in_A[..., :2] / safe_z).clamp(-10.0, 10.0).to(dtype)

        # PeakAwareLayer：基于粗略去畸变坐标做几何验证
        refined_warp_coarse, match_entropy, raw_sim_matrix, geo_scores = self.peak_aware_layer(
            feat_A, feat_B, pos_A, pos_B, Hc, Wc, temp, pos_B_for_geo=pos_B_for_geo
        )
        sim_matrix_for_kl = sim_matrix
        expected_feat_B = torch.bmm(prob_A_to_B.to(feat_B.dtype), feat_B)
        feat_diff = torch.abs(feat_A - expected_feat_B)
        feat_prod = feat_A * expected_feat_B

        pos_A_flat = pos_A
        warp_AB_coarse = refined_warp_coarse.reshape(B, Hc, Wc, 2)

        # Confidence Head
        conf_input = torch.cat(
            [feat_diff, feat_prod, top1_prob, top2_prob, margin, entropy, pos_A_flat],
            dim=-1
        )
        conf_logits_coarse = self.conf_head(conf_input).squeeze(-1).reshape(B, Hc, Wc)
        confidence_AB_coarse = torch.sigmoid(conf_logits_coarse)
        confidence_AB_coarse = _sanitize_tensor(
            confidence_AB_coarse, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0)
        )

        # WarpRefiner
        warp_fine_init, overlap_fine_init = upsample_warp_and_overlap(
            warp_AB_coarse, confidence_AB_coarse, out_hw=(Hf, Wf)
        )
        warp_AB_fine, overlap_logits_fine, overlap_fine = self.refiner_64(
            feat_A=feat_A_fine,
            feat_B=feat_B_fine,
            prev_warp=warp_fine_init,
            prev_overlap=overlap_fine_init,
        )
        conf_logits_fine = overlap_logits_fine.squeeze(-1)
        confidence_AB_fine = overlap_fine.squeeze(-1)
        confidence_AB_fine = _sanitize_tensor(
            confidence_AB_fine, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0)
        )

        # 有效重叠掩码（物理含义）
        # warp_AB_fine[b,i,j] = A(i,j) 在 B 中的归一化坐标
        # 只有落在 [-1,1]² 内的对应关系才是真正"有图可采样"的有效区域
        valid_overlap_mask = (
            (warp_AB_fine[..., 0] >= -1.0) & (warp_AB_fine[..., 0] <= 1.0) &
            (warp_AB_fine[..., 1] >= -1.0) & (warp_AB_fine[..., 1] <= 1.0)
        ).float()  # [B, Hf, Wf]，1=有效重叠，0=B 中无对应区域

        # H 在 refine 后计算，用精化 warp 和 ContextAwareInlierPredictor
        # 优势：
        #   a) warp_AB_fine 已经经过 WarpRefiner 修正，残差更小，H 更准
        #   b) ContextAwareInlierPredictor 用特征差异识别语义假匹配（重复纹理骗不过）
        #   c) Hartley normalization 确保 DLT 数值稳定，H 不退化为单位阵
        warp_fine_flat = warp_AB_fine.reshape(B, Hf * Wf, 2)
        pos_A_fine = self._get_token_coords(B, Hf, Wf, device, dtype)

        # 对应的 fine 特征（B, N_fine, C）
        feat_A_fine_flat = feat_A_fine.flatten(2).transpose(1, 2)
        feat_B_aligned_for_ransac = safe_grid_sample(
            feat_B_fine, warp_AB_fine, padding_mode='border', align_corners=False
        )
        H_proxy = self.h_proxy_head(
            feat_A_fine,
            feat_B_aligned_for_ransac,
            warp_AB_fine,
            confidence_AB_fine,
        )
        feat_B_fine_flat = feat_B_aligned_for_ransac.flatten(2).transpose(1, 2)

        inlier_weights = None
        if H_prior is not None:
            H_to_invert = H_prior.detach()

            base_stats = {
                "base_model_id": torch.full((B,), -1, device=device, dtype=torch.long),
                "base_p90": torch.zeros(B, device=device, dtype=torch.float32),
                "base_points": torch.zeros(B, device=device, dtype=torch.float32),
                "base_bins": torch.zeros(B, device=device, dtype=torch.float32),
                "base_quads": torch.zeros(B, device=device, dtype=torch.float32),
                "base_dst_span_x": torch.zeros(B, device=device, dtype=torch.float32),
                "base_dst_span_y": torch.zeros(B, device=device, dtype=torch.float32),
            }
        else:
            # H_to_invert 计算部分
            if self.use_inlier_predictor:
                inlier_weights = self.inlier_predictor(
                    pos_A_fine.detach(),
                    warp_fine_flat.detach(),
                    feat_A_fine_flat.float(),
                    feat_B_fine_flat.float(),
                )
                conf_fine_flat = confidence_AB_fine.reshape(B, Hf * Wf, 1).detach()
                combined_weights = (inlier_weights * conf_fine_flat).squeeze(-1)
            else:
                combined_weights = confidence_AB_fine.reshape(B, Hf * Wf).detach()
                inlier_weights = None

            # 先过滤原始置信度，彻底剥夺纯噪声的发言权
            valid_mask = (combined_weights > 0.1).float()  # [B, N]
            cw_filtered = combined_weights * valid_mask

            # 计算局部密度并进行空间均匀分配
            cw_2d = cw_filtered.reshape(B, 1, Hf, Wf)
            local_density = F.avg_pool2d(
                cw_2d, kernel_size=9, stride=1, padding=4, count_include_pad=False
            )

            # 密集区降权，稀疏区提权
            spatially_spread = cw_2d / (local_density + 1e-4)

            # 能量守恒归一化，保证空间重分配后，总的 Data Term 能量(权重总和)不变，从而与 reg_lambda 完美抗衡
            original_energy = cw_2d.view(B, -1).sum(dim=1, keepdim=True)
            new_energy = spatially_spread.view(B, -1).sum(dim=1, keepdim=True)
            scale_factor = original_energy / (new_energy + 1e-8)

            spatially_spread = spatially_spread * scale_factor.view(B, 1, 1, 1)

            # Clamp 防止个别极端孤立点被放大到毁掉整个 DLT (3.0 是个非常安全的物理上限)
            spatially_spread = spatially_spread.clamp(0, 3.0)

            combined_weights_final = spatially_spread.reshape(B, Hf * Wf)

            with torch.no_grad():
                H_to_invert, base_stats = solve_robust_base_transform_from_dense_flow(
                    src_pts=pos_A_fine.float(),
                    dst_pts=warp_fine_flat.detach().float(),
                    weights=combined_weights_final.detach().float(),
                    H_grid=Hf,
                    W_grid=Wf,
                    bins_y=8,
                    bins_x=8,
                    topk_per_bin=8,
                    min_points=48,
                    min_bins=14,
                    min_quads=3,
                    min_dst_span=0.45,
                    max_shift=0.75,
                    allow_similarity=True,
                    allow_affine=True,
                    return_stats=True,
                )

            H_to_invert = H_to_invert.detach()

        return {
            # 匹配矩阵
            'sim_matrix': sim_matrix,
            'sim_matrix_kl': sim_matrix_kl,
            'geo_scores': geo_scores,
            'top1_prob': top1_prob,

            # Coarse 输出
            'warp_AB_coarse': warp_AB_coarse,
            'confidence_AB_coarse': confidence_AB_coarse,
            'conf_logits_coarse': conf_logits_coarse,

            # Fine 输出
            'warp_AB': warp_AB_fine,
            'confidence_AB': confidence_AB_fine,
            'conf_logits': conf_logits_fine,

            # 蒸馏特征
            'distill_feat_A': distill_feat_A,
            'distill_feat_B': distill_feat_B,

            # Fine 特征
            'feat_A_64': feat_A_fine,
            'feat_B_64': feat_B_fine,

            # 动态尺寸信息
            'coarse_hw': (Hc, Wc),
            'fine_hw': (Hf, Wf),

            # Base transform
            'H_base': H_to_invert,
            'H_proxy': H_proxy,
            'inlier_weights': inlier_weights,
            'valid_overlap_mask': valid_overlap_mask,

            # 诊断信息
            'base_model_id': base_stats['base_model_id'],
            'base_p90': base_stats['base_p90'],
            'base_points': base_stats['base_points'],
            'base_bins': base_stats['base_bins'],
            'base_quads': base_stats['base_quads'],
            'base_dst_span_x': base_stats['base_dst_span_x'],
            'base_dst_span_y': base_stats['base_dst_span_y'],
        }

    @staticmethod
    def _token_coords(B: int, gs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 与 RoMa v2 的 get_normalized_grid 一致：align_corners=False 的像素中心坐标
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
        动态生成傅里叶位置编码

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
        动态生成 token 坐标

        Returns:
            (B, H*W, 2)
        """
        y = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        return coords.unsqueeze(0).expand(B, -1, -1)

    def _clamp_coords(self, xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """裁剪坐标到有效范围"""
        if H <= 1 or W <= 1:
            return xy
        x = xy[..., 0].clamp(min=-1.0 + 1.0 / W, max=1.0 - 1.0 / W)
        y = xy[..., 1].clamp(min=-1.0 + 1.0 / H, max=1.0 - 1.0 / H)
        return torch.stack([x, y], dim=-1)


class AgriStitcher(nn.Module):
    def __init__(self, matcher_config, decoder_config=None):
        super().__init__()
        self.matcher = AgriMatcher(**matcher_config)
        decoder_config = decoder_config or {}
        feat_channels = int(decoder_config.get("feat_channels", matcher_config.get("d_model", 128)))
        decoder_hidden = int(decoder_config.get("decoder_hidden", 128))
        decoder_blocks = int(decoder_config.get("decoder_blocks", 3))
        residual_scale = float(decoder_config.get("residual_scale", 0.08))
        self.stitch_decoder = StitchingDecoder(
            feat_channels=feat_channels,
            hidden=decoder_hidden,
            num_blocks=decoder_blocks,
            residual_scale=residual_scale,
        )

    def forward(self, img_A, img_B, H_prior=None):
        m_out = self.matcher(img_A, img_B, H_prior=H_prior)
        warp_AB = _sanitize_tensor(m_out['warp_AB'], nan=0.0, posinf=1.5, neginf=-1.5, clamp=(-1.5, 1.5))
        conf_AB = _sanitize_tensor(m_out['confidence_AB'], nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0))

        B, H_feat, W_feat, _ = warp_AB.shape
        device = warp_AB.device
        dtype = warp_AB.dtype

        H_img, W_img = img_A.shape[2], img_A.shape[3]
        H_base_safe = _sanitize_homography(m_out['H_base'].detach().to(torch.float32))
        H_base_safe = _sanitize_base_transform_no_inplace(H_base_safe, max_shift=0.85)
        base_grid = project_grid_with_h(H_base_safe, B, H_img, W_img, device)

        H_proxy = m_out.get('H_proxy')
        if H_proxy is not None:
            H_proxy_safe = _sanitize_base_transform_no_inplace(H_proxy.to(torch.float32), max_shift=0.85)
            H_proxy_grid_train = project_grid_with_h(H_proxy_safe, B, H_img, W_img, device)
        else:
            H_proxy_safe = None
            H_proxy_grid_train = None

        # 基于 H 将图像 B 和 fine 特征 B 对齐到 A 坐标系
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

        residual_flow, mask_logits, stitch_mask = self.stitch_decoder(
            img_A=img_A,
            img_B_warped=img_B_warped,
            feat_A=m_out['feat_A_64'],
            feat_B_warped=feat_B_aligned,
            base_grid=base_grid.to(img_A.dtype),
        )
        stitch_residual_flow = _sanitize_tensor(
            residual_flow, nan=0.0, posinf=1.0, neginf=-1.0, clamp=(-1.0, 1.0)
        )
        stitch_mask = _sanitize_tensor(
            stitch_mask, nan=0.0, posinf=1.0, neginf=0.0, clamp=(0.0, 1.0)
        )

        final_grid = _sanitize_tensor(
            base_grid + stitch_mask * stitch_residual_flow,
            nan=0.0, posinf=3.0, neginf=-3.0, clamp=(-3.0, 3.0)
        )
        valid_overlap = (
            (final_grid[..., 0] >= -1.0) & (final_grid[..., 0] <= 1.0) &
            (final_grid[..., 1] >= -1.0) & (final_grid[..., 1] <= 1.0)
        ).to(conf_AB.dtype)

        return {
            'dense_grid': final_grid,
            'stitch_residual_flow': stitch_residual_flow,
            'stitch_mask': stitch_mask,
            'stitch_mask_logits': mask_logits,
            'valid_overlap_mask': valid_overlap,
            'matcher_out': m_out,
            'H_mat': H_base_safe,
            'H_proxy': H_proxy,
            'H_proxy_mat': H_proxy_safe,
            'H_proxy_grid_train': H_proxy_grid_train,
            'H_base_grid_nograd': base_grid.detach(),
            # 直接暴露 base_grid 给可视化（只用 H 的纯单应网格，不含 decoder 残差）
            'H_base_grid': base_grid.detach(),
        }


if __name__ == "__main__":
    pass
