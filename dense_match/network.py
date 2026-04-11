import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_match.backbone import MobileViTBackbone
from dense_match.refine import (WarpRefiner,
                                upsample_warp_and_overlap,
                                make_grid, safe_grid_sample)
from dense_match.flow_to_tps import BypassTPSEstimator, TPSGridGenerator


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

        # --- NEW: Generate and register local window positional encoding ---
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
    def __init__(self, d_model=128, top_k=8, sinkhorn_iters=5):
        super().__init__()
        self.top_k = top_k
        self.sinkhorn_iters = sinkhorn_iters
        self.geometric_validator = LocalGeometricValidator(neighbor_radius=3)
        self.dustbin_score = nn.Parameter(torch.tensor(-1.0))
        self.geo_weight = nn.Parameter(torch.tensor(0.5))

    def optimal_transport(self, scores):
        # 强制转为 FP32，免疫 AMP 导致的 logsumexp 溢出 NaN
        scores_f32 = scores.float()
        B, M, N = scores_f32.shape
        device = scores_f32.device

        ds = self.dustbin_score.float()
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
        for _ in range(self.sinkhorn_iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

        log_assign = Z + u.unsqueeze(2) + v.unsqueeze(1)
        # 转回原始的 dtype (FP16 或 FP32) 以保持外层计算图兼容
        return log_assign[:, :M, :N].to(scores.dtype)

    def forward(self, feat_A, feat_B, pos_A, pos_B, H, W, current_temp, pos_B_for_geo=None):
        B, N, C = feat_A.shape

        if pos_B_for_geo is None:
            pos_B_for_geo = pos_B

        norm_A = F.normalize(feat_A, p=2, dim=-1)
        norm_B = F.normalize(feat_B, p=2, dim=-1)
        raw_sim_matrix = torch.bmm(norm_A, norm_B.transpose(1, 2))

        scores = raw_sim_matrix / current_temp
        log_assign = self.optimal_transport(scores)

        ot_prob_matrix = torch.exp(log_assign)

        # 计算全局概率分布，确保梯度能流向相似度矩阵的每一个角落
        prob_dist = ot_prob_matrix / (ot_prob_matrix.sum(dim=-1, keepdim=True) + 1e-8)

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
        combined_scores = topk_values + geo_weight * geo_scores
        soft_weights = F.softmax(combined_scores / current_temp, dim=-1)
        geo_refined_offset = (topk_positions * soft_weights.unsqueeze(-1)).sum(dim=2) - base_expected_pos_B.detach()

        refined_warp = base_expected_pos_B + geo_refined_offset

        entropy = -(prob_dist * (prob_dist + 1e-8).log()).sum(dim=-1)

        return refined_warp, entropy, raw_sim_matrix, geo_scores


def tensor_DLT(src_p, dst_p):
    bs = src_p.shape[0]
    device = src_p.device
    dtype = src_p.dtype

    ones = torch.ones(bs, 4, 1, device=device, dtype=dtype)
    xy1 = torch.cat((src_p, ones), 2)
    zeros = torch.zeros_like(xy1)

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_p.reshape(-1, 1, 2),
    ).reshape(bs, -1, 2)

    # Ah = b
    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(bs, -1, 1)

    # 加入极小的对角线扰动，防止极端情况(如4点共线)下 torch.inverse 报错
    I = torch.eye(8, device=device, dtype=dtype).unsqueeze(0)
    A = A + I * 1e-5

    # h = A^{-1}b
    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(bs, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(bs, 3, 3)
    return H


class ContextAwareInlierPredictor(nn.Module):
    """
    NG-RANSAC 风格的上下文感知内点权重预测器。
      1. 置换不变性 (Permutation Invariance)
         使用 1D 卷积（PointNet 范式）处理 [B, C, N] 的无序点集，
         打乱 N 维顺序不影响任何单点的权重输出。
      2. 全局上下文感知 (Global Context)
         全局最大池化提取"大部队运动趋势"，与每个点的局部特征拼接，
         让每个点知道自己是否符合全局共识（重复纹理假匹配的克星）。
      3. 特征-几何深度融合 (Feature-Geometry Fusion)
         输入同时包含：
           - 几何坐标差 [x_A, y_A, x_B, y_B]                (4 维)
           - 特征绝对差  |feat_A - feat_B|（按通道压缩后）  (feat_compress_dim 维)
           - 特征 Hadamard 积  feat_A * feat_B（同上）       (feat_compress_dim 维)
         语义不一致的假匹配即使残差合理，也会被特征差异识破。

    输入:
        pos_A:    [B, N, 2]
        pos_B:    [B, N, 2]
        feat_A:   [B, N, C]
        feat_B:   [B, N, C]
    输出:
        weights:  [B, N, 1]  ∈ (0, 1)，用于加权 DLT
    """

    def __init__(
            self,
            feat_dim: int = 128,
            feat_compress_dim: int = 32,
            hidden_dim: int = 128,
            num_layers: int = 3,
    ):
        super().__init__()
        self.feat_compress = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_compress_dim * 2),
            nn.LayerNorm(feat_compress_dim * 2),
            nn.GELU(),
            nn.Linear(feat_compress_dim * 2, feat_compress_dim),
        )

        # PointNet: 1D conv 处理无序点集 [B, C_in, N]
        # 输入维度：4 (坐标) + feat_compress_dim (特征融合)
        in_dim = 4 + feat_compress_dim
        layers = []
        prev = in_dim
        for i in range(num_layers):
            out = hidden_dim
            layers += [
                nn.Conv1d(prev, out, 1, bias=False),
                nn.BatchNorm1d(out),
                nn.ReLU(inplace=True),
            ]
            prev = out
        self.local_encoder = nn.Sequential(*layers)

        # 全局上下文融合后的输出头
        # 输入：local(hidden_dim) + global(hidden_dim) → 权重
        self.output_head = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )

        # 初始化：输出层偏置设为正值，让网络初期倾向于给所有点中等权重
        nn.init.constant_(self.output_head[-2].bias, 1.0)

    def forward(
            self,
            pos_A: torch.Tensor,  # [B, N, 2]
            pos_B: torch.Tensor,  # [B, N, 2]
            feat_A: torch.Tensor,  # [B, N, C]
            feat_B: torch.Tensor,  # [B, N, C]
    ) -> torch.Tensor:  # [B, N, 1]
        B, N, _ = pos_A.shape

        # 特征融合
        feat_diff = (feat_A - feat_B).abs()  # [B, N, C]
        feat_prod = feat_A * feat_B  # [B, N, C]
        feat_enc = self.feat_compress(
            torch.cat([feat_diff, feat_prod], dim=-1)  # [B, N, 2C]
        )  # [B, N, feat_compress_dim]

        # 几何+特征拼接
        coord_cat = torch.cat([pos_A, pos_B], dim=-1)  # [B, N, 4]
        x = torch.cat([coord_cat, feat_enc], dim=-1)  # [B, N, 4+feat_compress_dim]

        # PointNet: [B, N, C] → [B, C, N]
        x = x.transpose(1, 2).contiguous()  # [B, C_in, N]
        local_feat = self.local_encoder(x)  # [B, hidden_dim, N]

        # 全局最大池化
        global_feat = local_feat.max(dim=2, keepdim=True).values  # [B, hidden_dim, 1]
        global_feat = global_feat.expand(-1, -1, N)  # [B, hidden_dim, N]

        # 局部 + 全局融合 → 输出权重
        fused = torch.cat([local_feat, global_feat], dim=1)  # [B, 2*hidden_dim, N]
        weights = self.output_head(fused)  # [B, 1, N]
        return weights.transpose(1, 2).contiguous()  # [B, N, 1]


def hartley_normalize(pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = pts.float()
    B, N, _ = pts.shape
    device = pts.device

    centroid = pts.mean(dim=1, keepdim=True)
    pts_c = pts - centroid

    dist = (pts_c ** 2).sum(dim=-1).clamp(min=0.0).sqrt()
    mean_dist = dist.mean(dim=1, keepdim=True)

    is_degenerate = mean_dist < 1e-3
    mean_dist_safe = mean_dist.clamp(min=1e-3)
    scale = (2.0 ** 0.5) / mean_dist_safe
    scale = torch.where(is_degenerate, torch.ones_like(scale), scale)

    pts_norm = pts_c * scale.unsqueeze(-1)

    cx = centroid[:, 0, 0]
    cy = centroid[:, 0, 1]
    s  = scale[:, 0]
    zeros = torch.zeros_like(s)
    ones  = torch.ones_like(s)

    T = torch.stack([
        s,     zeros, -s * cx,
        zeros, s,     -s * cy,
        zeros, zeros, ones,
    ], dim=-1).reshape(B, 3, 3)

    return pts_norm, T


def solve_differentiable_dlt(src_pts, dst_pts, weights, reg_lambda=1e-4):
    B, N, _ = src_pts.shape
    device = src_pts.device

    # Hartley 归一化
    src_norm, T_src = hartley_normalize(src_pts)   # T_src: 列向量约定
    dst_norm, T_dst = hartley_normalize(dst_pts)

    # 构造 DLT 矩阵（行向量约定：[x',y',1] = [x,y,1] @ H_norm.T）
    sx, sy = src_norm[..., 0:1].float(), src_norm[..., 1:2].float()
    dx, dy = dst_norm[..., 0:1].float(), dst_norm[..., 1:2].float()
    zeros, ones = torch.zeros_like(sx), torch.ones_like(sx)

    row1 = torch.cat([sx, sy, ones, zeros, zeros, zeros, -dx*sx, -dx*sy], dim=-1)
    row2 = torch.cat([zeros, zeros, zeros, sx, sy, ones, -dy*sx, -dy*sy], dim=-1)
    A_full = torch.cat([row1, row2], dim=1)   # [B, 2N, 8]
    b_full = torch.cat([dx, dy], dim=1)       # [B, 2N, 1]

    # 3. 加权
    w_scaled = weights.float().squeeze(-1)
    w_diag   = torch.cat([w_scaled, w_scaled], dim=1).unsqueeze(-1)
    Aw   = A_full * w_diag
    AtWA = torch.bmm(Aw.transpose(1, 2), A_full)
    AtWb = torch.bmm(Aw.transpose(1, 2), b_full)

    reg_I = torch.eye(8, device=device, dtype=torch.float32).unsqueeze(0) * reg_lambda
    h_id  = torch.tensor([1.,0.,0.,0.,1.,0.,0.,0.],
                          device=device, dtype=torch.float32).view(1, 8, 1)
    AtWA = AtWA + reg_I
    AtWb = AtWb + reg_lambda * h_id

    with torch.amp.autocast('cuda', enabled=False):
        try:
            h8_final = torch.linalg.solve(AtWA.float(), AtWb.float()).squeeze(-1)
            if not torch.isfinite(h8_final).all():
                raise RuntimeError("Non-finite")
        except Exception:
            h8_final = h_id.squeeze(-1).expand(B, 8).float()

    h8_final = h8_final.to(src_pts.dtype)
    h1,h2,h3,h4,h5,h6,h7,h8_ = h8_final.unbind(dim=-1)
    ones_b = torch.ones_like(h1)
    # H_norm 满足（行向量约定）：dst_norm_homo = src_norm_homo @ H_norm.T
    H_norm = torch.stack([h1,h2,h3,h4,h5,h6,h7,h8_,ones_b],
                          dim=-1).reshape(B, 3, 3)

    T_src_f = T_src.float()
    T_dst_f = T_dst.float()

    # T_dst 解析逆：T^{-1} = [[1/s, 0, cx], [0, 1/s, cy], [0, 0, 1]]
    s_dst  = T_dst_f[:, 0, 0].clamp(min=1e-6)
    cx_dst = -T_dst_f[:, 0, 2] / s_dst                 # [B]
    cy_dst = -T_dst_f[:, 1, 2] / s_dst
    inv_s  = 1.0 / s_dst
    z      = torch.zeros_like(inv_s)
    o      = torch.ones_like(inv_s)
    T_dst_inv = torch.stack([
        inv_s, z,     cx_dst,
        z,     inv_s, cy_dst,
        z,     z,     o,
    ], dim=-1).reshape(B, 3, 3)

    H_actual = torch.bmm(T_dst_inv, torch.bmm(H_norm.float(), T_src_f))

    # 归一化 H[2,2] = 1
    H_actual = H_actual / (H_actual[:, 2:3, 2:3].abs().clamp(min=1e-8))

    h33 = H_actual[:, 2:3, 2:3]

    # 提取符号（防止0导致符号丢失）
    h33_sign = h33.sign()
    h33_sign = torch.where(h33_sign == 0, torch.ones_like(h33_sign), h33_sign)

    # 归一化：除以 h33 自身（带符号，防0），确保右下角永远是绝对的 +1
    safe_h33 = h33.abs().clamp(min=1e-8) * h33_sign
    H_actual = H_actual / safe_h33
    H_finite = torch.isfinite(H_actual).all(dim=(-2, -1))  # [B]
    h33_ok = H_actual[:, 2, 2].abs().clamp(1e-6, 1e6)
    h33_ok = (H_actual[:, 2, 2].abs() > 1e-4) & \
             (H_actual[:, 2, 2].abs() < 1e4)  # [B]
    valid = H_finite & h33_ok

    if not valid.all():
        eye = torch.eye(3, device=H_actual.device,
                        dtype=H_actual.dtype).unsqueeze(0).expand(B, -1, -1)
        H_actual = torch.where(
            valid.view(B, 1, 1).expand_as(H_actual),
            H_actual, eye
        )

    return H_actual.to(src_pts.dtype)


class AgriMatcher(nn.Module):
    def __init__(self, d_model=128, teacher_dim=100, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.refine_grid_size = grid_size * 2
        self._pe_cache = {}
        self.backbone = MobileViTBackbone(out_channels=d_model)

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
        )

        # ContextAwareInlierPredictor: NG-RANSAC 风格的语义引导权重头
        # 替代纯几何 margin 权重，解决重复纹理欺骗 IRLS 的漏洞
        self.inlier_predictor = ContextAwareInlierPredictor(
            feat_dim=d_model,
            feat_compress_dim=32,
            hidden_dim=128,
            num_layers=3,
        )

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
        feat_B = feat_B_coarse.flatten(2).transpose(1, 2)

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

        # grid_sample 需要 feat_B 在 [B, C, H, W] 格式
        feat_B_chw = feat_B_pos.transpose(1, 2).reshape(B, -1, Hc, Wc)
        # 将 B 的特征 warp 到 A 的空间位置
        feat_B_warped_chw = F.grid_sample(
            feat_B_chw.float(), warp_grid_tmp.float(),
            mode='bilinear', padding_mode='zeros', align_corners=False
        ).to(feat_B_pos.dtype)
        feat_B_warped = feat_B_warped_chw.reshape(B, -1, Hc * Wc).transpose(1, 2)  # [B, N, C]

        # 两层窗口注意力：在已对齐的坐标系内做局部精细匹配
        feat_A_pos, feat_B_warped = self.attn_layers[1](feat_A_pos, feat_B_warped, Hc, Wc)
        feat_A_pos, feat_B_warped = self.attn_layers[2](feat_A_pos, feat_B_warped, Hc, Wc)

        feat_A_pos, _ = self.attn_layers[3](feat_A_pos, feat_B_warped)

        feat_A = feat_A_pos
        feat_B = feat_B_warped

        pos_A = self._get_token_coords(B, Hc, Wc, device, dtype)  # [B, N, 2]
        pos_B = pos_A

        # 特征蒸馏
        distill_feat_A = self.feature_projector(feat_A)
        distill_feat_B = self.feature_projector(feat_B)

        temp = self.temperature.clamp(min=0.07, max=1.0).float()
        norm_A = F.normalize(feat_A.float(), p=2, dim=-1)
        norm_B = F.normalize(feat_B.float(), p=2, dim=-1)
        sim_matrix = torch.bmm(norm_A, norm_B.transpose(1, 2)) / temp
        prob_A_to_B = F.softmax(sim_matrix, dim=-1)  # fp32

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
            H_for_geo = solve_differentiable_dlt(
                pos_A.detach(), expected_pos_B_crude.detach(), margin.detach()
            )
            try:
                H_geo_inv = torch.linalg.inv(H_for_geo.float())
            except RuntimeError:
                H_geo_inv = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)

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

        # H 在 refine 后计算，用精化 warp 和 ContextAwareInlierPredictor
        # 优势：
        #   a) warp_AB_fine 已经经过 WarpRefiner 修正，残差更小，H 更准
        #   b) ContextAwareInlierPredictor 用特征差异识别语义假匹配（重复纹理骗不过）
        #   c) Hartley normalization 确保 DLT 数值稳定，H 不退化为单位阵
        warp_fine_flat = warp_AB_fine.reshape(B, Hf * Wf, 2)
        pos_A_fine = self._get_token_coords(B, Hf, Wf, device, dtype)

        # 对应的 fine 特征（B, N_fine, C）
        feat_A_fine_flat = feat_A_fine.flatten(2).transpose(1, 2)
        feat_B_aligned_for_ransac = safe_grid_sample(feat_B_fine, warp_AB_fine, align_corners=False)
        feat_B_fine_flat = feat_B_aligned_for_ransac.flatten(2).transpose(1, 2)

        if H_prior is not None:
            H_to_invert = H_prior
        else:
            inlier_weights = self.inlier_predictor(
                pos_A_fine.detach(),
                warp_fine_flat.detach(),
                feat_A_fine_flat.float(),
                feat_B_fine_flat.float(),
            )

            conf_fine_flat = confidence_AB_fine.reshape(B, Hf * Wf, 1).detach()
            combined_weights = (inlier_weights * conf_fine_flat).squeeze(-1)

            H_to_invert = solve_differentiable_dlt(
                pos_A_fine.detach(),
                warp_fine_flat.detach(),
                combined_weights.unsqueeze(-1),
            )

        return {
            # 匹配矩阵
            'sim_matrix': sim_matrix,
            'sim_matrix_kl': sim_matrix / temp,
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
            # 动态尺寸信息（供 Loss 使用）
            'coarse_hw': (Hc, Wc),
            'fine_hw': (Hf, Wf),
            # H_base 现在来自 refine 后的精化 warp + 语义引导权重
            'H_base': H_to_invert,
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
        key = (H, W, device, dtype)
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
        proj = F.linear(coords, self.pos_scale * self.omega)
        return torch.cat([proj.sin(), proj.cos()], dim=-1).unsqueeze(0)

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


class AgriTPSStitcher(nn.Module):
    def __init__(self, matcher_config, tps_config):
        super().__init__()
        self.matcher = AgriMatcher(**matcher_config)
        self.tps_estimator = BypassTPSEstimator(**tps_config)
        self.gs = tps_config['grid_size']  # 提取 grid_size

        self.grid_gen = TPSGridGenerator(grid_size=self.gs)

    def forward(self, img_A, img_B, H_prior=None):
        m_out = self.matcher(img_A, img_B, H_prior=H_prior)
        warp_AB = m_out['warp_AB']
        conf_AB = m_out['confidence_AB']

        B, H_feat, W_feat, _ = warp_AB.shape
        device = warp_AB.device
        dtype = warp_AB.dtype

        H_img, W_img = img_A.shape[2], img_A.shape[3]
        H_mat = m_out['H_base']

        grid_A_img = make_grid(B, H_img, W_img, device, torch.float32)
        src_pts_img = grid_A_img.reshape(B, -1, 2)
        ones_img = torch.ones(B, H_img * W_img, 1, device=device, dtype=torch.float32)
        coords_homo_img = torch.cat([src_pts_img, ones_img], dim=-1)
        projected_homo = torch.bmm(coords_homo_img, H_mat.transpose(1, 2))
        z_img = projected_homo[..., 2:3]
        sign_img = z_img.sign()
        sign_img = torch.where(sign_img == 0, torch.ones_like(sign_img), sign_img)
        denom = torch.where(z_img.abs() < 1e-3, 1e-3 * sign_img, z_img)

        base_grid = (projected_homo[..., :2] / denom).reshape(B, H_img, W_img, 2).clamp(-3., 3.)

        # warp_AB[b,i,j] = A位置(i,j)在B中的归一化坐标
        # 用 warp_AB 采样 feat_B，得到 A 坐标系下对应的 B 特征
        # padding_mode='zeros'：越界区域填0，FlowAggregator 可通过 coverage 图感知到这些区域
        feat_B_aligned = F.grid_sample(
            m_out['feat_B_64'],  # (B, C, Hf, Wf)，原始 B 特征图
            warp_AB.to(m_out['feat_B_64'].dtype),  # (B, Hf, Wf, 2)，对齐坐标
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )

        hc_wc = m_out.get('coarse_hw')
        tps_out = self.tps_estimator(
            warp_AB=warp_AB,
            confidence=conf_AB,
            feat_A=m_out['feat_A_64'],
            feat_B=feat_B_aligned,
            sim_matrix=m_out['sim_matrix'],
            warp_AB_coarse=m_out['warp_AB_coarse'],
            coarse_hw=hc_wc,
        )
        delta_cp_total = tps_out['delta_cp']

        y_cp = torch.linspace(-1. + 1. / self.gs, 1. - 1. / self.gs, self.gs, device=device, dtype=torch.float32)
        x_cp = torch.linspace(-1. + 1. / self.gs, 1. - 1. / self.gs, self.gs, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(y_cp, x_cp, indexing='ij')
        p_src = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2).expand(B, -1, -1)

        ones_cp = torch.ones(B, self.gs * self.gs, 1, device=device, dtype=torch.float32)
        p_src_homo = torch.cat([p_src, ones_cp], dim=-1)
        p_homo_proj = torch.bmm(p_src_homo, H_mat.transpose(1, 2))
        z_cp = p_homo_proj[..., 2:3]
        sign_cp = z_cp.sign()
        sign_cp = torch.where(sign_cp == 0, torch.ones_like(sign_cp), sign_cp)
        denom_cp = torch.where(z_cp.abs() < 1e-3, 1e-3 * sign_cp, z_cp)

        p_homo = (p_homo_proj[..., :2] / denom_cp).clamp(-10., 10.)

        p_target_total = p_src + delta_cp_total.float().reshape(B, 2, -1).permute(0, 2, 1)
        delta_cp_local = p_target_total - p_homo
        delta_cp_local_4d = delta_cp_local.permute(0, 2, 1).reshape(B, 2, self.gs, self.gs)

        tps_local_grid = self.grid_gen(delta_cp_local_4d, target_shape=(H_img, W_img))
        tps_residual_flow = tps_local_grid - grid_A_img
        final_grid = base_grid + tps_residual_flow

        if self.training:
            fold_info = self.tps_estimator.folding(delta_cp_local_4d, final_grid)
            tps_out.update(fold_info)

        return {
            'dense_grid': final_grid,
            'tps_residual_flow': tps_local_grid - base_grid,
            'delta_cp_local': delta_cp_local_4d,
            'delta_cp': delta_cp_total,
            'matcher_out': m_out,
            'H_mat': H_mat,
            'tps_out': tps_out,
        }


class LocalGeometricConsistency(nn.Module):
    def __init__(self):
        super().__init__()
        # 二阶拉普拉斯算子：检测非线性形变（仿射变换下，二阶导数为0）
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', lap.view(1, 1, 3, 3))

        # 一阶 Sobel 算子：检测位移突变
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_x.T.contiguous().view(1, 1, 3, 3))

    def forward(self, flow, confidence):
        """
        flow:       [B, H, W, 2]
        confidence: [B, H, W]
        只惩罚二阶拉普拉斯（非线性扭曲），允许一阶仿射/透视形变。
        """
        B, H, W, _ = flow.shape
        f = flow.permute(0, 3, 1, 2).float()  # [B, 2, H, W]

        lap = self.laplacian.to(device=f.device, dtype=f.dtype)

        if confidence.ndim == 3:
            conf = confidence.unsqueeze(1).float()
        elif confidence.shape[-1] == 1:
            conf = confidence.permute(0, 3, 1, 2).float()
        else:
            conf = confidence.unsqueeze(1).float()

        f_reshaped = f.reshape(B * 2, 1, H, W)

        # 只保留二阶惩罚，允许一阶透视形变
        f_padded = F.pad(f_reshaped, (1, 1, 1, 1), mode='replicate')

        # 卷积时不再 padding
        second_order = F.conv2d(f_padded, lap, padding=0).abs()
        second_order = second_order.reshape(B, 2, H, W).mean(1, keepdim=True)

        return (second_order * conf).mean()


class DistillationLoss(nn.Module):
    """
    支持动态尺寸的蒸馏损失
    """

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
        """TV 平滑损失必须作用在位移残差上，防止绝对坐标坍缩"""
        B, H, W, _ = warp.shape
        device = warp.device

        # 实时生成当前分辨率的归一化网格
        xs = (torch.arange(W, device=device, dtype=warp.dtype) + 0.5) * (2.0 / W) - 1.0
        ys = (torch.arange(H, device=device, dtype=warp.dtype) + 0.5) * (2.0 / H) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        identity_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # 求出真实的形变偏移量 (Flow)
        flow = warp - identity_grid

        # 在偏移量上惩罚突变 (二阶差分/Laplacian TV)
        # flow shape is (B, H, W, 2).
        # dim 1 is H (y-axis), dim 2 is W (x-axis)
        dx2 = torch.abs(flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :])
        dy2 = torch.abs(flow[:, 2:, :, :] - 2 * flow[:, 1:-1, :, :] + flow[:, :-2, :, :])

        return dx2.mean() + dy2.mean()

    @staticmethod
    def _weighted_huber(
            pred: torch.Tensor,
            target: torch.Tensor,
            weight: torch.Tensor,
            delta: float = 1.0,
    ) -> torch.Tensor:
        """加权 Huber 损失"""
        err = F.huber_loss(pred, target, reduction="none", delta=delta).sum(dim=-1)
        weight = weight.detach()
        return (err * weight).sum() / (weight.sum() + 1e-6)

    @staticmethod
    def _resize_warp(teacher_warp: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """调整 warp 尺寸"""
        teacher_warp_B = teacher_warp[..., -2:]
        return F.interpolate(
            teacher_warp_B.permute(0, 3, 1, 2),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

    @staticmethod
    def _resize_conf(teacher_conf: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """调整 confidence 尺寸"""
        if teacher_conf.ndim == 4 and teacher_conf.shape[-1] == 1:
            teacher_conf = teacher_conf[..., 0]
        return F.interpolate(
            teacher_conf.unsqueeze(1),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    @staticmethod
    def _clamp_normalized(xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """裁剪坐标"""
        if H <= 1 or W <= 1:
            return xy
        x = xy[..., 0].clamp(min=-1.0 + 1.0 / W, max=1.0 - 1.0 / W)
        y = xy[..., 1].clamp(min=-1.0 + 1.0 / H, max=1.0 - 1.0 / H)
        return torch.stack([x, y], dim=-1)

    def _warp_to_prob(self, warp_B: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        将 teacher warp 场转换为 coarse 匹配分布监督。
        支持非方形网格，但你的当前场景一般是 H=W=32。

        输入:
            warp_B[b, i, j] = A中(i,j)位置在B中的normalized对应坐标
        输出:
            soft_prob: (B, H*W, H*W)
        """
        B = warp_B.shape[0]
        N = H * W
        device = warp_B.device
        dtype = warp_B.dtype

        if H == 1 and W == 1:
            return torch.ones((B, 1, 1), device=device, dtype=dtype)

        # Token 坐标
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (2.0 / W) - 1.0
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (2.0 / H) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        token_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

        warp_flat = warp_B.reshape(B, N, 2)
        diff = warp_flat.unsqueeze(2) - token_coords.unsqueeze(0).unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1)

        sigma_x = 2.0 / W
        sigma_y = 2.0 / H
        sigma = 0.5 * (sigma_x + sigma_y)

        soft_prob = torch.exp(-dist2 / (2 * sigma ** 2 + 1e-8))
        soft_prob = soft_prob / (soft_prob.sum(dim=-1, keepdim=True) + 1e-8)

        return soft_prob

    def forward(
            self,
            stu_output: Dict[str, torch.Tensor],
            teacher_warp: torch.Tensor,
            teacher_conf: torch.Tensor,
            teacher_feat_A: torch.Tensor,
            teacher_feat_B: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # 获取动态尺寸
        Hc, Wc = stu_output.get('coarse_hw', (32, 32))
        Hf, Wf = stu_output.get('fine_hw', (64, 64))

        warp_coarse = stu_output["warp_AB_coarse"]
        conf_logits_coarse = stu_output["conf_logits_coarse"]
        warp_refine = stu_output["warp_AB"]
        conf_logits_refine = stu_output["conf_logits"]

        B = warp_coarse.shape[0]
        N_coarse = Hc * Wc

        #  特征蒸馏损失
        stu_feat_A = stu_output["distill_feat_A"]
        stu_feat_B = stu_output["distill_feat_B"]

        N_stu = stu_feat_A.shape[1]
        N_teacher = teacher_feat_A.shape[1]

        if N_stu != N_teacher:
            # 插值对齐到 teacher 尺寸
            teacher_gs = int(math.sqrt(N_teacher))
            d = stu_feat_A.shape[-1]

            stu_feat_A_2d = stu_feat_A.reshape(B, Hc, Wc, d).permute(0, 3, 1, 2)
            stu_feat_B_2d = stu_feat_B.reshape(B, Hc, Wc, d).permute(0, 3, 1, 2)

            stu_feat_A_aligned = F.interpolate(
                stu_feat_A_2d, size=(teacher_gs, teacher_gs), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1).reshape(B, N_teacher, d)

            stu_feat_B_aligned = F.interpolate(
                stu_feat_B_2d, size=(teacher_gs, teacher_gs), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1).reshape(B, N_teacher, d)
        else:
            stu_feat_A_aligned = stu_feat_A
            stu_feat_B_aligned = stu_feat_B

        def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """1 - cosine_similarity，均值，范围 [0, 2]"""
            a_norm = F.normalize(a.float(), p=2, dim=-1)
            b_norm = F.normalize(b.float(), p=2, dim=-1)
            # cosine sim: [B, N]
            cos_sim = (a_norm * b_norm).sum(dim=-1)
            return (1.0 - cos_sim).mean()

        loss_feat = (
                _cosine_loss(stu_feat_A_aligned, teacher_feat_A) +
                _cosine_loss(stu_feat_B_aligned, teacher_feat_B)
        )

        # Warp 损失
        teacher_warp_c = self._resize_warp(teacher_warp, (Hc, Wc))
        teacher_warp_r = self._resize_warp(teacher_warp, (Hf, Wf))
        teacher_warp_c = self._clamp_normalized(teacher_warp_c, Hc, Wc)
        teacher_warp_r = self._clamp_normalized(teacher_warp_r, Hf, Wf)

        teacher_conf_c = self._resize_conf(teacher_conf, (Hc, Wc)).clamp(0, 1)
        teacher_conf_r = self._resize_conf(teacher_conf, (Hf, Wf)).clamp(0, 1)

        conf_mean = teacher_conf_c.mean(dim=(1, 2), keepdim=True).clamp(min=0.1)
        teacher_conf_c_normalized = (teacher_conf_c / conf_mean).clamp(0, 2.0)

        loss_warp_coarse = self._weighted_huber(warp_coarse, teacher_warp_c, teacher_conf_c_normalized)
        loss_warp_refine = self._weighted_huber(warp_refine, teacher_warp_r, teacher_conf_r)

        # KL 损失
        sim_matrix = stu_output.get("sim_matrix_kl")
        if sim_matrix is not None and sim_matrix.shape[1] == N_coarse:
            teacher_prob_c = self._warp_to_prob(teacher_warp_c, Hc, Wc)

            # 使用适中的温度，保留软分布信息
            # sim_matrix 已经除过 current_temp（约0.1~1.0），再除 KL_TEMP=1.0 即可
            KL_TEMP = 1.0  # 不再额外压缩，sim_matrix 已含温度信息
            sim_for_kl = sim_matrix.float()
            log_prob = F.log_softmax(sim_for_kl / KL_TEMP, dim=-1)
            target_prob = teacher_prob_c.detach().float()
            kl_map = F.kl_div(log_prob, target_prob, reduction="none").sum(dim=-1)

            valid_weight = teacher_conf_c.reshape(B, N_coarse).detach()
            valid_mask = (valid_weight > self.conf_thresh_kl).float()
            valid_sum = valid_mask.sum().clamp(min=1.0)
            loss_kl = (kl_map * valid_mask).sum() / valid_sum
        else:
            loss_kl = torch.zeros(1, device=warp_coarse.device).squeeze()

        # Confidence 损失
        loss_conf_coarse = F.binary_cross_entropy_with_logits(
            conf_logits_coarse, teacher_conf_c.detach()
        )
        loss_conf_refine = F.binary_cross_entropy_with_logits(
            conf_logits_refine, teacher_conf_r.detach()
        )

        # TV 损失
        loss_smooth_coarse = self.total_variation_loss(warp_coarse)
        loss_smooth_refine = self.total_variation_loss(warp_refine)

        # 总损失
        total = (
                self.alpha * loss_feat +
                self.beta_coarse * loss_warp_coarse +
                self.beta_refine * loss_warp_refine +
                self.gamma * loss_kl +
                self.eta_coarse * loss_conf_coarse +
                self.eta_refine * loss_conf_refine +
                self.lambda_tv_coarse * loss_smooth_coarse +
                self.lambda_tv_refine * loss_smooth_refine
        )

        return {
            "total": total,
            "loss_feat": float(loss_feat.detach().item()),
            "loss_warp_coarse": float(loss_warp_coarse.detach().item()),
            "loss_warp_refine": float(loss_warp_refine.detach().item()),
            "loss_kl": float(loss_kl.detach().item()) if torch.is_tensor(loss_kl) else 0.0,
            "loss_conf_coarse": float(loss_conf_coarse.detach().item()),
            "loss_conf_refine": float(loss_conf_refine.detach().item()),
            "loss_smooth_coarse": float(loss_smooth_coarse.detach().item()),
            "loss_smooth_refine": float(loss_smooth_refine.detach().item()),
        }


if __name__ == "__main__":
    pass