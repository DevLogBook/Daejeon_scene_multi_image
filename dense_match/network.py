import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_match.backbone import MobileViTBackbone, generate_2d_sincos_pos_emb
from dense_match.refine import (WarpRefiner, 
                                upsample_warp_and_overlap, 
                                compute_cycle_consistency_loss, 
                                compute_photometric_loss,
                                safe_grid_sample,
                                make_grid,
                                ssim_map)
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
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.norm1_A = nn.LayerNorm(d_model)
        self.norm1_B = nn.LayerNorm(d_model)
        self.norm2_A = nn.LayerNorm(d_model)
        self.norm2_B = nn.LayerNorm(d_model)

        self.ffn_A = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_B = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3_A = nn.LayerNorm(d_model)
        self.norm3_B = nn.LayerNorm(d_model)

    def forward(self, feat_A, feat_B):
        # 自注意力
        feat_A = self.norm1_A(feat_A + self.self_attn(feat_A, feat_A, feat_A)[0])
        feat_B = self.norm1_B(feat_B + self.self_attn(feat_B, feat_B, feat_B)[0])

        # 交叉注意力
        feat_A_out = self.norm2_A(feat_A + self.cross_attn(feat_A, feat_B, feat_B)[0])
        feat_B_out = self.norm2_B(feat_B + self.cross_attn(feat_B, feat_A, feat_A)[0])

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
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

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

        # Local Cross Attention (A attends to B within the same window)
        # 这里的逻辑是：只在对应的空间窗口内寻找匹配，适合位移不大的精细化匹配
        attn_A, _ = self.mha(win_A, win_B, win_B)
        attn_B, _ = self.mha(win_B, win_A, win_A)

        # Window Reverse
        feat_A = self.window_reverse(attn_A, H, W, Hp, Wp)
        feat_B = self.window_reverse(attn_B, H, W, Hp, Wp)

        # Residual & FFN
        feat_A = self.norm(feat_A + short_cut_A)
        feat_A = self.norm(feat_A + self.ffn(feat_A))
        feat_B = self.norm(feat_B + short_cut_B)
        feat_B = self.norm(feat_B + self.ffn(feat_B))

        return feat_A, feat_B


class LocalGeometricValidator(nn.Module):
    """验证候选匹配的局部几何一致性 """
    def __init__(self, neighbor_radius=3):
        super().__init__()
        self.radius = neighbor_radius
    
    def forward(self, src_pos, candidate_positions, H, W):
        """
        src_pos: [B, N, 2]
        candidate_positions: [B, N, K, 2]
        H, W: 当前特征图的实际宽高
        """
        B, N, K, _ = candidate_positions.shape
        
        geometric_scores = []
        # 对 K 个候选峰值逐一评估
        for k in range(K):
            cand_pos = candidate_positions[:, :, k, :] # [B, N, 2]
            
            # 计算位移场 (Flow)
            displacement = cand_pos - src_pos # [B, N, 2]
            disp_2d = displacement.reshape(B, H, W, 2).permute(0, 3, 1, 2) # [B, 2, H, W]
            
            # 提取邻域位移
            disp_unfold = F.unfold(
                disp_2d,
                kernel_size=2*self.radius+1,
                padding=self.radius
            ) # [B, 2*(2r+1)^2, N]
            
            # 计算邻域位移的一致性：方差越小，局部越像仿射变换
            # 我们惩罚位移突变
            disp_var = disp_unfold.var(dim=1) # [B, N]
            
            # 转换为一致性得分，1e-4 防止除零
            score = 1.0 / (1.0 + disp_var * 100.0) 
            geometric_scores.append(score)
            
        return torch.stack(geometric_scores, dim=-1) # [B, N, K]


class MultiPeakAwareAttention(nn.Module):
    def __init__(self, d_model=128, top_k=4, temperature=0.05):
        super().__init__()
        self.top_k = top_k
        self.temp = temperature # 控制分布的尖锐程度
        self.geometric_validator = LocalGeometricValidator(neighbor_radius=3)
        
    def forward(self, feat_A, feat_B, pos_A, pos_B, H, W):
        B, N, C = feat_A.shape
        
        # 计算特征相似度
        norm_A = F.normalize(feat_A, p=2, dim=-1)
        norm_B = F.normalize(feat_B, p=2, dim=-1)
        sim_matrix = torch.bmm(norm_A, norm_B.transpose(1, 2))
        
        # 提取候选峰值
        topk_values, topk_indices = sim_matrix.topk(self.top_k, dim=-1)
        
        # 获取候选点的物理坐标 [-1, 1]
        topk_positions = torch.gather(
            pos_B.unsqueeze(1).expand(-1, N, -1, -1),
            dim=2,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
        )
        
        # 计算几何一致性评分
        geo_scores = self.geometric_validator(pos_A, topk_positions, H, W)
        
        # 可微消歧
        # 综合得分 = 外观相似度 + 几何一致性
        combined_scores = topk_values + 0.5 * geo_scores
        
        # 使用 Softmax 替代 argmax，允许梯度回传
        # 这里使用了较小的温度系数来模拟“选择”行为，同时保留梯度
        soft_weights = F.softmax(combined_scores / self.temp, dim=-1) 
        
        # 加权求和得到精细化坐标
        refined_warp = (topk_positions * soft_weights.unsqueeze(-1)).sum(dim=2)
        
        # 计算匹配熵
        prob_dist = F.softmax(sim_matrix / self.temp, dim=-1)
        entropy = -(prob_dist * (prob_dist + 1e-8).log()).sum(dim=-1)
        
        return refined_warp, entropy, sim_matrix


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
            MatchWindowAttention(d_model, num_heads=4, window_size=8),
            MatchWindowAttention(d_model, num_heads=4, window_size=8),
            MatchAttention(d_model, nhead=4), 
            MatchAttention(d_model, nhead=4)
        ])
        self.peak_aware_layer = MultiPeakAwareAttention(d_model=d_model, top_k=4)

        self.temperature = nn.Parameter(torch.tensor(0.7))
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
            nn.Linear(d_model * 2 + 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.refiner_64 = WarpRefiner(
            C=d_model,
            hidden=max(32, d_model // 2),
            num_blocks=1,
            delta_scale=0.25,
            residual_overlap=True,
        )

    def forward(self, img_A, img_B):
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
        feat_B = feat_B_coarse.flatten(2).transpose(1, 2)  # (B, N, d)
        
        # 动态位置编码（用 fp32 计算更稳定，再 cast 回输入 dtype）
        pos_embed = self._get_pos_embed(Hc, Wc, device, torch.float32).to(dtype)
        feat_A = feat_A + pos_embed
        feat_B = feat_B + pos_embed
        
        # Attention 层
        for i, layer in enumerate(self.attn_layers):
            if isinstance(layer, MatchWindowAttention):
                feat_A, feat_B = layer(feat_A, feat_B, Hc, Wc)
            else:
                feat_A, feat_B = layer(feat_A, feat_B)

        pos_A = self._get_token_coords(B, Hc, Wc, device, dtype) # [B, N, 2]
        pos_B = pos_A

        # 特征蒸馏
        distill_feat_A = self.feature_projector(feat_A)
        distill_feat_B = self.feature_projector(feat_B)
        
        # Similarity Matrix（关键数值路径强制 fp32，避免 AMP 下 NaN）
        temp = self.temperature.clamp(min=0.01).float()
        norm_A = F.normalize(feat_A.float(), p=2, dim=-1)
        norm_B = F.normalize(feat_B.float(), p=2, dim=-1)
        sim_matrix = torch.bmm(norm_A, norm_B.transpose(1, 2)) / temp
        prob_A_to_B = F.softmax(sim_matrix, dim=-1)  # fp32
        

        # 期望特征（用于置信度估计）
        refined_warp_coarse, match_entropy, sim_matrix = self.peak_aware_layer(
            feat_A, feat_B, pos_A, pos_B, Hc, Wc
        )
        expected_feat_B = torch.bmm(prob_A_to_B.to(feat_B.dtype), feat_B)
        feat_diff = torch.abs(feat_A - expected_feat_B)
        feat_prod = feat_A * expected_feat_B
        
        top2_vals = torch.topk(prob_A_to_B, k=2, dim=-1).values
        top1_prob = top2_vals[..., 0:1]
        top2_prob = top2_vals[..., 1:2]
        margin = top1_prob - top2_prob
        entropy = -(prob_A_to_B * torch.log(prob_A_to_B.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        
        warp_AB_coarse = refined_warp_coarse.reshape(B, Hc, Wc, 2)
        
        # Confidence
        conf_input = torch.cat(
            [feat_diff, feat_prod, top1_prob, top2_prob, margin, entropy],
            dim=-1
        )
        conf_logits_coarse = self.conf_head(conf_input).squeeze(-1).reshape(B, Hc, Wc)
        confidence_AB_coarse = torch.sigmoid(conf_logits_coarse)
        
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
        
        return {
            # 匹配矩阵
            'sim_matrix': sim_matrix,
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
            y = torch.linspace(-1.0 + 1.0/H, 1.0 - 1.0/H, H, device=device, dtype=dtype)
            x = torch.linspace(-1.0 + 1.0/W, 1.0 - 1.0/W, W, device=device, dtype=dtype)
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
        y = torch.linspace(-1.0 + 1.0/H, 1.0 - 1.0/H, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0 + 1.0/W, 1.0 - 1.0/W, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        return coords.unsqueeze(0).expand(B, -1, -1)

    def _clamp_coords(self, xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """裁剪坐标到有效范围"""
        if H <= 1 or W <= 1:
            return xy
        x = xy[..., 0].clamp(min=-1.0 + 1.0/W, max=1.0 - 1.0/W)
        y = xy[..., 1].clamp(min=-1.0 + 1.0/H, max=1.0 - 1.0/H)
        return torch.stack([x, y], dim=-1)


class AgriTPSStitcher(nn.Module):
    def __init__(self, matcher_config, tps_config):
        super().__init__()
        self.matcher = AgriMatcher(**matcher_config)
        self.tps_estimator = BypassTPSEstimator(**tps_config)
        self._grid_cache = {}
        
        # 初始化 TPS 网格生成器
        self.grid_gen = TPSGridGenerator(
            grid_size=tps_config['grid_size'],
            target_height=512,
            target_width=512
        )

    def forward(self, img_A, img_B):
        # 密集匹配 (AgriMatcher)
        # 内部已包含 Local Attention 和 Corr_5x5
        m_out = self.matcher(img_A, img_B)
        
        # 控制点回归 (BypassTPSEstimator)
        # 传入所有辅助信号：sim_matrix, coarse_warp, feat...
        hc_wc = m_out.get('coarse_hw')
        tps_out = self.tps_estimator(
            warp_AB=m_out['warp_AB'],
            confidence=m_out['confidence_AB'],
            feat_A=m_out['feat_A_64'],
            feat_B=m_out['feat_B_64'],
            sim_matrix=m_out['sim_matrix'],
            warp_AB_coarse=m_out['warp_AB_coarse'],
            coarse_hw=hc_wc
        )
        
        # 生成 TPS 密集网格
        dense_grid = self.grid_gen(tps_out['delta_cp'], target_shape=img_A.shape[-2:])

        # 4. 训练时：将生成的密集场传回进行折叠惩罚检查
        if self.training:
            fold_info = self.tps_estimator.folding(tps_out['delta_cp'], dense_grid)
            tps_out.update(fold_info)

        return {
            'dense_grid': dense_grid,      # 最终用于采样拼接的网格
            'delta_cp': tps_out['delta_cp'],
            'matcher_out': m_out,          # 用于计算蒸馏 Loss
            'tps_out': tps_out             # 用于计算折叠 Loss
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
        flow: [B, H, W, 2]
        confidence: [B, H, W]
        """
        B, H, W, _ = flow.shape
        f = flow.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        # 强制 buffer 与输入 f 的设备和 dtype 对齐，防止 AMP 或设备冲突
        sx = self.sobel_x.to(device=f.device, dtype=f.dtype)
        sy = self.sobel_y.to(device=f.device, dtype=f.dtype)
        lap = self.laplacian.to(device=f.device, dtype=f.dtype)
        
        # 确保 confidence 维度正确
        if confidence.ndim == 3:
            conf = confidence.unsqueeze(1)
        else:
            # 处理可能的 (B,H,W,1) 或 (B,1,H,W)
            conf = confidence.permute(0, 3, 1, 2) if confidence.shape[-1] == 1 else confidence

        # 统一形状处理 [B*2, 1, H, W]
        f_reshaped = f.reshape(B * 2, 1, H, W)
        
        # 一阶平滑惩罚（惩罚突变）
        grad_x = F.conv2d(f_reshaped, sx, padding=1).abs()
        grad_y = F.conv2d(f_reshaped, sy, padding=1).abs()
        first_order = (grad_x + grad_y).reshape(B, 2, H, W).mean(1, keepdim=True)
        
        # 二阶平滑惩罚（惩罚非仿射扭曲，保持局部刚性）
        second_order = F.conv2d(f_reshaped, lap, padding=1).abs()
        second_order = second_order.reshape(B, 2, H, W).mean(1, keepdim=True)
        
        # 置信度加权求均值
        return ((first_order + second_order) * conf).mean()

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
        """TV 平滑损失"""
        dx = torch.abs(warp[:, :, 1:, :] - warp[:, :, :-1, :])
        dy = torch.abs(warp[:, 1:, :, :] - warp[:, :-1, :, :])
        return dx.mean() + dy.mean()
    
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
        x = xy[..., 0].clamp(min=-1.0 + 1.0/W, max=1.0 - 1.0/W)
        y = xy[..., 1].clamp(min=-1.0 + 1.0/H, max=1.0 - 1.0/H)
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
        
        loss_feat = (
            F.mse_loss(stu_feat_A_aligned, teacher_feat_A) +
            F.mse_loss(stu_feat_B_aligned, teacher_feat_B)
        )
        
        # Warp 损失
        teacher_warp_c = self._resize_warp(teacher_warp, (Hc, Wc))
        teacher_warp_r = self._resize_warp(teacher_warp, (Hf, Wf))
        teacher_warp_c = self._clamp_normalized(teacher_warp_c, Hc, Wc)
        teacher_warp_r = self._clamp_normalized(teacher_warp_r, Hf, Wf)
        
        teacher_conf_c = self._resize_conf(teacher_conf, (Hc, Wc)).clamp(0, 1)
        teacher_conf_r = self._resize_conf(teacher_conf, (Hf, Wf)).clamp(0, 1)
        
        loss_warp_coarse = self._weighted_huber(warp_coarse, teacher_warp_c, teacher_conf_c)
        loss_warp_refine = self._weighted_huber(warp_refine, teacher_warp_r, teacher_conf_r)
        
        # KL 损失
        sim_matrix = stu_output.get("sim_matrix")
        if sim_matrix is not None and sim_matrix.shape[1] == N_coarse:
            teacher_prob_c = self._warp_to_prob(teacher_warp_c, Hc, Wc)
            log_prob = F.log_softmax(sim_matrix, dim=-1).float()
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


class AgriTPSLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 10.0, epsilon: float = 1e-3):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.epsilon = epsilon

    def charbonnier_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """鲁棒惩罚函数"""
        return torch.sqrt(diff ** 2 + self.epsilon ** 2)

    def compute_photometric_loss(self, img_A: torch.Tensor, warped_img_B: torch.Tensor, 
                                 mask: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """
        img_A, warped_img_B: (B, 3, H, W)
        mask: (B, 1, H, W) 重叠区掩码
        confidence: (B, 1, H, W) AgriMatcher 输出的置信度图
        """
        # 计算像素级差异
        diff = img_A - warped_img_B
        charb_diff = self.charbonnier_loss(diff) # (B, 3, H, W)
        
        # 沿通道维度求均值
        charb_diff = charb_diff.mean(dim=1, keepdim=True) # (B, 1, H, W)
        
        # 计算综合权重
        weight = mask * confidence
        
        # 加权求和并归一化
        loss = torch.sum(charb_diff * weight) / (torch.sum(weight) + 1e-6)
        return loss

    def compute_smoothness_loss(self, delta_C: torch.Tensor) -> torch.Tensor:
        """
        计算二阶网格平滑损失
        delta_C: (B, 2, grid_h, grid_w) GridNet 输出的控制点偏移
        """
        # 水平方向二阶差分: C[i, j+1] - 2C[i, j] + C[i, j-1]
        diff_x = delta_C[:, :, :, 2:] - 2 * delta_C[:, :, :, 1:-1] + delta_C[:, :, :, :-2]
        
        # 垂直方向二阶差分: C[i+1, j] - 2C[i, j] + C[i-1, j]
        diff_y = delta_C[:, :, 2:, :] - 2 * delta_C[:, :, 1:-1, :] + delta_C[:, :, :-2, :]
        
        loss_x = torch.mean(diff_x ** 2)
        loss_y = torch.mean(diff_y ** 2)
        
        return loss_x + loss_y

    def forward(self, img_A: torch.Tensor, warped_img_B: torch.Tensor, 
                delta_C: torch.Tensor, confidence: torch.Tensor) -> dict:
        
        # 自动计算有效重叠区掩码 (假设无像素区域为绝对 0)
        # 考虑到差值容差，只要 img_A 或 warped_img_B 有像素即可
        mask_A = (img_A.sum(dim=1, keepdim=True) > 0).float()
        mask_B = (warped_img_B.sum(dim=1, keepdim=True) > 0).float()
        overlap_mask = mask_A * mask_B

        # 如果 confidence 尺寸不匹配 (如 64x64)，需上采样到全图尺寸
        if confidence.shape[-2:] != img_A.shape[-2:]:
            confidence = F.interpolate(confidence, size=img_A.shape[-2:], mode='bilinear', align_corners=False)

        # 计算各项损失
        loss_photo = self.compute_photometric_loss(img_A, warped_img_B, overlap_mask, confidence)
        loss_smooth = self.compute_smoothness_loss(delta_C)

        total_loss = loss_photo + self.lambda_smooth * loss_smooth

        return {
            "total": total_loss,
            "photo": loss_photo,
            "smooth": loss_smooth
        }

if __name__ == "__main__":
    pass