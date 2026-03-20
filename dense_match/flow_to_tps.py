import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from enum import Enum
from typing import Dict, Optional, Tuple


# 工具模块
class _DSConv(nn.Module):
    """深度可分离卷积（内部辅助类）"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DetachStrategy(Enum):
    FULL_DETACH = "full_detach"
    NO_DETACH   = "no_detach"
    GATED       = "gated"   # 推荐：门控融合，同时解决detach争议和假匹配鲁棒性


# 置信度加权投票

class ConfidenceWeightedVotingDynamic(nn.Module):
    """
    支持动态分辨率的置信度加权投票。
    物理驱动，无可训练参数，可解释。

    数学：delta_ck = Σ_i [conf_i * G(p_i, c_k) * flow_i]
                        / Σ_i [conf_i * G(p_i, c_k)]
    其中 G 是以控制点间距为 sigma 的高斯核。

    缓存策略：按 (H, W, device, dtype) 缓存高斯核，
              大田场景通常只有1~3种分辨率，命中率高。
    """

    def __init__(
        self,
        grid_size: int = 10,
        sigma_scale: float = 0.7,
        min_weight_sum: float = 1e-3,
        max_cached_resolutions: int = 4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.sigma_scale = sigma_scale
        self.min_weight_sum = min_weight_sum
        self.max_cached_resolutions = max_cached_resolutions

        self.cp_spacing = 2.0 / max(grid_size - 1, 1)
        self.sigma = sigma_scale * self.cp_spacing

        # 预计算控制点坐标（固定，注册为buffer随模型保存）
        y_cp = torch.linspace(-1.0, 1.0, grid_size)
        x_cp = torch.linspace(-1.0, 1.0, grid_size)
        gy, gx = torch.meshgrid(y_cp, x_cp, indexing="ij")
        cp_coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)  # (N_cp, 2)
        self.register_buffer("cp_coords", cp_coords)

        # LRU 高斯核缓存（不注册为buffer，运行时动态管理）
        self._kernel_cache: OrderedDict = OrderedDict()

    def _get_flow_pixel_coords(
        self, H: int, W: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """生成 align_corners=False 的像素中心归一化坐标 (H*W, 2)"""
        y_f = torch.linspace(-1.0 + 1.0/H, 1.0 - 1.0/H, H, device=device, dtype=dtype)
        x_f = torch.linspace(-1.0 + 1.0/W, 1.0 - 1.0/W, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(y_f, x_f, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

    def _compute_gaussian_kernel(
        self, H: int, W: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        """计算并归一化高斯核 (H*W, N_cp)"""
        flow_coords = self._get_flow_pixel_coords(H, W, device, dtype)
        cp = self.cp_coords.to(device=device, dtype=dtype)

        # 用展开公式计算平方距离，避免大中间张量
        a_sq = (flow_coords ** 2).sum(1, keepdim=True)   # (N_f, 1)
        b_sq = (cp ** 2).sum(1, keepdim=True).t()        # (1, N_cp)
        ab   = torch.mm(flow_coords, cp.t())              # (N_f, N_cp)
        dist2 = (a_sq + b_sq - 2 * ab).clamp(min=0.0)

        kernel = torch.exp(-dist2 / (2 * self.sigma ** 2))
        kernel = kernel / (kernel.sum(1, keepdim=True) + 1e-8)  # Nadaraya-Watson
        return kernel

    def _get_cached_kernel(
        self, H: int, W: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (H, W, str(device), str(dtype))
        if key in self._kernel_cache:
            self._kernel_cache.move_to_end(key)
            return self._kernel_cache[key]
        kernel = self._compute_gaussian_kernel(H, W, device, dtype)
        if len(self._kernel_cache) >= self.max_cached_resolutions:
            self._kernel_cache.popitem(last=False)
        self._kernel_cache[key] = kernel
        return kernel

    def clear_cache(self):
        """切换设备/dtype时手动清空（避免stale缓存）"""
        self._kernel_cache.clear()

    def forward(
        self,
        warp_AB: torch.Tensor,    # (B, H, W, 2)
        confidence: torch.Tensor, # (B, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            delta_cp: (B, 2, grid_size, grid_size) 控制点偏移
            coverage: (B, grid_size, grid_size) 归一化覆盖度 [0,1]
        """
        B, H, W, _ = warp_AB.shape
        N_flow = H * W
        device, dtype = warp_AB.device, warp_AB.dtype

        kernel      = self._get_cached_kernel(H, W, device, dtype)   # (N_f, N_cp)
        flow_coords = self._get_flow_pixel_coords(H, W, device, dtype)

        # flow = 目标坐标 - 源坐标（位移向量）
        flow      = warp_AB.reshape(B, N_flow, 2) - flow_coords.unsqueeze(0)  # (B,N_f,2)
        conf_flat = confidence.reshape(B, N_flow)                               # (B,N_f)

        # 置信度调制核：w[b,i,k] = conf[b,i] * kernel[i,k]
        w   = conf_flat.unsqueeze(2) * kernel.unsqueeze(0)   # (B,N_f,N_cp)
        w_t = w.transpose(1, 2)                               # (B,N_cp,N_f)

        weight_sum    = w_t.sum(-1, keepdim=True)             # (B,N_cp,1)
        delta_cp_flat = torch.bmm(w_t, flow) / (weight_sum + self.min_weight_sum)
        # (B,N_cp,2)

        # 覆盖度图
        gs = self.grid_size
        coverage = weight_sum.squeeze(-1).reshape(B, gs, gs)
        coverage = coverage / (coverage.amax(dim=(1, 2), keepdim=True) + 1e-6)

        # (B,N_cp,2) → (B,2,gs,gs)
        delta_cp = delta_cp_flat.reshape(B, gs, gs, 2).permute(0, 3, 1, 2).contiguous()

        return delta_cp, coverage

    def extra_repr(self) -> str:
        return (f"grid_size={self.grid_size}, sigma={self.sigma:.4f}, "
                f"max_cached={self.max_cached_resolutions}")


# 辅助信号提取

class MatchingEntropyMap(nn.Module):
    """
    从 sim_matrix 计算匹配熵图。
    熵高 → 匹配模糊 → 不可信（大田纹理重复场景的假匹配信号）。
    输出归一化到 [0,1]，不反传梯度到 AgriMatcher。
    """

    def __init__(self, grid_size: int = 32, output_size: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.output_size = output_size

    def forward(self, sim_matrix: torch.Tensor, Hc: int, Wc: int) -> torch.Tensor:
        """
        Hc, Wc: Coarse 阶段特征图的实际高度和宽度
        """
        B, N, _ = sim_matrix.shape
        
        prob    = F.softmax(sim_matrix.float(), dim=-1)
        entropy = -(prob * torch.log(prob + 1e-8)).sum(-1)    # (B, N)

        max_ent = torch.log(torch.tensor(float(N), device=sim_matrix.device))
        entropy_norm = (entropy / (max_ent + 1e-8)).clamp(0.0, 1.0)

        entropy_map = entropy_norm.reshape(B, 1, Hc, Wc)

        return entropy_map.detach()


class CoarseFineConsistency(nn.Module):
    """
    Coarse-Fine Warp 一致性图。
    两级预测一致 → 区域可信；差异大 → refiner做了大幅修正 → 谨慎。
    输出 (B, 1, Hf, Wf) ∈ [0,1]，不反传梯度。
    """

    def __init__(self, tau: float = 0.05):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        warp_coarse: torch.Tensor,  # (B, Hc, Wc, 2)
        warp_fine:   torch.Tensor,  # (B, Hf, Wf, 2)
    ) -> torch.Tensor:
        Hf, Wf = warp_fine.shape[1:3]
        warp_coarse_up = F.interpolate(
            warp_coarse.permute(0, 3, 1, 2),
            size=(Hf, Wf), mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)

        diff        = (warp_fine - warp_coarse_up).norm(dim=-1, keepdim=True)
        consistency = torch.exp(-diff / self.tau).permute(0, 3, 1, 2)  # (B,1,Hf,Wf)
        return consistency.detach()


# FlowAggregator（可学习残差修正，GATED模式）

class FlowAggregator(nn.Module):
    """
    在投票初始解基础上，用特征信息做门控残差修正。

    设计要点：
    - GATED 融合：final = gate * vote_init + (1-gate) * learned_delta
      gate 接近1 → 信任投票；接近0 → 依赖学习（应对假匹配）
    - gate 初始化为高值（sigmoid(2)≈0.88），训练初期稳定
    - residual_head 初始化为0，确保训练初期 = 纯投票结果
    - in_ch 外部指定，支持任意辅助信号数量
    """

    def __init__(
        self,
        grid_size: int = 10,
        feat_channels: int = 128,
        hidden_ch: int = 48,
        in_ch: int = 5,          # 输入通道数（由外部按实际信号数决定）
        delta_scale: float = 0.2,
    ):
        super().__init__()
        self.grid_size  = grid_size
        self.hidden_ch  = hidden_ch
        cp_spacing      = 2.0 / max(grid_size - 1, 1)
        self.max_delta  = delta_scale * cp_spacing

        # 特征一致性头：(feat_A, feat_B) → 1通道一致性图
        self.consistency_head = nn.Sequential(
            nn.Conv2d(feat_channels * 2, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

        # 编码器（在flow图分辨率上处理，逐步下采样到控制点网格）
        self.encoder = nn.Sequential(
            _DSConv(in_ch, hidden_ch, stride=1),
            _DSConv(hidden_ch, hidden_ch, stride=2),
            _DSConv(hidden_ch, hidden_ch, stride=2),
        )
        self.adapt_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # 门控头：预测对投票结果的信任度 [0,1]
        # 初始化偏向1（训练早期信任投票，逐渐自适应）
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_ch + 2, hidden_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch // 2), nn.GELU(),
            nn.Conv2d(hidden_ch // 2, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_head[-2].bias, 2.0)  # sigmoid(2) ≈ 0.88

        # 残差头：预测学习到的偏移量
        # 初始化为0，确保训练初期输出 = 投票结果（稳定起点）
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden_ch + 2, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch), nn.GELU(),
            nn.Conv2d(hidden_ch, 2, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.residual_head[-2].weight)
        nn.init.zeros_(self.residual_head[-2].bias)

    def forward(
        self,
        feat_in: torch.Tensor,         # (B, in_ch, H, W) 已拼接的输入特征
        delta_cp_init: torch.Tensor,   # (B, 2, gs, gs) 投票初始解
        feat_A: torch.Tensor,          # (B, C, H, W)
        feat_B: torch.Tensor,          # (B, C, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            delta_cp_final: (B, 2, gs, gs)
            gate:           (B, 1, gs, gs) ∈ [0,1]
        """
        # 特征一致性（额外的1通道监督信号，通过consistency_head计算）
        # 注意：consistency已在BypassTPSEstimator中计算并concat进feat_in
        # 这里直接用feat_in编码即可

        encoded    = self.encoder(feat_in)             # (B, hid, ~H/4, ~W/4)
        encoded_gs = self.adapt_pool(encoded)          # (B, hid, gs, gs)

        # skip connection：将投票初始解拼入头部输入
        # 梯度可以流向 delta_cp_init（为未来可学习Voting参数保留通道）
        head_input = torch.cat([encoded_gs, delta_cp_init], dim=1)  # (B,hid+2,gs,gs)

        # 门控融合
        gate          = self.gate_head(head_input)                    # (B,1,gs,gs)
        learned_delta = self.residual_head(head_input) * self.max_delta  # (B,2,gs,gs)

        # final = gate * vote + (1-gate) * learned
        delta_cp_final = gate * delta_cp_init + (1.0 - gate) * learned_delta

        return delta_cp_final, gate


# 折叠惩罚（双层：控制点 + 密集场）

class _FoldingPenaltyCP(nn.Module):
    """控制点层面折叠惩罚（有限差分近似雅可比）"""

    def __init__(self, grid_size: int = 10, epsilon: float = 0.01):
        super().__init__()
        self.grid_size = grid_size
        self.epsilon   = epsilon

    def forward(self, delta_cp: torch.Tensor) -> torch.Tensor:
        """delta_cp: (B, 2, gs, gs) → fold_loss: scalar"""
        gs = self.grid_size
        y_cp = torch.linspace(-1., 1., gs, device=delta_cp.device, dtype=delta_cp.dtype)
        x_cp = torch.linspace(-1., 1., gs, device=delta_cp.device, dtype=delta_cp.dtype)
        gy, gx = torch.meshgrid(y_cp, x_cp, indexing="ij")
        src = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1,2,gs,gs)

        tgt = src + delta_cp          # (B,2,gs,gs)
        tx, ty = tgt[:, 0], tgt[:, 1]

        # 有限差分（取公共 gs-1 × gs-1 区域）
        dxdu = (tx[:, :, 1:] - tx[:, :, :-1])[:, :-1, :]   # (B,gs-1,gs-1)
        dydv = (ty[:, 1:, :] - ty[:, :-1, :])[:, :, :-1]   # (B,gs-1,gs-1)
        dxdv = (tx[:, 1:, :] - tx[:, :-1, :])[:, :, :-1]   # (B,gs-1,gs-1)
        dydu = (ty[:, :, 1:] - ty[:, :, :-1])[:, :-1, :]   # (B,gs-1,gs-1)

        det_J = dxdu * dydv - dxdv * dydu
        return F.relu(-det_J + self.epsilon).mean()


class FoldingPenaltyDense(nn.Module):
    """
    密集形变场层面的折叠检测与惩罚。

    优势：能检测TPS插值引入的"控制点之间的折叠"，
    比控制点层面更准确但计算量更大。
    通过 stride 下采样控制计算量。

    注意：此处使用归一化雅可比（除以像素间距）以实现尺度不变性。
    """

    def __init__(
        self,
        stride: int = 4,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.stride  = stride
        self.epsilon = epsilon

    def forward(
        self, dense_field: torch.Tensor,  # (B, H, W, 2)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            fold_loss: scalar
            fold_mask: (B, Hs-1, Ws-1) bool，True=折叠（用于诊断/可视化）
        """
        s     = self.stride
        field = dense_field.permute(0, 3, 1, 2)  # (B,2,H,W)

        # stride 下采样
        field_s = field[:, :, ::s, ::s]           # (B,2,Hs,Ws)
        Hs, Ws  = field_s.shape[2:]

        # 归一化坐标系下的像素间距
        du = 2.0 / max(Ws - 1, 1)
        dv = 2.0 / max(Hs - 1, 1)

        # 归一化有限差分（雅可比元素）
        dx_du = (field_s[:, 0, :, 1:] - field_s[:, 0, :, :-1]) / du   # (B,Hs,Ws-1)
        dy_du = (field_s[:, 1, :, 1:] - field_s[:, 1, :, :-1]) / du
        dx_dv = (field_s[:, 0, 1:, :] - field_s[:, 0, :-1, :]) / dv   # (B,Hs-1,Ws)
        dy_dv = (field_s[:, 1, 1:, :] - field_s[:, 1, :-1, :]) / dv

        # 取公共区域 (Hs-1, Ws-1)
        det_J = dx_du[:, :-1, :] * dy_dv[:, :, :-1] \
              - dx_dv[:, :, :-1] * dy_du[:, :-1, :]   # (B,Hs-1,Ws-1)

        fold_loss = F.relu(-det_J + self.epsilon).mean()
        fold_mask = (det_J < 0).detach()

        return fold_loss, fold_mask

    @staticmethod
    def compute_fold_ratio(fold_mask: torch.Tensor) -> float:
        return fold_mask.float().mean().item()


class CombinedFoldingPenalty(nn.Module):
    """
    组合折叠惩罚：控制点层面 + 密集场层面。
    支持外部调度 lambda_cp / lambda_dense（训练脚本按epoch调整）。
    """

    def __init__(
        self,
        grid_size: int = 10,
        dense_stride: int = 4,
        epsilon: float = 0.01,
        lambda_cp: float = 1.0,
        lambda_dense: float = 0.5,
    ):
        super().__init__()
        self.lambda_cp    = lambda_cp
        self.lambda_dense = lambda_dense
        self.cp_penalty   = _FoldingPenaltyCP(grid_size=grid_size, epsilon=epsilon)
        self.dense_penalty = FoldingPenaltyDense(stride=dense_stride, epsilon=epsilon)

    def forward(
        self,
        delta_cp: torch.Tensor,                       # (B,2,gs,gs)
        dense_field: Optional[torch.Tensor] = None,   # (B,H,W,2)
    ) -> Dict[str, torch.Tensor]:

        loss_cp = self.cp_penalty(delta_cp)

        if dense_field is not None:
            loss_dense, fold_mask = self.dense_penalty(dense_field)
            fold_ratio = FoldingPenaltyDense.compute_fold_ratio(fold_mask)
        else:
            loss_dense = torch.zeros(1, device=delta_cp.device, dtype=delta_cp.dtype).squeeze()
            fold_mask  = None
            fold_ratio = 0.0

        total = self.lambda_cp * loss_cp + self.lambda_dense * loss_dense

        return {
            "fold_loss":  total,
            "loss_cp":    loss_cp.detach().item(),
            "loss_dense": loss_dense.detach().item(),
            "fold_ratio": fold_ratio,
            "fold_mask":  fold_mask,
        }


class BypassTPSEstimator(nn.Module):
    """
    Bypass Global Homography TPS 估计器。

    流程：
    1. ConfidenceWeightedVotingDynamic → 物理投票初始解（无参数）
    2. 构建多信号输入特征（flow + conf + coverage + feat_consistency
                          [+ anti_entropy] [+ cf_consistency]）
    3. FlowAggregator（GATED门控残差修正）→ 最终控制点偏移
    4. CombinedFoldingPenalty（训练时约束，推理时跳过密集计算）
    """

    def __init__(
        self,
        grid_size: int = 10,
        feat_channels: int = 128,
        hidden_ch: int = 48,
        flow_map_size: int = 64,    # 期望的flow分辨率（仅用于文档，实际动态）
        sigma_scale: float = 0.7,
        delta_scale: float = 0.2,
        use_entropy: bool = True,
        use_cf_consistency: bool = True,
    ):
        super().__init__()
        self.grid_size         = grid_size
        self.use_entropy       = use_entropy
        self.use_cf_consistency = use_cf_consistency
        self.flow_map_size     = flow_map_size
        
        # ── 辅助信号提取 ──
        if use_entropy:
            self.entropy_extractor = MatchingEntropyMap(
                grid_size=32, output_size=flow_map_size
            )
        if use_cf_consistency:
            self.cf_consistency_module = CoarseFineConsistency()

        # ── 静态确定输入通道数 ──
        # base: flow(2) + conf(1) + coverage(1) + feat_consistency(1) = 5
        # 可选: anti_entropy(1) + cf_consistency(1)
        base_ch  = 5
        extra_ch = int(use_entropy) + int(use_cf_consistency)
        total_in_ch = base_ch + extra_ch   # 5, 6, 或 7（固定，不受运行时条件影响）

        # ── 物理投票（无参数）──
        self.voter = ConfidenceWeightedVotingDynamic(
            grid_size=grid_size,
            sigma_scale=sigma_scale,
        )

        # ── FlowAggregator（唯一实例，GATED模式）──
        self.aggregator = FlowAggregator(
            grid_size=grid_size,
            feat_channels=feat_channels,
            hidden_ch=hidden_ch,
            in_ch=total_in_ch,
            delta_scale=delta_scale,
        )

        # ── 折叠惩罚 ──
        self.folding = CombinedFoldingPenalty(grid_size=grid_size)


    def forward(
        self,
        warp_AB: torch.Tensor,                          # (B,Hf,Wf,2)  fine warp
        confidence: torch.Tensor,                       # (B,Hf,Wf)
        feat_A: torch.Tensor,                           # (B,C,Hf,Wf)
        feat_B: torch.Tensor,                           # (B,C,Hf,Wf)
        sim_matrix: Optional[torch.Tensor] = None,      # (B,N,N) coarse
        warp_AB_coarse: Optional[torch.Tensor] = None,  # (B,Hc,Wc,2)
        dense_field: Optional[torch.Tensor] = None,     # (B,H,W,2) TPS输出，用于密集折叠
        coarse_hw: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:

        B, H, W, _ = warp_AB.shape
        device, dtype = warp_AB.device, warp_AB.dtype

        # 物理投票初始解
        delta_cp_init, coverage = self.voter(warp_AB, confidence)
        # delta_cp_init: (B,2,gs,gs)  coverage: (B,gs,gs)

        # 构建输入特征
        # flow 向量
        y_f = torch.linspace(-1.+1./H, 1.-1./H, H, device=device, dtype=dtype)
        x_f = torch.linspace(-1.+1./W, 1.-1./W, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(y_f, x_f, indexing="ij")
        src      = torch.stack([gx, gy], dim=-1).unsqueeze(0)         # (1,H,W,2)
        flow_chw = (warp_AB - src).permute(0, 3, 1, 2) # (B, 2, H, W)

        # 特征一致性（feat_A & feat_B → 1通道）
        feat_consistency = self.aggregator.consistency_head(
            torch.cat([feat_A, feat_B], dim=1)
        )  # (B,1,H,W)

        # 覆盖度图上采样到flow分辨率
        cov_up = F.interpolate(
            coverage.unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        )  # (B,1,H,W)

        # 组装基础5通道
        channels = [
            flow_chw,                    # (B,2,H,W)
            confidence.unsqueeze(1),     # (B,1,H,W)
            cov_up,                      # (B,1,H,W)
            feat_consistency,            # (B,1,H,W)
        ]

        anti_entropy_map: Optional[torch.Tensor] = None
        cf_cons_map: Optional[torch.Tensor]      = None

        if self.use_entropy and sim_matrix is not None:
            if coarse_hw is not None:
                Hc, Wc = coarse_hw
                # 传入 Hc, Wc
                ent_map = self.entropy_extractor(sim_matrix, Hc, Wc)
            else:
                # 备选逻辑：如果没传，再尝试开方（为了兼容旧调用）
                gs = int(math.sqrt(sim_matrix.shape[1]))
                ent_map = self.entropy_extractor(sim_matrix, gs, gs)
            
            # 统一缩放到当前 Fine 分辨率 (H, W)
            anti_entropy_map = F.interpolate(1.0 - ent_map, size=(H, W), mode="bilinear")
            channels.append(anti_entropy_map)

        if self.use_cf_consistency:
            if warp_AB_coarse is not None:
                cf_map = self.cf_consistency_module(warp_AB_coarse, warp_AB)
                cf_cons_map = F.interpolate(
                    cf_map, size=(H, W), 
                    mode="bilinear", align_corners=False
                )
            else:
                # coarse warp 缺失时用全1填充
                cf_cons_map = torch.ones(B, 1, H, W, device=device, dtype=dtype)
            channels.append(cf_cons_map)

        feat_in = torch.cat(channels, dim=1)  # (B, total_in_ch, H, W)

        # FlowAggregator
        delta_cp, gate = self.aggregator(feat_in, delta_cp_init, feat_A, feat_B)

        if self.training:
            fold_out = self.folding(delta_cp, dense_field)
        else:
            fold_out = {
                "fold_loss":  torch.zeros(1, device=device, dtype=dtype).squeeze(),
                "loss_cp":    0.0,
                "loss_dense": 0.0,
                "fold_ratio": 0.0,
                "fold_mask":  None,
            }

        return {
            # 主输出
            "delta_cp":       delta_cp,        # (B,2,gs,gs) → 传给TPSGridGenerator
            "delta_cp_init":  delta_cp_init,   # (B,2,gs,gs) → 监督/可视化
            "coverage":       coverage,        # (B,gs,gs)   → 监督/可视化
            "gate":           gate,            # (B,1,gs,gs) → 可视化/分析
            # 折叠信息（训练时有效）
            "fold_loss":      fold_out["fold_loss"],   # scalar Tensor
            "fold_ratio":     fold_out["fold_ratio"],  # float，诊断用
            "fold_mask":      fold_out["fold_mask"],   # bool tensor 或 None
            # 辅助信号（可视化用）
            "anti_entropy":   anti_entropy_map,
            "cf_consistency": cf_cons_map,
        }

# dense_match/flow_to_tps.py

class TPSGridGenerator(nn.Module):
    def __init__(self, grid_size: int, **kwargs): # 移除固定 H, W 传参
        super().__init__()
        self.gs = grid_size
        
        # 1. 预计算源控制点 (保持不变)
        y, x = torch.meshgrid(torch.linspace(-1, 1, grid_size), 
                              torch.linspace(-1, 1, grid_size), indexing='ij')
        p_src = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
        self.register_buffer("p_src", p_src)

        # 2. 预计算 L 逆矩阵 (只跟控制点数量有关，保持不变)
        dist_cp = torch.cdist(p_src, p_src)
        K = self._tps_kernel(dist_cp)
        P = torch.cat([torch.ones(grid_size**2, 1), p_src], dim=1)
        L = torch.zeros(grid_size**2 + 3, grid_size**2 + 3)
        L[:grid_size**2, :grid_size**2] = K
        L[:grid_size**2, grid_size**2:] = P
        L[grid_size**2:, :grid_size**2] = P.t()
        self.register_buffer("L_inv", torch.inverse(L + torch.eye(L.shape[0]) * 1e-6))

    def _tps_kernel(self, r):
        r_sq = r**2
        return r_sq * torch.log(r_sq + 1e-8)

    def forward(self, delta_cp: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        target_shape: (H, W) 当前 batch 图片的实际高度和宽度
        """
        B = delta_cp.shape[0]
        H, W = target_shape # 获取动态尺寸
        device, dtype = delta_cp.device, delta_cp.dtype
        
        # 强制在 FP32 下计算核心矩阵运算防止 NaN
        with torch.cuda.amp.autocast(enabled=False):
            # 1. 计算 coeffs (保持不变)
            p_delta = delta_cp.permute(0, 2, 3, 1).reshape(B, -1, 2).float()
            p_tgt_cp = self.p_src.unsqueeze(0) + p_delta
            Y = torch.cat([p_tgt_cp, torch.zeros(B, 3, 2, device=device)], dim=1)
            coeffs = torch.matmul(self.L_inv, Y)

            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing='ij'
            )
            p_tgt_pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1) # (N_pix, 2)

            # 3. 计算距离矩阵 (N_pix, N_cp)
            dist_pix_cp = torch.cdist(p_tgt_pix, self.p_src)
            K_pix = self._tps_kernel(dist_pix_cp)
            
            # 4. 映射坐标
            P_pix = torch.cat([torch.ones(H * W, 1, device=device, dtype=dtype), p_tgt_pix], dim=1)
            grid = torch.matmul(K_pix, coeffs[:, :self.gs**2, :]) + \
                   torch.matmul(P_pix, coeffs[:, self.gs**2:, :])
               
        return grid.reshape(B, H, W, 2).to(dtype)