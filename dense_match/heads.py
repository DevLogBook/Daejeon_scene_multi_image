import torch
import torch.nn as nn

from dense_match.refine import make_grid


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
                nn.GroupNorm(num_groups=min(32, out // 4), num_channels=out),
                nn.ReLU(inplace=True),
            ]
            prev = out
        self.local_encoder = nn.Sequential(*layers)

        # 全局上下文融合后的输出头
        # 输入：local(hidden_dim) + global(hidden_dim) → 权重
        self.output_head = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, hidden_dim // 4), num_channels=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, (hidden_dim // 2) // 4), num_channels=hidden_dim // 2),
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


class GlobalSimilarityHead(nn.Module):
    def __init__(self, feat_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 4 + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
            self,
            feat_A: torch.Tensor,
            feat_B: torch.Tensor,
            warp_AB: torch.Tensor,
            confidence: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = feat_A.shape
        conf = torch.nan_to_num(confidence.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        conf_chw = conf.unsqueeze(1)
        denom = conf_chw.sum(dim=(2, 3)).clamp_min(1e-6)

        feat_A_f = torch.nan_to_num(feat_A.float(), nan=0.0, posinf=0.0, neginf=0.0)
        feat_B_f = torch.nan_to_num(feat_B.float(), nan=0.0, posinf=0.0, neginf=0.0)
        fa_mean = (feat_A_f * conf_chw).sum(dim=(2, 3)) / denom
        fb_mean = (feat_B_f * conf_chw).sum(dim=(2, 3)) / denom
        fa_max = feat_A_f.amax(dim=(2, 3))
        fb_max = feat_B_f.amax(dim=(2, 3))

        src_grid = make_grid(B, H, W, warp_AB.device, torch.float32)
        flow = torch.nan_to_num(warp_AB.float(), nan=0.0, posinf=1.5, neginf=-1.5).clamp(-1.5, 1.5) - src_grid
        flow_chw = flow.permute(0, 3, 1, 2)
        flow_mean = (flow_chw * conf_chw).sum(dim=(2, 3)) / denom
        conf_stats = torch.stack([
            conf.mean(dim=(1, 2)),
            conf.amax(dim=(1, 2)),
        ], dim=1)

        raw = self.net(torch.cat([fa_mean, fb_mean, fa_max, fb_max, flow_mean, conf_stats], dim=1))
        tx = raw[:, 0].tanh() * 0.75
        ty = raw[:, 1].tanh() * 0.75
        log_s = raw[:, 2].tanh() * 0.20
        theta = raw[:, 3].tanh() * 0.35

        s = torch.exp(log_s)
        c = torch.cos(theta)
        r = torch.sin(theta)
        z = torch.zeros_like(tx)
        o = torch.ones_like(tx)
        return torch.stack([
            s * c, -s * r, tx,
            s * r,  s * c, ty,
            z,      z,     o,
        ], dim=1).reshape(B, 3, 3).float()
