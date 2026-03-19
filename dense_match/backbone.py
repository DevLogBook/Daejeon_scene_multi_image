import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file


def generate_2d_sincos_pos_emb(embed_dim, grid_size):
    """
    生成类似 ViT 的二维正弦余弦位置编码
    embed_dim: 特征维度 (如 128)
    grid_size: 特征图的高宽 (如 32)
    返回: [1, grid_size*grid_size, embed_dim]
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0)  # [2, grid_size, grid_size]

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (10000 ** omega)

    out_w = torch.einsum('m,d->md', [grid[0].flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid[1].flatten(), omega])

    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w),
                         torch.sin(out_h), torch.cos(out_h)], dim=1)
    return pos_emb.unsqueeze(0)  # [1, 1024, 128]



class MobileViTBackbone(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()

        # features_only=True 允许我们提取中间层的特征图
        # out_indices=(2,) 表示我们只提取下采样 8 倍 (Stride=8) 的那一层特征
        local_weights_path = "backbones/model.safetensors"
        self.mobilevit = timm.create_model(
            'mobilevitv2_050',
            pretrained=False, 
            features_only=True,
            out_indices=(1, 2)
        )
    
        state_dict = load_file(local_weights_path)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("stages.", "stages_")
            new_state_dict[new_key] = v

        msg = self.mobilevit.load_state_dict(new_state_dict, strict=False)
        # Avoid UnicodeEncodeError on Windows consoles using GBK codepage.
        print("Backbone MobileViT load success!")

        # mobilevitv2_050 在 out_indices=2 时的通道数通常是 128 或 144
        # 我们用一个 dummy input 测试一下来获取精确通道数
        dummy_input = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            features = self.mobilevit(dummy_input)
            timm_ch_fine = features[0].shape[1]    # Stride=4 层的通道数
            timm_ch_coarse = features[1].shape[1]  # Stride=8 层的通道数

        # 为 Coarse 特征 (32x32) 设置维度对齐
        self.dim_align_coarse = nn.Sequential(
            nn.Conv2d(timm_ch_coarse, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 为 Fine 特征 (64x64) 设置维度对齐
        self.dim_align_fine = nn.Sequential(
            nn.Conv2d(timm_ch_fine, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
            # x shape: [B, 3, 256, 256]
            features = self.mobilevit(x)
            
            feat_fine_raw = features[0]    # [B, C1, 64, 64]
            feat_coarse_raw = features[1]  # [B, C2, 32, 32]

            feat_fine = self.dim_align_fine(feat_fine_raw)       # [B, 128, 64, 64]
            feat_coarse = self.dim_align_coarse(feat_coarse_raw) # [B, 128, 32, 32]

            return feat_coarse, feat_fine