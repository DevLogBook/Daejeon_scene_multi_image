import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file


class BiFPNLayer(nn.Module):
    """
    针对 3 个尺度 (Stride 4, 8, 16) 的轻量化 BiFPN 融合层
    采用加权双向融合，提升重复纹理下的特征判别力
    """

    def __init__(self, channels):
        super().__init__()
        self.epsilon = 1e-4

        self.conv4_up = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
        )
        self.conv8_up = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
        )
        self.conv8_down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
        )
        self.conv16_down = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.GELU()
        )

        self.w1 = nn.Parameter(torch.ones(2))  # p8, p16_up
        self.w2 = nn.Parameter(torch.ones(2))  # p4, p8_td_up
        self.w3 = nn.Parameter(torch.ones(3))  # p8, p8_td, p4_down (标准的 3 节点融合)
        self.w4 = nn.Parameter(torch.ones(2))  # p16, p8_down

    def forward(self, p4, p8, p16):
        # 使用 ReLU 保证权重为正（比 abs 计算略快），并加上 epsilon 避免除以 0
        w1 = F.relu(self.w1) + self.epsilon
        w2 = F.relu(self.w2) + self.epsilon
        w3 = F.relu(self.w3) + self.epsilon
        w4 = F.relu(self.w4) + self.epsilon

        # --- Top-Down Path ---
        p16_up = F.interpolate(p16, size=p8.shape[-2:], mode='bilinear', align_corners=False)
        p8_td = self.conv8_up((w1[0] * p8 + w1[1] * p16_up) / w1.sum())

        p8_td_up = F.interpolate(p8_td, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        p4_out = self.conv4_up((w2[0] * p4 + w2[1] * p8_td_up) / w2.sum())

        # --- Bottom-Up Path ---
        p4_down = F.max_pool2d(p4_out, kernel_size=3, stride=2, padding=1)
        # 补全了原始 p8 的融合，防止原始特征丢失
        p8_out = self.conv8_down((w3[0] * p8 + w3[1] * p8_td + w3[2] * p4_down) / w3.sum())

        p8_down = F.max_pool2d(p8_out, kernel_size=3, stride=2, padding=1)
        p16_out = self.conv16_down((w4[0] * p16 + w4[1] * p8_down) / w4.sum())

        return p4_out, p8_out, p16_out


class MobileViTBackbone(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()

        # 提取 Stride 4, 8, 16 三个层级的特征
        local_weights_path = "backbones/model.safetensors"
        self.mobilevit = timm.create_model(
            'mobilevitv2_050',
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3)  # (S4, S8, S16)
        )

        try:
            from safetensors.torch import load_file
            state_dict = load_file(local_weights_path)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("stages.", "stages_")
                new_state_dict[new_key] = v
            self.mobilevit.load_state_dict(new_state_dict, strict=False)
            print("Backbone MobileViT Stride16+BiFPN version loaded!")
        except Exception as e:
            print(f"[Warning] Loading weights failed: {e}")

        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.mobilevit(dummy_input)
            ch_s4 = features[0].shape[1]
            ch_s8 = features[1].shape[1]
            ch_s16 = features[2].shape[1]

        # 维度对齐层
        self.align_s4 = nn.Conv2d(ch_s4, out_channels, 1, bias=False)
        self.align_s8 = nn.Conv2d(ch_s8, out_channels, 1, bias=False)
        self.align_s16 = nn.Conv2d(ch_s16, out_channels, 1, bias=False)

        self.bifpn = BiFPNLayer(out_channels)

        # 最终输出校准
        self.out_fine = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.out_coarse = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        features = self.mobilevit(x)

        # 提取三级特征
        s4_raw = features[0]  # [B, ch_s4, H/4, W/4]
        s8_raw = features[1]  # [B, ch_s8, H/8, W/8]
        s16_raw = features[2]  # [B, ch_s16, H/16, W/16]

        # 初始维度对齐
        p4 = self.align_s4(s4_raw)
        p8 = self.align_s8(s8_raw)
        p16 = self.align_s16(s16_raw)

        # BiFPN 双向特征融合
        f4, f8, f16 = self.bifpn(p4, p8, p16)

        feat_fine = self.out_fine(f8)  # [B, 128, H/8, W/8]
        feat_coarse = self.out_coarse(f16)  # [B, 128, H/16, W/16]

        return feat_coarse, feat_fine