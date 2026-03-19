import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING
import platform
import pathlib

# 修复跨平台反序列化问题 (Linux -> Windows)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from dense_match.network import AgriMatcher
# REPO_ROOT = Path(__file__).resolve().parent


# if TYPE_CHECKING:  # pragma: no cover
   

# 辅助函数：图像加载与预处理
def load_rgb_tensor(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    
    # 显式计算等比例缩放后的尺寸，强制最长边等于 size
    w, h = img.size
    if w >= h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_h = size
        new_w = int(w * (size / h))
        
    # 传入 [new_h, new_w] 明确指定高和宽，彻底绕过 max_size 冲突
    img = TF.resize(img, [new_h, new_w], antialias=True)
    ten = TF.to_tensor(img)  # [C, H, W]
    
    # 补黑边 (Padding) 至严格的 size x size 正方形
    _, curr_h, curr_w = ten.shape
    pad_bottom = size - curr_h
    pad_right = size - curr_w
    
    # F.pad 的参数顺序是 (左, 右, 上, 下)
    ten = F.pad(ten, (0, pad_right, 0, pad_bottom), value=0.0)
    
    return ten

def visualize_stitching_result(img_A, img_B, warp_AB, confidence_AB=None, save_path='stitch_res.jpg', conf_thresh=0.5):
    """
    计算单应性矩阵，构建全景画布，并将两张图像拼接融合。
    """
    # 1. 张量转换为 NumPy 数组 (H, W, C)
    img_A_np = img_A[0].permute(1, 2, 0).cpu().detach().numpy()
    img_B_np = img_B[0].permute(1, 2, 0).cpu().detach().numpy()
    warp_np = warp_AB[0].cpu().detach().numpy()
    
    H_img, W_img, _ = img_A_np.shape
    gs = warp_AB.shape[1]

    if confidence_AB is not None:
        conf_np = confidence_AB[0].cpu().detach().numpy()
    else:
        conf_np = np.ones((gs, gs))

    # 2. 提取高置信度匹配点
    y_idx = np.linspace(-1, 1, gs)
    x_idx = np.linspace(-1, 1, gs)
    grid_x, grid_y = np.meshgrid(x_idx, y_idx)

    mask = conf_np > conf_thresh
    pts_A_norm = np.stack([grid_x[mask], grid_y[mask]], axis=-1)
    pts_B_norm = warp_np[mask]

    if len(pts_A_norm) < 4:
        print("有效匹配点不足，无法执行拼接。")
        return

    # 3. 坐标映射至像素空间
    pts_A_px = (pts_A_norm + 1.0) / 2.0 * np.array([W_img - 1, H_img - 1])
    pts_B_px = (pts_B_norm + 1.0) / 2.0 * np.array([W_img - 1, H_img - 1])
    
    pts_A_px = pts_A_px.astype(np.float32)
    pts_B_px = pts_B_px.astype(np.float32)

    # 4. 计算单应性矩阵 (由 B 映射至 A)
    H_matrix, inliers = cv2.findHomography(pts_B_px, pts_A_px, cv2.USAC_MAGSAC, 3.0)

    if H_matrix is None:
        print("单应性矩阵计算失败。")
        return

    # 5. 计算拼接后的大画布尺寸
    # 获取图像 B 的四个角点
    corners_B = np.float32([[0, 0], [0, H_img], [W_img, H_img], [W_img, 0]]).reshape(-1, 1, 2)
    # 将图像 B 的角点投影到图像 A 的坐标系中
    warped_corners_B = cv2.perspectiveTransform(corners_B, H_matrix)
    
    # 获取图像 A 的角点
    corners_A = np.float32([[0, 0], [0, H_img], [W_img, H_img], [W_img, 0]]).reshape(-1, 1, 2)
    
    # 合并所有角点以找到全局边界
    all_corners = np.concatenate((corners_A, warped_corners_B), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # 6. 计算平移矩阵，避免图像投影后出现负坐标导致被裁剪
    translation_x = -x_min
    translation_y = -y_min
    T_matrix = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    # 7. 图像投影与融合
    # 投影图像 B
    canvas_B = cv2.warpPerspective(img_B_np, T_matrix @ H_matrix, (canvas_w, canvas_h))
    # 投影图像 A (仅进行平移)
    canvas_A = cv2.warpPerspective(img_A_np, T_matrix, (canvas_w, canvas_h))

    # 创建单通道掩码 (判断哪些像素有内容，避免黑色背景干扰)
    mask_A = (np.sum(canvas_A, axis=2) > 0).astype(np.float32)[..., np.newaxis]
    mask_B = (np.sum(canvas_B, axis=2) > 0).astype(np.float32)[..., np.newaxis]

    # 计算重叠区域掩码
    overlap_mask = mask_A * mask_B

    # 设置透明度参数 (0.5 表示在重叠区 A 和 B 各占 50% 的权重)
    alpha = 0.5

    # 融合逻辑切分
    # 1. 仅存在图 A 的非重叠区域
    only_A_area = canvas_A * mask_A * (1 - overlap_mask)
    # 2. 仅存在图 B 的非重叠区域
    only_B_area = canvas_B * mask_B * (1 - overlap_mask)
    # 3. 重叠区域进行透明度加权融合
    blended_overlap = (canvas_A * alpha + canvas_B * (1 - alpha)) * overlap_mask

    # 组合为最终拼接图
    stitched_img = only_A_area + only_B_area + blended_overlap
    stitched_img = np.clip(stitched_img, 0, 1)

    # 8. 绘制结果
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Image A (Target)")
    plt.imshow(img_A_np)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Image B (Source)")
    plt.imshow(img_B_np)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Stitched Panorama")
    plt.imshow(stitched_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"拼接可视化已完成，结果保存至 {save_path}")

def main():
    parser = argparse.ArgumentParser(description="AgriMatcher Inference Script")
    parser.add_argument("--img_a", type=str, required=True, help="Path to the first image (Target)")
    parser.add_argument("--img_b", type=str, required=True, help="Path to the second image (Source)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the saved checkpoint (e.g., best.pt)")
    parser.add_argument("--image_size", type=int, default=256, help="Image size used during training")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--teacher_dim", type=int, default=80, help="Teacher feature dimension (must match training)")
    parser.add_argument("--out", type=str, default="visres.jpg", help="Output visualization path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 计算 grid_size 并初始化模型
    grid_size = args.image_size // 8
    print(f"Initializing AgriMatcher (image_size={args.image_size}, grid_size={grid_size}, teacher_dim={args.teacher_dim})...")
    
    student = AgriMatcher(
        d_model=args.d_model, 
        teacher_dim=args.teacher_dim, 
        grid_size=grid_size
    ).to(device)

    # 2. 加载权重
    print(f"Loading checkpoint from {args.ckpt}...")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt}")
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 使用 strict=False 加载，以防结构有微调
    missing_keys, unexpected_keys = student.load_state_dict(checkpoint["student"], strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
        
    student.eval()

    # 3. 加载并预处理图像
    print(f"Processing images: A={args.img_a}, B={args.img_b}...")
    img_a_tensor = load_rgb_tensor(Path(args.img_a), args.image_size).unsqueeze(0).to(device)
    img_b_tensor = load_rgb_tensor(Path(args.img_b), args.image_size).unsqueeze(0).to(device)

    # 4. 前向推理
    print("Running inference...")
    with torch.no_grad():
        # 根据训练脚本，使用 AMP 进行半精度推理
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output = student(img_a_tensor, img_b_tensor)
            
            warp_AB = output['warp_AB'].float() # 转回 float32 供 grid_sample 使用
            
            # 提取置信度，处理可能存在的 Logits 情况
            if 'confidence_AB' in output:
                confidence_AB = output['confidence_AB'].float()
            elif 'conf_logits' in output:
                # 如果网络输出的是 logits，则需要应用 sigmoid
                confidence_AB = torch.sigmoid(output['conf_logits']).float()
            else:
                confidence_AB = None

    # 5. 可视化结果
    print("Generating stitching visualization...")
    visualize_stitching_result(
        img_a_tensor, 
        img_b_tensor, 
        warp_AB, 
        confidence_AB, 
        save_path=args.out,
        # conf_thresh=args.conf_thresh
        conf_thresh=0.5
    )
    print("Done.")

if __name__ == "__main__":
    main()