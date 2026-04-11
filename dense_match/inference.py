import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import platform
import pathlib

# 修复跨平台反序列化问题 (Linux -> Windows)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

REPO_ROOT = Path(__file__).resolve().parent

project_root = REPO_ROOT.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from dense_match.network import AgriTPSStitcher

def torch_grid_to_cv2_map(grid_tensor: torch.Tensor, out_h: int, out_w: int) -> tuple:
    """
    将 PyTorch [-1, 1] (align_corners=False) 的 normalized grid 转换为 OpenCV 的绝对像素坐标
    """
    grid = grid_tensor.squeeze(0).cpu().numpy()  # [H, W, 2]

    if grid.shape[0] != out_h or grid.shape[1] != out_w:
        grid = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # 严格遵循 align_corners=False 的数学逆映射
    map_x = ((grid[..., 0] + 1.0) / 2.0) * out_w - 0.5
    map_y = ((grid[..., 1] + 1.0) / 2.0) * out_h - 0.5

    return map_x.astype(np.float32), map_y.astype(np.float32)


def generate_voronoi_seam_masks(mask_a: np.ndarray, mask_b: np.ndarray):
    """利用距离变换生成最佳接缝 (Voronoi Seam)"""
    dist_a = cv2.distanceTransform(mask_a, cv2.DIST_L2, 5)
    dist_b = cv2.distanceTransform(mask_b, cv2.DIST_L2, 5)

    blend_mask_a = (dist_a > dist_b).astype(np.uint8) * 255
    blend_mask_b = (dist_b >= dist_a).astype(np.uint8) * 255

    blend_mask_a = cv2.bitwise_and(blend_mask_a, mask_a)
    blend_mask_b = cv2.bitwise_and(blend_mask_b, mask_b)

    return blend_mask_a, blend_mask_b


def multi_band_blending(img_a: np.ndarray, img_b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
                        num_bands: int = 5):
    """多频段无缝融合"""
    H, W = img_a.shape[:2]
    bounding_rect = (0, 0, W, H)

    img_a_16s = img_a.astype(np.int16)
    img_b_16s = img_b.astype(np.int16)

    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(num_bands)
    blender.prepare(bounding_rect)

    blender.feed(img_a_16s, mask_a, (0, 0))
    blender.feed(img_b_16s, mask_b, (0, 0))

    dst = np.zeros((H, W, img_a.shape[2]), dtype=np.int16)
    dst_mask = np.zeros((H, W), dtype=np.uint8)
    result, result_mask = blender.blend(dst, dst_mask)

    result_8u = np.clip(result, 0, 255).astype(np.uint8)
    return result_8u


@torch.no_grad()
def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 初始化全景推理引擎 ({device})...")

    # 1. 实例化与加载权重
    student = AgriTPSStitcher(
        matcher_config={'d_model': 128, 'teacher_dim': 100, 'grid_size': 32},
        tps_config={'grid_size': 8, 'feat_channels': 128}
    ).to(device)
    student.eval()

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    student.load_state_dict(checkpoint['student'], strict=True)

    # 2. 读取原图
    img_a_orig = cv2.imread(str(args.img_a))
    img_b_orig = cv2.imread(str(args.img_b))
    H_a, W_a = img_a_orig.shape[:2]
    H_b, W_b = img_b_orig.shape[:2]

    # 3. 网络推理获取网格和全局 H
    net_h, net_w = 512, 512
    img_a_net = cv2.resize(img_a_orig, (net_w, net_h))
    img_b_net = cv2.resize(img_b_orig, (net_w, net_h))

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    tensor_a = torch.from_numpy((cv2.cvtColor(img_a_net, cv2.COLOR_BGR2RGB) / 255.0 - mean) / std).permute(2, 0,
                                                                                                           1).float().unsqueeze(
        0).to(device)
    tensor_b = torch.from_numpy((cv2.cvtColor(img_b_net, cv2.COLOR_BGR2RGB) / 255.0 - mean) / std).permute(2, 0,
                                                                                                           1).float().unsqueeze(
        0).to(device)

    start_time = time.time()
    out = student(tensor_a, tensor_b)
    dense_grid = out['dense_grid']
    H_mat = out['H_mat'][0].cpu().numpy()  # 网络输出的是 [-1,1] 归一化空间的 H
    print(f"⚡ 推理耗时: {(time.time() - start_time) * 1000:.2f} ms")

    # =========================================================================
    # 🌟 核心：计算真正的全景画布尺寸 (严格按照归一化空间映射)
    # =========================================================================
    H_inv = np.linalg.inv(H_mat)

    # 获取图 B 的 4 个像素角点，并转到 [-1, 1] 空间
    corners_b_pix = np.array([[0, 0], [W_b, 0], [W_b, H_b], [0, H_b]], dtype=np.float32)
    corners_b_norm = np.copy(corners_b_pix)
    corners_b_norm[:, 0] = (corners_b_norm[:, 0] + 0.5) * (2.0 / W_b) - 1.0
    corners_b_norm[:, 1] = (corners_b_norm[:, 1] + 0.5) * (2.0 / H_b) - 1.0

    # 投影到 A 的 [-1, 1] 空间
    pts_b = np.vstack([corners_b_norm.T, np.ones(4)])
    pts_a_norm = H_inv @ pts_b
    pts_a_norm = pts_a_norm[:2, :] / pts_a_norm[2, :]

    # 转回 A 的绝对像素空间
    corners_a_pix = pts_a_norm.T
    corners_a_pix[:, 0] = ((corners_a_pix[:, 0] + 1.0) / 2.0) * W_a - 0.5
    corners_a_pix[:, 1] = ((corners_a_pix[:, 1] + 1.0) / 2.0) * H_a - 0.5

    # 计算包含 A 和 B 的全景 Bounding Box
    all_x = np.concatenate([corners_a_pix[:, 0], [0, W_a]])
    all_y = np.concatenate([corners_a_pix[:, 1], [0, H_a]])
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)

    W_pano = int(np.ceil(max_x - min_x))
    H_pano = int(np.ceil(max_y - min_y))
    tx, ty = int(round(-min_x)), int(round(-min_y))
    print(f"🌍 动态扩充全景画布: {W_pano}x{H_pano}, 偏移基准: ({tx}, {ty})")

    # =========================================================================
    # 🌟 核心：生成 宏观(Homography) + 微观(TPS) 混合渲染网格
    # =========================================================================
    # 1. 铺设全景底网 (利用全局单应性)
    grid_y, grid_x = np.mgrid[0:H_pano, 0:W_pano].astype(np.float32)
    # 转为 A 像素坐标 -> A 归一化坐标
    nx_a = (grid_x - tx + 0.5) * (2.0 / W_a) - 1.0
    ny_a = (grid_y - ty + 0.5) * (2.0 / H_a) - 1.0

    # 映射到 B 归一化坐标 -> B 像素坐标
    pts_a_flat = np.stack([nx_a.ravel(), ny_a.ravel(), np.ones_like(nx_a.ravel())])
    pts_b_norm = H_mat @ pts_a_flat
    nx_b = (pts_b_norm[0] / pts_b_norm[2]).reshape(H_pano, W_pano)
    ny_b = (pts_b_norm[1] / pts_b_norm[2]).reshape(H_pano, W_pano)

    map_x_pano = ((nx_b + 1.0) / 2.0) * W_b - 0.5
    map_y_pano = ((ny_b + 1.0) / 2.0) * H_b - 0.5
    map_x_pano, map_y_pano = map_x_pano.astype(np.float32), map_y_pano.astype(np.float32)

    # 2. 局部高精度覆写：将含有水波纹的 TPS 形变嵌入重叠区
    map_x_tps, map_y_tps = torch_grid_to_cv2_map(dense_grid, H_a, W_a)
    map_x_pano[ty:ty + H_a, tx:tx + W_a] = map_x_tps
    map_y_pano[ty:ty + H_a, tx:tx + W_a] = map_y_tps

    # =========================================================================
    # 🌟 最终渲染与融合
    # =========================================================================
    warped_b_pano = cv2.remap(img_b_orig, map_x_pano, map_y_pano, interpolation=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT)

    img_a_pano = np.zeros((H_pano, W_pano, 3), dtype=np.uint8)
    img_a_pano[ty:ty + H_a, tx:tx + W_a] = img_a_orig

    # 生成精确 Mask
    mask_a_pano = np.zeros((H_pano, W_pano), dtype=np.uint8)
    mask_a_pano[ty:ty + H_a, tx:tx + W_a] = 255

    mask_b_orig = np.ones((H_b, W_b), dtype=np.uint8) * 255
    mask_b_pano = cv2.remap(mask_b_orig, map_x_pano, map_y_pano, interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT)

    print(f"✨ 执行多频段无缝融合...")
    blend_mask_a, blend_mask_b = generate_voronoi_seam_masks(mask_a_pano, mask_b_pano)
    final_panorama = multi_band_blending(img_a_pano, warped_b_pano, blend_mask_a, blend_mask_b, num_bands=5)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final_panorama)
    print(f"🎉 真正的全景拼接成功！已保存至: {out_path.absolute()}")

    # 依然保留 Overlap 用来发论文对比
    overlap = cv2.addWeighted(img_a_pano, 0.5, warped_b_pano, 0.5, 0)
    cv2.imwrite(str(out_path.with_name(f"{out_path.stem}_overlap{out_path.suffix}")), overlap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriTPSStitcher 真正的高清全景生成器")
    parser.add_argument("--img-a", type=str, required=True, help="基准图片")
    parser.add_argument("--img-b", type=str, required=True, help="待拼接图片")
    parser.add_argument("--ckpt", type=str, required=True, help="权重路径")
    parser.add_argument("--out", type=str, default="output/panorama.jpg", help="输出路径")
    args = parser.parse_args()
    inference(args)