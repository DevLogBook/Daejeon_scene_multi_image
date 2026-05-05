import argparse
import sys
import time
from pathlib import Path
import torch
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
from dense_match.network import AgriStitcher


def build_model_from_checkpoint(checkpoint: dict) -> AgriStitcher:
    ckpt_args = checkpoint.get("args", {}) or {}
    d_model = int(ckpt_args.get("d_model", 128))
    teacher_setting = ckpt_args.get("teacher_setting", "precise")
    teacher_dim = int(ckpt_args.get("teacher_dim", {"base": 80, "precise": 100}.get(teacher_setting, 100)))
    teacher_grid_size = int(ckpt_args.get("teacher_grid_size", 32))

    decoder_config = {
        "feat_channels": d_model,
        "residual_mode": ckpt_args.get("residual_mode", "mesh"),
        "mesh_size": int(ckpt_args.get("mesh_size", 12)),
        "max_residual_px": float(ckpt_args.get("max_residual_px", 4.0)),
        "decoder_hidden": int(ckpt_args.get("decoder_hidden", 128)),
        "decoder_blocks": int(ckpt_args.get("decoder_blocks", 3)),
        "residual_scale": float(ckpt_args.get("residual_scale", 0.08)),
    }

    return AgriStitcher(
        matcher_config={
            "d_model": d_model,
            "teacher_dim": teacher_dim,
            "grid_size": teacher_grid_size,
        },
        decoder_config=decoder_config,
    )


def torch_grid_to_cv2_map(
        grid_tensor: torch.Tensor,
        dst_h: int,
        dst_w: int,
        src_h: int,
        src_w: int,
) -> tuple:
    """
    将 PyTorch [-1, 1] (align_corners=False) grid 转为 OpenCV remap 坐标。

    grid_tensor 的空间维度属于输出图 A；grid 数值本身是采样图 B 的归一化坐标。
    """
    grid = grid_tensor.squeeze(0).detach().cpu().numpy()  # [H, W, 2]

    if grid.shape[0] != dst_h or grid.shape[1] != dst_w:
        grid = cv2.resize(grid, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    map_x = ((grid[..., 0] + 1.0) / 2.0) * src_w - 0.5
    map_y = ((grid[..., 1] + 1.0) / 2.0) * src_h - 0.5

    return map_x.astype(np.float32), map_y.astype(np.float32)


def normalized_to_pixel_xy(points_norm: np.ndarray, height: int, width: int) -> np.ndarray:
    points_pix = np.empty_like(points_norm, dtype=np.float64)
    points_pix[..., 0] = ((points_norm[..., 0] + 1.0) / 2.0) * width - 0.5
    points_pix[..., 1] = ((points_norm[..., 1] + 1.0) / 2.0) * height - 0.5
    return points_pix


def project_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    projected = points_h @ H.T
    denom = projected[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-8, np.sign(denom) * 1e-8 + (denom == 0) * 1e-8, denom)
    return projected[:, :2] / denom


def compute_panorama_canvas(H_mat: np.ndarray, H_a: int, W_a: int, H_b: int, W_b: int):
    try:
        H_inv = np.linalg.inv(H_mat)
    except np.linalg.LinAlgError:
        H_inv = np.eye(3, dtype=np.float64)

    corners_b_norm = np.array([
        [-1.0 + 1.0 / W_b, -1.0 + 1.0 / H_b],
        [1.0 - 1.0 / W_b, -1.0 + 1.0 / H_b],
        [1.0 - 1.0 / W_b, 1.0 - 1.0 / H_b],
        [-1.0 + 1.0 / W_b, 1.0 - 1.0 / H_b],
    ], dtype=np.float64)

    corners_b_in_a_norm = project_points(H_inv, corners_b_norm)
    corners_b_in_a_pix = normalized_to_pixel_xy(corners_b_in_a_norm, H_a, W_a)

    all_x = np.concatenate([corners_b_in_a_pix[:, 0], np.array([0.0, float(W_a - 1)])])
    all_y = np.concatenate([corners_b_in_a_pix[:, 1], np.array([0.0, float(H_a - 1)])])
    min_x, max_x = float(np.min(all_x)), float(np.max(all_x))
    min_y, max_y = float(np.min(all_y)), float(np.max(all_y))

    left = int(np.floor(min_x))
    top = int(np.floor(min_y))
    right = int(np.ceil(max_x))
    bottom = int(np.ceil(max_y))
    pano_w = right - left + 1
    pano_h = bottom - top + 1
    tx = -left
    ty = -top
    return pano_h, pano_w, tx, ty


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

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    student = build_model_from_checkpoint(checkpoint).to(device)
    student.eval()

    state_key = "student_ema" if args.use_ema and "student_ema" in checkpoint else "student"
    student.load_state_dict(checkpoint[state_key], strict=True)
    print(f"✅ 已加载权重: {args.ckpt} ({state_key})")

    # 读取原图
    img_a_orig = cv2.imread(str(args.img_a))
    img_b_orig = cv2.imread(str(args.img_b))
    if img_a_orig is None:
        raise FileNotFoundError(f"无法读取基准图片: {args.img_a}")
    if img_b_orig is None:
        raise FileNotFoundError(f"无法读取待拼接图片: {args.img_b}")
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
    stitch_mask = out.get("stitch_mask")
    stitch_residual_flow = out.get("stitch_residual_flow")
    print(f"⚡ 推理耗时: {(time.time() - start_time) * 1000:.2f} ms")

    # =========================================================================
    # 🌟 核心：计算真正的全景画布尺寸
    # H_mat 是归一化坐标（[-1,1]）空间的单应矩阵，约定 dst_norm = H @ src_norm
    # =========================================================================
    H_pano, W_pano, tx, ty = compute_panorama_canvas(H_mat, H_a, W_a, H_b, W_b)
    pano_megapixels = (H_pano * W_pano) / 1_000_000.0
    if pano_megapixels > args.max_pano_megapixels:
        raise RuntimeError(
            f"全景画布过大: {W_pano}x{H_pano} ({pano_megapixels:.1f} MP)，"
            f"超过 --max-pano-megapixels={args.max_pano_megapixels}。"
        )
    print(f"🌍 动态扩充全景画布: {W_pano}x{H_pano}, 偏移基准: ({tx}, {ty})")

    # =========================================================================
    # 🌟 核心：生成 宏观(Homography) + 微观(Decoder residual) 混合渲染网格
    # =========================================================================
    # 1. 铺设全景底网（利用全局单应性 H，在归一化空间推导）
    #    全景像素 (gx, gy) → A 归一化坐标 → H 映射 → B 归一化坐标 → B 像素坐标
    grid_y, grid_x = np.mgrid[0:H_pano, 0:W_pano].astype(np.float32)

    # 全景像素 → A 归一化坐标（align_corners=False）
    nx_a = (grid_x - tx + 0.5) * (2.0 / W_a) - 1.0
    ny_a = (grid_y - ty + 0.5) * (2.0 / H_a) - 1.0

    # A 归一化 → B 归一化（H: dst_B_norm = H @ src_A_norm）
    pts_a_flat = np.stack([nx_a.ravel(), ny_a.ravel(), np.ones_like(nx_a.ravel())])  # [3, HW]
    pts_b_norm = H_mat @ pts_a_flat                                                  # [3, HW]
    nx_b = (pts_b_norm[0] / pts_b_norm[2]).reshape(H_pano, W_pano)
    ny_b = (pts_b_norm[1] / pts_b_norm[2]).reshape(H_pano, W_pano)

    # B 归一化 → B 像素（align_corners=False）
    map_x_pano = ((nx_b + 1.0) / 2.0) * W_b - 0.5
    map_y_pano = ((ny_b + 1.0) / 2.0) * H_b - 0.5
    map_x_pano = map_x_pano.astype(np.float32)
    map_y_pano = map_y_pano.astype(np.float32)

    # 2. 局部高精度覆写：dense_grid 的输出尺寸对齐 A，采样坐标必须转换到 B 的像素尺寸。
    map_x_dense, map_y_dense = torch_grid_to_cv2_map(dense_grid, H_a, W_a, H_b, W_b)
    map_x_pano[ty:ty + H_a, tx:tx + W_a] = map_x_dense
    map_y_pano[ty:ty + H_a, tx:tx + W_a] = map_y_dense

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

    if stitch_mask is not None:
        stitch_mask_img = (stitch_mask[0, ..., 0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        stitch_mask_img = cv2.resize(stitch_mask_img, (W_a, H_a), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_path.with_name(f"{out_path.stem}_stitch_mask.png")), stitch_mask_img)
    if stitch_residual_flow is not None:
        residual_mag = stitch_residual_flow[0].detach().norm(dim=-1).cpu().numpy()
        residual_mag = cv2.resize(residual_mag, (W_a, H_a), interpolation=cv2.INTER_LINEAR)
        residual_mag = residual_mag / (residual_mag.max() + 1e-8)
        residual_mag = (residual_mag * 255.0).astype(np.uint8)
        cv2.imwrite(str(out_path.with_name(f"{out_path.stem}_stitch_residual.png")), residual_mag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriStitcher 真正的高清全景生成器")
    parser.add_argument("--img-a", type=str, required=True, help="基准图片")
    parser.add_argument("--img-b", type=str, required=True, help="待拼接图片")
    parser.add_argument("--ckpt", type=str, required=True, help="权重路径")
    parser.add_argument("--out", type=str, default="output/panorama.jpg", help="输出路径")
    parser.add_argument("--use-ema", action="store_true", help="如果 checkpoint 内有 student_ema，则使用 EMA 权重")
    parser.add_argument("--max-pano-megapixels", type=float, default=120.0, help="允许生成的最大全景画布像素数，单位 MP")
    args = parser.parse_args()
    inference(args)
