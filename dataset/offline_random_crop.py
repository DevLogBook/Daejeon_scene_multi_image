import os
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def check_bounds(M, crop_w, crop_h, orig_w, orig_h):
    """
    数学探测：检查经过仿射变换 M 后，目标图像是否完全落在原图的合法物理像素内
    """
    M_inv = cv2.invertAffineTransform(M)
    # 目标图像的四个角
    corners = np.array([[0, 0], [crop_w, 0], [0, crop_h], [crop_w, crop_h]], dtype=np.float32)
    # 逆映射回原图的坐标
    corners_orig = np.dot(M_inv[:, :2], corners.T).T + M_inv[:, 2]

    # 留 1 个像素的安全冗余，防止插值时触碰边界
    in_bounds = np.all((corners_orig[:, 0] >= 1) & (corners_orig[:, 0] < orig_w - 1) &
                       (corners_orig[:, 1] >= 1) & (corners_orig[:, 1] < orig_h - 1))
    return in_bounds


def offline_sift_crop_strict_no_pad(
        pair_txt: str,
        crop_out_dir1: str,
        crop_out_dir2: str,
        output_pair_txt: str = "uav_cropped_pairs.txt",
        crops_per_image: int = 10,
        scale_factor: float = 2.0,
        max_shift_ratio: float = 0.15,
        max_rotate_angle: float = 15.0,
        seed: int = 42
):
    """
    严格无填充裁剪法 (Strict No-Padding):
    1. 结合大感受野 (scale_factor) 和 Resize。
    2. 绝对不使用任何 Border Reflect / Constant Padding。
    3. 能支持旋转且不越界就旋转；否则自动退化为纯平移滑动。
    """
    count = 0
    random.seed(seed)
    out_dir1, out_dir2 = Path(crop_out_dir1), Path(crop_out_dir2)
    out_dir1.mkdir(parents=True, exist_ok=True)
    out_dir2.mkdir(parents=True, exist_ok=True)

    target_sizes = [(512, 512), (256, 256), (384, 384), (512, 384), (384, 512)]

    pairs = []
    with open(pair_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

    print(f"🌾 开始进行 [严格无填充 + 智能旋转] SIFT裁剪，共 {len(pairs)} 对...")

    sift = cv2.SIFT_create(nfeatures=100000)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    out_lines = []

    for idx, (path_a, path_b) in enumerate(tqdm(pairs, desc="Strict Cropping")):
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)

        if img_a is None or img_b is None:
            continue

        orig_h, orig_w = img_a.shape[:2]
        base_name_a, ext_a = Path(path_a).stem, Path(path_a).suffix
        base_name_b, ext_b = Path(path_b).stem, Path(path_b).suffix

        # --- 1. SIFT 寻找物理对应点 ---
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(gray_a, None)
        kp2, des2 = sift.detectAndCompute(gray_b, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for match_pts in matches if len(match_pts) == 2 for m, n in [match_pts] if
                        m.distance < 0.75 * n.distance]

        if len(good_matches) < 15:
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if mask is None:
            continue

        inliers = mask.ravel().tolist()
        inlier_matches = [m for i, m in enumerate(good_matches) if inliers[i] == 1]

        if len(inlier_matches) < 10:
            continue

        # --- 2. 严禁填充的裁剪与变换 ---
        successful_crops = 0
        attempts = 0

        while successful_crops < crops_per_image and attempts < crops_per_image * 3:
            attempts += 1

            match = random.choice(inlier_matches)
            pt_a = kp1[match.queryIdx].pt
            pt_b = kp2[match.trainIdx].pt

            target_th, target_tw = random.choice(target_sizes)
            crop_th = int(target_th * scale_factor)
            crop_tw = int(target_tw * scale_factor)

            if crop_th > orig_h or crop_tw > orig_w:
                continue  # 如果放大后比原图还大，直接放弃该尺寸

            # ==========================================
            # A 图：计算合法滑动窗口
            # ==========================================
            # 允许框存在的相对坐标区间，确保 SIFT 点在框内，且框在图内
            min_rx = max(0, pt_a[0] - orig_w + crop_tw)
            max_rx = min(crop_tw, pt_a[0])
            min_ry = max(0, pt_a[1] - orig_h + crop_th)
            max_ry = min(crop_th, pt_a[1])

            if min_rx > max_rx or min_ry > max_ry:
                continue

            # 优先将锚点放在正中心，如果在边界则用 clip 强行卡进合法区间
            rx = np.clip(crop_tw / 2.0, min_rx, max_rx)
            ry = np.clip(crop_th / 2.0, min_ry, max_ry)

            xa = int(pt_a[0] - rx)
            ya = int(pt_a[1] - ry)

            # 因为被严密卡过范围，这里切图绝不会越界
            crop_a_large = img_a[ya:ya + crop_th, xa:xa + crop_tw]
            crop_a_final = cv2.resize(crop_a_large, (target_tw, target_th), interpolation=cv2.INTER_AREA)

            dx = random.randint(int(-crop_tw * max_shift_ratio), int(crop_tw * max_shift_ratio))
            dy = random.randint(int(-crop_th * max_shift_ratio), int(crop_th * max_shift_ratio))

            # 期望锚点在 B 切图中的位置
            tx_ideal = rx + dx
            ty_ideal = ry + dy

            angle = random.uniform(-max_rotate_angle, max_rotate_angle)

            # 1. 尝试 旋转 + 平移
            M_B = cv2.getRotationMatrix2D(pt_b, angle, 1.0)
            M_B[0, 2] += tx_ideal - pt_b[0]
            M_B[1, 2] += ty_ideal - pt_b[1]

            # 2. 验证这个操作是否会越界
            if not check_bounds(M_B, crop_tw, crop_th, orig_w, orig_h):
                # 触发越界！放弃旋转，退化为纯平移
                # 重新计算纯平移下，必须满足边界约束的 tx 和 ty
                min_tx = max(0, pt_b[0] - orig_w + crop_tw)
                max_tx = min(crop_tw, pt_b[0])
                min_ty = max(0, pt_b[1] - orig_h + crop_th)
                max_ty = min(crop_th, pt_b[1])

                if min_tx > max_tx or min_ty > max_ty:
                    continue  # 极端情况，连平移都塞不下，直接放弃

                tx = np.clip(tx_ideal, min_tx, max_tx)
                ty = np.clip(ty_ideal, min_ty, max_ty)

                M_B = cv2.getRotationMatrix2D(pt_b, 0.0, 1.0)
                M_B[0, 2] += tx - pt_b[0]
                M_B[1, 2] += ty - pt_b[1]

            # 使用 CONSTANT 模式。因为前面已经经过严密的 check_bounds，这里绝对不可能触发 padding 边界！
            crop_b_large = cv2.warpAffine(img_b, M_B, (crop_tw, crop_th), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT)
            crop_b_final = cv2.resize(crop_b_large, (target_tw, target_th), interpolation=cv2.INTER_AREA)

            # --- 保存结果 ---
            save_name_a = f"{base_name_a}_crop{successful_crops:02d}_fov{scale_factor}x_{target_th}x{target_tw}_{count:04d}.jpg"
            save_name_b = f"{base_name_b}_crop{successful_crops:02d}_fov{scale_factor}x_{target_th}x{target_tw}_{count:04d}.jpg"
            save_path_a = out_dir1 / save_name_a
            save_path_b = out_dir2 / save_name_b

            cv2.imwrite(str(save_path_a), crop_a_final)
            cv2.imwrite(str(save_path_b), crop_b_final)
            count += 1

            out_lines.append(f"{save_path_a.resolve()} {save_path_b.resolve()}\n")
            successful_crops += 1

    with open(output_pair_txt, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

    print(f"✅ 严格无填充裁剪完成！成功生成 {len(out_lines)} 对图像。")


if __name__ == "__main__":
    offline_sift_crop_strict_no_pad(
        pair_txt="../pairs_2.txt",
        crop_out_dir1="../training_2/input1",
        crop_out_dir2="../training_2/input2",
        crops_per_image=5,
        scale_factor=2.0,
        max_shift_ratio=0.15,
        max_rotate_angle=10.0
    )