import cv2
import numpy as np

def robust_homography_estimation(img_A_shape, pts_A_norm, pts_B_norm, confidence, conf_thresh=0.7):
    """
    针对密集匹配点的鲁棒单应性矩阵估计与过滤
    pts_A_norm, pts_B_norm: 归一化坐标 [-1, 1], shape (N, 2)
    confidence: 每个点的置信度, shape (N,)
    """
    H_img, W_img = img_A_shape[:2]

    # 1. 置信度硬阈值过滤
    mask_conf = confidence > conf_thresh
    pts_A = pts_A_norm[mask_conf]
    pts_B = pts_B_norm[mask_conf]

    if len(pts_A) < 10:
        return None, None, None

    # 将归一化坐标转换为像素坐标
    pts_A_px = (pts_A + 1.0) / 2.0 * np.array([W_img - 1, H_img - 1])
    pts_B_px = (pts_B + 1.0) / 2.0 * np.array([W_img - 1, H_img - 1])

    # 2. 运动矢量场一致性过滤 (Flow Consistency Filter)
    # 计算所有匹配点的位移向量
    flow_vectors = pts_B_px - pts_A_px
    
    # 计算位移向量的幅度(距离)和角度
    magnitudes = np.linalg.norm(flow_vectors, axis=1)
    angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])

    # 计算中值
    med_mag = np.median(magnitudes)
    med_ang = np.median(angles)

    # 设定允许的容差范围 (例如：幅度偏差不超过中值的 2 倍，角度偏差不超过 30 度)
    # 对于无人机平移拍摄，大部分点的运动矢量应该是一致的
    mag_mask = np.abs(magnitudes - med_mag) < (1.5 * med_mag + 1e-5)
    
    # 角度容差处理 (需处理 -pi 到 pi 的边界跃变)
    angle_diff = np.abs(angles - med_ang)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    ang_mask = angle_diff < np.deg2rad(30)

    # 结合向量过滤掩码
    flow_mask = mag_mask & ang_mask
    
    pts_A_filtered = pts_A_px[flow_mask].astype(np.float32)
    pts_B_filtered = pts_B_px[flow_mask].astype(np.float32)

    if len(pts_A_filtered) < 8:
        return None, None, None

    # 3. 使用 MAGSAC++ 进行鲁棒单应性计算
    # cv2.USAC_MAGSAC 是 OpenCV 4.5+ 中引入的高级 RANSAC
    # ransacReprojThreshold 设为 3.0-5.0 像素，MAGSAC++ 内部会自适应处理
    H_matrix, inliers_mask = cv2.findHomography(
        pts_B_filtered, 
        pts_A_filtered, 
        cv2.USAC_MAGSAC, 
        ransacReprojThreshold=4.0,
        maxIters=10000,
        confidence=0.995
    )

    if H_matrix is None or inliers_mask is None:
        return None, None, None

    # 提取最终被 MAGSAC++ 认可的绝对内点
    inliers_mask = inliers_mask.ravel().astype(bool)
    final_inliers_A = pts_A_filtered[inliers_mask]
    final_inliers_B = pts_B_filtered[inliers_mask]

    return H_matrix, final_inliers_A, final_inliers_B