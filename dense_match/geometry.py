from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from dense_match.refine import make_grid


def _sanitize_tensor(
    x: torch.Tensor,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
    clamp: Tuple[float, float] | None = None,
) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if clamp is not None:
        x = x.clamp(min=clamp[0], max=clamp[1])
    return x


def _sanitize_homography(H: torch.Tensor) -> torch.Tensor:
    B = H.shape[0]
    eye = torch.eye(3, device=H.device, dtype=H.dtype).unsqueeze(0).expand(B, -1, -1)
    finite = torch.isfinite(H).all(dim=(-2, -1))
    h33 = H[:, 2, 2]
    valid = finite & (h33.abs() > 1e-4) & (h33.abs() < 1e4)
    H = torch.where(valid.view(B, 1, 1), H, eye)
    return _sanitize_tensor(H, nan=0.0, posinf=1.0, neginf=-1.0, clamp=(-1e4, 1e4))

def _sanitize_base_transform(H: torch.Tensor, max_shift: float = 0.85) -> torch.Tensor:
    B = H.shape[0]
    device = H.device
    dtype = H.dtype

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

    H = torch.nan_to_num(H, nan=0.0, posinf=1.0, neginf=-1.0)

    # 明确禁用透视
    H[:, 2, 0] = 0.0
    H[:, 2, 1] = 0.0
    H[:, 2, 2] = 1.0

    A = H[:, :2, :2]
    sx = torch.linalg.norm(A[:, :, 0], dim=1)
    sy = torch.linalg.norm(A[:, :, 1], dim=1)
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]

    tx = H[:, 0, 2].abs()
    ty = H[:, 1, 2].abs()

    valid = (
        torch.isfinite(H).all(dim=(-2, -1))
        & (sx > 0.65) & (sx < 1.40)
        & (sy > 0.65) & (sy < 1.40)
        & (det > 0.40) & (det < 2.00)
        & (tx < max_shift)
        & (ty < max_shift)
    )

    # fallback 不要直接 identity；保留安全平移
    fallback = eye.clone()
    fallback[:, 0, 2] = H[:, 0, 2].clamp(-max_shift, max_shift)
    fallback[:, 1, 2] = H[:, 1, 2].clamp(-max_shift, max_shift)

    return torch.where(valid.view(B, 1, 1), H, fallback)


def _sanitize_base_transform_no_inplace(H: torch.Tensor, max_shift: float = 0.85) -> torch.Tensor:
    B = H.shape[0]
    device = H.device
    dtype = H.dtype

    H0 = torch.nan_to_num(H, nan=0.0, posinf=1.0, neginf=-1.0)
    row0 = torch.stack([
        H0[:, 0, 0],
        H0[:, 0, 1],
        H0[:, 0, 2].clamp(-max_shift, max_shift),
    ], dim=1)
    row1 = torch.stack([
        H0[:, 1, 0],
        H0[:, 1, 1],
        H0[:, 1, 2].clamp(-max_shift, max_shift),
    ], dim=1)
    row2 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3).expand(B, 3)
    H_clean = torch.stack([row0, row1, row2], dim=1)

    A = H_clean[:, :2, :2]
    sx = torch.linalg.norm(A[:, :, 0], dim=1)
    sy = torch.linalg.norm(A[:, :, 1], dim=1)
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    valid = (
        torch.isfinite(H_clean).all(dim=(-2, -1))
        & (sx > 0.65) & (sx < 1.40)
        & (sy > 0.65) & (sy < 1.40)
        & (det > 0.40) & (det < 2.00)
    )

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    fallback = eye.clone()
    fallback[:, 0, 2] = H_clean[:, 0, 2]
    fallback[:, 1, 2] = H_clean[:, 1, 2]
    return torch.where(valid.view(B, 1, 1), H_clean, fallback)


def project_grid_with_h(
        H: torch.Tensor,
        B: int,
        H_img: int,
        W_img: int,
        device: torch.device,
) -> torch.Tensor:
    grid_A_img = make_grid(B, H_img, W_img, device, torch.float32)
    src_pts_img = grid_A_img.reshape(B, -1, 2)
    ones_img = torch.ones(B, H_img * W_img, 1, device=device, dtype=torch.float32)
    coords_homo_img = torch.cat([src_pts_img, ones_img], dim=-1)
    projected_homo = torch.bmm(coords_homo_img, H.float().transpose(1, 2))
    z_img = projected_homo[..., 2:3]
    sign_img = z_img.sign()
    sign_img = torch.where(sign_img == 0, torch.ones_like(sign_img), sign_img)
    denom = torch.where(z_img.abs() < 1e-3, 1e-3 * sign_img, z_img)
    grid = (projected_homo[..., :2] / denom).reshape(B, H_img, W_img, 2)
    return _sanitize_tensor(grid, nan=0.0, posinf=3.0, neginf=-3.0, clamp=(-3.0, 3.0))


def hartley_normalize(pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = pts.float()
    B, N, _ = pts.shape
    device = pts.device

    centroid = pts.mean(dim=1, keepdim=True)
    pts_c = pts - centroid

    dist = (pts_c ** 2).sum(dim=-1).add(1e-8).sqrt()
    mean_dist = dist.mean(dim=1, keepdim=True).clamp(min=1e-8)

    is_degenerate = mean_dist < 1e-3
    mean_dist_safe = mean_dist.clamp(min=1e-3)
    scale = (2.0 ** 0.5) / mean_dist_safe
    scale = torch.where(is_degenerate, torch.ones_like(scale), scale)

    pts_norm = pts_c * scale.unsqueeze(-1)

    cx = centroid[:, 0, 0]
    cy = centroid[:, 0, 1]
    s  = scale[:, 0]
    zeros = torch.zeros_like(s)
    ones  = torch.ones_like(s)

    T = torch.stack([
        s,     zeros, -s * cx,
        zeros, s,     -s * cy,
        zeros, zeros, ones,
    ], dim=-1).reshape(B, 3, 3)

    return pts_norm, T


def _safe_quantile(x: torch.Tensor, q: float, default: float = 0.0) -> torch.Tensor:
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return torch.tensor(default, device=x.device, dtype=torch.float32)
    return torch.quantile(x.float(), q)


def _make_identity_h(B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)


def _make_affine_h(theta: torch.Tensor) -> torch.Tensor:
    """
    theta:
      [6] or [B,6]
      x' = a*x + b*y + tx
      y' = c*x + d*y + ty
    """
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    B = theta.shape[0]
    H = torch.zeros(B, 3, 3, device=theta.device, dtype=theta.dtype)
    H[:, 0, 0] = theta[:, 0]
    H[:, 0, 1] = theta[:, 1]
    H[:, 0, 2] = theta[:, 2]
    H[:, 1, 0] = theta[:, 3]
    H[:, 1, 1] = theta[:, 4]
    H[:, 1, 2] = theta[:, 5]
    H[:, 2, 2] = 1.0
    return H


def _apply_h_to_xy(H: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    H:  [3,3] or [B,3,3]
    xy: [N,2] or [B,N,2]
    """
    squeeze_h = False
    squeeze_xy = False

    if H.ndim == 2:
        H = H.unsqueeze(0)
        squeeze_h = True
    if xy.ndim == 2:
        xy = xy.unsqueeze(0)
        squeeze_xy = True

    B, N, _ = xy.shape
    ones = torch.ones(B, N, 1, device=xy.device, dtype=xy.dtype)
    homo = torch.cat([xy, ones], dim=-1)
    proj = torch.bmm(homo, H.transpose(1, 2))

    z = proj[..., 2:3]
    sign = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    z_safe = torch.where(z.abs() < 1e-6, sign * 1e-6, z)

    out = proj[..., :2] / z_safe
    out = torch.nan_to_num(out, nan=0.0, posinf=3.0, neginf=-3.0)

    if squeeze_h and squeeze_xy:
        return out[0]
    return out


def _spatial_topk_indices(
    weights: torch.Tensor,
    valid: torch.Tensor,
    H: int,
    W: int,
    *,
    bins_y: int = 8,
    bins_x: int = 8,
    topk_per_bin: int = 8,
    min_weight: float = 0.03,
) -> torch.Tensor:
    """
    weights: [H*W]
    valid:   [H*W]
    返回空间均匀采样后的 flat indices。
    """
    device = weights.device
    selected = []

    w2d = weights.reshape(H, W)
    v2d = valid.reshape(H, W)

    for by in range(bins_y):
        y0 = int(round(by * H / bins_y))
        y1 = int(round((by + 1) * H / bins_y))
        if y1 <= y0:
            continue

        for bx in range(bins_x):
            x0 = int(round(bx * W / bins_x))
            x1 = int(round((bx + 1) * W / bins_x))
            if x1 <= x0:
                continue

            local_w = w2d[y0:y1, x0:x1]
            local_v = v2d[y0:y1, x0:x1] & (local_w > min_weight)

            if not local_v.any():
                continue

            flat_local = torch.nonzero(local_v.reshape(-1), as_tuple=False).squeeze(1)
            local_scores = local_w.reshape(-1)[flat_local]

            k = min(topk_per_bin, flat_local.numel())
            _, order = torch.topk(local_scores, k=k, largest=True, sorted=False)
            chosen_local = flat_local[order]

            yy = chosen_local // (x1 - x0)
            xx = chosen_local % (x1 - x0)
            chosen_global = (y0 + yy) * W + (x0 + xx)
            selected.append(chosen_global)

    if len(selected) == 0:
        return torch.empty(0, device=device, dtype=torch.long)

    return torch.cat(selected, dim=0).unique()


def _coverage_stats_from_indices(idx: torch.Tensor, H: int, W: int, bins_y: int = 8, bins_x: int = 8):
    """
    返回：
      covered_bins: 覆盖 bin 数
      covered_quads: 覆盖象限数
    """
    if idx.numel() == 0:
        return 0, 0

    y = idx // W
    x = idx % W

    by = torch.clamp((y * bins_y) // H, 0, bins_y - 1)
    bx = torch.clamp((x * bins_x) // W, 0, bins_x - 1)
    bins = by * bins_x + bx
    covered_bins = int(bins.unique().numel())

    q0 = ((y < H // 2) & (x < W // 2)).any()
    q1 = ((y < H // 2) & (x >= W // 2)).any()
    q2 = ((y >= H // 2) & (x < W // 2)).any()
    q3 = ((y >= H // 2) & (x >= W // 2)).any()
    covered_quads = int(q0) + int(q1) + int(q2) + int(q3)

    return covered_bins, covered_quads


def _fit_translation_irls(
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    *,
    iters: int = 5,
    huber: float = 0.08,
    max_shift: float = 0.75,
) -> torch.Tensor:
    """
    src, dst: [N,2]
    w:        [N]
    返回 [3,3]
    """
    device, dtype = src.device, src.dtype

    flow = dst.float() - src.float()
    w0 = w.float().clamp_min(0.0)

    if flow.shape[0] < 4 or w0.sum() < 1e-6:
        return torch.eye(3, device=device, dtype=dtype)

    rw = w0
    t = (flow * rw[:, None]).sum(dim=0) / rw.sum().clamp_min(1e-6)

    for _ in range(iters):
        res = torch.linalg.norm(flow - t[None, :], dim=-1)
        robust = torch.clamp(huber / (res + 1e-6), max=1.0)
        rw = w0 * robust
        if rw.sum() < 1e-6:
            break
        t = (flow * rw[:, None]).sum(dim=0) / rw.sum().clamp_min(1e-6)

    t = torch.nan_to_num(t, nan=0.0, posinf=max_shift, neginf=-max_shift)
    t = t.clamp(-max_shift, max_shift)

    H = torch.eye(3, device=device, dtype=torch.float32)
    H[0, 2] = t[0]
    H[1, 2] = t[1]
    return H.to(dtype)


def _fit_similarity_once(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Weighted Procrustes similarity:
      dst = s * R * src + t
    返回 [3,3]
    """
    device, dtype = src.device, src.dtype

    w = w.float().clamp_min(0.0)
    sw = w.sum().clamp_min(1e-6)

    src_f = src.float()
    dst_f = dst.float()

    mu_x = (src_f * w[:, None]).sum(dim=0) / sw
    mu_y = (dst_f * w[:, None]).sum(dim=0) / sw

    x = src_f - mu_x[None, :]
    y = dst_f - mu_y[None, :]

    var_x = (w * (x ** 2).sum(dim=-1)).sum() / sw
    if var_x < 1e-8:
        return _fit_translation_irls(src, dst, w).float()

    cov = torch.matmul((x * w[:, None]).transpose(0, 1), y) / sw

    try:
        U, S, Vh = torch.linalg.svd(cov)
    except RuntimeError:
        return _fit_translation_irls(src, dst, w).float()

    R = torch.matmul(Vh.transpose(0, 1), U.transpose(0, 1))

    if torch.linalg.det(R) < 0:
        Vh_fix = Vh.clone()
        Vh_fix[-1, :] *= -1
        R = torch.matmul(Vh_fix.transpose(0, 1), U.transpose(0, 1))

    scale = S.sum() / var_x
    scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)

    t = mu_y - scale * torch.matmul(R, mu_x)

    H = torch.eye(3, device=device, dtype=torch.float32)
    H[:2, :2] = scale * R
    H[:2, 2] = t
    return H.to(dtype)


def _fit_similarity_irls(
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    *,
    iters: int = 5,
    huber: float = 0.08,
) -> torch.Tensor:
    w0 = w.float().clamp_min(0.0)
    rw = w0

    H = _fit_similarity_once(src, dst, rw)

    for _ in range(iters):
        pred = _apply_h_to_xy(H.float(), src.float())
        res = torch.linalg.norm(pred - dst.float(), dim=-1)
        robust = torch.clamp(huber / (res + 1e-6), max=1.0)
        rw = w0 * robust

        if rw.sum() < 1e-6:
            break

        H = _fit_similarity_once(src, dst, rw)

    return H.to(src.dtype)


def _fit_affine_once(
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    *,
    reg: float = 1e-3,
) -> torch.Tensor:
    """
    Weighted affine LS:
      x' = a*x + b*y + tx
      y' = c*x + d*y + ty
    返回 [3,3]
    """
    device, dtype = src.device, src.dtype

    N = src.shape[0]
    if N < 6:
        return _fit_similarity_once(src, dst, w).to(dtype)

    src_f = src.float()
    dst_f = dst.float()
    w_f = w.float().clamp_min(0.0)

    x = src_f[:, 0]
    y = src_f[:, 1]
    xp = dst_f[:, 0]
    yp = dst_f[:, 1]

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    row_x = torch.stack([x, y, ones, zeros, zeros, zeros], dim=-1)
    row_y = torch.stack([zeros, zeros, zeros, x, y, ones], dim=-1)

    A = torch.cat([row_x, row_y], dim=0)
    b = torch.cat([xp, yp], dim=0).unsqueeze(-1)

    ww = torch.cat([w_f, w_f], dim=0).clamp_min(1e-6)
    ws = torch.sqrt(ww).unsqueeze(-1)

    Aw = A * ws
    bw = b * ws

    prior = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32).view(6, 1)

    AtA = torch.matmul(Aw.transpose(0, 1), Aw)
    Atb = torch.matmul(Aw.transpose(0, 1), bw)

    diag_mean = AtA.diagonal().mean().clamp_min(1.0)
    lam = float(reg) * diag_mean

    AtA = AtA + lam * torch.eye(6, device=device, dtype=torch.float32)
    Atb = Atb + lam * prior

    try:
        theta = torch.linalg.solve(AtA, Atb).squeeze(-1)
    except RuntimeError:
        return _fit_similarity_once(src, dst, w).to(dtype)

    theta = torch.nan_to_num(theta, nan=0.0, posinf=2.0, neginf=-2.0)
    return _make_affine_h(theta).squeeze(0).to(dtype)


def _fit_affine_irls(
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    *,
    iters: int = 5,
    huber: float = 0.08,
    reg: float = 1e-3,
) -> torch.Tensor:
    w0 = w.float().clamp_min(0.0)
    rw = w0

    H = _fit_affine_once(src, dst, rw, reg=reg)

    for _ in range(iters):
        pred = _apply_h_to_xy(H.float(), src.float())
        res = torch.linalg.norm(pred - dst.float(), dim=-1)
        robust = torch.clamp(huber / (res + 1e-6), max=1.0)
        rw = w0 * robust

        if rw.sum() < 1e-6:
            break

        H = _fit_affine_once(src, dst, rw, reg=reg)

    return H.to(src.dtype)


def _transform_metrics(H: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    H: [3,3]
    """
    A = H[:2, :2].float()
    col0 = A[:, 0]
    col1 = A[:, 1]

    sx = torch.linalg.norm(col0)
    sy = torch.linalg.norm(col1)
    det = torch.det(A)

    dot = torch.abs(torch.dot(col0, col1)) / (sx * sy + 1e-6)
    tx = H[0, 2].float().abs()
    ty = H[1, 2].float().abs()

    return {
        "sx": sx,
        "sy": sy,
        "det": det,
        "shear_cos": dot,
        "tx": tx,
        "ty": ty,
    }


def _is_physically_valid_base_transform(
    H: torch.Tensor,
    *,
    scale_min: float = 0.70,
    scale_max: float = 1.35,
    det_min: float = 0.45,
    det_max: float = 1.85,
    max_shear_cos: float = 0.55,
    max_shift: float = 0.85,
) -> bool:
    if not torch.isfinite(H).all().item():
        return False

    m = _transform_metrics(H)

    sx = float(m["sx"])
    sy = float(m["sy"])
    det = float(m["det"])
    shear = float(m["shear_cos"])
    tx = float(m["tx"])
    ty = float(m["ty"])

    return (
        scale_min <= sx <= scale_max
        and scale_min <= sy <= scale_max
        and det_min <= det <= det_max
        and shear <= max_shear_cos
        and tx <= max_shift
        and ty <= max_shift
    )


def _score_transform(
    H: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred = _apply_h_to_xy(H.float(), src.float())
    res = torch.linalg.norm(pred - dst.float(), dim=-1)
    res = torch.nan_to_num(res, nan=10.0, posinf=10.0, neginf=10.0)

    return {
        "median": _safe_quantile(res, 0.50, default=10.0),
        "p80": _safe_quantile(res, 0.80, default=10.0),
        "p90": _safe_quantile(res, 0.90, default=10.0),
        "mean": res.mean(),
    }


def solve_robust_base_transform_from_dense_flow(
    src_pts: torch.Tensor,
    dst_pts: torch.Tensor,
    weights: torch.Tensor,
    H_grid: int,
    W_grid: int,
    *,
    bins_y: int = 8,
    bins_x: int = 8,
    topk_per_bin: int = 8,
    min_points: int = 48,
    min_bins: int = 14,
    min_quads: int = 3,
    min_dst_span: float = 0.45,
    max_shift: float = 0.75,
    allow_affine: bool = True,
    allow_similarity: bool = True,
    return_stats: bool = False,
):
    """
    生产版 dense-flow → 稳定 base transform。

    输入:
      src_pts: [B,N,2] A normalized coords
      dst_pts: [B,N,2] predicted B normalized coords
      weights: [B,N] or [B,N,1]
      H_grid, W_grid: dst/src dense warp 的空间尺寸，例如 Hf,Wf

    输出:
      H_base: [B,3,3]
      stats:  可选，便于 TensorBoard 记录

    模型选择:
      1. translation 永远可用
      2. coverage 足够时尝试 similarity
      3. affine 只有在显著优于 similarity 且物理合法时才启用
      4. 禁用 full homography，避免重复纹理下的透视/scale 退化
    """
    assert src_pts.ndim == 3 and src_pts.shape[-1] == 2
    assert dst_pts.ndim == 3 and dst_pts.shape[-1] == 2

    if weights.ndim == 3:
        weights = weights.squeeze(-1)

    B, N, _ = src_pts.shape
    device = src_pts.device
    dtype = src_pts.dtype

    out_H = _make_identity_h(B, device, torch.float32)

    stat_model = torch.zeros(B, device=device, dtype=torch.long)          # 0=id, 1=translation, 2=similarity, 3=affine
    stat_p90 = torch.full((B,), 10.0, device=device, dtype=torch.float32)
    stat_points = torch.zeros(B, device=device, dtype=torch.float32)
    stat_bins = torch.zeros(B, device=device, dtype=torch.float32)
    stat_quads = torch.zeros(B, device=device, dtype=torch.float32)
    stat_dst_span_x = torch.zeros(B, device=device, dtype=torch.float32)
    stat_dst_span_y = torch.zeros(B, device=device, dtype=torch.float32)

    with torch.no_grad():
        src_all = torch.nan_to_num(src_pts.float(), nan=0.0, posinf=2.0, neginf=-2.0)
        dst_all = torch.nan_to_num(dst_pts.float(), nan=0.0, posinf=2.0, neginf=-2.0)
        w_all = torch.nan_to_num(weights.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        # 只允许落在合理 B 归一化坐标附近的点参与全局估计。
        # 允许 1.15 是为了保留边界附近的有效外推，但不会让大越界点支配估计。
        finite_valid = (
            torch.isfinite(src_all).all(dim=-1)
            & torch.isfinite(dst_all).all(dim=-1)
            & torch.isfinite(w_all)
            & (w_all > 0.0)
            & (dst_all[..., 0].abs() <= 1.15)
            & (dst_all[..., 1].abs() <= 1.15)
        )

        for b in range(B):
            src_b = src_all[b]
            dst_b = dst_all[b]
            w_b = w_all[b]
            valid_b = finite_valid[b]

            idx = _spatial_topk_indices(
                w_b,
                valid_b,
                H_grid,
                W_grid,
                bins_y=bins_y,
                bins_x=bins_x,
                topk_per_bin=topk_per_bin,
                min_weight=0.03,
            )

            if idx.numel() < min_points:
                # fallback: 如果 top-k 太少，直接在全图 valid 里拿权重最大的 min_points
                valid_idx = torch.nonzero(valid_b, as_tuple=False).squeeze(1)
                if valid_idx.numel() > 0:
                    k = min(max(min_points, idx.numel()), valid_idx.numel())
                    _, order = torch.topk(w_b[valid_idx], k=k, largest=True, sorted=False)
                    idx = valid_idx[order]

            if idx.numel() < 8:
                out_H[b] = torch.eye(3, device=device, dtype=torch.float32)
                stat_model[b] = 0
                continue

            src = src_b[idx]
            dst = dst_b[idx]
            w = w_b[idx].clamp_min(1e-6)

            covered_bins, covered_quads = _coverage_stats_from_indices(idx, H_grid, W_grid, bins_y, bins_x)
            stat_points[b] = float(idx.numel())
            stat_bins[b] = float(covered_bins)
            stat_quads[b] = float(covered_quads)

            dst_x_span = _safe_quantile(dst[:, 0], 0.95) - _safe_quantile(dst[:, 0], 0.05)
            dst_y_span = _safe_quantile(dst[:, 1], 0.95) - _safe_quantile(dst[:, 1], 0.05)
            stat_dst_span_x[b] = dst_x_span
            stat_dst_span_y[b] = dst_y_span

            coverage_ok = (
                idx.numel() >= min_points
                and covered_bins >= min_bins
                and covered_quads >= min_quads
                and float(dst_x_span) >= min_dst_span
                and float(dst_y_span) >= min_dst_span
            )

            # 1. Translation baseline: 永远拟合，永远安全。
            H_trans = _fit_translation_irls(
                src,
                dst,
                w,
                iters=6,
                huber=0.08,
                max_shift=max_shift,
            ).float()
            score_trans = _score_transform(H_trans, src, dst)

            best_H = H_trans
            best_score = score_trans
            best_model = 1

            if coverage_ok and allow_similarity:
                H_sim = _fit_similarity_irls(
                    src,
                    dst,
                    w,
                    iters=6,
                    huber=0.08,
                ).float()

                sim_valid = _is_physically_valid_base_transform(
                    H_sim,
                    scale_min=0.72,
                    scale_max=1.28,
                    det_min=0.52,
                    det_max=1.65,
                    max_shear_cos=0.20,
                    max_shift=max_shift,
                )
                score_sim = _score_transform(H_sim, src, dst)

                # similarity 必须比 translation 有明确收益才替换。
                if sim_valid and float(score_sim["p90"]) < float(score_trans["p90"]) * 0.92:
                    best_H = H_sim
                    best_score = score_sim
                    best_model = 2

                if allow_affine:
                    H_aff = _fit_affine_irls(
                        src,
                        dst,
                        w,
                        iters=6,
                        huber=0.08,
                        reg=3e-3,
                    ).float()

                    aff_valid = _is_physically_valid_base_transform(
                        H_aff,
                        scale_min=0.70,
                        scale_max=1.32,
                        det_min=0.50,
                        det_max=1.75,
                        max_shear_cos=0.42,
                        max_shift=max_shift,
                    )
                    score_aff = _score_transform(H_aff, src, dst)

                    # affine 必须显著优于当前 best，防止用多余自由度吃重复纹理假匹配。
                    if aff_valid and float(score_aff["p90"]) < float(best_score["p90"]) * 0.85:
                        best_H = H_aff
                        best_score = score_aff
                        best_model = 3

            # 最后一层硬拒绝：如果最佳模型 residual 仍然很差，退回 translation。
            # normalized 坐标下 0.20 已经是很大的误差；这里不要用坏 affine/H 污染 decoder。
            if float(best_score["p90"]) > 0.22:
                best_H = H_trans
                best_score = score_trans
                best_model = 1

            # translation 也差得离谱时，说明当前 warp 不可信；保留安全平移，不允许 scale。
            if float(best_score["p90"]) > 0.35:
                best_H = H_trans
                best_model = 1

            best_H[2, 0] = 0.0
            best_H[2, 1] = 0.0
            best_H[2, 2] = 1.0

            out_H[b] = torch.nan_to_num(best_H, nan=0.0, posinf=1.0, neginf=-1.0)
            stat_model[b] = int(best_model)
            stat_p90[b] = best_score["p90"]

    out_H = out_H.to(dtype)

    if not return_stats:
        return out_H

    stats = {
        "base_model_id": stat_model,
        "base_p90": stat_p90,
        "base_points": stat_points,
        "base_bins": stat_bins,
        "base_quads": stat_quads,
        "base_dst_span_x": stat_dst_span_x,
        "base_dst_span_y": stat_dst_span_y,
    }
    return out_H, stats
