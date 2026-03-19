import sys
import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, Subset
import platform
import pathlib

# 修复跨平台反序列化问题 (Linux -> Windows)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
REPO_ROOT = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_img_tensor(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_w, new_h = (size, int(h*size/w)) if w >= h else (int(w*size/h), size)
    img = TF.resize(img, [new_h, new_w], antialias=True)
    ten = TF.to_tensor(img)
    _, ch, cw = ten.shape
    return F.pad(ten, (0, size-cw, 0, size-ch), value=0.0)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor,
                 mask: torch.Tensor) -> float:
    """在有效重叠区计算PSNR"""
    diff = (img1 - img2) ** 2
    mse  = (diff * mask).sum() / (mask.sum() * 3 + 1e-8)
    if mse < 1e-10:
        return 100.0
    return float(-10 * torch.log10(mse))


def compute_ssim_simple(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """简化SSIM（不依赖外部库）"""
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu12 = mu1 * mu2
    s1 = F.avg_pool2d(img1**2, 11, 1, 5) - mu1_sq
    s2 = F.avg_pool2d(img2**2, 11, 1, 5) - mu2_sq
    s12 = F.avg_pool2d(img1*img2, 11, 1, 5) - mu12
    num = (2*mu12 + C1) * (2*s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    return float((num / den).mean())


def warp_image(img_b: torch.Tensor,
               warp_field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """对img_b应用warp场，返回warped图和有效mask"""
    warped = F.grid_sample(
        img_b, warp_field,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    ones = torch.ones_like(img_b[:, :1])
    mask = F.grid_sample(
        ones, warp_field,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return warped, (mask > 0.5).float()


def upsample_warp(warp_64: torch.Tensor, target_hw: Tuple[int,int]) -> torch.Tensor:
    """将64×64的warp场上采样到目标分辨率"""
    return F.interpolate(
        warp_64.permute(0, 3, 1, 2),
        size=target_hw, mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1)


# ══════════════════════════════════════════════════════════
# 核心诊断函数
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def diagnose_single_pair(
    img_a:       torch.Tensor,    # (1,3,H,W)
    img_b:       torch.Tensor,    # (1,3,H,W)
    matcher:     torch.nn.Module,
    bypass_tps:  torch.nn.Module,
    tps_gen:     torch.nn.Module,
    device:      torch.device,
    image_size:  int = 256,
) -> Dict:
    """
    对单个图片对执行全面诊断。
    返回包含所有中间结果和指标的字典。
    """
    img_a = img_a.to(device)
    img_b = img_b.to(device)
    H, W  = image_size, image_size

    # ── Step1: AgriMatcher 推理 ──
    m_out       = matcher(img_a, img_b)
    warp_64     = m_out["warp_AB"]           # (1,64,64,2)
    conf_64     = m_out["confidence_AB"]     # (1,64,64)
    feat_a64    = m_out["feat_A_64"]
    feat_b64    = m_out["feat_B_64"]
    warp_coarse = m_out.get("warp_AB_coarse")
    sim_matrix  = m_out.get("sim_matrix")

    # ── Step2: 直接用AgriMatcher warp做grid_sample（上界基准）──
    warp_256_direct = upsample_warp(warp_64, (H, W))  # (1,H,W,2)
    warped_b_direct, mask_direct = warp_image(img_b, warp_256_direct)

    # ── Step3: BypassTPS 推理 ──
    tps_out  = bypass_tps(
        warp_AB        = warp_64,
        confidence     = conf_64,
        feat_A         = feat_a64,
        feat_B         = feat_b64,
        sim_matrix     = sim_matrix,
        warp_AB_coarse = warp_coarse,
        dense_field    = None,
    )
    delta_cp  = tps_out["delta_cp"]          # (1,2,gs,gs)
    tps_field = tps_gen(delta_cp)            # (1,H,W,2)
    gate      = tps_out["gate"]              # (1,1,gs,gs)
    coverage  = tps_out["coverage"]         # (1,gs,gs)

    warped_b_tps, mask_tps = warp_image(img_b, tps_field)

    # ── Step4: 计算指标 ──
    # 4a: 重叠区掩码（两者共有）
    overlap_direct = mask_direct
    overlap_tps    = mask_tps
    mask_a         = (img_a.sum(1, keepdim=True) > 0).float()

    # 4b: PSNR（重叠区）
    psnr_direct = compute_psnr(img_a, warped_b_direct,
                               mask_a * overlap_direct)
    psnr_tps    = compute_psnr(img_a, warped_b_tps,
                               mask_a * overlap_tps)

    # 4c: 平均光度误差（L1）
    l1_direct = float((torch.abs(img_a - warped_b_direct) *
                        mask_a * overlap_direct).sum() /
                       ((mask_a * overlap_direct).sum() * 3 + 1e-8))
    l1_tps    = float((torch.abs(img_a - warped_b_tps) *
                        mask_a * overlap_tps).sum() /
                       ((mask_a * overlap_tps).sum() * 3 + 1e-8))

    # 4d: TPS field 与 AgriMatcher warp 的差异
    warp_256_tps = tps_field
    warp_diff_mag = (warp_256_tps - warp_256_direct).norm(dim=-1)
    warp_diff_mean = float(warp_diff_mag.mean())

    # 4e: delta_cp 幅度（TPS在AgriMatcher基础上做了多少修正）
    delta_cp_mag  = float(delta_cp.norm(dim=1).mean())

    # 4f: confidence 统计
    conf_mean = float(conf_64.mean())
    conf_std  = float(conf_64.std())

    # 4g: gate 统计
    gate_mean = float(gate.mean())

    # 4h: 重叠面积比例
    overlap_ratio_direct = float(overlap_direct.mean())
    overlap_ratio_tps    = float(overlap_tps.mean())

    # 4i: 特征余弦相似度（变形前 vs 变形后）
    warped_feat_b = F.grid_sample(
        feat_b64,
        F.interpolate(tps_field.permute(0,3,1,2), size=(64,64),
                      mode="bilinear", align_corners=False).permute(0,2,3,1),
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    feat_cos_after  = float(F.cosine_similarity(
        feat_a64, warped_feat_b, dim=1
    ).mean())
    feat_cos_before = float(F.cosine_similarity(
        feat_a64, feat_b64, dim=1
    ).mean())

    return {
        # 图像张量（用于可视化）
        "img_a":           img_a.cpu(),
        "img_b":           img_b.cpu(),
        "warped_b_direct": warped_b_direct.cpu(),
        "warped_b_tps":    warped_b_tps.cpu(),
        "conf_64":         conf_64.cpu(),
        "gate":            gate.cpu(),
        "coverage":        coverage.cpu(),
        "delta_cp":        delta_cp.cpu(),
        "mask_direct":     overlap_direct.cpu(),
        "mask_tps":        overlap_tps.cpu(),
        # 定量指标
        "psnr_direct":          psnr_direct,
        "psnr_tps":             psnr_tps,
        "l1_direct":            l1_direct,
        "l1_tps":               l1_tps,
        "warp_diff_mean":       warp_diff_mean,
        "delta_cp_mag":         delta_cp_mag,
        "conf_mean":            conf_mean,
        "conf_std":             conf_std,
        "gate_mean":            gate_mean,
        "overlap_ratio_direct": overlap_ratio_direct,
        "overlap_ratio_tps":    overlap_ratio_tps,
        "feat_cos_before":      feat_cos_before,
        "feat_cos_after":       feat_cos_after,
    }


# ══════════════════════════════════════════════════════════
# 可视化函数
# ══════════════════════════════════════════════════════════

def visualize_diagnosis(result: Dict, save_path: Path, idx: int):
    """
    生成单个样本的完整诊断可视化图。

    布局：
    行1: img_A | img_B | warped_B(direct) | warped_B(TPS)
    行2: diff(direct) | diff(TPS) | confidence | gate
    行3: 指标文字摘要
    """
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.1)

    def to_np(t):
        return t.squeeze().permute(1,2,0).clamp(0,1).numpy() \
               if t.dim() == 4 else t.squeeze().numpy()

    cmap_heat = plt.cm.RdYlGn

    # ── 行1：图像对比 ──
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(to_np(result["img_a"]))
    ax.set_title("img_A (Reference)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(to_np(result["img_b"]))
    ax.set_title("img_B (Original)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(to_np(result["warped_b_direct"]))
    ax.set_title(
        f"Warped_B (AgriMatcher直接)\n"
        f"PSNR={result['psnr_direct']:.2f}dB  L1={result['l1_direct']:.4f}",
        fontsize=9
    )
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(to_np(result["warped_b_tps"]))
    ax.set_title(
        f"Warped_B (BypassTPS)\n"
        f"PSNR={result['psnr_tps']:.2f}dB  L1={result['l1_tps']:.4f}",
        fontsize=9
    )
    ax.axis("off")

    # ── 行2：差异图和诊断图 ──
    diff_direct = torch.abs(result["img_a"] - result["warped_b_direct"])
    diff_tps    = torch.abs(result["img_a"] - result["warped_b_tps"])
    diff_direct_vis = diff_direct.squeeze().mean(0).numpy()
    diff_tps_vis    = diff_tps.squeeze().mean(0).numpy()
    vmax = max(diff_direct_vis.max(), diff_tps_vis.max(), 0.01)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(diff_direct_vis, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(f"差异图 (AgriMatcher)\n越黑越准", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(diff_tps_vis, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(f"差异图 (BypassTPS)\n越黑越准", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    conf_np = result["conf_64"].squeeze().numpy()
    im = ax.imshow(conf_np, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(
        f"AgriMatcher Confidence\n"
        f"mean={result['conf_mean']:.3f}  std={result['conf_std']:.3f}",
        fontsize=9
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 3])
    gate_np = result["gate"].squeeze().numpy()
    im = ax.imshow(gate_np, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title(
        f"BypassTPS Gate\n"
        f"mean={result['gate_mean']:.3f}  (1=信任投票，0=信任学习)",
        fontsize=9
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── 行3：delta_cp可视化 + 指标摘要 ──
    ax = fig.add_subplot(gs[2, 0])
    delta_mag = result["delta_cp"].norm(dim=1).squeeze().numpy()
    im = ax.imshow(delta_mag, cmap="plasma")
    ax.set_title(
        f"TPS 控制点偏移幅度\n"
        f"mean={result['delta_cp_mag']:.5f}",
        fontsize=9
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # coverage图
    ax = fig.add_subplot(gs[2, 1])
    cov_np = result["coverage"].squeeze().numpy()
    im = ax.imshow(cov_np, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("投票覆盖度\n(低=边缘/假匹配)", fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 指标摘要文字
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis("off")

    # 判断
    psnr_gap = result['psnr_tps'] - result['psnr_direct']
    if result['psnr_direct'] < 18:
        matcher_quality = "❌ 较差（AgriMatcher本身质量不足）"
    elif result['psnr_direct'] < 22:
        matcher_quality = "⚠️  一般（AgriMatcher有改进空间）"
    else:
        matcher_quality = "✅ 良好（AgriMatcher质量足够）"

    if result['delta_cp_mag'] < 0.001:
        tps_activity = "❌ TPS几乎无修正（delta_cp≈0，梯度未到达）"
    elif result['delta_cp_mag'] < 0.01:
        tps_activity = "⚠️  TPS修正量偏小"
    else:
        tps_activity = "✅ TPS有效修正"

    feat_improvement = result['feat_cos_after'] - result['feat_cos_before']
    if feat_improvement > 0.05:
        feat_quality = f"✅ TPS后特征相似度提升 +{feat_improvement:.3f}"
    elif feat_improvement > 0:
        feat_quality = f"⚠️  TPS后特征相似度微弱提升 +{feat_improvement:.3f}"
    else:
        feat_quality = f"❌ TPS后特征相似度未提升 {feat_improvement:.3f}"

    summary = (
        f"【诊断摘要 - 样本 #{idx}】\n\n"
        f"AgriMatcher 直接对齐质量：\n"
        f"  PSNR = {result['psnr_direct']:.2f} dB  →  {matcher_quality}\n\n"
        f"BypassTPS vs AgriMatcher：\n"
        f"  PSNR 差值 = {psnr_gap:+.2f} dB  "
        f"({'TPS改善' if psnr_gap>0 else 'TPS变差'})\n"
        f"  warp_field 差异 = {result['warp_diff_mean']:.5f}\n\n"
        f"TPS 活跃度：\n"
        f"  delta_cp 幅度 = {result['delta_cp_mag']:.5f}  →  {tps_activity}\n\n"
        f"特征对齐质量：\n"
        f"  变形前 cosine = {result['feat_cos_before']:.3f}\n"
        f"  变形后 cosine = {result['feat_cos_after']:.3f}\n"
        f"  →  {feat_quality}\n\n"
        f"Gate 均值：{result['gate_mean']:.3f}  "
        f"(接近1=几乎不用学习结果)"
    )
    ax.text(
        0.05, 0.95, summary,
        transform=ax.transAxes,
        fontsize=9.5, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    fig.suptitle(
        f"AgriMatcher + BypassTPS 诊断报告  #{idx}",
        fontsize=13, fontweight='bold'
    )
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════
# 统计聚合函数
# ══════════════════════════════════════════════════════════

def print_aggregate_report(all_results: list, output_dir: Path):
    """打印并保存统计摘要报告"""

    metrics = [
        "psnr_direct", "psnr_tps", "l1_direct", "l1_tps",
        "warp_diff_mean", "delta_cp_mag", "conf_mean", "gate_mean",
        "overlap_ratio_direct", "feat_cos_before", "feat_cos_after",
    ]

    stats = {}
    for m in metrics:
        vals = [r[m] for r in all_results]
        stats[m] = {
            "mean": np.mean(vals),
            "std":  np.std(vals),
            "min":  np.min(vals),
            "max":  np.max(vals),
        }

    # ── 打印 ──
    sep = "=" * 65
    print(f"\n{sep}")
    print("  AgriMatcher + BypassTPS 整体诊断报告")
    print(f"  样本数: {len(all_results)}")
    print(sep)

    print("\n【1. AgriMatcher 直接对齐质量（上界基准）】")
    print(f"  PSNR(direct):  {stats['psnr_direct']['mean']:.2f} ± "
          f"{stats['psnr_direct']['std']:.2f} dB")
    print(f"  L1(direct):    {stats['l1_direct']['mean']:.4f} ± "
          f"{stats['l1_direct']['std']:.4f}")
    print(f"  重叠面积比例:   {stats['overlap_ratio_direct']['mean']:.2%}")

    psnr_mean = stats['psnr_direct']['mean']
    if psnr_mean < 18:
        verdict_matcher = "❌ AgriMatcher质量不足，是主要瓶颈，建议重新训练Matcher"
    elif psnr_mean < 22:
        verdict_matcher = "⚠️  AgriMatcher质量一般，有改进空间，可尝试更多epoch"
    elif psnr_mean < 28:
        verdict_matcher = "✅ AgriMatcher质量良好，不是主要问题"
    else:
        verdict_matcher = "✅✅ AgriMatcher质量优秀"
    print(f"  结论：{verdict_matcher}")

    print("\n【2. BypassTPS 对齐质量对比】")
    psnr_gap = stats['psnr_tps']['mean'] - stats['psnr_direct']['mean']
    print(f"  PSNR(TPS):     {stats['psnr_tps']['mean']:.2f} ± "
          f"{stats['psnr_tps']['std']:.2f} dB")
    print(f"  PSNR gap:      {psnr_gap:+.2f} dB  "
          f"({'TPS改善Matcher' if psnr_gap > 0.5 else 'TPS无明显改善' if psnr_gap > -0.5 else 'TPS变差'})")
    print(f"  warp差异:       {stats['warp_diff_mean']['mean']:.5f}")

    print("\n【3. TPS活跃度诊断】")
    delta_mean = stats['delta_cp_mag']['mean']
    print(f"  delta_cp幅度:  {delta_mean:.5f}")
    if delta_mean < 0.001:
        verdict_tps = "❌ TPS几乎不动，BypassTPS训练无效，梯度未到达"
    elif delta_mean < 0.005:
        verdict_tps = "⚠️  TPS修正量很小，训练效果有限"
    elif delta_mean < 0.02:
        verdict_tps = "✅ TPS有适度修正"
    else:
        verdict_tps = "⚠️  TPS修正量偏大，注意折叠风险"
    print(f"  结论：{verdict_tps}")

    print("\n【4. Gate诊断（梯度屏蔽检测）】")
    gate_mean = stats['gate_mean']['mean']
    print(f"  Gate均值:      {gate_mean:.3f}")
    if gate_mean > 0.85:
        verdict_gate = "❌ Gate过高（≈完全信任投票），FlowAggregator未生效，梯度被屏蔽"
    elif gate_mean > 0.7:
        verdict_gate = "⚠️  Gate偏高，学习比例偏小"
    elif gate_mean > 0.3:
        verdict_gate = "✅ Gate适中，投票与学习均衡"
    else:
        verdict_gate = "⚠️  Gate偏低，投票初始解被过度覆盖"
    print(f"  结论：{verdict_gate}")

    print("\n【5. 特征对齐诊断】")
    print(f"  变形前cos相似度: {stats['feat_cos_before']['mean']:.3f}")
    print(f"  变形后cos相似度: {stats['feat_cos_after']['mean']:.3f}")
    feat_delta = stats['feat_cos_after']['mean'] - stats['feat_cos_before']['mean']
    print(f"  TPS后提升:      {feat_delta:+.3f}")

    print(f"\n【6. Confidence质量】")
    print(f"  conf均值:       {stats['conf_mean']['mean']:.3f}")
    print(f"  (若conf普遍>0.8 可能存在大量假匹配)")

    print(f"\n{sep}")
    print("【综合诊断结论】")
    issues = []
    if psnr_mean < 20:
        issues.append("主要瓶颈：AgriMatcher质量不足（PSNR<20dB），建议重训Matcher")
    if delta_mean < 0.001:
        issues.append("主要问题：BypassTPS梯度未到达，gate初始化需修复")
    if gate_mean > 0.85:
        issues.append("主要问题：Gate过高（梯度屏蔽），需将gate_init_bias改为0.0")
    if psnr_gap < -1.0:
        issues.append("注意：TPS使对齐变差，说明delta_cp方向错误")
    if feat_delta < 0:
        issues.append("注意：TPS后特征相似度降低，TPS训练方向可能有误")

    if not issues:
        print("  ✅ 各模块指标正常，需要更多训练epoch")
    else:
        for issue in issues:
            print(f"  ❗ {issue}")
    print(sep)

    # 保存为文本文件
    report_path = output_dir / "diagnosis_report.txt"
    lines = []
    for key, s in stats.items():
        lines.append(f"{key}: mean={s['mean']:.6f} std={s['std']:.6f} "
                     f"min={s['min']:.6f} max={s['max']:.6f}")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n详细统计已保存至: {report_path}")


# ══════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════

def build_argparser():
    p = argparse.ArgumentParser("AgriMatcher + BypassTPS 诊断工具")
    p.add_argument("--pairs-file",    type=Path, required=True)
    p.add_argument("--matcher-ckpt",  type=Path, required=True)
    p.add_argument("--bypass-ckpt",   type=Path, default=None,
                   help="若不提供，只诊断AgriMatcher")
    p.add_argument("--output-dir",    type=Path, default=Path("diag_output"))
    p.add_argument("--num-samples",   type=int,  default=50)
    p.add_argument("--image-size",    type=int,  default=256)
    p.add_argument("--d-model",       type=int,  default=128)
    p.add_argument("--teacher-dim",   type=int,  default=None)
    p.add_argument("--grid-size",     type=int,  default=10)
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--device",        type=str,  default="cuda")
    p.add_argument("--save-vis",      action="store_true", default=True)
    p.add_argument("--max-vis",       type=int,  default=20,
                   help="最多保存多少张可视化图（避免磁盘占用过大）")
    return p


def main():
    args   = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(REPO_ROOT))
    from dense_match.network import AgriMatcher, TPSGridGenerator
    from flow_to_tps import BypassTPSEstimator

    # ── 加载 AgriMatcher ──
    ckpt_m  = torch.load(args.matcher_ckpt, map_location=device, weights_only=False)
    sd      = ckpt_m.get("student", ckpt_m.get("model", {}))
    teacher_dim = (args.teacher_dim or
                   int(sd["feature_projector.weight"].shape[0]))
    grid_size_m = int(sd["pos_embed"].shape[1] ** 0.5)

    matcher = AgriMatcher(
        d_model=args.d_model, teacher_dim=teacher_dim, grid_size=grid_size_m,
    ).to(device)
    matcher.load_state_dict(sd, strict=True)
    matcher.eval()
    print(f"[Diag] AgriMatcher loaded: teacher_dim={teacher_dim}, gs={grid_size_m}")

    # ── 加载 BypassTPS（可选）──
    bypass_tps = None
    tps_gen    = None
    if args.bypass_ckpt and args.bypass_ckpt.exists():
        bypass_tps = BypassTPSEstimator(
            grid_size=args.grid_size, feat_channels=args.d_model,
        ).to(device)
        ckpt_b = torch.load(args.bypass_ckpt, map_location=device, weights_only=False)
        bypass_tps.load_state_dict(ckpt_b["model"], strict=True)
        bypass_tps.eval()

        tps_gen = TPSGridGenerator(
            out_h=args.image_size, out_w=args.image_size,
            grid_size=args.grid_size,
        ).to(device)
        print(f"[Diag] BypassTPS loaded from {args.bypass_ckpt}")
    else:
        print("[Diag] No bypass_ckpt provided, only diagnosing AgriMatcher")

    # ── 加载数据 ──
    # 简单读取pairs_file，随机采样
    base_dir = args.pairs_file.parent
    all_pairs = []
    for line in args.pairs_file.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 2:
            continue
        a = Path(parts[0])
        b = Path(parts[1])
        a = (base_dir/a).resolve() if not a.is_absolute() else a
        b = (base_dir/b).resolve() if not b.is_absolute() else b
        all_pairs.append((a, b))

    random.shuffle(all_pairs)
    sample_pairs = all_pairs[:args.num_samples]
    print(f"[Diag] Diagnosing {len(sample_pairs)} pairs...")

    # ── 诊断循环 ──
    all_results = []
    for idx, (path_a, path_b) in enumerate(sample_pairs):
        img_a = load_img_tensor(path_a, args.image_size).unsqueeze(0)
        img_b = load_img_tensor(path_b, args.image_size).unsqueeze(0)

        # 若没有bypass_ckpt，用identity TPS做占位
        if bypass_tps is None:
            # 创建 dummy bypass（只跑AgriMatcher诊断）
            result = _diagnose_matcher_only(
                img_a, img_b, matcher, device, args.image_size
            )
        else:
            result = diagnose_single_pair(
                img_a, img_b, matcher, bypass_tps, tps_gen,
                device, args.image_size,
            )

        all_results.append(result)

        # 保存可视化
        if args.save_vis and idx < args.max_vis:
            vis_path = vis_dir / f"sample_{idx:04d}.png"
            visualize_diagnosis(result, vis_path, idx)

        # 进度
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(sample_pairs)}] "
                  f"psnr_direct={result['psnr_direct']:.2f}  "
                  f"psnr_tps={result['psnr_tps']:.2f}  "
                  f"delta_cp={result['delta_cp_mag']:.5f}  "
                  f"gate={result['gate_mean']:.3f}")

    # ── 打印汇总报告 ──
    print_aggregate_report(all_results, args.output_dir)


@torch.no_grad()
def _diagnose_matcher_only(img_a, img_b, matcher, device, image_size):
    """只诊断AgriMatcher，没有BypassTPS时使用"""
    img_a = img_a.to(device)
    img_b = img_b.to(device)
    H, W  = image_size, image_size

    m_out   = matcher(img_a, img_b)
    warp_64 = m_out["warp_AB"]
    conf_64 = m_out["confidence_AB"]
    feat_a  = m_out["feat_A_64"]
    feat_b  = m_out["feat_B_64"]

    warp_256 = upsample_warp(warp_64, (H, W))
    warped_b, mask = warp_image(img_b, warp_256)
    mask_a   = (img_a.sum(1, keepdim=True) > 0).float()

    psnr    = compute_psnr(img_a, warped_b, mask_a * mask)
    l1      = float((torch.abs(img_a - warped_b) * mask_a * mask).sum() /
                    ((mask_a * mask).sum() * 3 + 1e-8))
    cos_bef = float(F.cosine_similarity(feat_a, feat_b, dim=1).mean())
    warped_feat_b = F.grid_sample(
        feat_b,
        F.interpolate(warp_256.permute(0,3,1,2), size=(64,64),
                      mode="bilinear", align_corners=False).permute(0,2,3,1),
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    cos_aft = float(F.cosine_similarity(feat_a, warped_feat_b, dim=1).mean())

    # dummy TPS结果（与direct相同）
    return {
        "img_a": img_a.cpu(), "img_b": img_b.cpu(),
        "warped_b_direct": warped_b.cpu(), "warped_b_tps": warped_b.cpu(),
        "conf_64": conf_64.cpu(),
        "gate": torch.ones(1,1,10,10) * 0.5,
        "coverage": torch.ones(1,10,10) * 0.5,
        "delta_cp": torch.zeros(1,2,10,10),
        "mask_direct": mask.cpu(), "mask_tps": mask.cpu(),
        "psnr_direct": psnr, "psnr_tps": psnr,
        "l1_direct": l1, "l1_tps": l1,
        "warp_diff_mean": 0.0, "delta_cp_mag": 0.0,
        "conf_mean": float(conf_64.mean()), "conf_std": float(conf_64.std()),
        "gate_mean": 0.5, "overlap_ratio_direct": float(mask.mean()),
        "overlap_ratio_tps": float(mask.mean()),
        "feat_cos_before": cos_bef, "feat_cos_after": cos_aft,
    }


if __name__ == "__main__":
    main()