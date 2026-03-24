import argparse
import torch


def _fmt(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor{tuple(x.shape)} {x.dtype} {x.device}"
    return repr(x)


def main():
    from dense_match.network import AgriMatcher, AgriTPSStitcher

    torch.manual_seed(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--b", type=int, default=1)
    args = ap.parse_args()

    # 你可以用不同分辨率复现维度问题，例如：
    # python debug_shape.py --h 384 --w 512
    B, C, H, W = args.b, 3, args.h, args.w
    img_a = torch.randn(B, C, H, W)
    img_b = torch.randn(B, C, H, W)

    matcher = AgriMatcher(d_model=128, teacher_dim=100, grid_size=32).eval()
    with torch.no_grad():
        out = matcher(img_a, img_b)
    print("== AgriMatcher outputs ==")
    for k in [
        "warp_AB",
        "confidence_AB",
        "warp_AB_coarse",
        "confidence_AB_coarse",
        "sim_matrix",
        "distill_feat_A",
        "distill_feat_B",
        "feat_A_64",
        "feat_B_64",
        "coarse_hw",
        "fine_hw",
    ]:
        if k in out:
            print(f"{k}: {_fmt(out[k])}")

    stitcher = AgriTPSStitcher(
        matcher_config=dict(d_model=128, teacher_dim=100, grid_size=32),
        tps_config=dict(
            grid_size=10,
            feat_channels=128,
        ),
    ).eval()
    with torch.no_grad():
        out2 = stitcher(img_a, img_b)
    print("\n== AgriTPSStitcher outputs ==")
    for k in ["dense_grid", "delta_cp"]:
        if k in out2:
            print(f"{k}: {_fmt(out2[k])}")


if __name__ == "__main__":
    main()

