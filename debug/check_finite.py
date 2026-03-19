import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    args = ap.parse_args()

    from dense_match.network import AgriMatcher

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = AgriMatcher().to(device).train()

    x = torch.randn(1, 3, args.h, args.w, device=device)
    y = torch.randn(1, 3, args.h, args.w, device=device)

    use_amp = bool(args.amp and device.type == "cuda")
    with torch.amp.autocast("cuda", enabled=use_amp):
        out = m(x, y)

    def fin(name: str, t: torch.Tensor):
        ok = bool(torch.isfinite(t).all().item())
        print(name, ok, tuple(t.shape), t.dtype)

    fin("warp_AB", out["warp_AB"])
    fin("confidence_AB", out["confidence_AB"])
    fin("sim_matrix", out["sim_matrix"])


if __name__ == "__main__":
    main()

