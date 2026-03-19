import argparse
import torch


def main():
    from dense_match.network import AgriTPSStitcher, DistillationLoss
    from dense_match.train import (
        compute_photometric_loss,
        extract_teacher_features_ds,
        make_teacher_inputs,
    )
    from romav2 import RoMaV2

    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--teacher-setting", type=str, default="turbo")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (device.type == "cuda") and False

    teacher = RoMaV2(RoMaV2.Cfg(setting=args.teacher_setting)).to(device).eval()
    if device.type == "cpu":
        # Some backbones load bf16 weights; force fp32 for CPU execution.
        teacher = teacher.float()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 512, 512, device=device)
        img_a_lr, _, _, _ = make_teacher_inputs(dummy, dummy, teacher)
        feat_a_flat, _ = extract_teacher_features_ds(teacher, img_a_lr, img_a_lr, 32)
        teacher_dim = feat_a_flat.shape[-1]

    student = AgriTPSStitcher(
        matcher_config={"d_model": 128, "teacher_dim": teacher_dim, "grid_size": 32},
        tps_config={"grid_size": 10, "feat_channels": 128},
    ).to(device)
    student.train()
    if device.type == "cpu":
        student = student.float()

    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    loss_fn = DistillationLoss().to(device)

    for it in range(2):
        img_a = torch.randn(1, 3, args.h, args.w, device=device)
        img_b = torch.randn(1, 3, args.h, args.w, device=device)

        with torch.no_grad():
            img_a_lr, img_b_lr, img_a_hr, img_b_hr = make_teacher_inputs(img_a, img_b, teacher)
            t_out = teacher(img_a_lr, img_b_lr, img_a_hr, img_b_hr)
            t_feat_a, t_feat_b = extract_teacher_features_ds(teacher, img_a_lr, img_b_lr, 32)

        with torch.amp.autocast("cuda", enabled=use_amp):
            stu_out = student(img_a, img_b)

            teacher_conf = t_out["confidence_AB"]
            if isinstance(teacher_conf, torch.Tensor) and teacher_conf.ndim == 4 and teacher_conf.shape[-1] >= 1:
                teacher_conf = teacher_conf[..., 0]

            d = loss_fn(
                stu_output=stu_out["matcher_out"],
                teacher_warp=t_out["warp_AB"],
                teacher_conf=teacher_conf,
                teacher_feat_A=t_feat_a,
                teacher_feat_B=t_feat_b,
            )
            p = compute_photometric_loss(
                img_a, img_b, stu_out["dense_grid"], stu_out["matcher_out"]["confidence_AB"]
            )
            f = stu_out["tps_out"].get("fold_loss", torch.tensor(0.0, device=device))
            total = d["total"] + p + f

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        print(f"iter={it} total={float(total.detach().cpu()):.6f}")

    print("done")


if __name__ == "__main__":
    main()

