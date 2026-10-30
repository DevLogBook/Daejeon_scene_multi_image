"""Continue GRM matching training with GT-F-filtered SIFT/Teacher pseudo labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import GRMMatcherConfig
from .imc_dataset import IMCPairDataset
from .matcher import GRMMatcher
from .real_adaptation import GRMStage25BCoarseAndRefinementLoss
from .stage25 import GRMStage25Config, GRMStage25ConsistencyLoss, make_ema_teacher
from .train_stage25 import (
    _atomic_torch_save, _checkpoint_state, _configure_real_phase,
    _make_optimizer, _run_epoch, _seed_everything,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("dataset/imc2022_grm"))
    p.add_argument("--output-dir", type=Path, default=Path("training_runs/grm_imc"))
    checkpoint = p.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--init-checkpoint", type=Path)
    checkpoint.add_argument("--resume", type=Path)
    p.add_argument("--scope", choices=["fine", "mid_fine", "coarse_refine"], default="coarse_refine")
    p.add_argument("--label-source", choices=["sift", "teacher"], default="sift")
    p.add_argument("--stage25-config", type=Path)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--coarse-learning-rate", type=float, default=3e-6)
    p.add_argument("--epipolar-threshold-px", type=float, default=2.)
    p.add_argument("--max-empty-batches", type=int, default=20)
    p.add_argument("--min-val-supervised-fraction", type=float, default=.25)
    p.add_argument("--limit-train", type=int, default=0, help="Smoke tests only; zero uses full split")
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--cache-only", action="store_true", help="Build SIFT cache for train/val without training")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def data_signature(folder):
    digest = hashlib.sha256()
    for name in ("splits.json", "train/manifest.jsonl", "val/manifest.jsonl"):
        digest.update((folder / name).read_bytes())
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if min(args.epochs, args.batch_size, args.max_empty_batches) <= 0 or args.num_workers < 0:
        raise ValueError("Invalid epochs, batch size, empty-batch limit or worker count")
    if min(args.learning_rate, args.coarse_learning_rate, args.epipolar_threshold_px) <= 0:
        raise ValueError("Learning rates and epipolar threshold must be positive")
    if min(args.limit_train, args.limit_val) < 0 or not 0 < args.min_val_supervised_fraction <= 1:
        raise ValueError("Invalid sample limits or validation coverage threshold")
    _seed_everything(args.seed)
    # Avoid severe CPU oversubscription on small-batch local matching.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw = torch.load(args.resume or args.init_checkpoint, map_location="cpu", weights_only=False)
    config = GRMMatcherConfig.from_dict(raw["config"])
    config.use_amp = False  # Existing GRM FP16 path is not validated on the local GTX 1650 Ti.
    phase = "stage2.5b" if args.scope == "coarse_refine" else "stage2.5a"
    stage_values = json.loads(args.stage25_config.read_text(encoding="utf-8")) if args.stage25_config else {}
    adaptation = GRMStage25Config(**stage_values)
    adaptation.homography_loss_weight = adaptation.source_replay_weight = 0.
    dataset_args = dict(label_source=args.label_source, epipolar_threshold_px=args.epipolar_threshold_px)
    train = IMCPairDataset(args.data_dir, "train", config, limit=args.limit_train, **dataset_args)
    val = IMCPairDataset(args.data_dir, "val", config, limit=args.limit_val, **dataset_args)
    # Check manifests rather than trusting a declared split name.
    scenes_a, scenes_b = {r["scene"] for r in train.records}, {r["scene"] for r in val.records}
    if scenes_a & scenes_b:
        raise ValueError("IMC train/validation scenes overlap")
    contract = {"data_signature": data_signature(args.data_dir), "scope": args.scope, "label_source": args.label_source,
                "batch_size": args.batch_size, "seed": args.seed,
                "limit_train": args.limit_train, "limit_val": args.limit_val,
                "epipolar_threshold_px": args.epipolar_threshold_px,
                "min_val_supervised_fraction": args.min_val_supervised_fraction,
                "learning_rate": args.learning_rate, "coarse_learning_rate": args.coarse_learning_rate,
                "stage25_config": adaptation.to_dict()}
    if args.resume and raw.get("imc_contract") != contract:
        raise ValueError("Resume data/config differs from checkpoint; use the original options or --init-checkpoint for a new run")
    student = GRMMatcher(config).to(device)
    student.load_state_dict(raw["student"] if args.resume else raw["model"], strict=True)
    _configure_real_phase(student, phase, "fine" if args.scope == "fine" else "mid_fine")
    teacher = make_ema_teacher(student)
    if args.resume:
        teacher.load_state_dict(raw["teacher"], strict=True)
    optimizer = _make_optimizer(student, phase, args.learning_rate, args.coarse_learning_rate, 1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = GRMStage25BCoarseAndRefinementLoss(config, adaptation) if phase == "stage2.5b" else GRMStage25ConsistencyLoss(config, adaptation)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            generator=torch.Generator().manual_seed(args.seed + 1))
    start, step, best = 0, 0, math.inf
    if args.resume:
        optimizer.load_state_dict(raw["optimizer"])
        start, step, best = int(raw["epoch"]) + 1, int(raw["global_step"]), float(raw["best_validation_loss"])
        rng = raw["rng"]
        torch.set_rng_state(rng["torch"])
        generator.set_state(rng["loader"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        if device.type == "cuda" and rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    summary = {"train_pairs": len(train), "val_pairs": len(val), "canvas": [train.height, train.width],
               "device": str(device), "scope": args.scope, "teacher": "fixed", "amp": False,
               "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
               "supervision": f"GT-F filtered {args.label_source} pseudo correspondence; not dense GT"}
    print(json.dumps(summary, indent=2), flush=True)
    if args.cache_only:
        if args.label_source != "sift":
            raise ValueError("--cache-only requires --label-source sift")
        from tqdm import tqdm
        for name, dataset in (("train", train), ("val", val)):
            counts = [int(dataset[i]["sparse_valid"].sum()) for i in tqdm(range(len(dataset)), desc=name)]
            print(json.dumps({"split": name, "pairs": len(counts), "mean_matches": float(np.mean(counts)),
                              "pairs_ge_8": sum(n >= 8 for n in counts)}), flush=True)
        return
    if args.audit_only:
        for dataset in (train, val):
            batch = dataset[0]
            assert torch.isfinite(batch["image_a"]).all() and torch.isfinite(batch["F_ab"]).all()
            print("READ_OK", batch["pair_id"], tuple(batch["image_a"].shape),
                  "sparse_matches", int(batch["sparse_valid"].sum()) if "sparse_valid" in batch else "teacher", flush=True)
        return
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory for a new run (or --resume)")
    if start >= args.epochs:
        raise ValueError("--epochs is the total target epoch count; no epochs remain")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run.json").write_text(json.dumps({**summary, **contract,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2), encoding="utf-8")
    for epoch in range(start, args.epochs):
        common = dict(student=student, teacher=teacher, loss_module=loss, optimizer=optimizer,
            scaler=scaler, device=device, matcher_config=config, stage25_config=adaptation,
            phase=phase, amp_enabled=False, gradient_clip=1., max_empty_batches=args.max_empty_batches,
            output_dir=args.output_dir, epoch=epoch, writer=None, log_every=10,
            progress_enabled=args.progress, teacher_mode="fixed", epipolar_threshold_px=args.epipolar_threshold_px)
        train_metrics, step = _run_epoch(loader=train_loader, train=True, global_step=step, **common)
        # Fix stochastic validation augmentation while preserving the training RNG stream.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed + 999)
            val_metrics, _ = _run_epoch(loader=val_loader, train=False, global_step=0, **common)
        coverage_ok = val_metrics["supervised_batch_fraction"] >= args.min_val_supervised_fraction
        score = val_metrics["supervised_loss"]
        improved = coverage_ok and math.isfinite(score) and score < best
        if improved:
            best = score
        state = _checkpoint_state(student, teacher, optimizer, scaler, config, adaptation,
            phase, epoch, step, best, train_metrics, val_metrics, "fixed",
            "fine" if args.scope == "fine" else "mid_fine")
        state["imc_contract"] = contract
        state["rng"] = {"torch": torch.get_rng_state(), "loader": generator.get_state(),
            "python": random.getstate(), "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        _atomic_torch_save(state, args.output_dir / "latest.pt")
        if improved:
            _atomic_torch_save(state, args.output_dir / "best_imc.pt")
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics,
                  "best_updated": improved, "validation_coverage_ok": coverage_ok}
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if train_metrics["optimizer_updated"] == 0 or not coverage_ok:
            raise RuntimeError("No training updates or insufficient validation supervision; inspect metrics.jsonl. No usable best model is guaranteed.")


if __name__ == "__main__":
    main()
