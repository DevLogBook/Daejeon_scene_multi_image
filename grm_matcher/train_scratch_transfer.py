"""Random-init IMC pretraining and supervised agricultural homography transfer.

Unlike train_imc/train_stage25, this entry point never constructs a Teacher.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import GRMMatcherConfig
from .imc_dataset import IMCPairDataset
from .matcher import GRMMatcher
from .architecture_upgrade import v2_config, load_v1_weights
from .scratch_transfer import (
    ResizedHomographyDataset, ScratchTransferObjective, check_homography_splits,
    configure_optimizer, lr_multiplier,
)
from .train_stage25 import _atomic_torch_save, _seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("imc", "agri"), required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--init-checkpoint", type=Path)
    group.add_argument("--resume", type=Path)
    group.add_argument("--scratch", action="store_true", help="Agriculture only: random initialization using exact synthetic H labels")
    p.add_argument("--config", type=Path, help="Architecture overrides for new random-init IMC or agriculture --scratch runs")
    p.add_argument("--require-architecture", choices=("v1", "v2", "v3"),
                   help="Fail before loading data if checkpoint/config architecture differs")
    p.add_argument("--upgrade-v2", action="store_true",
                   help="Start a new agricultural run from a v1 IMC/agri checkpoint; reset optimizer and schedule")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--imc-lr", type=float, default=2e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--coarse-lr", type=float, default=2e-5)
    p.add_argument("--refinement-lr", type=float, default=5e-5)
    p.add_argument("--geometry-lr", type=float, default=1e-4)
    p.add_argument("--min-lr-ratio", type=float, default=.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.)
    p.add_argument("--epipolar-weight", type=float, default=.1)
    p.add_argument("--label-source", choices=("sift", "roma"), default="sift")
    p.add_argument("--teacher-cache", type=Path)
    p.add_argument("--imc-sampling", choices=("uniform", "scene_curriculum"), default="uniform")
    p.add_argument("--curriculum-epochs", type=int, default=10)
    p.add_argument("--epipolar-threshold-px", type=float, default=2.)
    p.add_argument("--anchor-weight", type=float, default=.25)
    p.add_argument("--agri-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default=None)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--max-empty-batches", type=int, default=100)
    p.add_argument("--min-val-supervised-fraction", type=float, default=.25)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=20,
                   help="Progress interval in microbatches, for both training and validation; 0 logs only first/last")
    p.add_argument("--stop-after-epochs", type=int, default=0,
                   help="Run only this many additional epochs, preserving the planned LR schedule (smoke tests)")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument('--synthetic-robustness', action='store_true', help='V3 online exact dense Flow augmentation stage')
    p.add_argument('--augment-config', type=Path, help='JSON overrides for RobustAugmentConfig')
    args = p.parse_args(argv)
    if args.synthetic_robustness and args.stage != 'agri':
        p.error('--synthetic-robustness is agriculture only')
    if args.augment_config and not args.synthetic_robustness:
        p.error('--augment-config requires --synthetic-robustness')
    if args.curriculum_epochs < 1:
        p.error("--curriculum-epochs must be positive")
    if args.stage != "imc" and args.imc_sampling != "uniform":
        p.error("--imc-sampling only applies to IMC pretraining")
    if args.stage == "imc" and args.label_source == "roma" and not args.teacher_cache:
        p.error("--label-source roma requires --teacher-cache")
    args.epochs = args.epochs if args.epochs is not None else (30 if args.stage == "imc" else 20)
    args.warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else (2 if args.stage == "imc" else 1)
    if args.stage == "imc" and args.init_checkpoint:
        p.error("IMC starts randomly: --init-checkpoint is prohibited; use --resume only for the same scratch run")
    if args.scratch and args.stage != "agri":
        p.error("--scratch applies to agriculture; IMC already initializes randomly")
    if args.stage == "agri" and not (args.init_checkpoint or args.resume or args.scratch):
        p.error("Agriculture requires --scratch, --init-checkpoint or --resume")
    if args.config and (args.resume or args.init_checkpoint or (args.stage != "imc" and not args.scratch)):
        p.error("--config is only for new random initialization")
    if args.upgrade_v2 and (args.stage != "agri" or not args.init_checkpoint or args.resume):
        p.error("--upgrade-v2 requires --stage agri --init-checkpoint and a new output directory")
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.max_empty_batches) <= 0:
        p.error("Epochs, batch/accumulation and empty-batch limit must be positive")
    if min(args.num_workers, args.limit_train, args.limit_val, args.save_every, args.stop_after_epochs, args.log_every) < 0:
        p.error("Workers, limits and intervals cannot be negative")
    if not 0 <= args.warmup_epochs < args.epochs or not 0 < args.min_lr_ratio <= 1:
        p.error("Warmup must be shorter than the full schedule; min LR ratio must be in (0,1]")
    if min(args.imc_lr, args.backbone_lr, args.coarse_lr, args.refinement_lr, args.geometry_lr,
           args.gradient_clip, args.epipolar_threshold_px) <= 0:
        p.error("Learning rates, clipping and F threshold must be positive")
    if min(args.weight_decay, args.epipolar_weight, args.anchor_weight) < 0:
        p.error("Loss weights and decay must be nonnegative")
    if not 0 < args.min_val_supervised_fraction <= 1:
        p.error("Invalid validation coverage threshold")
    return args


def dataset_signature(root, stage):
    digest = hashlib.sha256()
    names = ["train/manifest.jsonl", "val/manifest.jsonl"]
    if stage == "imc":
        names += ["splits.json"]
    for name in names:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def rng_state(generator):
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    generator.set_state(state["loader"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def monitor(args, message, event=None):
    """Flush human-readable output to stdout and persist structured progress."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    root = getattr(args, "output_dir", None)
    if root is not None and Path(root).is_dir():
        with (Path(root) / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if event is not None:
            with (Path(root) / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp": timestamp, **event}) + "\n")


def metric_text(metrics):
    names = ("loss", "coarse", "refinement", "epipolar", "mid", "fine", "anchor",
             "rerank", "role", "inlier", "matchability", "calibration", "confidence",
             "confidence_brier", "cycle", "overlap", "valid_count", "reachable_fraction",
             "coarse_initial_error_mean", "flow_fraction", "augmentation_strength")
    parts = [f"{key}={metrics[key]:.4f}" for key in names if key in metrics]
    if "pck3_predicted" in metrics:
        parts.append(f"candidate_PCK3={metrics['pck3_predicted']:.2%}")
    if "inlier_tp" in metrics:
        tp, fp, fn = (metrics[key] for key in ("inlier_tp", "inlier_fp", "inlier_fn"))
        parts.append(f"inlier_P@0.5={tp / max(tp + fp, 1e-8):.2%}")
        parts.append(f"inlier_R@0.5={tp / max(tp + fn, 1e-8):.2%}")
    return " ".join(parts)


def append_epoch_csv(path, record):
    row = {"epoch": record["epoch"], "best_updated": record["best_updated"],
           "best_validation_loss": record["best_validation_loss"]}
    for group in ("train", "validation", "lr"):
        row.update({f"{group}/{key}": value for key, value in record[group].items()})
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(model, objective, loader, optimizer, args, device, epoch,
              schedule_step, total_steps, warmup_steps, *, train):
    model.train(train)
    optimizer.zero_grad(set_to_none=True)
    totals, active, empty, pending, updates = {}, 0, 0, 0, 0
    batches = 0
    start = time.monotonic()
    phase = "TRAIN" if train else "VAL"
    last_norm = None
    monitor(args, f"[{phase}] epoch={epoch + 1}/{getattr(args, 'epochs', '?')} START batches={len(loader)} (waiting for data)")
    for index, cpu_batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                 for key, value in cpu_batch.items()}
        with torch.set_grad_enabled(train):
            loss, metrics = (objective.imc(model, batch, train) if args.stage == "imc"
                             else objective.agri(model, batch, train, epoch))
        batches += 1
        if loss is None:
            empty += 1
            if train and empty >= args.max_empty_batches:
                monitor(args, f"[ERROR] {empty} consecutive batches without supervision at batch={batches}")
                raise RuntimeError("Too many consecutive batches without reliable supervision; inspect SIFT/cache/data rather than relaxing GT checks")
        else:
            empty = 0
            if not torch.isfinite(loss):
                values = {key: float(value) for key, value in metrics.items()}
                bad = [key for key, value in values.items() if not math.isfinite(value)]
                event = {"phase": "nonfinite_loss", "stage": args.stage,
                         "epoch": epoch + 1, "batch": batches, "optimizer_step": schedule_step,
                         "pair_id": cpu_batch.get("pair_id"), "nonfinite_components": bad,
                         "metrics": {key: value if math.isfinite(value) else str(value)
                                     for key, value in values.items()}}
                monitor(args, "[NONFINITE] " + json.dumps(event), event)
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1} batch={batches}; "
                    f"components={bad}; pairs={cpu_batch.get('pair_id')}; see progress.jsonl")
            active += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + float(value)
            if train:
                loss.backward()
                pending += 1
        boundary = (index + 1) % args.accumulation_steps == 0 or index + 1 == len(loader)
        if train and boundary and pending:
            # Average only real supervised microbatches, including partial last windows.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
            last_norm = float(norm)
            factor = lr_multiplier(schedule_step, total_steps, warmup_steps, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * factor
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            schedule_step += 1
            updates += 1
        if batches == 1 or batches == len(loader) or (args.log_every > 0 and batches % args.log_every == 0):
            elapsed = time.monotonic() - start
            averages = {key: value / active for key, value in totals.items()} if active else {}
            if args.stage == "agri" and totals.get("predicted_matches", 0.) > 0:
                averages["pck3_predicted"] = totals["correct_at_3px"] / totals["predicted_matches"]
            # No LR has been used yet before the first optimizer update.
            lrs = {g["name"]: g["lr"] for g in optimizer.param_groups} if schedule_step else {}
            eta = elapsed / batches * (len(loader) - batches)
            gpu = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
            event = {"stage": args.stage, "phase": phase.lower(), "epoch": epoch + 1,
                "batch": batches, "total_batches": len(loader), "optimizer_step": schedule_step,
                "averages": averages, "skipped_batches": batches - active,
                "supervised_batch_fraction": active / batches, "lr": lrs,
                "gradient_norm": last_norm, "seconds": elapsed, "eta_seconds": eta,
                "seconds_per_batch": elapsed / batches, "gpu_peak_gib": gpu}
            lr_text = ",".join(f"{k}:{v:.2e}" for k, v in lrs.items()) or "pending-first-update"
            extras = f" grad_norm={last_norm:.3f}" if last_norm is not None else ""
            extras += f" gpu_peak={gpu:.2f}GiB" if gpu is not None else ""
            monitor(args, f"[{phase}] epoch={epoch + 1}/{getattr(args, 'epochs', '?')} "
                f"batch={batches}/{len(loader)} ({batches / len(loader):.1%}) step={schedule_step} "
                f"avg[{metric_text(averages)}] supervised={active / batches:.1%} skipped={batches - active} "
                f"lr={lr_text}{extras} sec/batch={elapsed / batches:.2f} ETA={eta / 60:.1f}min", event)
    if not batches:
        raise RuntimeError("Empty DataLoader")
    if not active or (train and updates == 0):
        raise RuntimeError("Epoch had no effective training/validation supervision")
    means = {key: value / active for key, value in totals.items()}
    means.update(supervised_batch_fraction=active / batches, optimizer_updates=updates,
                 batches=batches, seconds=time.monotonic() - start)
    if args.stage == "agri":
        means["pck3_predicted"] = totals.get("correct_at_3px", 0.) / max(1., totals.get("predicted_matches", 0.))
        if "inlier_tp" in totals:
            tp, fp, fn = (totals[key] for key in ("inlier_tp", "inlier_fp", "inlier_fn"))
            means["inlier_precision_at_05"] = tp / max(1., tp + fp)
            means["inlier_recall_at_05"] = tp / max(1., tp + fn)
    return means, schedule_step


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw = torch.load(args.resume or args.init_checkpoint, map_location="cpu", weights_only=False) if (args.resume or args.init_checkpoint) else None
    if raw:
        if raw.get("training_recipe") not in ('scratch_imc_exact_h_v1', 'v3_exact_dense_robustness_v1'):
            raise ValueError("Checkpoint is not from this scratch-pretraining recipe")
        expected_recipe = 'v3_exact_dense_robustness_v1' if args.synthetic_robustness else 'scratch_imc_exact_h_v1'
        if args.resume and raw['training_recipe'] != expected_recipe:
            raise ValueError('Resume must keep the same supervision recipe; use --init-checkpoint for a new stage')
        if raw['training_recipe'] == 'v3_exact_dense_robustness_v1' and not args.synthetic_robustness:
            raise ValueError('Use the robustness entry point for this checkpoint')
        if args.resume and raw["stage"] != args.stage:
            raise ValueError("Resume cannot switch stages; use --init-checkpoint")
        if args.init_checkpoint and raw["stage"] != "imc" and not ((args.upgrade_v2 or args.synthetic_robustness) and raw["stage"] == "agri"):
            raise ValueError("Agricultural initialization must come from the scratch IMC stage")
        config = GRMMatcherConfig.from_dict(raw["config"])
        if args.upgrade_v2:
            if config.architecture_version != "v1":
                raise ValueError("--upgrade-v2 is only for v1 source checkpoints")
            config = v2_config(config)
    else:
        values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        # Architecture constructors use random initialization only; no pretrained download.
        config = GRMMatcherConfig.from_dict(values)
    if args.require_architecture is not None and config.architecture_version != args.require_architecture:
        raise ValueError(f"Expected {args.require_architecture}, received {config.architecture_version}; do not silently mix architectures")
    config.use_amp = False
    if args.synthetic_robustness and config.architecture_version != 'v3':
        raise ValueError('Exact dense robustness augmentation currently supports V3 only')
    if args.stage == "imc":
        train_data = IMCPairDataset(args.data_dir, "train", config, limit=args.limit_train,
                                    label_source=args.label_source, teacher_cache=args.teacher_cache, epipolar_threshold_px=args.epipolar_threshold_px)
        val_data = IMCPairDataset(args.data_dir, "val", config, limit=args.limit_val,
                                  label_source=args.label_source, teacher_cache=args.teacher_cache, epipolar_threshold_px=args.epipolar_threshold_px)
        if args.label_source == "roma":
            from .teacher_cache import pair_key
            teacher_meta = json.loads((args.teacher_cache / "teacher.json").read_text(encoding="utf-8"))
            if teacher_meta.get("teacher") != "RoMaV2":
                raise ValueError("Expected versioned RoMaV2 cache")
            for name, expected in teacher_meta["inputs"].items():
                actual = hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f"Teacher cache dataset mismatch: {name}")
            for dataset in (train_data, val_data):
                missing = [r["pair_id"] for r in dataset.records if not (args.teacher_cache / (pair_key(r,dataset.height,dataset.width)+".npz")).is_file()]
                if missing:
                    raise FileNotFoundError(f"Generate RoMa labels first: {len(missing)} missing; example {missing[0]}")
        metadata = json.loads((args.data_dir / "splits.json").read_text(encoding="utf-8"))
        train_scenes, val_scenes, test_scenes = [set(metadata[f"{s}_scenes"]) for s in ("train", "val", "test")]
        if train_scenes & val_scenes or train_scenes & test_scenes or val_scenes & test_scenes:
            raise ValueError("Scene split overlap")
        if not {r["scene"] for r in train_data.records} <= train_scenes or not {r["scene"] for r in val_data.records} <= val_scenes:
            raise ValueError("Manifest scene assignments contradict split metadata")
    else:
        metadata = check_homography_splits(args.data_dir)
        train_data = ResizedHomographyDataset(args.data_dir, "train", config, args.agri_size, args.limit_train)
        val_data = ResizedHomographyDataset(args.data_dir, "val", config, args.agri_size, args.limit_val)
    augmentation = None
    if args.synthetic_robustness:
        from .robust_synthetic import RobustAugmentConfig, RobustSyntheticDataset
        values = json.loads(args.augment_config.read_text(encoding='utf-8')) if args.augment_config else {}
        values.setdefault('seed',args.seed)
        augmentation = RobustAugmentConfig(**values)
        train_data = RobustSyntheticDataset(train_data,config,augmentation,train=True)
        val_data = RobustSyntheticDataset(val_data,config,augmentation,train=False)
    if args.cache_only:
        if args.stage != "imc":
            raise ValueError("--cache-only is only for IMC")
        for name, dataset in (("train", train_data), ("val", val_data)):
            counts = []
            for i in range(len(dataset)):
                counts.append(int(dataset[i]["sparse_valid"].sum()))
                if (i + 1) % 100 == 0:
                    print(name, i + 1, "/", len(dataset), flush=True)
            print(json.dumps({"split": name, "pairs": len(counts), "mean_matches": float(np.mean(counts)),
                              "pairs_ge_8": sum(c >= 8 for c in counts)}), flush=True)
        return
    model = GRMMatcher(config).to(device)
    if raw:
        if args.upgrade_v2:
            load_v1_weights(model, raw["model"])
        else:
            model.load_state_dict(raw["model"], strict=True)
    optimizer = configure_optimizer(model, args.stage, imc_lr=args.imc_lr, backbone_lr=args.backbone_lr,
        coarse_lr=args.coarse_lr, refinement_lr=args.refinement_lr, geometry_lr=args.geometry_lr,
        weight_decay=args.weight_decay)
    # During supervised H training, do not let confidence filtering hide all
    # candidates from the newly initialized geometry heads. Evaluation uses
    # the same fixed training sampler; deployed config is not overwritten.
    if args.stage == "agri" and config.architecture_version == "v1":
        sampler = model.candidate_sampler.config
        sampler.fixed_score_threshold = 0.
        sampler.use_quantile_threshold = False
        sampler.use_dynamic_query_budget = False
        sampler.allow_underfilled_budget = False
    objective = ScratchTransferObjective(config, args.epipolar_weight, args.anchor_weight,
                                         exact_dense=augmentation is not None)
    generator = torch.Generator().manual_seed(args.seed)
    curriculum = None
    if args.stage == "imc" and args.imc_sampling == "scene_curriculum":
        from .imc_sampling import IMCCurriculumSampler
        curriculum = IMCCurriculumSampler(train_data.records, args.seed, args.curriculum_epochs)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=curriculum is None,
                              sampler=curriculum, generator=generator,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed + 1), num_workers=args.num_workers)
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_steps, warmup_steps = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    contract_keys = ("stage", "epochs", "warmup_epochs", "batch_size", "accumulation_steps", "imc_lr",
        "backbone_lr", "coarse_lr", "refinement_lr", "geometry_lr", "min_lr_ratio", "weight_decay",
        "gradient_clip", "epipolar_weight", "epipolar_threshold_px", "anchor_weight", "agri_size",
        "seed", "limit_train", "limit_val", "min_val_supervised_fraction")
    contract = {key: getattr(args, key) for key in contract_keys}
    contract.update(data_signature=dataset_signature(args.data_dir, args.stage), total_steps=total_steps)
    if augmentation is not None:
        contract.update(synthetic_robustness=augmentation.to_dict(), augmentation_recipe='v3_exact_dense_v1')
        signature=hashlib.sha256()
        for filename in ('robust_synthetic.py','training_v3.py','scratch_transfer.py'):
            signature.update(Path(__file__).with_name(filename).read_bytes())
        contract['robustness_code_signature']=signature.hexdigest()
    # Leave the legacy default contract unchanged for existing run resumes.
    if curriculum is not None:
        contract.update(imc_sampling="scene_curriculum_v1", curriculum_epochs=args.curriculum_epochs)
    if args.stage == "imc" and args.label_source == "roma":
        contract.update(label_source="roma", teacher_signature=hashlib.sha256((args.teacher_cache/"teacher.json").read_bytes()).hexdigest())
    summary = {"stage": args.stage, "architecture_version": config.architecture_version,
        "initialization": "resume" if args.resume else ("v1 weights upgraded to v2" if args.upgrade_v2 else ("IMC checkpoint" if raw else "random; no pretrained weights")),
        "train_pairs": len(train_data), "val_pairs": len(val_data), "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs, "effective_batch_size": args.batch_size * args.accumulation_steps,
        "precision": "FP32", "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_lrs": {g["name"]: g["peak_lr"] for g in optimizer.param_groups},
        "classification_mode": config.classification_mode,
        "geometry_supervision": ("correspondence and candidate-ranking supervision; no classification" if config.classification_mode == "none" else "none in IMC; GT inlier/outlier in agriculture" if config.classification_mode == "inlier"
                                 else "none in IMC; Global/False only for pure-H agriculture")}
    if config.architecture_version == "v3":
        summary.update(v3_revision=config.v3_revision,
            geometry_supervision="no graph, internal H, reranker or Role head",
            confidence_head=config.v3_confidence_head,
            confidence_supervision="known-pair error <= 3 input pixels; score only; no hard filtering",
            v3_loss_weights={"coarse": 1., "anchor": args.anchor_weight, "predicted_refinement": 1.,
                "confidence": config.v3_confidence_weight, "cycle": .05, "overlap": .1,
                "epipolar": args.epipolar_weight if args.stage == "imc" else 0.})
    if augmentation is not None:
        summary.update(initialization='same-V3 checkpoint; new optimizer' if args.init_checkpoint else summary['initialization'],
                       supervision='exact bidirectional dense maps, including target local deformation; no Teacher',
                       synthetic_robustness=augmentation.to_dict(),
                       validation='fixed augmented validation; original source split preserved')
    print(json.dumps(summary, indent=2), flush=True)
    if args.audit_only:
        for dataset in (train_data, val_data):
            sample = dataset[0]
            if not torch.isfinite(sample["image_a"]).all():
                raise ValueError("Nonfinite input")
            print("READ_OK", tuple(sample["image_a"].shape), flush=True)
        return
    start, step, best = 0, 0, math.inf
    if args.resume:
        if raw["contract"] != contract:
            raise ValueError("Resume contract changed. Keep epochs/LR/data/accumulation unchanged; epochs defines the original cosine horizon")
        optimizer.load_state_dict(raw["optimizer"])
        start, step, best = raw["epoch"] + 1, raw["schedule_step"], raw["best_validation_loss"]
        restore_rng(raw["rng"], generator)
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("New runs need an empty output directory; use --resume for an interrupted run")
    if start >= args.epochs:
        raise ValueError("The requested run has already finished")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_info = {**summary, "contract": contract, "data": metadata,
                "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None}
    (args.output_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    monitor(args, f"[RUN] stage={args.stage} initialization={summary['initialization']} "
        f"train_pairs={len(train_data)} val_pairs={len(val_data)} effective_batch={summary['effective_batch_size']} "
        f"device={device} output={args.output_dir.resolve()} start_epoch={start + 1}")
    end = min(args.epochs, start + args.stop_after_epochs) if args.stop_after_epochs else args.epochs
    for epoch in range(start, end):
        if augmentation is not None:
            train_data.set_epoch(epoch)
            monitor(args,'[AUGMENTATION MIX] '+json.dumps(train_data.severity_counts),
                    {'phase':'augmentation_mix','epoch':epoch+1,'counts':train_data.severity_counts})
        if curriculum is not None:
            curriculum.set_epoch(epoch)
        common = dict(model=model, objective=objective, optimizer=optimizer, args=args, device=device,
                      epoch=epoch, total_steps=total_steps, warmup_steps=warmup_steps)
        train_metrics, step = run_epoch(loader=train_loader, schedule_step=step, train=True, **common)
        if curriculum is not None:
            monitor(args, "[SAMPLING] " + json.dumps(curriculum.last_summary),
                    {"phase": "sampling", **curriculum.last_summary})
        val_metrics, _ = run_epoch(loader=val_loader, schedule_step=step, train=False, **common)
        coverage_ok = val_metrics["supervised_batch_fraction"] >= args.min_val_supervised_fraction
        score = val_metrics["loss"]
        improved = coverage_ok and math.isfinite(score) and score < best
        if improved:
            best = score
        state = {"training_recipe": 'v3_exact_dense_robustness_v1' if augmentation is not None else 'scratch_imc_exact_h_v1', "stage": args.stage,
            "epoch": epoch, "schedule_step": step, "global_step": step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config.to_dict(),
            "contract": contract, "rng": rng_state(generator), "best_validation_loss": best,
            "train_metrics": train_metrics, "validation_metrics": val_metrics,
            "provenance": raw.get("provenance") if args.resume else
                {"random_init": raw is None, "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                 "parent": raw.get("provenance") if raw else None},
            "geometry_heads_trained": args.stage == "agri" and config.architecture_version != "v3", "local_role_positive_supervision": False,
            "confidence_head_trained": config.architecture_version == "v3" and config.v3_confidence_head,
            "inlier_head_trained": args.stage == "agri" and config.classification_mode == "inlier"}
        _atomic_torch_save(state, args.output_dir / "latest.pt")
        if improved:
            _atomic_torch_save(state, args.output_dir / f"best_{args.stage}.pt")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            _atomic_torch_save(state, args.output_dir / f"epoch_{epoch + 1:04d}.pt")
        record = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics,
                  "lr": {g["name"]: g["lr"] for g in optimizer.param_groups}, "best_updated": improved,
                  "best_validation_loss": best}
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        append_epoch_csv(args.output_dir / "metrics.csv", record)
        monitor(args, f"[EPOCH END] {epoch + 1}/{args.epochs} train[{metric_text(train_metrics)}] "
            f"val[{metric_text(val_metrics)}] best_val_loss={best:.4f} best_updated={improved} "
            f"time={(train_metrics['seconds'] + val_metrics['seconds']) / 60:.1f}min")
        monitor(args, f"[CHECKPOINT] saved={args.output_dir / 'latest.pt'}" +
            (f" NEW BEST={args.output_dir / f'best_{args.stage}.pt'}" if improved else ""))
        if not coverage_ok:
            raise RuntimeError("Validation supervision coverage too low; latest saved, inspect labels before resuming")
    monitor(args, f"[FINISHED] completed_epoch={end}/{args.epochs} best_val_loss={best:.4f} "
        f"best_checkpoint={args.output_dir / f'best_{args.stage}.pt'}")


if __name__ == "__main__":
    main()
