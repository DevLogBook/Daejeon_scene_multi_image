"""Standalone three-stage training entry point for GRM-Matcher."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import shutil
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split

from .config import GRMMatcherConfig
from .coordinates import image_xy_to_feature_xy, image_xy_to_normalized_grid
from .homography_pair_dataset import (
    GRMHomographyPairDataset,
    GRMResolutionBucketBatchSampler,
    parse_bucket_batch_sizes,
)
from .losses import GRMMatcherLoss
from .joint_training import (
    GRMJointMixedBatchLoader,
    apply_joint_checkpoint_gate,
    joint_source_name,
    merge_joint_validation_metrics,
)
from .matcher import GRMMatcher
from .outputs import GRMMatcherAuxiliary, GRMMatcherOutput
from .role_targets import build_geometry_role_targets
from .stage2_metrics import (
    GRMStage2Predictions,
    compute_stage2_metric_components,
)
from .synthetic_dataset import (
    GRMTargetDomainSyntheticDataset,
    SyntheticGRMDataConfig,
    SyntheticGRMDataset,
)
from .visualization import (
    visualize_candidate_entropy,
    visualize_matches,
    visualize_role_ground_truth_comparison,
    visualize_spatial_coverage,
    visualize_topk_candidates,
)


_STAGE_BEST_METRICS: Dict[str, tuple[tuple[str, str, str], ...]] = {
    "stage1": (("candidate_recall", "max", "best_candidate_recall.pt"),),
    "stage2": (("pixel_error", "min", "best_pixel_error.pt"),),
    "stage3": (("role_f1", "max", "best_role_f1.pt"),),
    "joint": (
        ("candidate_recall", "max", "best_candidate_recall.pt"),
        ("pixel_error", "min", "best_pixel_error.pt"),
        ("role_f1", "max", "best_role_f1.pt"),
        ("joint_score", "max", "best_joint.pt"),
    ),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone GRM-Matcher.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/grm_training"))
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON matcher config.")
    parser.add_argument(
        "--dataset_type",
        choices=(
            "homography_pairs",
            "synthetic",
            "role_augmented_homography",
            "joint_mixed",
        ),
        default="homography_pairs",
        help=(
            "role_augmented_homography reads split-safe input1 textures from a "
            "Homography manifest and adds exact H/local-flow/occlusion Role GT."
        ),
    )
    parser.add_argument(
        "--role_data_config",
        type=Path,
        default=None,
        help=(
            "Optional JSON overrides for Stage-3 controlled geometry. "
            "Defaults to stable small local deformations and occlusions."
        ),
    )
    parser.add_argument(
        "--role_val_samples_per_source",
        type=int,
        default=1,
        help="Deterministic controlled-geometry variants per validation source image.",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument(
        "--buckets",
        nargs="*",
        default=None,
        help="Optional paired-data resolution buckets, e.g. --buckets 512x512 768x768.",
    )
    parser.add_argument(
        "--bucket_batch_size",
        action="append",
        default=None,
        metavar="BUCKET:BATCH",
        help="Per-resolution batch size; repeat for multiple buckets.",
    )
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3", "joint"), default="stage1")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--backbone", choices=("conv", "hybrid_mobilevit"), default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--joint_backbone_lr", type=float, default=0.0)
    parser.add_argument("--joint_coarse_lr", type=float, default=1.0e-6)
    parser.add_argument("--joint_refinement_lr", type=float, default=3.0e-6)
    parser.add_argument("--joint_geometry_lr", type=float, default=1.0e-5)
    parser.add_argument("--joint_homography_batches", type=int, default=2)
    parser.add_argument("--joint_role_batches", type=int, default=1)
    parser.add_argument(
        "--joint_geometry_detach_epochs",
        type=int,
        default=2,
        help=(
            "Joint warm-up epochs during which graph/rerank/Role inputs are "
            "detached from Stage-1/2 predictions."
        ),
    )
    parser.add_argument("--joint_max_recall_drop", type=float, default=0.002)
    parser.add_argument("--joint_max_pixel_error_ratio", type=float, default=1.05)
    parser.add_argument("--joint_max_pck3_drop", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--samples_per_source", type=int, default=20, help="Synthetic dataset mode only."
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.1, help="Synthetic dataset mode only."
    )
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume the same stage, including optimizer/scaler, epoch, and best metric.",
    )
    checkpoint_group.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights from another stage; epoch and optimizer start fresh.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--amp_init_scale",
        type=float,
        default=8192.0,
        help="Initial CUDA GradScaler scale. A conservative value reduces startup overflows.",
    )
    parser.add_argument(
        "--amp_growth_interval",
        type=int,
        default=2000,
        help="Successful AMP steps required before GradScaler grows its scale.",
    )
    parser.add_argument(
        "--max_consecutive_amp_overflows",
        type=int,
        default=16,
        help="Persistent overflow limit at the minimum scale; zero disables fatal escalation.",
    )
    parser.add_argument(
        "--min_amp_scale",
        type=float,
        default=1.0,
        help="Scale floor used to distinguish persistent instability from normal AMP backoff.",
    )
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--visualize_every", type=int, default=500)
    parser.add_argument("--histogram_every", type=int, default=500)
    parser.add_argument("--max_visualized_matches", type=int, default=100)
    parser.add_argument(
        "--save_every_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep an immutable epoch_XXXX.pt checkpoint after every validated epoch.",
    )
    parser.add_argument(
        "--best_snapshot_interval",
        type=int,
        default=10,
        help="Save a separate cumulative stage-best snapshot every N epochs; zero disables.",
    )
    return parser.parse_args(argv)


def _bucket_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    bucket_batch_sizes: Mapping[str, int],
    device: torch.device,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    bucket_to_indices = getattr(dataset, "bucket_to_indices", None)
    if not isinstance(bucket_to_indices, Mapping):
        raise TypeError("Bucketed GRM datasets must expose bucket_to_indices.")
    sampler = GRMResolutionBucketBatchSampler(
        bucket_to_indices,
        args.batch_size,
        bucket_batch_sizes,
        shuffle=shuffle,
        drop_last=False,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def _joint_parameter_group(name: str) -> str:
    if name.startswith("backbone."):
        return "backbone"
    if name.startswith(
        (
            "fpn.coarse_lateral.",
            "fpn.coarse_output.",
            "coarse_transformer.",
            "correlation.",
        )
    ):
        return "coarse"
    if name == "fpn._fine_alpha_logit" or name.startswith(
        (
            "fpn.mid_lateral.",
            "fpn.coarse_to_mid.",
            "fpn.mid_output.",
            "fpn.fine_lateral.",
            "fpn.mid_to_fine.",
            "fpn.fine_output.",
            "mid_refiner.",
            "fine_refiner.",
        )
    ):
        return "refinement"
    if name.startswith(
        ("token_encoder.", "geometry_graph.", "rerank_head.", "role_head.")
    ):
        return "geometry"
    raise ValueError(f"Joint optimizer cannot classify trainable parameter {name!r}.")


def _build_stage_optimizer(
    model: GRMMatcher,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    if args.stage != "joint":
        parameters = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            [{"params": parameters, "lr": args.learning_rate, "name": "default"}],
            weight_decay=args.weight_decay,
        )

    learning_rates = {
        "backbone": float(args.joint_backbone_lr),
        "coarse": float(args.joint_coarse_lr),
        "refinement": float(args.joint_refinement_lr),
        "geometry": float(args.joint_geometry_lr),
    }
    grouped: Dict[str, list[nn.Parameter]] = {
        name: [] for name in learning_rates
    }
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = _joint_parameter_group(parameter_name)
        if learning_rates[group_name] <= 0.0:
            parameter.requires_grad_(False)
            continue
        grouped[group_name].append(parameter)
    parameter_groups = [
        {"params": grouped[name], "lr": lr, "name": name}
        for name, lr in learning_rates.items()
        if lr > 0.0 and grouped[name]
    ]
    if not parameter_groups:
        raise RuntimeError("Joint optimizer has no positive-learning-rate parameters.")
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    if args.dataset_type == "synthetic" and not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val_fraction must lie in (0, 1).")
    if args.role_val_samples_per_source <= 0:
        raise ValueError("role_val_samples_per_source must be positive.")
    if (
        args.dataset_type == "role_augmented_homography"
        and args.stage not in {"stage3", "joint"}
    ):
        raise ValueError(
            "role_augmented_homography is intended for stage3 or joint training."
        )
    if args.dataset_type == "joint_mixed" and args.stage != "joint":
        raise ValueError("dataset_type=joint_mixed requires --stage joint.")
    joint_lrs = (
        args.joint_backbone_lr,
        args.joint_coarse_lr,
        args.joint_refinement_lr,
        args.joint_geometry_lr,
    )
    if any(value < 0.0 for value in joint_lrs) or args.joint_geometry_lr <= 0.0:
        raise ValueError(
            "Joint learning rates must be nonnegative and joint_geometry_lr positive."
        )
    if args.joint_homography_batches <= 0 or args.joint_role_batches <= 0:
        raise ValueError("Joint batch ratios must be positive.")
    if args.joint_geometry_detach_epochs < 0:
        raise ValueError("joint_geometry_detach_epochs must be nonnegative.")
    if args.joint_max_recall_drop < 0.0 or args.joint_max_pck3_drop < 0.0:
        raise ValueError("Joint validation metric-drop tolerances must be nonnegative.")
    if args.joint_max_pixel_error_ratio < 1.0:
        raise ValueError("joint_max_pixel_error_ratio must be at least one.")
    for name in ("log_every", "visualize_every", "histogram_every"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative; zero disables it.")
    if args.max_visualized_matches <= 0:
        raise ValueError("max_visualized_matches must be positive.")
    if args.best_snapshot_interval < 0:
        raise ValueError("best_snapshot_interval must be nonnegative; zero disables it.")
    if args.amp_init_scale <= 0.0 or args.amp_growth_interval <= 0:
        raise ValueError("amp_init_scale and amp_growth_interval must be positive.")
    if args.max_consecutive_amp_overflows < 0 or args.min_amp_scale <= 0.0:
        raise ValueError(
            "max_consecutive_amp_overflows must be nonnegative and min_amp_scale positive."
        )
    _seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_requested = config.use_amp if args.amp is None else args.amp
    amp_enabled = bool(amp_requested and device.type == "cuda")

    bucket_batch_sizes = parse_bucket_batch_sizes(args.bucket_batch_size)
    dataset_metadata: Dict[str, object] = {}
    validation_loaders: Dict[str, object] = {}
    if args.dataset_type in {
        "homography_pairs",
        "role_augmented_homography",
        "joint_mixed",
    }:
        train_pair_dataset = GRMHomographyPairDataset(
            args.data_dir, args.train_split, config, args.buckets
        )
        validation_pair_dataset = GRMHomographyPairDataset(
            args.data_dir, args.val_split, config, args.buckets
        )
        if args.dataset_type in {"role_augmented_homography", "joint_mixed"}:
            role_train_config = _load_role_data_config(
                args.role_data_config,
                config,
                seed=args.seed,
            )
            role_validation_config = replace(
                role_train_config,
                samples_per_source=args.role_val_samples_per_source,
                seed=args.seed + 1_000_003,
            )
            role_train_dataset = GRMTargetDomainSyntheticDataset(
                train_pair_dataset.source_image_paths(),
                role_train_config,
                config,
            )
            role_validation_dataset = GRMTargetDomainSyntheticDataset(
                validation_pair_dataset.source_image_paths(),
                role_validation_config,
                config,
            )
            dataset_metadata = {
                "role_train_config": role_train_config.to_dict(),
                "role_validation_config": role_validation_config.to_dict(),
                "train_source_images": len(train_pair_dataset.source_image_paths()),
                "validation_source_images": len(
                    validation_pair_dataset.source_image_paths()
                ),
                "geometry_direction": (
                    "H_ab maps synthetic A coordinates to B; "
                    "gt_dense_mapping = H_ab(A) + local_flow(A)"
                ),
            }
        if args.dataset_type == "joint_mixed":
            homography_train_loader = _bucket_loader(
                train_pair_dataset,
                args,
                bucket_batch_sizes,
                device,
                shuffle=True,
                seed=args.seed,
            )
            role_train_loader = _bucket_loader(
                role_train_dataset,
                args,
                bucket_batch_sizes,
                device,
                shuffle=True,
                seed=args.seed + 101,
            )
            train_loader = GRMJointMixedBatchLoader(
                homography_train_loader,
                role_train_loader,
                homography_batches=args.joint_homography_batches,
                role_batches=args.joint_role_batches,
            )
            validation_loaders = {
                "homography": _bucket_loader(
                    validation_pair_dataset,
                    args,
                    bucket_batch_sizes,
                    device,
                    shuffle=False,
                    seed=args.seed,
                ),
                "role_augmented": _bucket_loader(
                    role_validation_dataset,
                    args,
                    bucket_batch_sizes,
                    device,
                    shuffle=False,
                    seed=args.seed + 101,
                ),
            }
            training_size = len(train_pair_dataset) + len(role_train_dataset)
            validation_size = (
                len(validation_pair_dataset) + len(role_validation_dataset)
            )
            dataset_metadata.update(
                {
                    "joint_homography_train_samples": len(train_pair_dataset),
                    "joint_role_train_samples": len(role_train_dataset),
                    "joint_homography_validation_samples": len(
                        validation_pair_dataset
                    ),
                    "joint_role_validation_samples": len(role_validation_dataset),
                    "joint_batch_ratio": {
                        "homography": args.joint_homography_batches,
                        "role_augmented": args.joint_role_batches,
                    },
                }
            )
        else:
            if args.dataset_type == "role_augmented_homography":
                train_dataset = role_train_dataset
                validation_dataset = role_validation_dataset
            else:
                train_dataset = train_pair_dataset
                validation_dataset = validation_pair_dataset
            train_loader = _bucket_loader(
                train_dataset,
                args,
                bucket_batch_sizes,
                device,
                shuffle=True,
                seed=args.seed,
            )
            validation_loader = _bucket_loader(
                validation_dataset,
                args,
                bucket_batch_sizes,
                device,
                shuffle=False,
                seed=args.seed,
            )
            validation_loaders = {"main": validation_loader}
            training_size = len(train_dataset)
            validation_size = len(validation_dataset)
    else:
        image_paths = _find_images(args.data_dir)
        data_config = SyntheticGRMDataConfig(
            output_height=config.input_height,
            output_width=config.input_width,
            samples_per_source=args.samples_per_source,
            seed=args.seed,
            global_residual_threshold=config.global_residual_threshold,
            local_residual_threshold=config.local_residual_threshold,
        )
        dataset = SyntheticGRMDataset(image_paths, data_config)
        dataset_metadata = {"synthetic_config": data_config.to_dict()}
        validation_size = max(1, int(round(len(dataset) * args.val_fraction)))
        training_size = len(dataset) - validation_size
        if training_size <= 0:
            raise ValueError("Dataset is too small for the requested validation split.")
        train_dataset, validation_dataset = random_split(
            dataset,
            (training_size, validation_size),
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        validation_loaders = {"main": validation_loader}

    model = GRMMatcher(config).to(device)
    _configure_stage(
        model,
        args.stage,
        joint_backbone_trainable=args.joint_backbone_lr > 0.0,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError(f"Stage {args.stage} has no trainable parameters.")
    optimizer = _build_stage_optimizer(model, args)
    loss_module = GRMMatcherLoss(config)
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=args.amp_init_scale,
        growth_interval=args.amp_growth_interval,
        enabled=amp_enabled,
    )
    writer = _make_writer(args.output_dir, args.tensorboard)
    _log_run_metadata(
        writer,
        model,
        config,
        args,
        amp_enabled,
        device,
        dataset_metadata=dataset_metadata,
    )

    start_epoch = 0
    best = _initial_stage_best(args.stage)
    joint_baseline: Optional[Dict[str, float]] = None
    checkpoint_path = args.resume if args.resume is not None else args.init_checkpoint
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        resume_training = args.resume is not None
        start_epoch, best = _apply_checkpoint(
            checkpoint,
            model,
            optimizer,
            scaler,
            requested_stage=args.stage,
            resume_training=resume_training,
        )
        if resume_training:
            saved_baseline = checkpoint.get("joint_baseline")
            if isinstance(saved_baseline, Mapping):
                joint_baseline = {
                    str(name): float(value)
                    for name, value in saved_baseline.items()
                    if isinstance(value, (int, float))
                }
            print(
                f"Resumed {args.stage} from {checkpoint_path} at epoch {start_epoch}.",
                flush=True,
            )
        else:
            print(
                f"Initialized {args.stage} model weights from {checkpoint_path}; "
                "optimizer, scaler, epoch, and best metric were reset.",
                flush=True,
            )
    if start_epoch >= args.epochs:
        raise ValueError(
            f"No epochs would run: checkpoint resumes at epoch {start_epoch}, but "
            f"--epochs={args.epochs}. --epochs is the total epoch count for the current "
            "stage; increase it, or use --init_checkpoint to start a new stage at epoch 0."
        )

    global_step = start_epoch * len(train_loader)
    if args.stage == "joint" and joint_baseline is None:
        joint_baseline = _evaluate_validation_loaders(
            model,
            loss_module,
            validation_loaders,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
            global_step=global_step,
            epoch=-1,
            writer=writer,
            visualize=False,
        )
        joint_baseline = {
            name: float(joint_baseline[name])
            for name in (
                "candidate_recall",
                "pixel_error",
                "fine_top1_pck_3",
                "role_f1",
            )
        }
        print(
            json.dumps(
                {"joint_baseline": joint_baseline},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if writer is not None:
            for name, value in joint_baseline.items():
                writer.add_scalar(f"joint_baseline/{name}", value, 0)
    print(
        json.dumps(
            {
                "device": str(device),
                "amp_enabled": amp_enabled,
                "amp_init_scale": args.amp_init_scale if amp_enabled else None,
                "amp_growth_interval": args.amp_growth_interval if amp_enabled else None,
                "stage": args.stage,
                "dataset_type": args.dataset_type,
                "dataset_metadata": dataset_metadata,
                "selected_buckets": args.buckets,
                "bucket_batch_sizes": bucket_batch_sizes,
                "train_samples": training_size,
                "validation_samples": validation_size,
                "train_batches": len(train_loader),
                "validation_batches": {
                    name: len(loader)  # type: ignore[arg-type]
                    for name, loader in validation_loaders.items()
                },
                "optimizer_groups": {
                    str(group.get("name", index)): {
                        "learning_rate": float(group["lr"]),
                        "parameters": sum(
                            parameter.numel() for parameter in group["params"]
                        ),
                    }
                    for index, group in enumerate(optimizer.param_groups)
                },
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "tensorboard_dir": str(args.output_dir / "tensorboard")
                if writer is not None
                else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs):
        train_metrics, global_step = _run_epoch(
            model,
            loss_module,
            train_loader,
            optimizer,
            scaler,
            device,
            args.stage,
            amp_enabled,
            args.gradient_clip,
            args.output_dir,
            train=True,
            writer=writer,
            global_step=global_step,
            epoch=epoch,
            progress_enabled=args.progress,
            log_every=args.log_every,
            visualize_enabled=args.visualize,
            visualize_every=args.visualize_every,
            histogram_every=args.histogram_every,
            max_visualized_matches=args.max_visualized_matches,
            max_consecutive_amp_overflows=args.max_consecutive_amp_overflows,
            min_amp_scale=args.min_amp_scale,
            detach_geometry_inputs=(
                args.stage == "joint"
                and epoch < args.joint_geometry_detach_epochs
            ),
        )
        epoch_number = epoch + 1
        if args.save_every_epoch:
            recovery_state = _checkpoint_state(
                model,
                optimizer,
                scaler,
                config,
                args.stage,
                epoch,
                best,
                training_metadata=dataset_metadata,
                joint_baseline=joint_baseline,
            )
            recovery_state["checkpoint_kind"] = "post_train_pre_validation"
            recovery_state["train_metrics"] = dict(train_metrics)
            _atomic_torch_save(recovery_state, args.output_dir / "latest_train.pt")
        validation_metrics = _evaluate_validation_loaders(
            model,
            loss_module,
            validation_loaders,
            optimizer,
            scaler,
            device,
            args,
            amp_enabled,
            writer=writer,
            global_step=global_step,
            epoch=epoch,
            visualize=args.visualize,
        )
        if args.stage == "joint":
            if joint_baseline is None:
                raise RuntimeError("Joint baseline was not initialized.")
            validation_metrics = apply_joint_checkpoint_gate(
                validation_metrics,
                joint_baseline,
                maximum_recall_drop=args.joint_max_recall_drop,
                maximum_pixel_error_ratio=args.joint_max_pixel_error_ratio,
                maximum_pck3_drop=args.joint_max_pck3_drop,
            )
        for name, value in validation_metrics.items():
            if writer is not None:
                writer.add_scalar(f"validation/{name}", value, epoch)
        print(
            json.dumps(
                {"epoch": epoch, "train": train_metrics, "validation": validation_metrics},
                ensure_ascii=False,
            )
        )

        improved_checkpoints = _update_stage_best_metrics(
            args.stage, validation_metrics, best
        )
        state = _checkpoint_state(
            model,
            optimizer,
            scaler,
            config,
            args.stage,
            epoch,
            best,
            training_metadata=dataset_metadata,
            joint_baseline=joint_baseline,
        )
        state["checkpoint_kind"] = "post_validation"
        state["train_metrics"] = dict(train_metrics)
        state["validation_metrics"] = dict(validation_metrics)
        if args.save_every_epoch:
            epoch_path = args.output_dir / _epoch_checkpoint_name(epoch_number)
            _atomic_torch_save(state, epoch_path)
            _atomic_copy(epoch_path, args.output_dir / "latest.pt")
            for checkpoint_name in improved_checkpoints.values():
                _atomic_copy(epoch_path, args.output_dir / checkpoint_name)
        else:
            _atomic_torch_save(state, args.output_dir / "latest.pt")
            for checkpoint_name in improved_checkpoints.values():
                _atomic_torch_save(state, args.output_dir / checkpoint_name)
        if (
            args.best_snapshot_interval > 0
            and epoch_number % args.best_snapshot_interval == 0
        ):
            saved_snapshots = _save_stage_best_snapshots(
                args.output_dir, args.stage, epoch_number
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "epoch_number": epoch_number,
                        "best_snapshots": [str(path) for path in saved_snapshots],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if writer is not None:
        writer.close()


def _evaluate_validation_loaders(
    model: GRMMatcher,
    loss_module: GRMMatcherLoss,
    validation_loaders: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    amp_enabled: bool,
    *,
    global_step: int,
    epoch: int,
    writer: object | None,
    visualize: bool,
) -> Dict[str, float]:
    results: Dict[str, Dict[str, float]] = {}
    for validation_name, loader in validation_loaders.items():
        source_override = (
            validation_name
            if validation_name in {"homography", "role_augmented"}
            else None
        )
        metrics, _ = _run_epoch(
            model,
            loss_module,
            loader,  # type: ignore[arg-type]
            optimizer,
            scaler,
            device,
            args.stage,
            amp_enabled,
            args.gradient_clip,
            args.output_dir,
            train=False,
            writer=writer,
            global_step=global_step,
            epoch=epoch,
            progress_enabled=args.progress,
            log_every=args.log_every,
            visualize_enabled=visualize,
            visualize_every=args.visualize_every,
            histogram_every=0,
            max_visualized_matches=args.max_visualized_matches,
            max_consecutive_amp_overflows=args.max_consecutive_amp_overflows,
            min_amp_scale=args.min_amp_scale,
            split_name=f"validation_{validation_name}",
            joint_source_override=source_override,
        )
        results[validation_name] = metrics
    if set(results) == {"homography", "role_augmented"}:
        return merge_joint_validation_metrics(
            results["homography"],
            results["role_augmented"],
        )
    if len(results) != 1:
        raise RuntimeError(
            f"Unsupported validation loader groups: {sorted(results)}."
        )
    return next(iter(results.values()))


def _optimizer_group_gradient_norms(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """Compute pre-clipping L2 gradient norms for each optimizer group."""

    result: Dict[str, float] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        squared = 0.0
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            squared += float(gradient.square().sum().cpu())
        name = str(group.get("name", f"group_{group_index}"))
        result[name] = math.sqrt(max(squared, 0.0))
    return result


def _run_epoch(
    model: GRMMatcher,
    loss_module: GRMMatcherLoss,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    stage: str,
    amp_enabled: bool,
    gradient_clip: float,
    output_dir: Path,
    *,
    train: bool,
    writer: object | None,
    global_step: int,
    epoch: int,
    progress_enabled: bool,
    log_every: int,
    visualize_enabled: bool,
    visualize_every: int,
    histogram_every: int,
    max_visualized_matches: int,
    max_consecutive_amp_overflows: int,
    min_amp_scale: float,
    split_name: Optional[str] = None,
    detach_geometry_inputs: bool = False,
    joint_source_override: Optional[str] = None,
) -> tuple[Dict[str, float], int]:
    model.train(train)
    if train and hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)  # type: ignore[attr-defined]
    else:
        batch_sampler = getattr(loader, "batch_sampler", None)
        if train and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
    accumulated: Dict[str, float] = {}
    batch_count = 0
    consecutive_amp_overflows = 0
    context = torch.enable_grad if train else torch.no_grad
    split = split_name or ("train" if train else "validation")
    progress = _progress_iterator(
        loader,
        enabled=progress_enabled,
        description=f"Epoch {epoch + 1} {split}",
    )
    with context():
        for batch_index, cpu_batch in enumerate(progress):
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
                for key, value in cpu_batch.items()
            }
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                output, auxiliary = model(
                    batch["image_a"],
                    batch["image_b"],
                    return_auxiliary=True,
                    detach_geometry_inputs=detach_geometry_inputs and train,
                )
                losses = loss_module(
                    output,
                    batch,
                    auxiliary,
                    stage=stage,
                    joint_source=joint_source_override or joint_source_name(batch),
                )
            if not torch.isfinite(losses["total"]):
                diagnostic = {
                    "batch_index": batch_index,
                    "stage": stage,
                    "losses": {name: value.detach().cpu() for name, value in losses.items()},
                    "batch": {
                        key: value.detach().cpu()
                        for key, value in batch.items()
                        if isinstance(value, Tensor)
                    },
                    "model": model.state_dict(),
                }
                torch.save(diagnostic, output_dir / "nan_batch.pt")
                raise FloatingPointError(
                    f"Non-finite training loss at batch {batch_index}; saved nan_batch.pt."
                )

            gradient_norm = 0.0
            group_gradient_norms: Dict[str, float] = {}
            amp_overflow = 0.0
            if train:
                scale_before = float(scaler.get_scale())
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                group_gradient_norms = _optimizer_group_gradient_norms(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    gradient_clip,
                    # GradScaler must see transient FP16 overflow so it can
                    # skip the optimizer step and reduce its scale.
                    error_if_nonfinite=not amp_enabled,
                )
                gradient_is_finite = bool(torch.isfinite(norm).item())
                if gradient_is_finite:
                    gradient_norm = float(norm.detach().cpu())
                if writer is not None and _is_log_step(global_step, histogram_every):
                    if gradient_is_finite:
                        _log_parameter_histograms(writer, model, global_step)
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                # A decreasing scale is GradScaler's authoritative signal that it
                # skipped the optimizer step. Startup backoff is expected and is
                # not, by itself, evidence that model training has diverged.
                step_was_skipped = bool(amp_enabled and scale_after < scale_before)
                overflow_detected = not gradient_is_finite or step_was_skipped
                if overflow_detected:
                    amp_overflow = 1.0
                    consecutive_amp_overflows += 1
                    print(
                        f"AMP gradient overflow at batch {batch_index}; "
                        f"GradScaler scale {scale_before:g} -> {scale_after:g} "
                        f"({consecutive_amp_overflows}/"
                        f"{max_consecutive_amp_overflows or 'disabled'} consecutive).",
                        flush=True,
                    )
                    if _amp_overflow_is_fatal(
                        consecutive_amp_overflows,
                        scale_after,
                        max_consecutive_amp_overflows,
                        min_amp_scale,
                    ):
                        diagnostic = {
                            "batch_index": batch_index,
                            "stage": stage,
                            "global_step": global_step,
                            "amp_scale_before": scale_before,
                            "amp_scale_after": scale_after,
                            "consecutive_amp_overflows": consecutive_amp_overflows,
                            "max_consecutive_amp_overflows": max_consecutive_amp_overflows,
                            "min_amp_scale": min_amp_scale,
                            "losses": {
                                name: value.detach().cpu()
                                for name, value in losses.items()
                            },
                            "batch": {
                                key: value.detach().cpu()
                                for key, value in batch.items()
                                if isinstance(value, Tensor)
                            },
                            "model": model.state_dict(),
                        }
                        diagnostic_path = output_dir / "amp_overflow_batch.pt"
                        torch.save(diagnostic, diagnostic_path)
                        raise FloatingPointError(
                            "Persistent AMP gradient overflows continued after the "
                            f"scale reached {scale_after:g}; "
                            f"saved {diagnostic_path}."
                        )
                else:
                    consecutive_amp_overflows = 0

            with torch.no_grad():
                monitoring = _monitoring_metrics(
                    output, auxiliary, batch, model.config, stage=stage
                )
            source_name = (
                joint_source_override or joint_source_name(batch)
                if stage == "joint"
                else None
            )
            values = {
                **{f"loss_{name}": float(value.detach().cpu()) for name, value in losses.items() if name != "zero"},
                **monitoring,
                "gradient_norm": gradient_norm,
                **{
                    f"gradient_norm_{name}": value
                    for name, value in group_gradient_norms.items()
                },
                "amp_overflow": amp_overflow,
            }
            if stage == "joint":
                values["joint_geometry_inputs_detached"] = float(
                    detach_geometry_inputs and train
                )
                values["joint_source_role_augmented"] = float(
                    source_name == "role_augmented"
                )
            for name, value in values.items():
                accumulated[name] = accumulated.get(name, 0.0) + value
                if train and writer is not None and _is_log_step(global_step, log_every):
                    writer.add_scalar(f"train/{name}", value, global_step)
            if train and writer is not None and _is_log_step(global_step, log_every):
                for group_index, group in enumerate(optimizer.param_groups):
                    group_name = str(group.get("name", f"group_{group_index}"))
                    writer.add_scalar(
                        f"train/learning_rate_{group_name}",
                        float(group["lr"]),
                        global_step,
                    )
                writer.add_scalar(
                    "train/amp_scale", float(scaler.get_scale()), global_step
                )

            should_visualize = visualize_enabled and visualize_every > 0 and (
                (_is_log_step(global_step, visualize_every) if train else batch_index == 0)
            )
            if should_visualize:
                visualization_step = global_step if train else epoch
                _write_training_visualizations(
                    output,
                    auxiliary,
                    batch,
                    output_dir,
                    split=split,
                    step=visualization_step,
                    writer=writer,
                    max_matches=max_visualized_matches,
                    correspondence_correct_threshold=(
                        model.config.correspondence_correct_threshold
                    ),
                )

            if hasattr(progress, "set_postfix") and (
                log_every == 0 or batch_index % max(log_every, 1) == 0
            ):
                if stage == "stage2":
                    progress.set_postfix(
                        loss=f"{values['loss_total']:.4f}",
                        coarse=f"{values['coarse_top1_pixel_error']:.2f}",
                        mid=f"{values['mid_top1_pixel_error']:.2f}",
                        fine=f"{values['fine_top1_pixel_error']:.2f}",
                        pck3=f"{values['fine_top1_pck_3']:.3f}",
                        reach=f"{values['fine_top1_reachable_ratio']:.3f}",
                        grad=f"{gradient_norm:.2f}" if train else "-",
                    )
                elif stage == "stage3":
                    progress.set_postfix(
                        loss=f"{values['loss_total']:.4f}",
                        role_f1=f"{values['role_f1']:.3f}",
                        correct=f"{values['correspondence_correct_ratio']:.3f}",
                        gt_local=f"{values['target_role_local_ratio']:.3f}",
                        grad=f"{gradient_norm:.2f}" if train else "-",
                    )
                elif stage == "joint":
                    progress.set_postfix(
                        loss=f"{values['loss_total']:.4f}",
                        recall=f"{values['candidate_recall']:.3f}",
                        fine=f"{values['fine_top1_pixel_error']:.2f}",
                        pck3=f"{values['fine_top1_pck_3']:.3f}",
                        rerank=f"{values['rerank_accuracy']:.3f}",
                        role_f1=f"{values['role_f1']:.3f}",
                        grad=f"{gradient_norm:.2f}" if train else "-",
                    )
                else:
                    progress.set_postfix(
                        loss=f"{values['loss_total']:.4f}",
                        recall=f"{values['candidate_recall']:.3f}",
                        px=f"{values['pixel_error']:.2f}",
                        grad=f"{gradient_norm:.2f}" if train else "-",
                    )
            batch_count += 1
            if train:
                global_step += 1
    if batch_count == 0:
        raise RuntimeError("DataLoader produced no batches.")
    return {name: value / batch_count for name, value in accumulated.items()}, global_step


def _is_log_step(step: int, interval: int) -> bool:
    return interval > 0 and step % interval == 0


def _amp_overflow_is_fatal(
    consecutive_overflows: int,
    current_scale: float,
    max_consecutive_overflows: int,
    min_scale: float,
) -> bool:
    """Escalate only persistent overflows after dynamic loss scaling has backed off."""

    return (
        max_consecutive_overflows > 0
        and consecutive_overflows >= max_consecutive_overflows
        and current_scale <= min_scale
    )


def _progress_iterator(
    loader: DataLoader,
    *,
    enabled: bool,
    description: str,
) -> Iterable[Mapping[str, object]]:
    if not enabled:
        return loader
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(
            "tqdm is not installed; continuing without a progress bar. "
            "Install requirements_grm_matcher.txt to enable it.",
            flush=True,
        )
        return loader
    return tqdm(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        mininterval=0.5,
    )


def _log_run_metadata(
    writer: object | None,
    model: GRMMatcher,
    config: GRMMatcherConfig,
    args: argparse.Namespace,
    amp_enabled: bool,
    device: torch.device,
    *,
    dataset_metadata: Optional[Mapping[str, object]] = None,
) -> None:
    if writer is None:
        return
    config_values = config.to_dict()
    argument_values = vars(args)
    metadata = {
        "config": config_values,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in argument_values.items()
        },
        "environment": {
            "torch_version": torch.__version__,
            "device": str(device),
            "amp_enabled": amp_enabled,
        },
        "dataset": dict(dataset_metadata or {}),
    }
    writer.add_text(
        "run/configuration",
        "```json\n" + json.dumps(metadata, indent=2, ensure_ascii=False) + "\n```",
        0,
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    writer.add_scalar("run/parameters_total", total_parameters, 0)
    writer.add_scalar("run/parameters_trainable", trainable_parameters, 0)
    hparams: Dict[str, object] = {
        "stage": args.stage,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "amp_enabled": amp_enabled,
        "backbone": config.backbone_name,
        "input_height": config.input_height,
        "input_width": config.input_width,
        "max_source_queries": config.max_source_queries,
        "topk_candidates": config.topk_candidates,
    }
    writer.add_hparams(hparams, {"hparam/initialized": 1.0})


def _log_parameter_histograms(
    writer: object,
    model: GRMMatcher,
    global_step: int,
) -> None:
    """Record trainable parameter/gradient distributions and per-tensor norms."""

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tag = name.replace(".", "/")
        parameter_value = parameter.detach().float()
        writer.add_histogram(f"parameters/{tag}", parameter_value, global_step)
        writer.add_scalar(
            f"parameter_norm/{tag}",
            float(torch.linalg.vector_norm(parameter_value).cpu()),
            global_step,
        )
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            writer.add_histogram(f"gradients/{tag}", gradient, global_step)
            writer.add_scalar(
                f"gradient_norm/{tag}",
                float(torch.linalg.vector_norm(gradient).cpu()),
                global_step,
            )


def _write_training_visualizations(
    output: GRMMatcherOutput,
    auxiliary: GRMMatcherAuxiliary,
    batch: Mapping[str, Tensor],
    output_dir: Path,
    *,
    split: str,
    step: int,
    writer: object | None,
    max_matches: int,
    correspondence_correct_threshold: float = 5.0,
) -> None:
    """Save representative matching figures and mirror them to TensorBoard."""

    visualization_dir = output_dir / "visualizations" / split
    stem = f"{split}_{step:08d}"
    figures = {
        "matches": visualize_matches(
            batch["image_a"],
            batch["image_b"],
            output.src_xy,
            output.dst_xy,
            output.p_match,
            output.p_global,
            output.p_local,
            output.p_false,
            output.candidate_entropy,
            output.valid_mask,
            max_matches=max_matches,
            save_path=visualization_dir / f"{stem}_matches.png",
        ),
        "topk": visualize_topk_candidates(
            batch["image_a"],
            batch["image_b"],
            auxiliary.candidate_src_xy[:, :, 0],
            auxiliary.fine_dst_xy,
            auxiliary.rerank_probabilities,
            auxiliary.candidate_valid_mask.any(dim=-1),
            max_queries=min(max_matches, 32),
            save_path=visualization_dir / f"{stem}_topk.png",
        ),
        "entropy": visualize_candidate_entropy(
            batch["image_a"],
            output.src_xy,
            output.candidate_entropy,
            output.valid_mask,
            save_path=visualization_dir / f"{stem}_entropy.png",
        ),
        "coverage": visualize_spatial_coverage(
            batch["image_a"],
            output.src_xy,
            output.valid_mask,
            save_path=visualization_dir / f"{stem}_coverage.png",
        ),
    }
    role_target_keys = {
        "gt_dense_mapping",
        "gt_overlap_a",
        "gt_global_mask",
        "gt_local_mask",
    }
    if role_target_keys.issubset(batch):
        role_targets = build_geometry_role_targets(
            output.src_xy,
            output.dst_xy,
            batch,
            correspondence_correct_threshold=correspondence_correct_threshold,
        )
        figures["role_gt_comparison"] = visualize_role_ground_truth_comparison(
            batch["image_a"],
            batch["image_b"],
            output.src_xy,
            output.dst_xy,
            role_targets.gt_dst_xy,
            output.p_global,
            output.p_local,
            output.p_false,
            role_targets.labels,
            role_targets.correspondence_error,
            output.valid_mask,
            max_matches=max_matches,
            save_path=visualization_dir / f"{stem}_role_gt.png",
        )
    if writer is not None:
        for name, figure in figures.items():
            writer.add_figure(f"{split}/visualization_{name}", figure, step, close=True)
    else:
        from matplotlib import pyplot as plt

        for figure in figures.values():
            plt.close(figure)


def _monitoring_metrics(
    output: GRMMatcherOutput,
    auxiliary: GRMMatcherAuxiliary,
    batch: Mapping[str, Tensor],
    config: GRMMatcherConfig,
    *,
    stage: str,
) -> Dict[str, float]:
    query_src = auxiliary.candidate_src_xy[:, :, 0]
    query_gt = _sample_mapping(batch["gt_dense_mapping"], query_src)
    target_feature = image_xy_to_feature_xy(query_gt, config.coarse_stride).round().long()
    target_index = target_feature[..., 1] * auxiliary.coarse_width + target_feature[..., 0]
    inside = (
        (target_feature[..., 0] >= 0)
        & (target_feature[..., 0] < auxiliary.coarse_width)
        & (target_feature[..., 1] >= 0)
        & (target_feature[..., 1] < auxiliary.coarse_height)
    )
    recall_valid = inside & auxiliary.candidate_valid_mask.any(dim=-1)
    recalled = (auxiliary.candidate_indices == target_index[..., None]).any(dim=-1)
    candidate_recall = float(
        recalled[recall_valid].float().mean().detach().cpu()
    ) if recall_valid.any() else 0.0

    stage2_metrics: Dict[str, float] = {}
    if stage in {"stage2", "joint"}:
        stage2_metrics = compute_stage2_metric_components(
            GRMStage2Predictions.from_auxiliary(auxiliary), batch, config
        ).metrics()
        # Stage-2 monitoring and best-checkpoint selection use the Stage-1
        # top-1 candidate directly. They never consult geometry, reranking, or
        # role predictions, all of which are intentionally untrained here.
        if stage == "stage2":
            return {
                "candidate_recall": candidate_recall,
                "pixel_error": stage2_metrics["fine_top1_pixel_error"],
                "role_f1": 0.0,
                **stage2_metrics,
            }

    role_targets = build_geometry_role_targets(
        output.src_xy,
        output.dst_xy,
        batch,
        correspondence_correct_threshold=config.correspondence_correct_threshold,
    )
    valid = output.valid_mask & role_targets.overlap_mask
    error = role_targets.correspondence_error
    pixel_error = float(error[valid].mean().detach().cpu()) if valid.any() else 1.0e6

    predicted_role = torch.stack(
        (output.p_global, output.p_local, output.p_false), dim=-1
    ).argmax(dim=-1)
    role_metrics = _role_classification_metrics(
        predicted_role[output.valid_mask],
        role_targets.labels[output.valid_mask],
    )
    role_f1 = role_metrics["role_f1"]
    candidate_error = torch.linalg.vector_norm(
        auxiliary.fine_dst_xy - role_targets.gt_dst_xy[:, :, None],
        dim=-1,
    ).masked_fill(
        ~auxiliary.candidate_valid_mask,
        torch.finfo(auxiliary.fine_dst_xy.dtype).max,
    )
    best_candidate_error, best_candidate_index = candidate_error.min(dim=-1)
    rerank_valid = (
        role_targets.overlap_mask
        & auxiliary.candidate_valid_mask.any(dim=-1)
        & (best_candidate_error <= config.correspondence_correct_threshold)
    )
    rerank_accuracy = (
        float(
            (auxiliary.selected_indices[rerank_valid] == best_candidate_index[rerank_valid])
            .float()
            .mean()
            .detach()
            .cpu()
        )
        if rerank_valid.any()
        else 0.0
    )
    role_valid = output.valid_mask
    if role_valid.any():
        target_ratios = [
            float(
                (role_targets.labels[role_valid] == class_index)
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for class_index in range(3)
        ]
        predicted_ratios = [
            float(
                (predicted_role[role_valid] == class_index)
                .float()
                .mean()
                .detach()
                .cpu()
            )
            for class_index in range(3)
        ]
        correct_ratio = float(
            role_targets.correct_mask[role_valid].float().mean().detach().cpu()
        )
    else:
        target_ratios = [0.0, 0.0, 0.0]
        predicted_ratios = [0.0, 0.0, 0.0]
        correct_ratio = 0.0
    overlap_pixels = batch["gt_overlap_a"].bool()
    global_pixels = batch["gt_global_mask"].bool()
    local_pixels = batch["gt_local_mask"].bool()
    total_pixels = float(overlap_pixels.numel())
    dataset_overlap_ratio = float(overlap_pixels.float().mean().detach().cpu())
    dataset_global_ratio = float(global_pixels.sum().detach().cpu()) / max(
        float(overlap_pixels.sum().detach().cpu()), 1.0
    )
    dataset_local_ratio = float(local_pixels.sum().detach().cpu()) / max(
        float(overlap_pixels.sum().detach().cpu()), 1.0
    )
    metrics = {
        "candidate_recall": candidate_recall,
        "pixel_error": (
            stage2_metrics["fine_top1_pixel_error"]
            if stage == "joint"
            else pixel_error
        ),
        "final_reranked_pixel_error": pixel_error,
        "rerank_accuracy": rerank_accuracy,
        "role_f1": role_f1,
        **role_metrics,
        "correspondence_correct_ratio": correct_ratio,
        "target_role_global_ratio": target_ratios[0],
        "target_role_local_ratio": target_ratios[1],
        "target_role_false_ratio": target_ratios[2],
        "predicted_role_global_ratio": predicted_ratios[0],
        "predicted_role_local_ratio": predicted_ratios[1],
        "predicted_role_false_ratio": predicted_ratios[2],
        "dataset_overlap_pixel_ratio": dataset_overlap_ratio,
        "dataset_global_pixel_ratio_in_overlap": dataset_global_ratio,
        "dataset_local_pixel_ratio_in_overlap": dataset_local_ratio,
        "dataset_nonoverlap_pixel_ratio": (
            1.0 - float(overlap_pixels.sum().detach().cpu()) / max(total_pixels, 1.0)
        ),
    }
    if stage == "joint":
        metrics.update(stage2_metrics)
    return metrics


def _sample_mapping(field: Tensor, points_xy: Tensor) -> Tensor:
    grid = image_xy_to_normalized_grid(
        points_xy.float(), field.shape[-2], field.shape[-1]
    ).unsqueeze(2)
    return F.grid_sample(
        field.float(), grid, align_corners=False, mode="bilinear", padding_mode="zeros"
    ).squeeze(-1).transpose(1, 2)


def _macro_f1(predicted: Tensor, target: Tensor) -> float:
    return _role_classification_metrics(predicted, target)["role_f1"]


def _role_classification_metrics(
    predicted: Tensor,
    target: Tensor,
) -> Dict[str, float]:
    if predicted.numel() == 0:
        return {
            "role_f1": 0.0,
            **{
                f"role_{name}_{metric}": 0.0
                for name in ("global", "local", "false")
                for metric in ("precision", "recall", "f1")
            },
        }
    values: list[Tensor] = []
    result: Dict[str, float] = {}
    for class_index, class_name in enumerate(("global", "local", "false")):
        true_positive = ((predicted == class_index) & (target == class_index)).sum().float()
        false_positive = ((predicted == class_index) & (target != class_index)).sum().float()
        false_negative = ((predicted != class_index) & (target == class_index)).sum().float()
        precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
        recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
        f1 = 2.0 * true_positive / (
            2.0 * true_positive + false_positive + false_negative
        ).clamp_min(1.0)
        values.append(f1)
        result[f"role_{class_name}_precision"] = float(
            precision.detach().cpu()
        )
        result[f"role_{class_name}_recall"] = float(recall.detach().cpu())
        result[f"role_{class_name}_f1"] = float(f1.detach().cpu())
    result["role_f1"] = float(torch.stack(values).mean().detach().cpu())
    return result


def _configure_stage(
    model: GRMMatcher,
    stage: str,
    *,
    joint_backbone_trainable: bool = True,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: Iterable[nn.Module]
    if stage == "stage1":
        modules = (model.backbone, model.fpn, model.coarse_transformer, model.correlation)
    elif stage == "stage2":
        # Stage 1 only supervises F_coarse, so these refinement-specific FPN
        # branches have not learned useful local descriptors yet. Train them in
        # Stage 2 while keeping backbone and the complete coarse path frozen.
        modules = (
            model.fpn.mid_lateral,
            model.fpn.coarse_to_mid,
            model.fpn.mid_output,
            model.fpn.fine_lateral,
            model.fpn.mid_to_fine,
            model.fpn.fine_output,
            model.mid_refiner,
            model.fine_refiner,
        )
    elif stage == "stage3":
        modules = (
            model.token_encoder,
            model.geometry_graph,
            model.rerank_head,
            model.role_head,
        )
    elif stage == "joint":
        modules = (model,)
    else:
        raise ValueError(f"Unknown training stage: {stage}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if stage == "joint" and not joint_backbone_trainable:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    if stage == "stage2":
        model.fpn._fine_alpha_logit.requires_grad_(True)


def _load_config(args: argparse.Namespace) -> GRMMatcherConfig:
    values: Dict[str, object] = {}
    if args.config is not None:
        values = json.loads(args.config.read_text(encoding="utf-8"))
    if args.height is not None:
        values["input_height"] = args.height
    if args.width is not None:
        values["input_width"] = args.width
    if args.backbone is not None:
        values["backbone_name"] = args.backbone
    return GRMMatcherConfig.from_dict(values)


def _load_role_data_config(
    path: Optional[Path],
    matcher_config: GRMMatcherConfig,
    *,
    seed: int,
) -> SyntheticGRMDataConfig:
    """Load exact Stage-3 role geometry without allowing coordinate drift."""

    values: Dict[str, object] = {
        "samples_per_source": 4,
        "local_flow_probability": 0.70,
        "local_region_min": 1,
        "local_region_max": 3,
        "local_flow_min_pixels": 4.0,
        "local_flow_max_pixels": 18.0,
        "occlusion_probability": 0.35,
        "max_occlusions": 2,
        "blur_probability": 0.35,
        "noise_std": 0.01,
    }
    if path is not None:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("role_data_config JSON must contain an object.")
        values.update(loaded)
    # These values define the matcher/data coordinate contract and cannot be
    # overridden by a stale augmentation JSON file.
    values.update(
        {
            "output_height": matcher_config.input_height,
            "output_width": matcher_config.input_width,
            "seed": seed,
            "global_residual_threshold": (
                matcher_config.global_residual_threshold
            ),
            "local_residual_threshold": matcher_config.local_residual_threshold,
        }
    )
    config = SyntheticGRMDataConfig(**values)
    if config.local_flow_max_pixels >= matcher_config.local_residual_threshold:
        raise ValueError(
            "Stage-3 local_flow_max_pixels must remain below "
            "local_residual_threshold so exact Local correspondences are not "
            "mislabelled False."
        )
    return config


def _find_images(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Training data directory does not exist: {data_dir}")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found under {data_dir}")
    return paths


def _initial_stage_best(stage: str) -> Dict[str, float]:
    """Return fresh best values only for metrics meaningful to ``stage``."""
    try:
        specifications = _STAGE_BEST_METRICS[stage]
    except KeyError as error:
        raise ValueError(f"Unknown training stage: {stage}") from error
    return {
        metric: (-math.inf if direction == "max" else math.inf)
        for metric, direction, _ in specifications
    }


def _apply_checkpoint(
    checkpoint: Mapping[str, object],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    requested_stage: str,
    resume_training: bool,
) -> tuple[int, Dict[str, float]]:
    """Load a checkpoint using either same-stage resume or weight-init semantics."""
    if "model" not in checkpoint:
        raise KeyError("Checkpoint is missing required 'model' state.")
    if resume_training:
        checkpoint_stage = checkpoint.get("stage")
        if checkpoint_stage != requested_stage:
            raise ValueError(
                f"--resume requires a checkpoint from the same stage, but checkpoint stage "
                f"is {checkpoint_stage!r} and requested stage is {requested_stage!r}. "
                "Use --init_checkpoint for cross-stage weight initialization."
            )
        if "optimizer" not in checkpoint or "epoch" not in checkpoint:
            raise KeyError(
                "A resumable checkpoint must contain 'optimizer' and 'epoch' states."
            )

    model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]
    if not resume_training:
        return 0, _initial_stage_best(requested_stage)

    optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore[arg-type]
    scaler_state = checkpoint.get("scaler")
    if scaler_state:
        scaler.load_state_dict(scaler_state)  # type: ignore[arg-type]

    best = _initial_stage_best(requested_stage)
    saved_best = checkpoint.get("best")
    if isinstance(saved_best, Mapping):
        for metric in best:
            value = saved_best.get(metric)
            if isinstance(value, (int, float)):
                best[metric] = float(value)
    return int(checkpoint["epoch"]) + 1, best


def _update_stage_best_metrics(
    stage: str,
    validation_metrics: Mapping[str, float],
    best: Dict[str, float],
) -> Dict[str, str]:
    """Update stage-relevant best metrics and return metric-to-file saves."""
    try:
        specifications = _STAGE_BEST_METRICS[stage]
    except KeyError as error:
        raise ValueError(f"Unknown training stage: {stage}") from error

    improved: Dict[str, str] = {}
    for metric, direction, checkpoint_name in specifications:
        if metric not in validation_metrics:
            raise KeyError(f"Validation metrics are missing stage metric {metric!r}.")
        if (
            stage == "joint"
            and metric == "joint_score"
            and float(validation_metrics.get("joint_eligible", 0.0)) < 0.5
        ):
            continue
        value = float(validation_metrics[metric])
        previous = best[metric]
        is_improved = value > previous if direction == "max" else value < previous
        if is_improved:
            best[metric] = value
            improved[metric] = checkpoint_name
    return improved


def _make_writer(output_dir: Path, enabled: bool) -> object | None:
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging requested but tensorboard is not installed. "
            "Install requirements_grm_matcher.txt or pass --no-tensorboard."
        ) from error
    return SummaryWriter(log_dir=str(output_dir / "tensorboard"))


def _epoch_checkpoint_name(epoch_number: int) -> str:
    """Return a human-facing one-based immutable epoch checkpoint name."""
    if epoch_number <= 0:
        raise ValueError("epoch_number must be positive.")
    return f"epoch_{epoch_number:04d}.pt"


def _stage_best_snapshot_name(best_checkpoint_name: str, epoch_number: int) -> str:
    if epoch_number <= 0:
        raise ValueError("epoch_number must be positive.")
    path = Path(best_checkpoint_name)
    return f"{path.stem}_epoch_{epoch_number:04d}{path.suffix}"


def _save_stage_best_snapshots(
    output_dir: Path,
    stage: str,
    epoch_number: int,
) -> list[Path]:
    """Copy cumulative stage-best checkpoints into milestone-specific files."""
    try:
        specifications = _STAGE_BEST_METRICS[stage]
    except KeyError as error:
        raise ValueError(f"Unknown training stage: {stage}") from error
    saved: list[Path] = []
    for _, _, best_checkpoint_name in specifications:
        source = output_dir / best_checkpoint_name
        if not source.is_file():
            print(
                f"Warning: cannot create milestone best snapshot because {source} "
                "does not exist (the validation metric may never have been finite).",
                flush=True,
            )
            continue
        destination = output_dir / _stage_best_snapshot_name(
            best_checkpoint_name, epoch_number
        )
        _atomic_copy(source, destination)
        saved.append(destination)
    return saved


def _atomic_torch_save(state: Mapping[str, object], destination: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt the old file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(dict(state), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    """Atomically publish a copy of an already serialized checkpoint."""
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_state(
    model: GRMMatcher,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: GRMMatcherConfig,
    stage: str,
    epoch: int,
    best: Mapping[str, float],
    *,
    training_metadata: Optional[Mapping[str, object]] = None,
    joint_baseline: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config.to_dict(),
        "stage": stage,
        "epoch": epoch,
        "best": dict(best),
        "training_metadata": dict(training_metadata or {}),
        "joint_baseline": (
            dict(joint_baseline) if joint_baseline is not None else None
        ),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
