"""Real-pair adaptation: Stage 2.5-A, optional 2.5-B, and Stage 3.5."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .config import GRMMatcherConfig
from .homography_pair_dataset import (
    GRMHomographyPairDataset,
    GRMResolutionBucketBatchSampler,
    parse_bucket_batch_sizes,
)
from .matcher import GRMMatcher
from .real_pair_dataset import GRMRealPairDataset
from .real_adaptation import (
    GRMStage25BCoarseAndRefinementLoss,
    GRMStage35GeometryLoss,
    build_geometry_role_pseudo_labels,
    forward_student_coarse,
    transform_geometry_role_pseudo_labels,
)
from .stage25 import (
    GRMStage25Config,
    GRMStage25ConsistencyLoss,
    GRMStage25PseudoLabels,
    GRMStage25StudentOutput,
    build_dense_mapping_supervision_labels,
    build_homography_supervision_labels,
    forward_student_refinement,
    generate_teacher_bidirectional_pseudo_labels,
    generate_teacher_pseudo_labels,
    make_ema_teacher,
    make_student_augmentation,
    merge_refinement_valid_masks,
    transform_pseudo_labels,
    update_ema_teacher,
)
from .synthetic_dataset import (
    GRMTargetDomainSyntheticDataset,
    SyntheticGRMDataConfig,
)
from .visualization import visualize_stage25_refinement


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed/EMA Teacher/Student adaptation on real image pairs."
    )
    parser.add_argument(
        "--phase",
        choices=("stage2.5a", "stage2.5b", "stage3.5"),
        default="stage2.5a",
        help="2.5-A local refinement (default), optional 2.5-B coarse, or 3.5 roles.",
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--source_data_dir",
        type=Path,
        default=None,
        help="Optional exact-GT Homography dataset used for Stage-2 source replay.",
    )
    parser.add_argument("--source_train_split", type=str, default="train")
    parser.add_argument("--source_buckets", nargs="*", default=None)
    parser.add_argument("--source_batch_size", type=int, default=None)
    parser.add_argument("--source_replay_interval", type=int, default=1)
    parser.add_argument(
        "--source_replay_mode",
        choices=("auto", "none", "homography", "target_synthetic"),
        default="auto",
        help=(
            "Stage-2.5A replay source. 'auto' preserves the old behavior: use "
            "'homography' when --source_data_dir is supplied, otherwise disable replay."
        ),
    )
    parser.add_argument(
        "--synthetic_config",
        type=Path,
        default=None,
        help="Optional JSON overrides for SyntheticGRMDataConfig.",
    )
    parser.add_argument(
        "--synthetic_image_dir",
        type=Path,
        default=None,
        help=(
            "Optional image directory for target_synthetic replay. If omitted, "
            "unique input1/input2 images from the real training split are used."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/grm_stage25"))
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--init_checkpoint", type=Path)
    checkpoint_group.add_argument("--resume", type=Path)
    parser.add_argument("--stage25_config", type=Path, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument(
        "--train_difficulty",
        choices=("all", "simple", "hard", "negative"),
        default="all",
    )
    parser.add_argument(
        "--val_difficulty",
        choices=("all", "simple", "hard", "negative"),
        default="all",
    )
    parser.add_argument(
        "--pair_buckets",
        nargs="+",
        choices=("global", "residual", "hard_positive", "negative"),
        default=None,
        help="Pair-quality buckets from manifest/labels; phase-safe defaults apply.",
    )
    parser.add_argument("--buckets", nargs="*", default=None)
    parser.add_argument("--bucket_batch_size", action="append", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2.0e-5)
    parser.add_argument("--coarse_learning_rate", type=float, default=3.0e-6)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--max_empty_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--stage25a_train_scope",
        choices=("fine", "mid_fine"),
        default="fine",
        help="Stage-2.5A-v2 defaults to fine-refiner-only adaptation.",
    )
    parser.add_argument(
        "--teacher_mode",
        choices=("fixed", "ema"),
        default=None,
        help="Defaults to fixed for Stage 2.5-A and EMA for later phases.",
    )
    parser.add_argument("--visualize_every", type=int, default=1)
    parser.add_argument("--visualize_max_matches", type=int, default=48)
    parser.add_argument("--save_every_epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="Validate manifests/masks and print split/bucket/batch counts without training.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    raw_checkpoint = torch.load(
        args.resume if args.resume is not None else args.init_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(raw_checkpoint, Mapping):
        raise TypeError("Stage-2/2.5 checkpoint must be a mapping.")
    matcher_values = raw_checkpoint.get("config")
    if not isinstance(matcher_values, Mapping):
        raise KeyError("Checkpoint does not contain a matcher config mapping.")
    matcher_config = GRMMatcherConfig.from_dict(matcher_values)
    stage25_config = _load_stage25_config(args.stage25_config, raw_checkpoint)
    amp_requested = matcher_config.use_amp if args.amp is None else args.amp
    amp_enabled = bool(amp_requested and device.type == "cuda")

    student = GRMMatcher(matcher_config).to(device)
    _configure_real_phase(student, args.phase, args.stage25a_train_scope)
    teacher = make_ema_teacher(student).to(device)
    optimizer = _make_optimizer(
        student,
        args.phase,
        args.learning_rate,
        args.coarse_learning_rate,
        args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_validation = math.inf
    if args.resume is not None:
        checkpoint_phase = raw_checkpoint.get("phase", raw_checkpoint.get("stage"))
        compatible_legacy = args.phase == "stage2.5a" and checkpoint_phase == "stage2.5"
        if checkpoint_phase != args.phase and not compatible_legacy:
            raise ValueError(
                f"--resume requires phase={args.phase!r}, got {checkpoint_phase!r}. "
                "Use --init_checkpoint when moving between adaptation phases."
            )
        checkpoint_scope = raw_checkpoint.get("stage25a_train_scope")
        if (
            args.phase == "stage2.5a"
            and checkpoint_scope is not None
            and checkpoint_scope != args.stage25a_train_scope
        ):
            raise ValueError(
                "--resume must keep the same --stage25a_train_scope; "
                f"checkpoint={checkpoint_scope!r}, requested={args.stage25a_train_scope!r}."
            )
        student.load_state_dict(_require_state(raw_checkpoint, "student"), strict=True)
        teacher.load_state_dict(_require_state(raw_checkpoint, "teacher"), strict=True)
        optimizer.load_state_dict(_require_state(raw_checkpoint, "optimizer"))
        scaler_state = raw_checkpoint.get("scaler")
        if isinstance(scaler_state, Mapping):
            scaler.load_state_dict(dict(scaler_state))
        start_epoch = int(raw_checkpoint.get("epoch", -1)) + 1
        global_step = int(raw_checkpoint.get("global_step", 0))
        best_validation = float(raw_checkpoint.get("best_validation_loss", math.inf))
    else:
        state = raw_checkpoint.get("model", raw_checkpoint)
        if not isinstance(state, Mapping):
            raise TypeError("Initial checkpoint model state must be a mapping.")
        student.load_state_dict(state, strict=True)
        teacher.load_state_dict(state, strict=True)
    teacher.eval()
    checkpoint_teacher_mode = (
        raw_checkpoint.get("teacher_mode") if args.resume is not None else None
    )
    teacher_mode = args.teacher_mode or (
        str(checkpoint_teacher_mode)
        if checkpoint_teacher_mode in {"fixed", "ema"}
        else ("fixed" if args.phase == "stage2.5a" else "ema")
    )

    pair_buckets = (
        args.pair_buckets
        if args.pair_buckets is not None
        else _default_pair_buckets(args.phase)
    )
    train_dataset = GRMRealPairDataset(
        args.data_dir, args.train_split, matcher_config, args.buckets,
        difficulty=args.train_difficulty,
        selected_pair_buckets=pair_buckets,
    )
    validation_dataset = GRMRealPairDataset(
        args.data_dir, args.val_split, matcher_config, args.buckets,
        difficulty=args.val_difficulty,
        selected_pair_buckets=pair_buckets,
    )
    bucket_sizes = parse_bucket_batch_sizes(args.bucket_batch_size)
    train_sampler = GRMResolutionBucketBatchSampler(
        train_dataset.bucket_to_indices,
        args.batch_size,
        bucket_sizes,
        shuffle=True,
        drop_last=False,
        seed=args.seed,
    )
    validation_sampler = GRMResolutionBucketBatchSampler(
        validation_dataset.bucket_to_indices,
        args.batch_size,
        bucket_sizes,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    source_loader: Optional[DataLoader] = None
    source_sampler: Optional[GRMResolutionBucketBatchSampler] = None
    source_dataset_size = 0
    source_replay_mode = _resolve_source_replay_mode(args)
    synthetic_config_summary: Optional[dict[str, object]] = None
    if args.phase == "stage2.5a" and source_replay_mode == "homography":
        if args.source_data_dir is None:
            raise ValueError(
                "--source_replay_mode homography requires --source_data_dir."
            )
        source_dataset = GRMHomographyPairDataset(
            args.source_data_dir,
            args.source_train_split,
            matcher_config,
            args.source_buckets,
        )
        source_dataset_size = len(source_dataset)
        source_sampler = GRMResolutionBucketBatchSampler(
            source_dataset.bucket_to_indices,
            args.source_batch_size or args.batch_size,
            shuffle=True,
            drop_last=False,
            seed=args.seed + 7919,
        )
        source_loader = DataLoader(
            source_dataset,
            batch_sampler=source_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    elif args.phase == "stage2.5a" and source_replay_mode == "target_synthetic":
        image_sources = (
            _discover_image_sources(args.synthetic_image_dir)
            if args.synthetic_image_dir is not None
            else train_dataset.source_image_paths()
        )
        synthetic_config = _load_synthetic_config(
            args.synthetic_config, matcher_config, args.seed + 7919
        )
        source_dataset = GRMTargetDomainSyntheticDataset(
            image_sources,
            synthetic_config,
            matcher_config,
        )
        synthetic_config_summary = synthetic_config.to_dict()
        source_dataset_size = len(source_dataset)
        source_sampler = GRMResolutionBucketBatchSampler(
            source_dataset.bucket_to_indices,
            args.source_batch_size or args.batch_size,
            shuffle=True,
            drop_last=False,
            seed=args.seed + 7919,
        )
        source_loader = DataLoader(
            source_dataset,
            batch_sampler=source_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    startup_summary = {
        "stage": args.phase,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "source_replay_samples": source_dataset_size,
        "source_replay_batches": len(source_loader) if source_loader is not None else 0,
        "train_batches": len(train_loader),
        "validation_batches": len(validation_loader),
        "resolution_buckets_train": {
            key: len(value) for key, value in train_dataset.bucket_to_indices.items()
        },
        "resolution_buckets_validation": {
            key: len(value) for key, value in validation_dataset.bucket_to_indices.items()
        },
        "pair_buckets": pair_buckets,
        "train_pair_bucket_counts": train_dataset.pair_bucket_counts,
        "validation_pair_bucket_counts": validation_dataset.pair_bucket_counts,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in student.parameters()
            if parameter.requires_grad
        ),
        "ema_decay": stage25_config.ema_decay,
        "teacher_mode": teacher_mode,
        "stage25a_train_scope": args.stage25a_train_scope,
        "source_replay_mode": source_replay_mode,
        "source_replay_interval": args.source_replay_interval,
        "synthetic_config": synthetic_config_summary,
    }
    print(json.dumps(startup_summary, ensure_ascii=False), flush=True)
    if args.audit_only:
        return
    writer = _make_writer(args.output_dir, args.tensorboard)
    if writer is not None and hasattr(writer, "add_text"):
        writer.add_text(
            "config/stage25",
            json.dumps(stage25_config.to_dict(), ensure_ascii=False, indent=2),
            0,
        )
        writer.add_text(
            "config/startup",
            json.dumps(startup_summary, ensure_ascii=False, indent=2),
            0,
        )
    if args.phase == "stage2.5a":
        loss_module: torch.nn.Module = GRMStage25ConsistencyLoss(
            matcher_config, stage25_config
        )
    elif args.phase == "stage2.5b":
        loss_module = GRMStage25BCoarseAndRefinementLoss(
            matcher_config, stage25_config
        )
    else:
        loss_module = GRMStage35GeometryLoss(stage25_config)
    if start_epoch >= args.epochs:
        raise ValueError(
            f"No epochs would run: resume starts at {start_epoch}, --epochs={args.epochs}."
        )
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if source_sampler is not None:
            source_sampler.set_epoch(epoch)
        train_metrics, global_step = _run_epoch(
            student,
            teacher,
            loss_module,
            train_loader,
            optimizer,
            scaler,
            device,
            matcher_config,
            stage25_config,
            args.phase,
            amp_enabled,
            args.gradient_clip,
            args.max_empty_batches,
            args.output_dir,
            train=True,
            epoch=epoch,
            global_step=global_step,
            writer=writer,
            log_every=args.log_every,
            progress_enabled=args.progress,
            source_loader=source_loader,
            source_replay_interval=args.source_replay_interval,
            teacher_mode=teacher_mode,
            visualize_every=args.visualize_every,
            visualize_max_matches=args.visualize_max_matches,
        )
        validation_metrics, _ = _run_epoch(
            student,
            teacher,
            loss_module,
            validation_loader,
            optimizer,
            scaler,
            device,
            matcher_config,
            stage25_config,
            args.phase,
            amp_enabled,
            args.gradient_clip,
            args.max_empty_batches,
            args.output_dir,
            train=False,
            epoch=epoch,
            global_step=global_step,
            writer=writer,
            log_every=args.log_every,
            progress_enabled=args.progress,
            source_loader=None,
            source_replay_interval=args.source_replay_interval,
            teacher_mode=teacher_mode,
            visualize_every=args.visualize_every,
            visualize_max_matches=args.visualize_max_matches,
        )
        for name, value in validation_metrics.items():
            if writer is not None:
                writer.add_scalar(f"validation/{name}", value, epoch)
        _log_trainable_parameter_histograms(writer, student, epoch + 1)
        print(
            json.dumps(
                {"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics},
                ensure_ascii=False,
            ),
            flush=True,
        )
        improved = validation_metrics["loss_total"] < best_validation
        if improved:
            best_validation = validation_metrics["loss_total"]
        state = _checkpoint_state(
            student,
            teacher,
            optimizer,
            scaler,
            matcher_config,
            stage25_config,
            args.phase,
            epoch,
            global_step,
            best_validation,
            train_metrics,
            validation_metrics,
            teacher_mode,
            args.stage25a_train_scope,
            source_replay_mode,
            synthetic_config_summary,
        )
        _atomic_torch_save(state, args.output_dir / "latest.pt")
        if args.save_every_epoch:
            _atomic_torch_save(state, args.output_dir / f"epoch_{epoch + 1:04d}.pt")
        if improved:
            _atomic_torch_save(state, args.output_dir / f"best_{args.phase}.pt")
    if writer is not None:
        writer.close()


def _run_epoch(
    student: GRMMatcher,
    teacher: GRMMatcher,
    loss_module: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    matcher_config: GRMMatcherConfig,
    stage25_config: GRMStage25Config,
    phase: str,
    amp_enabled: bool,
    gradient_clip: float,
    max_empty_batches: int,
    output_dir: Path,
    *,
    train: bool,
    epoch: int,
    global_step: int,
    writer: object | None,
    log_every: int,
    progress_enabled: bool,
    source_loader: Optional[DataLoader] = None,
    source_replay_interval: int = 1,
    teacher_mode: str = "ema",
    visualize_every: int = 0,
    visualize_max_matches: int = 48,
    epipolar_threshold_px: Optional[float] = None,
) -> tuple[Dict[str, float], int]:
    _set_trainable_modes(student, train)
    teacher.eval()
    accumulated: Dict[str, float] = {}
    batch_count = 0
    consecutive_empty = 0
    supervised_loss_sum = 0.0
    supervised_batches = 0
    source_iterator = iter(source_loader) if train and source_loader is not None else None
    iterator: Iterable[Mapping[str, object]] = loader
    if progress_enabled:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                loader,
                desc=f"{phase} epoch {epoch + 1} {'train' if train else 'validation'}",
                unit="batch",
            )
        except ImportError:
            pass
    context = torch.enable_grad if train else torch.no_grad
    with context():
        for batch_index, cpu_batch in enumerate(iterator):
            image_a = _tensor(cpu_batch, "image_a").to(device, non_blocking=True)
            image_b = _tensor(cpu_batch, "image_b").to(device, non_blocking=True)
            valid_a = _tensor(cpu_batch, "valid_a").to(device, non_blocking=True)
            valid_b = _tensor(cpu_batch, "valid_b").to(device, non_blocking=True)
            for valid_name, overlap_name, usable_name in (
                ("a", "overlap_a", "overlap_usable_a"),
                ("b", "overlap_b", "overlap_usable_b"),
            ):
                overlap = cpu_batch.get(overlap_name)
                usable = cpu_batch.get(usable_name)
                if not isinstance(overlap, Tensor) or not isinstance(usable, Tensor):
                    continue
                overlap_mask = overlap.to(device, non_blocking=True).bool()
                use_mask = usable.to(device, non_blocking=True).bool().view(-1, 1, 1, 1)
                effective = torch.where(
                    use_mask, overlap_mask, torch.ones_like(overlap_mask)
                )
                if valid_name == "a":
                    valid_a = valid_a & effective
                else:
                    valid_b = valid_b & effective
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                if "sparse_a" in cpu_batch:
                    if phase == "stage3.5":
                        raise ValueError("Sparse IMC labels do not supervise geometry roles")
                    from .imc_dataset import sparse_bidirectional_labels
                    teacher_bidirectional = sparse_bidirectional_labels(
                        cpu_batch, device, matcher_config, stage25_config
                    )
                    teacher_labels = teacher_bidirectional.forward
                elif phase == "stage2.5b":
                    teacher_bidirectional = generate_teacher_bidirectional_pseudo_labels(
                        teacher, image_a, image_b, stage25_config,
                        valid_a, valid_b, global_step,
                    )
                    teacher_labels = teacher_bidirectional.forward
                else:
                    teacher_labels = generate_teacher_pseudo_labels(
                        teacher, image_a, image_b, stage25_config,
                        valid_a, valid_b, global_step,
                    )
                    teacher_geometry = (
                        build_geometry_role_pseudo_labels(
                            teacher, teacher_labels, stage25_config
                        )
                        if phase == "stage3.5"
                        else None
                    )
                    if phase == "stage3.5" and teacher_geometry is not None:
                        pair_label = _tensor(cpu_batch, "pair_label").to(
                            device, non_blocking=True
                        )
                        negative_pair = pair_label == 0
                        if negative_pair.any():
                            base_valid = teacher_labels.base_valid_mask
                            if base_valid is None:
                                base_valid = teacher_labels.valid_mask
                            roles = teacher_geometry.role_labels.clone()
                            roles[negative_pair] = -1
                            roles[negative_pair[:, None] & base_valid] = 2
                            teacher_geometry.role_labels = roles
            epipolar_metrics = {}
            if epipolar_threshold_px is not None:
                from .imc_dataset import filter_epipolar_labels

                if phase == "stage3.5" or epipolar_threshold_px <= 0:
                    raise ValueError("GT-F filtering supports matching phases with a positive threshold")
                fundamental = _tensor(cpu_batch, "F_ab").to(device)
                before = int(teacher_labels.valid_mask.sum().item())
                teacher_labels = filter_epipolar_labels(
                    teacher_labels, fundamental, epipolar_threshold_px
                )
                if phase == "stage2.5b":
                    teacher_bidirectional.backward = filter_epipolar_labels(
                        teacher_bidirectional.backward, fundamental.transpose(1, 2),
                        epipolar_threshold_px,
                    )
                epipolar_metrics = {
                    "labels_before_F": float(before),
                    "labels_after_F": float(teacher_labels.valid_mask.sum().item()),
                }
            augmented = make_student_augmentation(
                image_a, image_b, matcher_config, stage25_config
            )
            height, width = image_a.shape[-2:]
            labels = transform_pseudo_labels(
                teacher_labels, augmented.transform_a, augmented.transform_b,
                height, width, matcher_config, stage25_config,
            )
            homography_labels = None
            if phase == "stage2.5a" and "H_ab" in cpu_batch:
                h_ab = _tensor(cpu_batch, "H_ab").to(device, non_blocking=True)
                h_ba = _tensor(cpu_batch, "H_ba").to(device, non_blocking=True)
                homography_valid = _tensor(cpu_batch, "homography_valid").to(
                    device, non_blocking=True
                )
                pair_mask = _pair_bucket_mask(
                    cpu_batch.get("pair_bucket"),
                    stage25_config.homography_supervision_pair_buckets,
                    image_a.shape[0],
                    device,
                )
                raw_homography_labels = build_homography_supervision_labels(
                    teacher_labels,
                    h_ab,
                    h_ba,
                    homography_valid,
                    stage25_config.max_homography_cycle_error,
                    pair_mask,
                )
                homography_labels = transform_pseudo_labels(
                    raw_homography_labels,
                    augmented.transform_a,
                    augmented.transform_b,
                    height,
                    width,
                    matcher_config,
                    stage25_config,
                )
            backward_labels = None
            geometry_labels = None
            if phase == "stage2.5b":
                backward_labels = transform_pseudo_labels(
                    teacher_bidirectional.backward,
                    augmented.transform_b,
                    augmented.transform_a,
                    height,
                    width,
                    matcher_config,
                    stage25_config,
                )
            elif phase == "stage3.5":
                if teacher_geometry is None:
                    raise RuntimeError("Stage 3.5 Teacher geometry labels were not built.")
                geometry_labels = transform_geometry_role_pseudo_labels(
                    teacher_geometry,
                    augmented.transform_a,
                    augmented.transform_b,
                    height,
                    width,
                    matcher_config,
                    stage25_config,
                )
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                if phase == "stage2.5a":
                    forward_labels = (
                        merge_refinement_valid_masks(labels, homography_labels)
                        if homography_labels is not None
                        else labels
                    )
                    prediction = forward_student_refinement(
                        student, augmented.image_a, augmented.image_b, forward_labels
                    )
                    pseudo_losses = loss_module(prediction, labels)
                    homography_losses = (
                        loss_module(
                            prediction,
                            homography_labels,
                            apply_absolute_reliability=False,
                        )
                        if homography_labels is not None
                        else None
                    )
                    source_losses = None
                    if (
                        train
                        and source_iterator is not None
                        and batch_index % source_replay_interval == 0
                    ):
                        try:
                            source_batch = next(source_iterator)
                        except StopIteration:
                            if source_loader is None:
                                raise RuntimeError("Source replay loader unexpectedly vanished.")
                            source_iterator = iter(source_loader)
                            source_batch = next(source_iterator)
                        source_losses = _source_replay_loss(
                            student,
                            teacher,
                            loss_module,
                            source_batch,
                            device,
                            matcher_config,
                            stage25_config,
                            amp_enabled,
                            global_step,
                        )
                    losses = _combine_stage25a_losses(
                        pseudo_losses,
                        homography_losses,
                        source_losses,
                        stage25_config,
                    )
                elif phase == "stage2.5b":
                    if backward_labels is None:
                        raise RuntimeError("Stage 2.5-B backward labels are missing.")
                    prediction = forward_student_coarse(
                        student, augmented.image_a, augmented.image_b
                    )
                    refinement_prediction = forward_student_refinement(
                        student, augmented.image_a, augmented.image_b, labels
                    )
                    losses = loss_module(
                        prediction, refinement_prediction, labels, backward_labels
                    )
                else:
                    if geometry_labels is None:
                        raise RuntimeError("Stage 3.5 geometry labels are missing.")
                    matcher_result = student(
                        augmented.image_a, augmented.image_b, return_auxiliary=True
                    )
                    output, auxiliary = matcher_result
                    losses = loss_module(output, auxiliary, geometry_labels)
            if phase == "stage3.5":
                valid_count = int(
                    (losses["rerank_count"] + losses["role_count"]).detach().item()
                )
            else:
                valid_count = int(losses["valid_count"].detach().item())
                if epipolar_threshold_px is not None and phase == "stage2.5b":
                    valid_count += int(losses["refine_valid_count"].detach().item())
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Non-finite adaptation loss")
            optimizer_updated = False
            gradient_norm = 0.0
            if valid_count == 0:
                consecutive_empty += 1
                if train and consecutive_empty >= max_empty_batches:
                    diagnostic = {
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "global_step": global_step,
                        "pair_id": cpu_batch.get("pair_id"),
                        "teacher_valid_count": teacher_labels.valid_mask.sum(dim=1).cpu(),
                        "student_valid_count": labels.valid_mask.sum(dim=1).cpu(),
                    }
                    path = output_dir / "empty_pseudo_batch.pt"
                    torch.save(diagnostic, path)
                    raise RuntimeError(
                        f"{consecutive_empty} consecutive batches contained no reliable "
                        f"pseudo labels; saved {path}. Relax filters only after inspection."
                    )
            else:
                consecutive_empty = 0
                supervised_batches += 1
                supervised_loss_sum += float(losses["total"].detach().item())
                if train:
                    scale_before = float(scaler.get_scale())
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in student.parameters()
                            if parameter.requires_grad
                        ],
                        gradient_clip,
                        error_if_nonfinite=not amp_enabled,
                    )
                    if torch.isfinite(norm):
                        gradient_norm = float(norm.detach().cpu())
                    scaler.step(optimizer)
                    scaler.update()
                    overflow = amp_enabled and float(scaler.get_scale()) < scale_before
                    optimizer_updated = not overflow and bool(torch.isfinite(norm).item())
                    if optimizer_updated and teacher_mode == "ema":
                        update_ema_teacher(teacher, student, stage25_config.ema_decay)
            if phase == "stage3.5" and geometry_labels is not None:
                pseudo_mask = geometry_labels.role_labels >= 0
            else:
                pseudo_mask = labels.valid_mask
            pseudo_ratio = float(pseudo_mask.float().mean().detach().cpu())
            metrics = {
                **epipolar_metrics,
                **{
                    f"loss_{name}": float(value.detach().cpu())
                    for name, value in losses.items()
                },
                "pseudo_ratio": pseudo_ratio,
                "pseudo_count": float(valid_count),
                "mean_confidence": _masked_scalar(labels.confidence, labels.valid_mask),
                "mean_cycle_error": _masked_scalar(labels.cycle_error, labels.valid_mask),
                "mean_fine_std": _masked_scalar(labels.fine_std, labels.valid_mask),
                "gradient_norm": gradient_norm,
                "optimizer_updated": float(optimizer_updated),
            }
            if (
                phase == "stage2.5a"
                and not train
                and batch_index == 0
                and visualize_every > 0
                and (epoch + 1) % visualize_every == 0
            ):
                _write_stage25_visualization(
                    writer,
                    output_dir,
                    epoch,
                    matcher_config,
                    image_a,
                    image_b,
                    labels,
                    prediction,
                    homography_labels,
                    visualize_max_matches,
                )
            for name, value in metrics.items():
                accumulated[name] = accumulated.get(name, 0.0) + value
                if (
                    train
                    and writer is not None
                    and log_every > 0
                    and global_step % log_every == 0
                ):
                    writer.add_scalar(f"train/{name}", value, global_step)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    loss=f"{metrics['loss_total']:.4f}",
                    pseudo=valid_count,
                    ratio=f"{pseudo_ratio:.3f}",
                    cycle=f"{metrics['mean_cycle_error']:.2f}",
                    grad=f"{gradient_norm:.2f}" if train else "-",
                )
            batch_count += 1
            if train:
                global_step += 1
    if batch_count == 0:
        raise RuntimeError("Stage-2.5 DataLoader produced no batches.")
    metrics = {name: value / batch_count for name, value in accumulated.items()}
    metrics["supervised_batch_fraction"] = supervised_batches / batch_count
    metrics["supervised_loss"] = supervised_loss_sum / supervised_batches if supervised_batches else math.inf
    return metrics, global_step


def _source_replay_loss(
    student: GRMMatcher,
    teacher: GRMMatcher,
    loss_module: torch.nn.Module,
    cpu_batch: Mapping[str, object],
    device: torch.device,
    matcher_config: GRMMatcherConfig,
    stage25_config: GRMStage25Config,
    amp_enabled: bool,
    global_step: int,
) -> Dict[str, Tensor]:
    """Train local refinement on exact source-domain dense correspondences."""

    image_a = _tensor(cpu_batch, "image_a").to(device, non_blocking=True)
    image_b = _tensor(cpu_batch, "image_b").to(device, non_blocking=True)
    valid_a = (
        _tensor(cpu_batch, "valid_a").to(device, non_blocking=True)
        & _tensor(cpu_batch, "gt_overlap_a").to(device, non_blocking=True)
    )
    valid_b = (
        _tensor(cpu_batch, "valid_b").to(device, non_blocking=True)
        & _tensor(cpu_batch, "gt_overlap_b").to(device, non_blocking=True)
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=amp_enabled,
    ):
        teacher_labels = generate_teacher_pseudo_labels(
            teacher,
            image_a,
            image_b,
            stage25_config,
            valid_a,
            valid_b,
            global_step,
        )
        dense_mapping = _tensor(cpu_batch, "gt_dense_mapping").to(
            device, non_blocking=True
        ).float()
        exact_labels = build_dense_mapping_supervision_labels(
            teacher_labels,
            dense_mapping,
            _tensor(cpu_batch, "gt_overlap_a").to(device, non_blocking=True),
            _tensor(cpu_batch, "gt_overlap_b").to(device, non_blocking=True),
        )
    augmented = make_student_augmentation(
        image_a, image_b, matcher_config, stage25_config
    )
    height, width = image_a.shape[-2:]
    exact_labels = transform_pseudo_labels(
        exact_labels,
        augmented.transform_a,
        augmented.transform_b,
        height,
        width,
        matcher_config,
        stage25_config,
    )
    prediction = forward_student_refinement(
        student, augmented.image_a, augmented.image_b, exact_labels
    )
    return loss_module(
        prediction,
        exact_labels,
        apply_absolute_reliability=False,
    )


def _combine_stage25a_losses(
    pseudo: Mapping[str, Tensor],
    homography: Optional[Mapping[str, Tensor]],
    source: Optional[Mapping[str, Tensor]],
    config: GRMStage25Config,
) -> Dict[str, Tensor]:
    """Combine independently interpretable Stage-2.5A-v2 supervision streams."""

    zero = pseudo["total"] * 0.0
    h_total = homography["total"] if homography is not None else zero
    source_total = source["total"] if source is not None else zero
    h_count = homography["valid_count"] if homography is not None else zero.detach()
    source_count = source["valid_count"] if source is not None else zero.detach()
    total = (
        config.pseudo_loss_weight * pseudo["total"]
        + config.homography_loss_weight * h_total
        + config.source_replay_weight * source_total
    )
    return {
        "total": total,
        "pseudo_total": pseudo["total"],
        "pseudo_unscaled_total": pseudo["unscaled_total"],
        "homography_total": h_total,
        "source_replay_total": source_total,
        "mid_coordinate": pseudo["mid_coordinate"],
        "fine_coordinate": pseudo["fine_coordinate"],
        "mid_nll": pseudo["mid_nll"],
        "fine_nll": pseudo["fine_nll"],
        "batch_reliability": pseudo["batch_reliability"],
        "mean_absolute_confidence": pseudo["mean_absolute_confidence"],
        "covered_region_count": pseudo["covered_region_count"],
        "pseudo_valid_count": pseudo["valid_count"],
        "homography_valid_count": h_count,
        "source_replay_valid_count": source_count,
        "valid_count": pseudo["valid_count"] + h_count + source_count,
    }


def _pair_bucket_mask(
    values: object,
    accepted: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if isinstance(values, str):
        names = [values]
    elif isinstance(values, Sequence):
        names = [str(value) for value in values]
    else:
        names = ["unknown"] * batch_size
    if len(names) != batch_size:
        raise ValueError("pair_bucket batch length does not match image batch size.")
    accepted_set = set(accepted)
    return torch.tensor(
        [name in accepted_set for name in names],
        device=device,
        dtype=torch.bool,
    )


def _set_trainable_modes(model: GRMMatcher, train: bool) -> None:
    """Keep frozen modules in eval mode while enabling trainable leaf modules."""

    model.eval()
    if not train:
        return
    for module in model.modules():
        module.training = any(
            parameter.requires_grad
            for parameter in module.parameters(recurse=False)
        )


def _display_image(image: Tensor, config: GRMMatcherConfig) -> Tensor:
    value = image.detach().float()
    if config.input_range == "imagenet_normalized":
        mean = value.new_tensor(config.normalization_mean).view(1, 3, 1, 1)
        std = value.new_tensor(config.normalization_std).view(1, 3, 1, 1)
        value = value * std + mean
    return value.clamp(0.0, 1.0)


def _write_stage25_visualization(
    writer: object | None,
    output_dir: Path,
    epoch: int,
    matcher_config: GRMMatcherConfig,
    image_a: Tensor,
    image_b: Tensor,
    labels: GRMStage25PseudoLabels,
    prediction: GRMStage25StudentOutput,
    homography_labels: GRMStage25PseudoLabels | None,
    max_matches: int,
) -> None:
    path = output_dir / "visualizations" / f"stage25a_epoch_{epoch + 1:04d}.png"
    valid = labels.valid_mask
    if homography_labels is not None:
        valid = valid | homography_labels.valid_mask
    figure = visualize_stage25_refinement(
        _display_image(image_a, matcher_config),
        _display_image(image_b, matcher_config),
        labels.src_xy,
        labels.coarse_dst_xy,
        labels.fine_dst_xy,
        prediction.mid.dst_xy,
        prediction.fine.dst_xy,
        prediction.mid.local_probability,
        prediction.fine.local_probability,
        valid,
        (
            homography_labels.fine_dst_xy
            if homography_labels is not None
            else None
        ),
        (
            homography_labels.valid_mask
            if homography_labels is not None
            else None
        ),
        max_matches=max_matches,
        save_path=path,
    )
    if writer is not None and hasattr(writer, "add_figure"):
        writer.add_figure("validation/stage25a_refinement", figure, epoch + 1)
    from matplotlib import pyplot as plt

    plt.close(figure)


def _log_trainable_parameter_histograms(
    writer: object | None,
    model: GRMMatcher,
    epoch: int,
) -> None:
    if writer is None or not hasattr(writer, "add_histogram"):
        return
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            writer.add_histogram(
                f"parameters/{name}",
                parameter.detach().float().cpu(),
                epoch,
            )


def _checkpoint_state(
    student: GRMMatcher,
    teacher: GRMMatcher,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    matcher_config: GRMMatcherConfig,
    stage25_config: GRMStage25Config,
    phase: str,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    teacher_mode: str = "ema",
    stage25a_train_scope: str = "mid_fine",
    source_replay_mode: str = "none",
    synthetic_config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    if teacher_mode not in {"fixed", "ema"}:
        raise ValueError("teacher_mode must be 'fixed' or 'ema'.")
    exported_model = student if teacher_mode == "fixed" else teacher
    return {
        "stage": phase,
        "phase": phase,
        "epoch": epoch,
        "global_step": global_step,
        # A fixed Teacher never receives adaptation updates, so export Student.
        "model": exported_model.state_dict(),
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "teacher_mode": teacher_mode,
        "stage25a_train_scope": stage25a_train_scope,
        "source_replay_mode": source_replay_mode,
        "synthetic_config": (
            dict(synthetic_config) if synthetic_config is not None else None
        ),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": matcher_config.to_dict(),
        "stage25_config": stage25_config.to_dict(),
        "best_validation_loss": best_validation_loss,
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
    }


def _load_stage25_config(
    path: Optional[Path], checkpoint: Mapping[str, object]
) -> GRMStage25Config:
    if path is not None:
        values = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(checkpoint.get("stage25_config"), Mapping):
        values = dict(checkpoint["stage25_config"])  # type: ignore[arg-type]
    else:
        values = {}
    return GRMStage25Config(**values)


def _resolve_source_replay_mode(args: argparse.Namespace) -> str:
    if args.source_replay_mode != "auto":
        return str(args.source_replay_mode)
    return "homography" if args.source_data_dir is not None else "none"


def _load_synthetic_config(
    path: Optional[Path],
    matcher_config: GRMMatcherConfig,
    seed: int,
) -> SyntheticGRMDataConfig:
    values: dict[str, object] = {}
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("Synthetic configuration JSON must contain an object.")
        values.update(raw)
    values.setdefault("output_height", matcher_config.input_height)
    values.setdefault("output_width", matcher_config.input_width)
    values.setdefault("seed", seed)
    return SyntheticGRMDataConfig(**values)


def _discover_image_sources(directory: Path) -> tuple[Path, ...]:
    root = directory.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Synthetic texture directory does not exist: {root}")
    paths = tuple(
        sorted(
            (
                path.resolve()
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in GRMRealPairDataset.image_extensions
            ),
            key=lambda path: str(path),
        )
    )
    if not paths:
        raise RuntimeError(f"No supported texture images found below {root}.")
    return paths


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.epochs <= 0 or args.num_workers < 0:
        raise ValueError("batch_size/epochs must be positive and num_workers nonnegative.")
    if (
        args.learning_rate <= 0.0
        or args.coarse_learning_rate <= 0.0
        or args.weight_decay < 0.0
    ):
        raise ValueError("Learning rates must be positive and weight_decay nonnegative.")
    if args.gradient_clip <= 0.0 or args.max_empty_batches <= 0:
        raise ValueError("gradient_clip and max_empty_batches must be positive.")
    if args.log_every < 0:
        raise ValueError("log_every must be nonnegative.")
    if args.source_batch_size is not None and args.source_batch_size <= 0:
        raise ValueError("source_batch_size must be positive when provided.")
    if args.source_replay_interval <= 0:
        raise ValueError("source_replay_interval must be positive.")
    if args.visualize_every < 0 or args.visualize_max_matches <= 0:
        raise ValueError("visualize_every must be nonnegative and max matches positive.")
    replay_mode = _resolve_source_replay_mode(args)
    if args.phase != "stage2.5a" and replay_mode != "none":
        raise ValueError("Source replay is currently supported only for --phase stage2.5a.")
    if replay_mode == "homography" and args.source_data_dir is None:
        raise ValueError("--source_replay_mode homography requires --source_data_dir.")
    if replay_mode == "target_synthetic" and args.source_data_dir is not None:
        raise ValueError(
            "--source_data_dir is only used by homography replay; remove it when "
            "using --source_replay_mode target_synthetic."
        )
    if (
        args.synthetic_config is not None or args.synthetic_image_dir is not None
    ) and replay_mode != "target_synthetic":
        raise ValueError(
            "--synthetic_config/--synthetic_image_dir require "
            "--source_replay_mode target_synthetic."
        )


def _default_pair_buckets(phase: str) -> list[str]:
    if phase == "stage2.5a":
        return ["global", "residual", "hard_positive"]
    if phase == "stage2.5b":
        # Coarse self-training starts only from the strongest geometric pairs.
        return ["global", "residual"]
    if phase == "stage3.5":
        return ["global", "residual", "hard_positive", "negative"]
    raise ValueError(f"Unknown real adaptation phase: {phase!r}.")


def _configure_real_phase(
    model: GRMMatcher,
    phase: str,
    stage25a_train_scope: str = "mid_fine",
) -> None:
    """Open only modules justified by the selected real-domain phase."""

    if stage25a_train_scope not in {"fine", "mid_fine"}:
        raise ValueError("stage25a_train_scope must be 'fine' or 'mid_fine'.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    refinement_modules = (
        model.fpn.mid_lateral,
        model.fpn.coarse_to_mid,
        model.fpn.mid_output,
        model.fpn.fine_lateral,
        model.fpn.mid_to_fine,
        model.fpn.fine_output,
        model.mid_refiner,
        model.fine_refiner,
    )
    if phase == "stage2.5a":
        modules = (
            (model.fine_refiner,)
            if stage25a_train_scope == "fine"
            else refinement_modules
        )
    elif phase == "stage2.5b":
        late_backbone = []
        if hasattr(model.backbone, "stage_c4"):
            late_backbone.append(model.backbone.stage_c4)
        else:
            late_backbone.extend(
                (model.backbone.stage_c4_local, model.backbone.stage_c4_mobilevit)
            )
        modules = (
            *late_backbone,
            model.fpn.coarse_lateral,
            model.fpn.coarse_output,
            model.coarse_transformer,
            model.correlation,
            *refinement_modules,
        )
    elif phase == "stage3.5":
        modules = (
            model.token_encoder,
            model.geometry_graph,
            model.rerank_head,
            *([model.inlier_head] if model.config.classification_mode == "inlier" else
              [model.role_head] if model.config.classification_mode == "role" else []),
        )
    else:
        raise ValueError(f"Unknown real adaptation phase: {phase!r}.")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if phase == "stage2.5b" or (
        phase == "stage2.5a" and stage25a_train_scope == "mid_fine"
    ):
        model.fpn._fine_alpha_logit.requires_grad_(True)


def _make_optimizer(
    model: GRMMatcher,
    phase: str,
    learning_rate: float,
    coarse_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if phase != "stage2.5b":
        parameters = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    coarse_prefixes = (
        "backbone.stage_c4",
        "fpn.coarse_lateral",
        "fpn.coarse_output",
        "coarse_transformer",
        "correlation",
    )
    coarse_parameters = []
    refinement_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(coarse_prefixes):
            coarse_parameters.append(parameter)
        else:
            refinement_parameters.append(parameter)
    if not coarse_parameters or not refinement_parameters:
        raise RuntimeError("Stage 2.5-B parameter groups are unexpectedly empty.")
    return torch.optim.AdamW(
        (
            {"params": coarse_parameters, "lr": coarse_learning_rate, "group": "coarse"},
            {"params": refinement_parameters, "lr": learning_rate, "group": "refinement"},
        ),
        weight_decay=weight_decay,
    )


def _require_state(checkpoint: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = checkpoint.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(f"Stage-2.5 resume checkpoint is missing {name!r} state.")
    return value


def _tensor(batch: Mapping[str, object], name: str) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"Batch field {name!r} must be a Tensor.")
    return value


def _masked_scalar(values: Tensor, mask: Tensor) -> float:
    return float(values[mask].mean().detach().cpu()) if mask.any() else 0.0


def _make_writer(output_dir: Path, enabled: bool) -> object | None:
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("TensorBoard is required unless --no-tensorboard is used.") from error
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    return writer


def _atomic_torch_save(state: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), temporary)
    temporary.replace(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
