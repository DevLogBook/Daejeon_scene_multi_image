import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

import sys

REPO_ROOT = Path(__file__).resolve().parent
project_root = REPO_ROOT.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dense_match.network import AgriTPSStitcher, DistillationLoss, LocalGeometricConsistency
from dense_match.refine import compute_photometric_loss, compute_cycle_consistency_loss
from dataset.dataset import CachedTeacherDataset

from train import (
    EMALossTracker,
    _prepare_teacher_targets,
    compute_loss_bundle,
    save_checkpoint,
    load_checkpoint,
    visualize_results,
    cached_collate_fn,
)
from dense_match.utils import (
    STAGE_DEFS, get_stage, interpolate_loss_weights,
    apply_freeze_state, build_optimizer_and_scheduler,
)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """将各卡的标量 tensor 求平均，用于准确的 Log 记录"""
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


@torch.no_grad()
def validate_ddp(
        student: nn.Module,
        val_loader: DataLoader,
        distill_loss_fn: nn.Module,
        geo_loss_fn: nn.Module,
        w: Dict[str, float],
        device: torch.device,
        use_amp: bool,
        max_batches: Optional[int] = None,  # 改为 None = 跑全量
) -> Dict[str, float]:
    was_training = student.training
    student.eval()

    keys = ('distill', 'fold', 'photo', 'cycle', 'geo', 'tps_smooth', 'total', 'fold_ratio')
    totals = {k: 0.0 for k in keys}
    count = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        img_a, img_b, t_out, t_feat_a, t_feat_b = _prepare_teacher_targets(batch, device)
        bundle = compute_loss_bundle(
            student=student, img_a=img_a, img_b=img_b,
            t_out=t_out, t_feat_a=t_feat_a, t_feat_b=t_feat_b,
            distill_loss_fn=distill_loss_fn, geo_loss_fn=geo_loss_fn,
            w=w, device=device, amp_enabled=use_amp, use_cycle=False,
        )
        for k in keys:
            v = bundle.get(k, 0.0)
            totals[k] += float(v.detach().item()) if torch.is_tensor(v) else float(v)
        count += 1

    if was_training:
        student.train()

    # 跨卡汇总
    count_t = torch.tensor([float(count)], device=device)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    total_count = float(count_t.item())

    reduced = {}
    for k in keys:
        t = torch.tensor([totals[k]], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[k] = float(t.item()) / max(total_count, 1e-6)

    return reduced


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("AgriMatcher DDP Training")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints_ddp"))
    p.add_argument("--log-dir", type=Path, default=Path("runs/agrimatch_ddp"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--teacher-grid-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4, help="每张 GPU 的 batch size")
    p.add_argument("--epochs", type=int, default=65)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--force-amp", action="store_true")
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--teacher-setting", type=str, default="precise")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta-coarse", type=float, default=1.0)
    p.add_argument("--beta-refine", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--eta-coarse", type=float, default=0.5)
    p.add_argument("--eta-refine", type=float, default=1.0)
    p.add_argument("--lambda-tv-coarse", type=float, default=0.01)
    p.add_argument("--lambda-tv-refine", type=float, default=0.05)
    p.add_argument("--conf-thresh-kl", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--val-interval", type=int, default=200)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    main_proc = is_main_process()

    args = build_argparser().parse_args()

    # 每卡用不同种子，避免数据增强完全同步
    seed = args.seed + local_rank
    torch.manual_seed(seed)
    random.seed(seed)

    # AMP 检查（同一台机器 GPU 型号一致，各卡判断结果相同，无需同步）
    use_amp = args.amp
    if device.type == "cuda" and use_amp and not args.force_amp:
        major, minor = torch.cuda.get_device_capability(local_rank)
        if major < 8:
            if main_proc:
                print(f"[AMP] Auto-disabled (capability={major}.{minor}). Use --force-amp to override.")
            use_amp = False

    writer = SummaryWriter(log_dir=str(args.log_dir)) if main_proc else None

    if main_proc:
        print(f"[Dataset] DDP 缓存模式: {args.cache_dir}")

    train_ds = CachedTeacherDataset(args.cache_dir, args.val_ratio, return_split='train')
    val_ds   = CachedTeacherDataset(args.cache_dir, args.val_ratio, return_split='val')

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=local_rank, shuffle=False)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": cached_collate_fn,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 2 if args.num_workers > 0 else None,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, sampler=val_sampler,   **loader_kwargs)

    actual_teacher_dim = {'base': 80, 'precise': 100}[args.teacher_setting]
    base_student = AgriTPSStitcher(
        matcher_config={'d_model': args.d_model, 'teacher_dim': actual_teacher_dim,
                        'grid_size': args.teacher_grid_size},
        tps_config={'grid_size': 8, 'feat_channels': args.d_model}
    ).to(device)

    distill_loss_fn = DistillationLoss(
        alpha=args.alpha, beta_coarse=args.beta_coarse, beta_refine=args.beta_refine,
        gamma=args.gamma, eta_coarse=args.eta_coarse, eta_refine=args.eta_refine,
        lambda_tv_coarse=args.lambda_tv_coarse, lambda_tv_refine=args.lambda_tv_refine,
        conf_thresh_kl=args.conf_thresh_kl,
    ).to(device)
    geo_loss_fn  = LocalGeometricConsistency().to(device)
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_tracker = EMALossTracker(alpha=args.ema_alpha)

    start_epoch   = 1
    global_step   = 0
    best_val_loss = float("inf")
    current_stage: Optional[Dict] = None
    optimizer     = None
    scheduler     = None

    if args.resume:
        resumed_stage = get_stage(1)
        apply_freeze_state(base_student, resumed_stage)
        optimizer, scheduler = build_optimizer_and_scheduler(
            base_student, resumed_stage, len(train_loader), args.weight_decay)
        start_epoch, global_step, best_val_loss, _ema = load_checkpoint(
            args.resume, base_student, optimizer, scheduler,
            scaler if use_amp else None, device)
        loss_tracker.ema_values['total'] = _ema
        current_stage = resumed_stage
        if main_proc:
            print(f"Resumed from epoch={start_epoch}, step={global_step}, best_val={best_val_loss:.4f}")

    # find_unused_parameters=False：冻结参数不进优化器，不会产生 unused param，
    # 与 no_sync 梯度累加同时使用不会产生冲突。
    student = DDP(
        base_student,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    steps_per_epoch = (len(train_loader) + args.accum_steps - 1) // args.accum_steps

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        new_stage = get_stage(epoch)
        if new_stage is not current_stage:
            # 确保所有进程同步到阶段切换点
            dist.barrier()
            if main_proc:
                print(f"\n{'═'*60}\n[Stage Transition] Epoch {epoch} → Stage: {new_stage['name']}\n{'═'*60}")
            current_stage = new_stage

            apply_freeze_state(student.module, current_stage)
            optimizer, scheduler = build_optimizer_and_scheduler(
                student.module, current_stage, steps_per_epoch, args.weight_decay)

            if main_proc:
                for g in optimizer.param_groups:
                    n = sum(p.numel() for p in g['params'])
                    print(f"  Group '{g.get('name','?')}': lr={g['lr']:.2e}, params={n:,}")

        w = interpolate_loss_weights(current_stage, epoch)
        use_cycle_this_epoch = w.get("cycle", 0.0) > 0.01

        if main_proc:
            for k, v in w.items():
                writer.add_scalar(f"LossWeights/{k}", v, epoch)
            print(f"\n🚀 Epoch {epoch}/{args.epochs} | Stage: {current_stage['name']}")
            print(f"   Weights: {', '.join(f'{k}={v:.3f}' for k, v in w.items() if v > 0)}")

        student.train()
        # 保持冻结模块处于 eval 状态
        if "backbone" in current_stage.get("freeze", []) or not current_stage.get("unfreeze_backbone", True):
            student.module.matcher.backbone.eval()
        if "aggregator" in current_stage.get("freeze", []):
            student.module.tps_estimator.eval()

        pbar = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch}", disable=not main_proc,
        )

        for i, batch in pbar:
            img_a, img_b, t_out, t_feat_a, t_feat_b = _prepare_teacher_targets(batch, device)

            # no_sync 挂起梯度同步，最后一步再触发 all-reduce
            is_accumulating = (i + 1) % args.accum_steps != 0 and (i + 1) != len(train_loader)

            def forward_backward():
                bundle = compute_loss_bundle(
                    student=student, img_a=img_a, img_b=img_b,
                    t_out=t_out, t_feat_a=t_feat_a, t_feat_b=t_feat_b,
                    distill_loss_fn=distill_loss_fn, geo_loss_fn=geo_loss_fn,
                    w=w, device=device, amp_enabled=use_amp,
                    use_cycle=use_cycle_this_epoch,
                )
                loss_for_backward = bundle['total'] / args.accum_steps
                scaler.scale(loss_for_backward).backward()
                return bundle

            if is_accumulating:
                with student.no_sync():
                    bundle = forward_backward()
            else:
                bundle = forward_backward()

            if not is_accumulating:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

                has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in student.parameters()
                )
                if has_nan_grad:
                    if main_proc:
                        print(f"🛡️ [GradShield] NaN gradient at step {global_step}! Skipping.")
                    # NaN 时只更新 scaler，不执行 optimizer.step
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if main_proc and global_step % 10 == 0:
                    total_loss = bundle['total']
                    reduced_total = reduce_tensor(total_loss).item()
                    smoothed = loss_tracker.update({'total': reduced_total})
                    pbar.set_postfix({
                        'Loss':  f"{smoothed['total_ema']:.4f}",
                        'Fold':  f"{float(bundle['fold'].detach()):.4f}",
                        'Photo': f"{float(bundle['photo'].detach() if torch.is_tensor(bundle['photo']) else bundle['photo']):.4f}",
                    })

                if main_proc and global_step % args.log_interval == 0:
                    # 记录全部子项 Loss（与单卡版对齐）
                    for k in ('distill', 'fold', 'photo', 'cycle', 'geo', 'tps_smooth', 'total'):
                        v = bundle.get(k, 0.0)
                        val = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                        writer.add_scalar(f"Train/{k}", val, global_step)
                    for j, g in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"LR/group_{j}_{g.get('name','')}", g['lr'], global_step)
                    with torch.no_grad():
                        visualize_results(img_a, img_b, bundle['stu_out'], global_step, writer)

                if global_step % args.val_interval == 0:
                    val_losses = validate_ddp(
                        student=student, val_loader=val_loader,
                        distill_loss_fn=distill_loss_fn, geo_loss_fn=geo_loss_fn,
                        w=w, device=device, use_amp=use_amp,
                        max_batches=None,  # 跑全量验证集
                    )

                    if main_proc:
                        val_total = val_losses['total']
                        pbar.write(
                            f"📊 [val] step={global_step} | total={val_total:.4f} | "
                            f"photo={val_losses['photo']:.4f} | fold={val_losses['fold']:.4f}"
                        )
                        for k, v in val_losses.items():
                            writer.add_scalar(f"Val/{k}", v, global_step)

                        is_best = val_total < best_val_loss
                        if is_best:
                            best_val_loss = val_total

                        save_checkpoint(
                            args.save_dir, epoch, global_step, student.module,
                            optimizer, scheduler, scaler if use_amp else None, args,
                            train_loss_ema=loss_tracker.get_ema('total'),
                            val_loss=val_total, best_val_loss=best_val_loss, is_best=is_best,
                        )

                    # 同步 best_val_loss 到所有进程（防止进程间状态漂移）
                    bvl_t = torch.tensor([best_val_loss], device=device)
                    dist.broadcast(bvl_t, src=0)
                    best_val_loss = float(bvl_t.item())

        if main_proc:
            print(f"[epoch {epoch}] Done. EMA={loss_tracker.get_ema('total'):.4f}")

    if main_proc:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()