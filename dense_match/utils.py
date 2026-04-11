from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_match.network import AgriTPSStitcher

STAGE_DEFS = [
    # 目标：让 Matcher + WarpRefiner 在纯监督信号下收敛，
    #       Aggregator 冻结，保证"投票初始解"质量。
    {
        "name": "SUPERVISED_WARMUP",
        "epoch_start": 1,
        "epoch_end": 20,
        "freeze": ["aggregator"],          # backbone + matcher 可训练
        "unfreeze_backbone": True,         # backbone 以低 LR 微调
        "scheduler": "onecycle",
        "max_lr": {
            "backbone": 2e-5,              # 预训练权重，极小 LR 保护
            "matcher":  2e-4,
            "aggregator": 0.0,             # 冻结，不放入优化器
        },
        # 固定权重（不插值）
        "loss_weights": {
            "distill": 1.0,
            "fold":    0.1,   # fold 权重保守：此阶段 Aggregator 冻结，fold_loss 来自投票初始解
            "photo":   0.0,   # 关键：Stage 1 禁止 photo_loss
            "ssim":    0.0,
            "cycle":   0.0,
            "geo":     0.15,
            "tps_smooth": 0.1,
        },
    },
    # 目标：Aggregator 从一张"白纸"开始学习，
    #       用 OneCycleLR 给它足够能量，photo_loss 缓慢升温。
    {
        "name": "AGGREGATOR_ACTIVATION",
        "epoch_start": 21,
        "epoch_end": 40,
        "freeze": [],                       # 全部解冻
        "unfreeze_backbone": False,         # backbone 保持冻结状态（避免大幅震荡）
        "scheduler": "onecycle",
        # 注意：backbone 此阶段冻结（requires_grad=False），不放入优化器
        "max_lr": {
            "backbone":   0.0,             # 冻结
            "matcher":    5e-5,            # Matcher 已有梯度历史，低 LR 精调
            "aggregator": 2e-4,            # Aggregator 新生，给够能量
        },
        # 阶段内线性插值
        "loss_weights_start": {
            "distill": 0.8,
            "fold":    0.3,
            "photo":   0.0,
            "ssim":    0.0,
            "cycle":   0.0,
            "geo":     0.15,
            "tps_smooth": 0.3,
        },
        "loss_weights_end": {
            "distill": 0.5,
            "fold":    0.5,
            "photo":   0.8,   # 20 个 Epoch 后到 0.8
            "ssim":    0.0,
            "cycle":   0.0,
            "geo":     0.1,
            "tps_smooth": 0.5,
        },
    },
    # 目标：全网络端到端精调，引入 SSIM 和（可选）Cycle Loss，
    #       Backbone 以极小 LR 解冻，Cosine 缓慢收敛。
    {
        "name": "END_TO_END_FINETUNE",
        "epoch_start": 41,
        "epoch_end": 80,      # 可通过 --epochs 覆盖
        "freeze": [],
        "unfreeze_backbone": True,
        "scheduler": "cosine",
        "max_lr": {
            "backbone":   5e-6,   # 非常保守，只让预训练权重微调
            "matcher":    2e-5,
            "aggregator": 5e-5,
        },
        "loss_weights_start": {
            "distill": 0.4,
            "fold":    0.5,
            "photo":   0.8,
            "ssim":    0.0,   # SSIM 从 0 开始，前 10 个 Epoch 内线性升温
            "cycle":   0.0,
            "geo":     0.08,
            "tps_smooth": 0.2,
        },
        "loss_weights_end": {
            "distill": 0.3,
            "fold":    0.4,
            "photo":   0.6,
            "ssim":    0.3,
            "cycle":   0.0,   # Cycle Loss 风险较高，默认关闭；如需开启，从 0.1 起
            "geo":     0.05,
            "tps_smooth": 0.1,
        },
    },
]


def get_stage(epoch: int) -> Dict:
    """根据 epoch 返回当前阶段配置（超出范围则使用最后一个阶段）"""
    for s in STAGE_DEFS:
        if s["epoch_start"] <= epoch <= s["epoch_end"]:
            return s
    return STAGE_DEFS[-1]


def interpolate_loss_weights(stage: Dict, epoch: int) -> Dict[str, float]:
    """在阶段内对 Loss 权重做线性插值，消除阶跃跳变"""
    if "loss_weights" in stage:
        return dict(stage["loss_weights"])

    e_start = stage["epoch_start"]
    e_end = stage["epoch_end"]
    t = (epoch - e_start) / max(e_end - e_start, 1)
    t = float(max(0.0, min(1.0, t)))

    ws = stage["loss_weights_start"]
    we = stage["loss_weights_end"]
    return {k: ws[k] + (we[k] - ws[k]) * t for k in ws}



def _get_module_param_groups(model: AgriTPSStitcher, max_lr: Dict[str, float]) -> List[Dict]:
    """
    按子模块分组参数，支持每组独立 LR。
    只把 lr > 0 的组放入优化器（冻结模块不产生梯度，放入也无意义且浪费内存）。
    """
    groups = [
        {
            "name": "backbone",
            "params": list(model.matcher.backbone.parameters()),
            "lr": max_lr.get("backbone", 0.0),
        },
        {
            "name": "matcher",
            "params": [
                p for n, p in model.matcher.named_parameters()
                if "backbone" not in n
            ],
            "lr": max_lr.get("matcher", 2e-4),
        },
        {
            "name": "aggregator",
            "params": list(model.tps_estimator.parameters()),
            "lr": max_lr.get("aggregator", 2e-4),
        },
    ]
    return groups


def apply_freeze_state(model: AgriTPSStitcher, stage: Dict) -> None:
    """
    按阶段配置冻结/解冻模块。
    冻结时同时调用 .eval()，防止 BatchNorm / LayerNorm 统计量被小 batch 破坏。
    """
    frozen = set(stage.get("freeze", []))
    unfreeze_backbone = stage.get("unfreeze_backbone", True)

    # Backbone
    backbone_frozen = "backbone" in frozen or not unfreeze_backbone
    for p in model.matcher.backbone.parameters():
        p.requires_grad = not backbone_frozen
    if backbone_frozen:
        model.matcher.backbone.eval()

    # Matcher（不含 backbone）
    matcher_frozen = "matcher" in frozen
    for n, p in model.matcher.named_parameters():
        if "backbone" not in n:
            p.requires_grad = not matcher_frozen

    # Aggregator（BypassTPSEstimator）
    aggregator_frozen = "aggregator" in frozen
    for p in model.tps_estimator.parameters():
        p.requires_grad = not aggregator_frozen
    if aggregator_frozen:
        model.tps_estimator.eval()

    frozen_names = []
    if backbone_frozen:
        frozen_names.append("backbone")
    if matcher_frozen:
        frozen_names.append("matcher")
    if aggregator_frozen:
        frozen_names.append("aggregator")
    print(f"[Freeze] Frozen: {frozen_names or 'none'}")


def build_optimizer_and_scheduler(
    model: AgriTPSStitcher,
    stage: Dict,
    steps_per_epoch: int,
    weight_decay: float = 1e-4,
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    每次进入新阶段时调用，重建优化器和调度器。
    OneCycleLR 自带 warmup + annealing，适合新模块激活。
    Cosine 适合后期精细收敛。
    """
    param_groups = _get_module_param_groups(model, stage["max_lr"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    total_steps = (stage["epoch_end"] - stage["epoch_start"] + 1) * steps_per_epoch

    if stage["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=0.25,        # 前 25% warmup
            anneal_strategy="cos",
            div_factor=10,         # 起始 LR = max_lr / 10
            final_div_factor=500,  # 结束 LR = max_lr / 5000（比较温柔的结尾）
        )
    elif stage["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-8,
        )
    else:
        raise ValueError(f"Unknown scheduler: {stage['scheduler']}")

    return optimizer, scheduler

def update_optimizer_and_scheduler(
    model: AgriTPSStitcher, stage: Dict, steps_per_epoch: int,
    weight_decay: float = 1e-4, optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[torch.optim.Optimizer, Any]:

    param_groups_cfg = _get_module_param_groups(model, stage["max_lr"])

    if optimizer is None:
        optimizer = torch.optim.AdamW(param_groups_cfg, weight_decay=weight_decay)
    else:
        for g, cfg in zip(optimizer.param_groups, param_groups_cfg):
            g['lr'] = cfg['lr']

    total_steps = (stage["epoch_end"] - stage["epoch_start"] + 1) * steps_per_epoch
    if stage["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups_cfg],
            total_steps=total_steps,
            pct_start=0.25,  # 前 25% warmup
            anneal_strategy="cos",
            div_factor=10,  # 起始 LR = max_lr / 10
            final_div_factor=500,  # 结束 LR = max_lr / 5000（比较温柔的结尾）
        )
    elif stage["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-8,
        )
    else:
        raise ValueError(f"Unknown scheduler: {stage['scheduler']}")
    return optimizer, scheduler


def make_teacher_inputs(
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        teacher: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    img_a_lr = F.interpolate(
        img_a, size=(teacher.H_lr, teacher.W_lr),
        mode="bicubic", align_corners=False, antialias=True,
    )
    img_b_lr = F.interpolate(
        img_b, size=(teacher.H_lr, teacher.W_lr),
        mode="bicubic", align_corners=False, antialias=True,
    )

    if teacher.H_hr is None or teacher.W_hr is None:
        return img_a_lr, img_b_lr, None, None

    img_a_hr = F.interpolate(
        img_a, size=(teacher.H_hr, teacher.W_hr),
        mode="bicubic", align_corners=False, antialias=True,
    )
    img_b_hr = F.interpolate(
        img_b, size=(teacher.H_hr, teacher.W_hr),
        mode="bicubic", align_corners=False, antialias=True,
    )
    return img_a_lr, img_b_lr, img_a_hr, img_b_hr


@torch.no_grad()
def extract_teacher_features_ds(
        teacher: Any,
        img_a_lr: torch.Tensor,
        img_b_lr: torch.Tensor,
        grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """提取 teacher 特征并下采样到指定尺寸"""
    f_list_a = teacher.f(img_a_lr)
    f_list_b = teacher.f(img_b_lr)

    if f_list_a[0].shape[1] < f_list_a[0].shape[-1]:
        feat_a = torch.cat([x.float() for x in f_list_a], dim=1)
        feat_b = torch.cat([x.float() for x in f_list_b], dim=1)
        feat_a_ds = F.interpolate(
            feat_a, size=(grid_size, grid_size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        feat_b_ds = F.interpolate(
            feat_b, size=(grid_size, grid_size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
    else:
        feat_a = torch.cat([x.float() for x in f_list_a], dim=-1)
        feat_b = torch.cat([x.float() for x in f_list_b], dim=-1)
        feat_a_ds = F.interpolate(
            feat_a.permute(0, 3, 1, 2), size=(grid_size, grid_size),
            mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        feat_b_ds = F.interpolate(
            feat_b.permute(0, 3, 1, 2), size=(grid_size, grid_size),
            mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

    bsz = feat_a_ds.shape[0]
    teacher_dim = feat_a_ds.shape[-1]
    feat_a_flat = feat_a_ds.reshape(bsz, grid_size * grid_size, teacher_dim)
    feat_b_flat = feat_b_ds.reshape(bsz, grid_size * grid_size, teacher_dim)
    return feat_a_flat, feat_b_flat
