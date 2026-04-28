from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_match.network import AgriStitcher

STAGE_DEFS = [
    # STAGE 1: 基础对齐与内点识别 (Epoch 1-20)
    # 目标：Aggregator 冻结。让 Matcher 学会特征匹配，
    #       同时让 inlier_predictor 学会区分真假匹配。
    {
        "name": "SUPERVISED_WARMUP",
        "epoch_start": 1,
        "epoch_end": 20,
        "freeze": ["stitch_decoder"],
        "use_inlier_predictor": True,
        "unfreeze_backbone": True,
        "scheduler": "onecycle",
        "max_lr": {
            "backbone": 2e-5,
            "matcher": 2e-4,
            "stitch_decoder": 0.0,
            "inlier_predictor": 1e-4,
        },
        "loss_weights_start": {
            "distill": 1.0,
            "photo": 0.0,
            "ssim": 0.0,
            "cycle": 0.0,
            "geo": 0.15,
            "h_distill": 0.03,
            "h_match": 0.01,
            "h_residual_budget": 0.0,
            "residual_distill": 0.0,
            "stitch_residual": 0.0,
            "stitch_mask": 0.0,
            "stitch_mask_target": 0.03,
            "area_penalty": 0.0,
            "inlier": 1.0,
        },
        "loss_weights_end": {
            "distill": 1.0,
            "photo": 0.0,
            "ssim": 0.0,
            "cycle": 0.0,
            "geo": 0.15,
            "h_distill": 0.05,
            "h_match": 0.02,
            "h_residual_budget": 0.0,
            "residual_distill": 0.0,
            "stitch_residual": 0.0,
            "stitch_mask": 0.0,
            "stitch_mask_target": 0.03,
            "area_penalty": 0.0,
            "inlier": 1.0,
        },
    },

    # STAGE 2: 变形场激活与光度蒸馏 (Epoch 21-40)
    # 目标：Aggregator 解冻，开始从学习。
    #       利用已经学好的 H 矩阵，配合逐渐升温的 Photo Loss 寻找最优解。
    {
        "name": "STITCH_DECODER_ACTIVATION",
        "epoch_start": 21,
        "epoch_end": 40,
        "freeze": [],  # 全部解冻
        "use_inlier_predictor": True,
        "unfreeze_backbone": False,
        "scheduler": "onecycle",
        "max_lr": {
            "backbone": 0.0,
            "matcher": 5e-5,
            "stitch_decoder": 2e-4,
            "inlier_predictor": 3e-5,
        },
        # 阶段内线性插值
        "loss_weights_start": {
            "distill": 0.8,
            "photo": 0.0,
            "ssim": 0.0,
            "cycle": 0.0,
            "geo": 0.15,
            "h_distill": 0.08,
            "h_match": 0.03,
            "h_residual_budget": 0.02,
            "residual_distill": 0.02,
            "stitch_residual": 0.15,
            "stitch_mask": 0.05,
            "stitch_mask_target": 0.04,
            "area_penalty": 0.0,
            "inlier": 0.5,
        },
        "loss_weights_end": {
            "distill": 0.5,
            "photo": 0.25,
            "ssim": 0.0,
            "cycle": 0.0,
            "geo": 0.1,
            "h_distill": 0.05,
            "h_match": 0.02,
            "h_residual_budget": 0.08,
            "residual_distill": 0.08,
            "stitch_residual": 0.1,
            "stitch_mask": 0.05,
            "stitch_mask_target": 0.08,
            "area_penalty": 0.25,
            "inlier": 0.3,
        },
    },

    # STAGE 3: 全网络端到端收敛 (Epoch 41-80)
    # 目标：解冻所有模块，引入极小 LR 和余弦退火。
    #       开启感知级 Loss (SSIM, Cycle) 进行像素级打磨。
    {
        "name": "END_TO_END_FINETUNE",
        "epoch_start": 41,
        "epoch_end": 80,
        "freeze": [],
        "use_inlier_predictor": True,
        "unfreeze_backbone": True,
        "scheduler": "cosine",
        "max_lr": {
            "backbone": 5e-6,  # 极微弱地调优骨干网络
            "matcher": 2e-5,
            "stitch_decoder": 5e-5,
            "inlier_predictor": 1e-5,
        },
        "loss_weights_start": {
            "distill": 0.4,
            "photo": 0.25,
            "ssim": 0.0,
            "cycle": 0.05,
            "geo": 0.08,
            "h_distill": 0.04,
            "h_match": 0.015,
            "h_residual_budget": 0.08,
            "residual_distill": 0.06,
            "stitch_residual": 0.1,
            "stitch_mask": 0.05,
            "stitch_mask_target": 0.08,
            "area_penalty": 0.25,
            "inlier": 0.2,
        },
        "loss_weights_end": {
            "distill": 0.2,
            "photo": 0.45,
            "ssim": 0.3,
            "cycle": 0.1,
            "geo": 0.05,
            "h_distill": 0.02,
            "h_match": 0.01,
            "h_residual_budget": 0.06,
            "residual_distill": 0.04,
            "stitch_residual": 0.05,
            "stitch_mask": 0.03,
            "stitch_mask_target": 0.10,
            "area_penalty": 0.5,
            "inlier": 0.1,
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
    """
    在阶段内对 Loss 权重做线性插值。
    """
    if "loss_weights" in stage:
        return dict(stage["loss_weights"])

    e_start = stage["epoch_start"]
    e_end = stage["epoch_end"]
    span = max(e_end - e_start, 1)

    t = (epoch - e_start + 0.5) / span
    t = float(max(0.0, min(1.0, t)))

    ws = stage["loss_weights_start"]
    we = stage["loss_weights_end"]
    return {k: ws[k] + (we[k] - ws[k]) * t for k in ws}



def _get_module_param_groups(model: AgriStitcher, max_lr: Dict[str, float]) -> List[Dict]:
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
                if "backbone" not in n and "inlier_predictor" not in n
            ],
            "lr": max_lr.get("matcher", 2e-4),
        },
        {
            "name": "inlier_predictor",
            "params": list(model.matcher.inlier_predictor.parameters()),
            "lr": max_lr.get("inlier_predictor", 0.0),
        },
        {
            "name": "stitch_decoder",
            "params": list(model.stitch_decoder.parameters()),
            "lr": max_lr.get("stitch_decoder", 2e-4),
        },
    ]
    return groups


def apply_freeze_state(model: AgriStitcher, stage: Dict) -> None:
    frozen = set(stage.get("freeze", []))
    stitch_decoder_key = "stitch_decoder"
    unfreeze_backbone = stage.get("unfreeze_backbone", True)
    use_inlier_predictor = stage.get("use_inlier_predictor", False)

    model.matcher.use_inlier_predictor = use_inlier_predictor

    backbone_frozen = "backbone" in frozen or not unfreeze_backbone
    for p in model.matcher.backbone.parameters():
        p.requires_grad = not backbone_frozen
    if backbone_frozen:
        model.matcher.backbone.eval()

    inlier_frozen = ("inlier_predictor" in frozen) or not use_inlier_predictor
    for p in model.matcher.inlier_predictor.parameters():
        p.requires_grad = not inlier_frozen
    if inlier_frozen:
        model.matcher.inlier_predictor.eval()

    matcher_frozen = "matcher" in frozen
    for n, p in model.matcher.named_parameters():
        if "backbone" not in n and "inlier_predictor" not in n:
            p.requires_grad = not matcher_frozen

    stitch_decoder_frozen = stitch_decoder_key in frozen
    for p in model.stitch_decoder.parameters():
        p.requires_grad = not stitch_decoder_frozen
    if stitch_decoder_frozen:
        model.stitch_decoder.eval()

    frozen_names = [n for n, flag in [
        ("backbone", backbone_frozen), ("inlier_predictor", inlier_frozen),
        ("matcher", matcher_frozen), ("stitch_decoder", stitch_decoder_frozen)
    ] if flag]
    print(f"[Freeze] Frozen: {frozen_names or 'none'}")


def build_optimizer_and_scheduler(
    model: AgriStitcher,
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
        model, stage, steps_per_epoch, weight_decay=1e-4, optimizer=None
):
    param_groups_cfg = _get_module_param_groups(model, stage["max_lr"])
    optimizer = torch.optim.AdamW(param_groups_cfg, weight_decay=weight_decay)

    total_steps = (stage["epoch_end"] - stage["epoch_start"] + 1) * steps_per_epoch
    if stage["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups_cfg],
            total_steps=total_steps,
            pct_start=0.25,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=500,
        )
    elif stage["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-8,
        )
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
