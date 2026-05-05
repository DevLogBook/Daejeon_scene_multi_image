from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_match.network import AgriStitcher

STAGE_DEFS = [
    # STAGE 1: supervised warm-up for base alignment and inlier prediction.
    # The stitch decoder is frozen so the matcher and inlier head learn a stable
    # global correspondence model before dense residual deformation is enabled.
    {
        "name": "SUPERVISED_WARMUP",
        "epoch_start": 1,
        "epoch_end": 12,
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
            "fold": 0.0,
            "photo_conf_threshold": 0.0,
            "inlier_sigma_teacher": 0.04,
            "inlier_sigma_h": 0.06,
            "inlier_conf_thresh": 0.5,
            "inlier_coverage": 0.03,
            "inlier_coverage_bins": 4,
            "inlier_coverage_min_mass": 0.03,
            "inlier": 0.5,
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
            "fold": 0.0,
            "photo_conf_threshold": 0.0,
            "inlier_sigma_teacher": 0.04,
            "inlier_sigma_h": 0.06,
            "inlier_conf_thresh": 0.5,
            "inlier_coverage": 0.03,
            "inlier_coverage_bins": 4,
            "inlier_coverage_min_mass": 0.03,
            "inlier": 0.4,
        },
    },

    # STAGE 2: activate the residual decoder with conservative photometric terms.
    # The H path remains supervised while the mesh/dense residual learns only the
    # part of the warp that the global transform cannot explain.
    {
        "name": "STITCH_DECODER_ACTIVATION",
        "epoch_start": 13,
        "epoch_end": 20,
        "freeze": [],  # Unfreeze all trainable modules selected by learning rate.
        "use_inlier_predictor": True,
        "unfreeze_backbone": False,
        "scheduler": "onecycle",
        "max_lr": {
            "backbone": 0.0,
            "matcher": 5e-5,
            "stitch_decoder": 2e-4,
            "inlier_predictor": 3e-5,
        },
        # Linearly interpolate loss weights within the stage.
        "loss_weights_start": {
            "distill": 0.8,
            "photo": 0.0,
            "ssim": 0.0,
            "cycle": 0.0,
            "geo": 0.15,
            "h_distill": 0.04,
            "h_match": 0.015,
            "h_residual_budget": 0.02,
            "residual_distill": 0.0,
            "stitch_residual": 0.15,
            "stitch_mask": 0.05,
            "stitch_mask_target": 0.04,
            "area_penalty": 0.0,
            "fold": 0.01,
            "photo_conf_threshold": 0.4,
            "inlier_sigma_teacher": 0.04,
            "inlier_sigma_h": 0.06,
            "inlier_conf_thresh": 0.5,
            "inlier_coverage": 0.02,
            "inlier_coverage_bins": 4,
            "inlier_coverage_min_mass": 0.03,
            "inlier": 0.3,
        },
        "loss_weights_end": {
            "distill": 0.5,
            "photo": 0.08,
            "ssim": 0.0,
            "cycle": 0.0,
            "geo": 0.1,
            "h_distill": 0.02,
            "h_match": 0.005,
            "h_residual_budget": 0.08,
            "residual_distill": 0.02,
            "stitch_residual": 0.1,
            "stitch_mask": 0.05,
            "stitch_mask_target": 0.08,
            "area_penalty": 0.0,
            "fold": 0.02,
            "photo_conf_threshold": 0.4,
            "inlier_sigma_teacher": 0.04,
            "inlier_sigma_h": 0.06,
            "inlier_conf_thresh": 0.5,
            "inlier_coverage": 0.02,
            "inlier_coverage_bins": 4,
            "inlier_coverage_min_mass": 0.03,
            "inlier": 0.2,
        },
    },

    # STAGE 3: end-to-end fine-tuning with small learning rates and cosine decay.
    # Photometric and perceptual terms are increased only after the geometry has
    # become stable enough to avoid residual-field collapse.
    {
        "name": "END_TO_END_FINETUNE",
        "epoch_start": 21,
        "epoch_end": 80,
        "freeze": [],
        "use_inlier_predictor": True,
        "unfreeze_backbone": True,
        "scheduler": "cosine",
        "max_lr": {
            "backbone": 5e-6,  # Very small LR for backbone fine-tuning.
            "matcher": 2e-5,
            "stitch_decoder": 5e-5,
            "inlier_predictor": 1e-5,
        },
        "loss_weights_start": {
            "distill": 0.4,
            "photo": 0.08,
            "ssim": 0.0,
            "cycle": 0.05,
            "geo": 0.08,
            "h_distill": 0.02,
            "h_match": 0.005,
            "h_residual_budget": 0.08,
            "residual_distill": 0.02,
            "stitch_residual": 0.1,
            "stitch_mask": 0.05,
            "stitch_mask_target": 0.08,
            "area_penalty": 0.0,
            "fold": 0.02,
            "photo_conf_threshold": 0.5,
            "inlier_sigma_teacher": 0.04,
            "inlier_sigma_h": 0.06,
            "inlier_conf_thresh": 0.5,
            "inlier_coverage": 0.01,
            "inlier_coverage_bins": 4,
            "inlier_coverage_min_mass": 0.03,
            "inlier": 0.1,
        },
        "loss_weights_end": {
            "distill": 0.2,
            "photo": 0.45,
            "ssim": 0.3,
            "cycle": 0.1,
            "geo": 0.05,
            "h_distill": 0.02,
            "h_match": 0.005,
            "h_residual_budget": 0.06,
            "residual_distill": 0.015,
            "stitch_residual": 0.05,
            "stitch_mask": 0.03,
            "stitch_mask_target": 0.10,
            "area_penalty": 0.0,
            "fold": 0.03,
            "photo_conf_threshold": 0.5,
            "inlier_sigma_teacher": 0.04,
            "inlier_sigma_h": 0.06,
            "inlier_conf_thresh": 0.5,
            "inlier_coverage": 0.0,
            "inlier_coverage_bins": 4,
            "inlier_coverage_min_mass": 0.03,
            "inlier": 0.05,
        },
    },
]


def get_stage(epoch: int) -> Dict:
    """Return the active stage for `epoch`; epochs beyond the range use the last stage."""
    for s in STAGE_DEFS:
        if s["epoch_start"] <= epoch <= s["epoch_end"]:
            return s
    return STAGE_DEFS[-1]


def interpolate_loss_weights(stage: Dict, epoch: int) -> Dict[str, float]:
    """
    Linearly interpolate loss weights within a stage.
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
    Group parameters by subsystem so each group can use an independent LR.

    Groups with non-positive LR are omitted to avoid optimizer state for frozen
    modules and to make the stage schedule explicit.
    """
    stitch_params = list(model.stitch_decoder.parameters()) if model.stitch_decoder is not None else []
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
            "params": stitch_params,
            "lr": max_lr.get("stitch_decoder", 2e-4),
        },
    ]
    return [g for g in groups if g["params"]]


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

    stitch_decoder_frozen = stitch_decoder_key in frozen or model.stitch_decoder is None
    if model.stitch_decoder is not None:
        for p in model.stitch_decoder.parameters():
            p.requires_grad = not stitch_decoder_frozen
    if stitch_decoder_frozen and model.stitch_decoder is not None:
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
    Rebuild the optimizer and scheduler when the training stage changes.

    OneCycleLR is used when new modules are activated. Cosine decay is used for
    late fine-tuning after the geometry has stabilized.
    """
    param_groups = _get_module_param_groups(model, stage["max_lr"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    total_steps = (stage["epoch_end"] - stage["epoch_start"] + 1) * steps_per_epoch

    if stage["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=0.25,        # First 25% of steps are warm-up.
            anneal_strategy="cos",
            div_factor=10,         # Initial LR = max_lr / 10.
            final_div_factor=500,  # Final LR = max_lr / (10 * 500).
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
    """Extract teacher features and downsample them to the requested grid size."""
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
