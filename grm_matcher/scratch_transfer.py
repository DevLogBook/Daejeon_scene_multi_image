"""Supervision and data adapters for scratch IMC -> exact-H transfer training."""
from __future__ import annotations

import json
import math
import posixpath
from pathlib import Path

import torch
import torch.nn.functional as F

from .coordinates import image_xy_to_normalized_grid
from .homography_pair_dataset import GRMHomographyPairDataset
from .imc_dataset import sampson_squared, sparse_bidirectional_labels
from .losses import GRMMatcherLoss
from .real_adaptation import GRMStage25CoarsePseudoLoss, forward_student_coarse
from .stage25 import (
    GRMStage25Config, GRMStage25ConsistencyLoss, forward_student_refinement,
    make_student_augmentation, transform_points, transform_pseudo_labels,
    _warp_image,
)


GEOMETRY_PREFIXES = ("token_encoder.", "geometry_graph.", "rerank_head.", "role_head.", "inlier_head.", "geometry_gain")


def lr_multiplier(step, total_steps, warmup_steps, minimum_ratio=.05):
    """Linear warmup from 10% to peak, then cosine to minimum_ratio."""
    if total_steps <= 0 or step < 0 or not 0 < minimum_ratio <= 1:
        raise ValueError("Invalid learning-rate schedule")
    warmup = min(max(0, warmup_steps), max(0, total_steps - 1))
    if step < warmup:
        return .1 + .9 * step / max(1, warmup - 1)
    progress = min(1., max(0., (step - warmup) / max(1, total_steps - warmup - 1)))
    return minimum_ratio + (1 - minimum_ratio) * .5 * (1 + math.cos(math.pi * progress))


def configure_optimizer(model, stage, *, imc_lr=2e-4, backbone_lr=1e-5,
                        coarse_lr=2e-5, refinement_lr=5e-5, geometry_lr=1e-4,
                        weight_decay=1e-4):
    """Scratch trains all matching layers; H fine-tuning trains the whole GRM."""
    groups = {}
    for name, parameter in model.named_parameters():
        geometry = name.startswith(GEOMETRY_PREFIXES)
        parameter.requires_grad_(stage == "agri" or not geometry)
        if not parameter.requires_grad:
            continue
        if stage == "imc":
            group, rate = "matching", imc_lr
        elif geometry:
            group, rate = "geometry", geometry_lr
        elif name.startswith("backbone."):
            group, rate = "backbone", backbone_lr
        elif name.startswith(("coarse_transformer.", "correlation.", "fpn.coarse_lateral.", "fpn.coarse_output.")):
            group, rate = "coarse", coarse_lr
        else:
            group, rate = "refinement", refinement_lr
        groups.setdefault(group, {"params": [], "lr": rate, "peak_lr": rate, "name": group})["params"].append(parameter)
    return torch.optim.AdamW(list(groups.values()), weight_decay=weight_decay)


def check_homography_splits(root):
    """Compare source identifiers in manifests, without loading test images."""
    sources, counts = {}, {}
    for split in ("train", "val", "test"):
        path = Path(root) / split / "manifest.jsonl"
        if not path.is_file() and split == "test":
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or any(not row.get("source_image") for row in rows):
            raise ValueError(f"{split}: source_image is required to check source split isolation")
        # Crop/scene directories can legitimately reuse names such as 1.JPG.
        # Preserve the source namespace; still reject the same normalized path.
        sources[split] = {posixpath.normpath(r["source_image"].replace("\\", "/")).casefold() for r in rows}
        counts[split] = {"pairs": len(rows), "source_names": len(sources[split])}
    keys = list(sources)
    for i, first in enumerate(keys):
        for second in keys[i + 1:]:
            overlap = sources[first] & sources[second]
            if overlap:
                raise ValueError(f"Source identifiers overlap between {first}/{second}: {sorted(overlap)[:5]}")
    return counts


class ResizedHomographyDataset(GRMHomographyPairDataset):
    """Reuse original image/mask/H loading; rebuild exact targets after resizing."""
    def __init__(self, root, split, config, size=512, limit=0):
        super().__init__(root, split, config)
        if size <= 0 or size % config.coarse_stride:
            raise ValueError("Agricultural size must be a positive multiple of coarse stride")
        self.size = size
        if limit:
            self.records = self.records[:limit]
        self.bucket_to_indices = {f"{size}x{size}": list(range(len(self.records)))}

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["pair_id"] = str(self.records[index].get("pair_id", f"sample_index:{index}"))
        old_h, old_w = sample["image_a"].shape[-2:]
        size = self.size
        transform = torch.tensor([[size / old_w, 0., (size / old_w - 1) / 2],
                                  [0., size / old_h, (size / old_h - 1) / 2],
                                  [0., 0., 1.]], dtype=torch.float64)
        h = transform @ sample["H_ab"].double() @ torch.linalg.inv(transform)
        h = (h / h[2, 2]).float()
        for key in ("image_a", "image_b"):
            sample[key] = F.interpolate(sample[key][None], (size, size), mode="bilinear", align_corners=False)[0]
        for key in ("valid_a", "valid_b", "gt_overlap_a"):
            sample[key] = F.interpolate(sample[key][None].float(), (size, size), mode="nearest")[0] > .5
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        points = torch.stack((x, y), -1).float().reshape(1, -1, 2)
        target = transform_points(points, h[None]).reshape(size, size, 2)
        reverse = transform_points(points, torch.linalg.inv(h)[None]).reshape(size, size, 2)
        def visible(mapping, target_mask):
            inside = ((mapping >= 0) & (mapping <= size - 1)).all(-1)
            grid = image_xy_to_normalized_grid(mapping, size, size)[None]
            sampled = F.grid_sample(target_mask[None].float(), grid, mode="nearest", align_corners=False)[0, 0] > .5
            return inside[None] & sampled[None]
        overlap_a = sample["valid_a"] & sample["gt_overlap_a"] & visible(target, sample["valid_b"])
        overlap_b = sample["valid_b"] & visible(reverse, overlap_a)
        sample.update(H_ab=h, gt_dense_mapping=target.permute(2, 0, 1).contiguous(),
                      gt_overlap_a=overlap_a, gt_overlap_b=overlap_b,
                      gt_global_mask=overlap_a.clone(), gt_local_mask=torch.zeros_like(overlap_a),
                      local_flow=torch.zeros(2, size, size))
        return sample


def exact_h_labels(batch, config, adaptation):
    """Evenly distributed coarse source anchors with exact H targets; no Teacher."""
    image = batch["image_a"]
    b, _, height, width = image.shape
    stride = config.coarse_stride
    y, x = torch.meshgrid(torch.arange(height // stride, device=image.device),
                          torch.arange(width // stride, device=image.device), indexing="ij")
    source = (torch.stack((x, y), -1).float().reshape(-1, 2) + .5) * stride - .5
    # Spread anchors throughout the grid rather than taking the first rows.
    indices = torch.linspace(0, len(source) - 1, min(len(source), config.max_source_queries), device=image.device).round().long()
    source = source[indices][None].expand(b, -1, -1)
    target = transform_points(source, batch["H_ab"])
    grid = image_xy_to_normalized_grid(source, height, width).unsqueeze(2)
    overlap = F.grid_sample(batch["gt_overlap_a"].float(), grid, mode="nearest", align_corners=False).squeeze(1).squeeze(-1) > .5
    valid = overlap & torch.isfinite(target).all(-1)
    valid &= (target[..., 0] >= 0) & (target[..., 0] <= width - 1) & (target[..., 1] >= 0) & (target[..., 1] <= height - 1)
    sparse = {"image_a": image, "sparse_a": source, "sparse_b": target,
              "sparse_valid": valid, "sparse_confidence": valid.float()}
    return sparse_bidirectional_labels(sparse, image.device, config, adaptation).forward


class ScratchTransferObjective:
    def __init__(self, config, epipolar_weight=.1, anchor_weight=.25, exact_dense=False):
        self.config = config
        self.epipolar_weight, self.anchor_weight = epipolar_weight, anchor_weight
        # SIFT/known-H labels do not need GRM Teacher confidence gating.
        self.adaptation = GRMStage25Config(use_absolute_batch_reliability=False,
            homography_loss_weight=0., source_replay_weight=0., min_pseudo_matches=1 if exact_dense else 8,
            coarse_min_pseudo_matches=1 if exact_dense else 8, coarse_min_covered_regions=1)
        self.coarse = GRMStage25CoarsePseudoLoss(config, self.adaptation)
        self.refine = GRMStage25ConsistencyLoss(config, self.adaptation)
        self.dense = GRMMatcherLoss(config)

    def imc(self, model, batch, train):
        a, b = batch["image_a"], batch["image_b"]
        labels = sparse_bidirectional_labels(batch, a.device, self.config, self.adaptation)
        fundamental = batch["F_ab"].float()
        if train:
            aug = make_student_augmentation(a, b, self.config, self.adaptation)
            a, b = aug.image_a, aug.image_b
            ta, tb = aug.transform_a, aug.transform_b
        else:
            ta = tb = torch.eye(3, device=a.device)[None].expand(a.shape[0], -1, -1)
        height, width = a.shape[-2:]
        forward = transform_pseudo_labels(labels.forward, ta, tb, height, width, self.config, self.adaptation)
        backward = transform_pseudo_labels(labels.backward, tb, ta, height, width, self.config, self.adaptation)
        count = int(forward.valid_mask.sum() + backward.valid_mask.sum())
        if not count:
            return None, {"valid_count": 0.}
        valid_a, valid_b = batch.get("valid_a"), batch.get("valid_b")
        if train:
            valid_a = _warp_image(valid_a.float(), ta) > .999 if valid_a is not None else None
            valid_b = _warp_image(valid_b.float(), tb) > .999 if valid_b is not None else None
        if self.config.architecture_version == "v3":
            from .training_v3 import supervised_v3
            f_new = torch.linalg.inv(tb).transpose(1, 2) @ fundamental @ torch.linalg.inv(ta)
            return supervised_v3(self, model, a, b, forward, backward, valid_a, valid_b, fundamental=f_new)
        coarse = forward_student_coarse(model, a, b, valid_a, valid_b)
        coarse_losses = self.coarse(coarse, forward, backward)
        prediction = forward_student_refinement(model, a, b, forward, valid_a, valid_b)
        refinement = self.refine(prediction, forward, apply_absolute_reliability=False)
        # Direct differentiable F supervision of predicted fine coordinates.
        f_new = torch.linalg.inv(tb).transpose(1, 2) @ fundamental @ torch.linalg.inv(ta)
        distance = sampson_squared(forward.src_xy, prediction.fine.dst_xy, f_new)
        mask = forward.valid_mask
        epipolar = (torch.sqrt(1 + distance[mask]) - 1).mean() if mask.any() else prediction.fine.dst_xy.sum() * 0
        total = coarse_losses["total"] + refinement["total"] + self.epipolar_weight * epipolar
        metrics = {"loss": total.detach(), "coarse": coarse_losses["total"].detach(),
                   "refinement": refinement["total"].detach(), "epipolar": epipolar.detach(),
                   "valid_count": float(count)}
        # Keep successful epoch/CSV schemas compatible with existing resumes.
        if not torch.isfinite(total):
            metrics.update({f"coarse_{k}": v.detach() for k, v in coarse_losses.items() if k != "total"})
            metrics.update({f"refine_{k}": v.detach() for k, v in refinement.items() if k != "total"})
        return total, metrics

    def agri(self, model, batch, train, epoch):
        if self.config.architecture_version == "v3":
            from .training_v3 import agricultural_v3
            return agricultural_v3(self, model, batch)
        output, auxiliary = model(batch["image_a"], batch["image_b"], return_auxiliary=True,
                                  detach_geometry_inputs=train and epoch < 2,
                                  valid_a=batch.get("valid_a"), valid_b=batch.get("valid_b"))
        losses = self.dense(output, batch, auxiliary, stage="joint")
        # Always available exact anchors prevent local-refinement starvation
        # when learned coarse candidates miss GT early in transfer.
        labels = exact_h_labels(batch, self.config, self.adaptation)
        predicted = forward_student_refinement(model, batch["image_a"], batch["image_b"], labels,
            batch.get("valid_a"), batch.get("valid_b"))
        anchor = self.refine(predicted, labels, apply_absolute_reliability=False)
        total = losses["total"] + self.anchor_weight * anchor["total"]
        grid = image_xy_to_normalized_grid(output.src_xy, batch["image_a"].shape[-2], batch["image_a"].shape[-1]).unsqueeze(2)
        overlap = F.grid_sample(batch["gt_overlap_a"].float(), grid, align_corners=False).squeeze(1).squeeze(-1) > .5
        gt = transform_points(output.src_xy, batch["H_ab"])
        error = torch.linalg.vector_norm(output.dst_xy - gt, dim=-1)
        mask = overlap & output.valid_mask
        correct = ((error <= 3) & mask).sum()
        metrics = {name: value.detach() for name, value in losses.items() if name != "zero"}
        metrics.update(loss=total.detach(), anchor=anchor["total"].detach(),
                       valid_count=float(labels.valid_mask.sum()),
                       predicted_matches=float(mask.sum()), correct_at_3px=float(correct),
                       global_gt_count=float((mask & (error <= self.config.correspondence_correct_threshold)).sum()),
                       false_gt_count=float((output.valid_mask & (~overlap | (error > self.config.correspondence_correct_threshold))).sum()),
                       local_gt_count=0.)
        if self.config.classification_mode == "inlier":
            metrics["inlier_gt_count"] = metrics.pop("global_gt_count")
            metrics["outlier_gt_count"] = metrics.pop("false_gt_count")
            metrics.pop("local_gt_count")
            truth = self.dense._inlier_targets(output, batch).bool()
            accepted = (output.p_inlier >= .5) & output.valid_mask
            metrics.update(inlier_tp=float((accepted & truth).sum()),
                           inlier_fp=float((accepted & ~truth).sum()),
                           inlier_fn=float((~accepted & truth & output.valid_mask).sum()))
        elif self.config.classification_mode == "none":
            for key in ("global_gt_count", "false_gt_count", "local_gt_count"):
                metrics.pop(key)
        return total, metrics
