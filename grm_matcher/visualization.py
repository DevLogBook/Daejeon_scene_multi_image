"""Matplotlib visualizations for matches, hypotheses, entropy, roles, and geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import torch
from torch import Tensor


GRM_ROLE_COLORS = {0: "#20bf55", 1: "#ffd23f", 2: "#e63946"}


def visualize_matches(
    image_a: Tensor | np.ndarray,
    image_b: Tensor | np.ndarray,
    src_xy: Tensor,
    dst_xy: Tensor,
    p_match: Tensor,
    p_global: Optional[Tensor] = None,
    p_local: Optional[Tensor] = None,
    p_false: Optional[Tensor] = None,
    candidate_entropy: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    *,
    max_matches: int = 100,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Show source/target points, lines, probability, roles, and entropy."""

    a = _image_numpy(image_a)
    b = _image_numpy(image_b)
    src, dst, probability, valid = _point_inputs(src_xy, dst_xy, p_match, valid_mask)
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if indices.numel() > max_matches:
        order = torch.topk(probability[indices], max_matches).indices
        indices = indices[order]
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    axes[0].imshow(a)
    axes[1].imshow(b)
    axes[0].set_title("Source A")
    axes[1].set_title("Target B")
    role_probabilities = None
    if p_global is not None and p_local is not None and p_false is not None:
        role_probabilities = torch.stack(
            (_first_batch(p_global), _first_batch(p_local), _first_batch(p_false)), dim=-1
        ).detach().cpu()
    entropy = (
        _first_batch(candidate_entropy).detach().cpu()
        if candidate_entropy is not None
        else None
    )
    for display_index, index_tensor in enumerate(indices):
        index = int(index_tensor)
        role = int(role_probabilities[index].argmax()) if role_probabilities is not None else 0
        color = GRM_ROLE_COLORS[role] if role_probabilities is not None else plt.cm.viridis(
            float(probability[index].clamp(0.0, 1.0))
        )
        axes[0].scatter(src[index, 0], src[index, 1], s=15, c=[color])
        axes[1].scatter(dst[index, 0], dst[index, 1], s=15, c=[color])
        figure.add_artist(
            ConnectionPatch(
                xyA=src[index].tolist(),
                xyB=dst[index].tolist(),
                coordsA="data",
                coordsB="data",
                axesA=axes[0],
                axesB=axes[1],
                color=color,
                linewidth=0.8,
                alpha=0.65,
            )
        )
        if display_index < 24:
            label = f"p={float(probability[index].detach().cpu()):.2f}"
            if role_probabilities is not None:
                label += f" G/L/F={role_probabilities[index, 0]:.2f}/{role_probabilities[index, 1]:.2f}/{role_probabilities[index, 2]:.2f}"
            if entropy is not None:
                label += f" H={entropy[index].detach().item():.2f}"
            axes[1].annotate(label, dst[index].tolist(), fontsize=5, color=color)
    for axis in axes:
        axis.set_axis_off()
    _save(figure, save_path)
    return figure


def visualize_topk_candidates(
    image_a: Tensor | np.ndarray,
    image_b: Tensor | np.ndarray,
    src_xy: Tensor,
    candidate_dst_xy: Tensor,
    candidate_probs: Tensor,
    valid_mask: Optional[Tensor] = None,
    *,
    max_queries: int = 32,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Draw all retained target hypotheses for spatially selected queries."""

    a, b = _image_numpy(image_a), _image_numpy(image_b)
    src = _first_batch(src_xy).detach().cpu()
    candidates = _first_batch(candidate_dst_xy).detach().cpu()
    probabilities = _first_batch(candidate_probs).detach().cpu()
    valid = (
        torch.ones(src.shape[0], dtype=torch.bool)
        if valid_mask is None
        else _first_batch(valid_mask).detach().cpu().bool()
    )
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)[:max_queries]
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    axes[0].imshow(a)
    axes[1].imshow(b)
    for index_tensor in indices:
        index = int(index_tensor)
        axes[0].scatter(*src[index].tolist(), c="cyan", s=20)
        for candidate_index in range(candidates.shape[1]):
            alpha = 0.2 + 0.8 * float(probabilities[index, candidate_index])
            point = candidates[index, candidate_index]
            axes[1].scatter(*point.tolist(), c="cyan", s=10 + 35 * alpha, alpha=alpha)
            axes[1].annotate(
                f"{candidate_index + 1}:{float(probabilities[index, candidate_index]):.2f}",
                point.tolist(),
                fontsize=5,
                color="cyan",
            )
    axes[0].set_title("Source queries")
    axes[1].set_title("Top-k target hypotheses")
    for axis in axes:
        axis.set_axis_off()
    _save(figure, save_path)
    return figure


def visualize_candidate_entropy(
    image_a: Tensor | np.ndarray,
    src_xy: Tensor,
    candidate_entropy: Tensor,
    valid_mask: Optional[Tensor] = None,
    *,
    save_path: Optional[str | Path] = None,
) -> Figure:
    image = _image_numpy(image_a)
    src = _first_batch(src_xy).detach().cpu()
    entropy = _first_batch(candidate_entropy).detach().cpu()
    valid = _valid_mask(valid_mask, src.shape[0])
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(image)
    scatter = axis.scatter(
        src[valid, 0], src[valid, 1], c=entropy[valid], cmap="magma", s=30, edgecolors="white", linewidths=0.25
    )
    figure.colorbar(scatter, ax=axis, label="Candidate entropy")
    axis.set_title("Top-k ambiguity")
    axis.set_axis_off()
    _save(figure, save_path)
    return figure


def visualize_geometry_roles(
    image_a: Tensor | np.ndarray,
    image_b: Tensor | np.ndarray,
    src_xy: Tensor,
    dst_xy: Tensor,
    p_global: Tensor,
    p_local: Tensor,
    p_false: Tensor,
    valid_mask: Optional[Tensor] = None,
    *,
    save_path: Optional[str | Path] = None,
) -> Figure:
    probabilities = torch.stack((p_global, p_local, p_false), dim=-1)
    roles = _first_batch(probabilities).argmax(dim=-1)
    confidence = _first_batch(probabilities).amax(dim=-1)
    figure = visualize_matches(
        image_a,
        image_b,
        src_xy,
        dst_xy,
        confidence,
        p_global,
        p_local,
        p_false,
        valid_mask=valid_mask,
        save_path=None,
    )
    figure.suptitle("Geometry roles: Global=green, Local=yellow, False=red")
    _save(figure, save_path)
    return figure


def visualize_role_ground_truth_comparison(
    image_a: Tensor | np.ndarray,
    image_b: Tensor | np.ndarray,
    src_xy: Tensor,
    predicted_dst_xy: Tensor,
    gt_dst_xy: Tensor,
    p_global: Tensor,
    p_local: Tensor,
    p_false: Tensor,
    target_roles: Tensor,
    correspondence_error: Tensor,
    valid_mask: Tensor,
    *,
    max_matches: int = 100,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Compare predicted geometry roles with exact dense-GT roles.

    The upper panel shows model endpoints and predicted role colours.  The
    lower panel uses exact GT endpoints and role labels generated with the
    configured correspondence-correctness threshold.  This makes a wrong
    correspondence that happens to lie within the broad Local residual band
    visually distinguishable from a genuinely correct Local match.
    """

    a = _image_numpy(image_a)
    b = _image_numpy(image_b)
    canvas = np.concatenate((a, b), axis=1)
    target_offset = float(a.shape[1])
    src = _first_batch(src_xy).detach().float().cpu()
    predicted_dst = _first_batch(predicted_dst_xy).detach().float().cpu()
    gt_dst = _first_batch(gt_dst_xy).detach().float().cpu()
    probabilities = torch.stack(
        (
            _first_batch(p_global),
            _first_batch(p_local),
            _first_batch(p_false),
        ),
        dim=-1,
    ).detach().float().cpu()
    predicted_roles = probabilities.argmax(dim=-1)
    target = _first_batch(target_roles).detach().long().cpu()
    errors = _first_batch(correspondence_error).detach().float().cpu()
    valid = _first_batch(valid_mask).detach().bool().cpu()
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if indices.numel() > max_matches:
        confidence = probabilities.amax(dim=-1)
        indices = indices[torch.topk(confidence[indices], max_matches).indices]

    figure, axes = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)
    for axis in axes:
        axis.imshow(canvas)
        axis.axvline(target_offset, color="white", linewidth=0.7, alpha=0.65)
        axis.set_axis_off()

    for index_tensor in indices:
        index = int(index_tensor)
        predicted_color = GRM_ROLE_COLORS[int(predicted_roles[index])]
        target_color = GRM_ROLE_COLORS[int(target[index])]
        axes[0].plot(
            [float(src[index, 0]), float(predicted_dst[index, 0]) + target_offset],
            [float(src[index, 1]), float(predicted_dst[index, 1])],
            color=predicted_color,
            linewidth=0.7,
            alpha=0.65,
        )
        axes[0].scatter(
            float(predicted_dst[index, 0]) + target_offset,
            float(predicted_dst[index, 1]),
            c=predicted_color,
            s=10,
        )
        axes[1].plot(
            [float(src[index, 0]), float(gt_dst[index, 0]) + target_offset],
            [float(src[index, 1]), float(gt_dst[index, 1])],
            color=target_color,
            linewidth=0.7,
            alpha=0.65,
        )
        axes[1].scatter(
            float(gt_dst[index, 0]) + target_offset,
            float(gt_dst[index, 1]),
            c=target_color,
            s=10,
        )
        # A magenta cross exposes the model endpoint when it differs from GT.
        if float(errors[index]) > 1.0:
            axes[1].scatter(
                float(predicted_dst[index, 0]) + target_offset,
                float(predicted_dst[index, 1]),
                c="#d946ef",
                marker="x",
                s=14,
                linewidths=0.7,
            )

    selected_correct = (
        predicted_roles[indices] == target[indices]
    ).float() if indices.numel() else torch.zeros(1)
    mean_error = (
        float(errors[indices].mean()) if indices.numel() else float("nan")
    )
    axes[0].set_title(
        "Predicted roles and predicted endpoints "
        f"(display role accuracy={float(selected_correct.mean()):.3f})"
    )
    axes[1].set_title(
        "GT roles and exact endpoints "
        f"(model endpoint shown as magenta × when error > 1 px; mean error={mean_error:.2f}px)"
    )
    legend = [
        Line2D([0], [0], color=GRM_ROLE_COLORS[0], label="Global"),
        Line2D([0], [0], color=GRM_ROLE_COLORS[1], label="Local"),
        Line2D([0], [0], color=GRM_ROLE_COLORS[2], label="False"),
        Line2D(
            [0], [0], color="#d946ef", marker="x", linestyle="None",
            label="Predicted endpoint differs from GT",
        ),
    ]
    axes[0].legend(handles=legend[:3], loc="lower right")
    axes[1].legend(handles=legend, loc="lower right")
    _save(figure, save_path)
    return figure


def visualize_seed_h_residuals(
    image_b: Tensor | np.ndarray,
    dst_xy: Tensor,
    residual_xy: Tensor,
    valid_mask: Optional[Tensor] = None,
    *,
    save_path: Optional[str | Path] = None,
) -> Figure:
    image = _image_numpy(image_b)
    dst = _first_batch(dst_xy).detach().cpu()
    residual = _first_batch(residual_xy).detach().cpu()
    valid = _valid_mask(valid_mask, dst.shape[0])
    magnitude = torch.linalg.vector_norm(residual, dim=-1)
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(image)
    quiver = axis.quiver(
        dst[valid, 0],
        dst[valid, 1],
        residual[valid, 0],
        residual[valid, 1],
        magnitude[valid],
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=1.0,
    )
    figure.colorbar(quiver, ax=axis, label="Seed H residual (px)")
    axis.set_title("Seed Homography residual vectors")
    axis.set_axis_off()
    _save(figure, save_path)
    return figure


def visualize_spatial_coverage(
    image_a: Tensor | np.ndarray,
    src_xy: Tensor,
    reliable_mask: Tensor,
    *,
    grid_rows: int = 8,
    grid_cols: int = 8,
    save_path: Optional[str | Path] = None,
) -> Figure:
    image = _image_numpy(image_a)
    height, width = image.shape[:2]
    src = _first_batch(src_xy).detach().cpu()
    reliable = _first_batch(reliable_mask).detach().cpu().bool()
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(image)
    for row in range(grid_rows):
        for col in range(grid_cols):
            inside = (
                reliable
                & (src[:, 0] >= col * width / grid_cols)
                & (src[:, 0] < (col + 1) * width / grid_cols)
                & (src[:, 1] >= row * height / grid_rows)
                & (src[:, 1] < (row + 1) * height / grid_rows)
            )
            color = "#20bf55" if inside.any() else "#e63946"
            axis.add_patch(
                Rectangle(
                    (col * width / grid_cols, row * height / grid_rows),
                    width / grid_cols,
                    height / grid_rows,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.22,
                )
            )
    axis.scatter(src[reliable, 0], src[reliable, 1], s=12, c="white")
    axis.set_title("Reliable-match spatial coverage")
    axis.set_axis_off()
    _save(figure, save_path)
    return figure


def visualize_stage25_refinement(
    image_a: Tensor | np.ndarray,
    image_b: Tensor | np.ndarray,
    src_xy: Tensor,
    coarse_dst_xy: Tensor,
    teacher_dst_xy: Tensor,
    student_mid_xy: Tensor,
    student_fine_xy: Tensor,
    mid_probability: Tensor,
    fine_probability: Tensor,
    valid_mask: Tensor,
    homography_dst_xy: Optional[Tensor] = None,
    homography_valid_mask: Optional[Tensor] = None,
    *,
    max_matches: int = 48,
    save_path: Optional[str | Path] = None,
) -> Figure:
    """Visualize fixed Stage-2.5 anchors, refinement paths, and local distributions."""

    a = _image_numpy(image_a)
    b = _image_numpy(image_b)
    src = _first_batch(src_xy).detach().float().cpu()
    coarse = _first_batch(coarse_dst_xy).detach().float().cpu()
    teacher = _first_batch(teacher_dst_xy).detach().float().cpu()
    mid = _first_batch(student_mid_xy).detach().float().cpu()
    fine = _first_batch(student_fine_xy).detach().float().cpu()
    valid = _first_batch(valid_mask).detach().cpu().bool()
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)[:max_matches]

    figure, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    canvas = np.concatenate((a, b), axis=1)
    axes[0, 0].imshow(canvas)
    target_offset = float(a.shape[1])
    for index_tensor in indices:
        index = int(index_tensor)
        axes[0, 0].plot(
            [float(src[index, 0]), float(fine[index, 0]) + target_offset],
            [float(src[index, 1]), float(fine[index, 1])],
            color="#00d4ff",
            linewidth=0.65,
            alpha=0.6,
        )
        axes[0, 0].scatter(
            float(fine[index, 0]) + target_offset,
            float(fine[index, 1]),
            c="#00d4ff",
            s=8,
        )
    axes[0, 0].set_title("Student fine matches (A -> B)")
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(b)
    for index_tensor in indices:
        index = int(index_tensor)
        path = torch.stack((coarse[index], mid[index], fine[index]))
        axes[0, 1].plot(
            path[:, 0],
            path[:, 1],
            color="#00d4ff",
            linewidth=1.0,
            alpha=0.75,
        )
        axes[0, 1].scatter(*coarse[index].tolist(), c="#ff9f1c", s=10)
        axes[0, 1].scatter(*teacher[index].tolist(), c="#e71d36", s=10)
        axes[0, 1].scatter(*fine[index].tolist(), c="#00d4ff", s=10)
    if homography_dst_xy is not None and homography_valid_mask is not None:
        homography = _first_batch(homography_dst_xy).detach().float().cpu()
        h_valid = _first_batch(homography_valid_mask).detach().cpu().bool()
        h_indices = torch.nonzero(valid & h_valid, as_tuple=False).squeeze(-1)[:max_matches]
        if h_indices.numel() > 0:
            axes[0, 1].scatter(
                homography[h_indices, 0],
                homography[h_indices, 1],
                marker="+",
                c="#20bf55",
                s=30,
                linewidths=1.0,
            )
    axes[0, 1].set_title(
        "B refinement: coarse=orange, Teacher=red, Student=cyan, H=green +"
    )
    axes[0, 1].set_axis_off()

    mid_prob = _first_batch(mid_probability).detach().float().cpu()
    fine_prob = _first_batch(fine_probability).detach().float().cpu()
    selected = indices if indices.numel() > 0 else torch.arange(mid_prob.shape[0])[:1]
    mid_mean = mid_prob[selected].mean(dim=0)
    fine_mean = fine_prob[selected].mean(dim=0)
    mid_image = axes[1, 0].imshow(mid_mean, cmap="magma")
    fine_image = axes[1, 1].imshow(fine_mean, cmap="magma")
    figure.colorbar(mid_image, ax=axes[1, 0], fraction=0.046)
    figure.colorbar(fine_image, ax=axes[1, 1], fraction=0.046)
    axes[1, 0].set_title("Mean mid local probability")
    axes[1, 1].set_title("Mean fine local probability")
    for axis in axes[1]:
        axis.set_xlabel("dx cell")
        axis.set_ylabel("dy cell")
    _save(figure, save_path)
    return figure


def _point_inputs(
    src_xy: Tensor,
    dst_xy: Tensor,
    probability: Tensor,
    valid_mask: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    src = _first_batch(src_xy).detach().cpu()
    dst = _first_batch(dst_xy).detach().cpu()
    probability = _first_batch(probability).detach().cpu()
    return src, dst, probability, _valid_mask(valid_mask, src.shape[0])


def _valid_mask(valid_mask: Optional[Tensor], count: int) -> Tensor:
    return (
        torch.ones(count, dtype=torch.bool)
        if valid_mask is None
        else _first_batch(valid_mask).detach().cpu().bool()
    )


def _first_batch(value: Tensor) -> Tensor:
    return value[0] if value.ndim >= 2 else value


def _image_numpy(image: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, Tensor):
        value = image.detach().float().cpu()
        if value.ndim == 4:
            value = value[0]
        if value.ndim != 3 or value.shape[0] != 3:
            raise ValueError("Tensor images must be [3,H,W] or [B,3,H,W].")
        return value.permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("NumPy images must be [H,W,3].")
    return value.astype(np.float32) / 255.0 if value.dtype == np.uint8 else np.clip(value, 0.0, 1.0)


def _save(figure: Figure, save_path: Optional[str | Path]) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
