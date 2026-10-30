"""Synthetic A-to-B homography plus continuous local-deformation dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .config import GRMMatcherConfig
from .coordinates import image_xy_to_normalized_grid
from .seed_geometry import apply_homography


ImageSource = Union[str, Path, Tensor]


@dataclass
class SyntheticGRMDataConfig:
    output_height: int = 512
    output_width: int = 512
    samples_per_source: int = 20
    seed: int = 1234
    max_rotation_degrees: float = 8.0
    min_scale: float = 0.94
    max_scale: float = 1.06
    max_translation_fraction: float = 0.12
    max_perspective: float = 2.0e-4
    local_region_min: int = 1
    local_region_max: int = 3
    local_flow_min_pixels: float = 4.0
    local_flow_max_pixels: float = 18.0
    local_flow_probability: float = 0.50
    global_residual_threshold: float = 3.0
    local_residual_threshold: float = 24.0
    inverse_iterations: int = 6
    hard_negative_count: int = 128
    max_occlusions: int = 2
    occlusion_probability: float = 0.35
    blur_probability: float = 0.35
    noise_std: float = 0.01
    brightness_delta: float = 0.08
    contrast_min: float = 0.85
    contrast_max: float = 1.15
    channel_gain_min: float = 0.85
    channel_gain_max: float = 1.15

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("Synthetic output size must be positive.")
        if self.output_height % 16 != 0 or self.output_width % 16 != 0:
            raise ValueError("Synthetic output size must be divisible by 16.")
        if self.samples_per_source <= 0 or self.hard_negative_count <= 0:
            raise ValueError("samples_per_source and hard_negative_count must be positive.")
        if not 0.0 < self.min_scale <= self.max_scale:
            raise ValueError("Synthetic scale range is invalid.")
        if not 0 <= self.local_region_min <= self.local_region_max:
            raise ValueError("Synthetic local region range is invalid.")
        if not 0.0 < self.global_residual_threshold < self.local_residual_threshold:
            raise ValueError("Synthetic role residual thresholds are invalid.")
        for name in (
            "local_flow_probability",
            "occlusion_probability",
            "blur_probability",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.local_flow_min_pixels <= 0.0 or (
            self.local_flow_max_pixels < self.local_flow_min_pixels
        ):
            raise ValueError("Synthetic local-flow magnitude range is invalid.")
        if self.noise_std < 0.0 or self.brightness_delta < 0.0:
            raise ValueError("Synthetic noise/brightness controls must be nonnegative.")
        if not 0.0 < self.contrast_min <= self.contrast_max:
            raise ValueError("Synthetic contrast range is invalid.")
        if not 0.0 < self.channel_gain_min <= self.channel_gain_max:
            raise ValueError("Synthetic channel-gain range is invalid.")
        if self.max_occlusions < 0:
            raise ValueError("max_occlusions must be nonnegative.")


class SyntheticGRMDataset(Dataset[dict[str, Tensor]]):
    """Create paired views from one source image.

    ``H_ab`` always maps coordinates in A to coordinates in B.  The exact
    dense mapping is ``H_ab(A_xy) + local_flow(A_xy)``.  B is rendered by
    fixed-point inversion of that mapping, then independently photometrically
    perturbed and partially occluded.
    """

    def __init__(
        self,
        image_sources: Sequence[ImageSource],
        config: SyntheticGRMDataConfig | None = None,
    ) -> None:
        if not image_sources:
            raise ValueError("SyntheticGRMDataset needs at least one source image.")
        self.image_sources = list(image_sources)
        self.config = config or SyntheticGRMDataConfig()

    def __len__(self) -> int:
        return len(self.image_sources) * self.config.samples_per_source

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.config.seed + index)
        source = self.image_sources[index % len(self.image_sources)]
        image_a = self._load_and_crop(source, generator)
        height, width = image_a.shape[-2:]
        h_ab = self._sample_homography(height, width, generator)
        local_flow = self._sample_local_flow(height, width, generator)

        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        a_grid = torch.stack((xx, yy), dim=-1)
        global_mapping = apply_homography(
            a_grid.reshape(1, -1, 2), h_ab.unsqueeze(0)
        ).reshape(height, width, 2)
        dense_mapping = global_mapping + local_flow.permute(1, 2, 0)

        image_b, inverse_a_grid, overlap_b = self._render_b_from_a(
            image_a, h_ab, local_flow, a_grid
        )
        image_a = self._photometric_augment(image_a, generator, allow_occlusion=False)[0]
        image_b, occluded_b = self._photometric_augment(
            image_b, generator, allow_occlusion=True
        )
        overlap_b = overlap_b & ~occluded_b[0]

        overlap_a = _inside(dense_mapping, height, width)
        mapped_occlusion = F.grid_sample(
            occluded_b[None].float(),
            image_xy_to_normalized_grid(dense_mapping, height, width)[None],
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )[0, 0] > 0.5
        overlap_a = overlap_a & ~mapped_occlusion

        residual = torch.linalg.vector_norm(local_flow, dim=0)
        global_mask = (
            overlap_a & (residual < self.config.global_residual_threshold)
        )
        local_mask = (
            overlap_a
            & (residual >= self.config.global_residual_threshold)
            & (residual <= self.config.local_residual_threshold)
        )
        hard_src, hard_dst, hard_type, hard_valid = self._make_hard_negatives(
            dense_mapping, overlap_a, generator
        )
        return {
            "image_a": image_a.clamp(0.0, 1.0),
            "image_b": image_b.clamp(0.0, 1.0),
            "H_ab": h_ab,
            "gt_dense_mapping": dense_mapping.permute(2, 0, 1).contiguous(),
            "gt_overlap_a": overlap_a.unsqueeze(0),
            "gt_overlap_b": overlap_b.unsqueeze(0),
            "gt_global_mask": global_mask.unsqueeze(0),
            "gt_local_mask": local_mask.unsqueeze(0),
            "local_flow": local_flow,
            "hard_negative_src_xy": hard_src,
            "hard_negative_dst_xy": hard_dst,
            "hard_negative_type": hard_type,
            "hard_negative_valid": hard_valid,
            "valid_a": torch.ones(1, height, width, dtype=torch.bool),
            "valid_b": torch.ones(1, height, width, dtype=torch.bool),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "local_flow_active": torch.tensor(
                bool(torch.linalg.vector_norm(local_flow, dim=0).amax() > 0.0)
            ),
        }

    def _load_and_crop(self, source: ImageSource, generator: torch.Generator) -> Tensor:
        if isinstance(source, Tensor):
            image = source.detach().cpu().float()
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError("Tensor sources must be [3, H, W].")
            if image.max() > 1.0:
                image = image / 255.0
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Synthetic source image does not exist: {path}")
            with Image.open(path) as pil_image:
                rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
            image = torch.from_numpy(rgb).permute(2, 0, 1)

        target_h, target_w = self.config.output_height, self.config.output_width
        source_h, source_w = image.shape[-2:]
        scale = max(target_h / source_h, target_w / source_w, 1.0)
        if scale > 1.0:
            resized_h = int(math.ceil(source_h * scale))
            resized_w = int(math.ceil(source_w * scale))
            image = F.interpolate(
                image[None], size=(resized_h, resized_w), mode="bilinear", align_corners=False
            )[0]
        max_y = image.shape[-2] - target_h
        max_x = image.shape[-1] - target_w
        offset_y = int(torch.randint(max_y + 1, (1,), generator=generator).item())
        offset_x = int(torch.randint(max_x + 1, (1,), generator=generator).item())
        return image[:, offset_y : offset_y + target_h, offset_x : offset_x + target_w]

    def _sample_homography(
        self, height: int, width: int, generator: torch.Generator
    ) -> Tensor:
        angle = math.radians(
            _uniform(generator, -self.config.max_rotation_degrees, self.config.max_rotation_degrees)
        )
        scale = _uniform(generator, self.config.min_scale, self.config.max_scale)
        tx = _uniform(generator, -1.0, 1.0) * self.config.max_translation_fraction * width
        ty = _uniform(generator, -1.0, 1.0) * self.config.max_translation_fraction * height
        cosine, sine = math.cos(angle) * scale, math.sin(angle) * scale
        cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
        p_x = _uniform(generator, -self.config.max_perspective, self.config.max_perspective)
        p_y = _uniform(generator, -self.config.max_perspective, self.config.max_perspective)
        h = torch.tensor(
            [
                [cosine, -sine, tx + cx - cosine * cx + sine * cy],
                [sine, cosine, ty + cy - sine * cx - cosine * cy],
                [p_x, p_y, 1.0 - p_x * cx - p_y * cy],
            ],
            dtype=torch.float32,
        )
        return h / h[2, 2]

    def _sample_local_flow(
        self, height: int, width: int, generator: torch.Generator
    ) -> Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        flow = torch.zeros(2, height, width)
        if _uniform(generator, 0.0, 1.0) >= self.config.local_flow_probability:
            return flow
        region_count = int(
            torch.randint(
                self.config.local_region_min,
                self.config.local_region_max + 1,
                (1,),
                generator=generator,
            ).item()
        )
        for _ in range(region_count):
            center_x = _uniform(generator, 0.15 * width, 0.85 * width)
            center_y = _uniform(generator, 0.15 * height, 0.85 * height)
            sigma = _uniform(generator, 0.10, 0.24) * min(height, width)
            magnitude = _uniform(
                generator,
                self.config.local_flow_min_pixels,
                self.config.local_flow_max_pixels,
            )
            angle = _uniform(generator, -math.pi, math.pi)
            weight = torch.exp(
                -((xx - center_x).square() + (yy - center_y).square()) / (2.0 * sigma * sigma)
            )
            flow[0] += math.cos(angle) * magnitude * weight
            flow[1] += math.sin(angle) * magnitude * weight
        norm = torch.linalg.vector_norm(flow, dim=0)
        maximum = self.config.local_residual_threshold * 0.95
        flow = flow * (maximum / norm.clamp_min(maximum)).clamp(max=1.0).unsqueeze(0)
        return flow

    def _render_b_from_a(
        self,
        image_a: Tensor,
        h_ab: Tensor,
        local_flow: Tensor,
        b_grid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        height, width = image_a.shape[-2:]
        inverse_h = torch.linalg.inv(h_ab).unsqueeze(0)
        flat_b = b_grid.reshape(1, -1, 2)
        a_estimate = apply_homography(flat_b, inverse_h).reshape(height, width, 2)
        for _ in range(self.config.inverse_iterations):
            flow_grid = image_xy_to_normalized_grid(a_estimate, height, width)[None]
            sampled_flow = F.grid_sample(
                local_flow[None],
                flow_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )[0].permute(1, 2, 0)
            a_estimate = apply_homography(
                (b_grid - sampled_flow).reshape(1, -1, 2), inverse_h
            ).reshape(height, width, 2)
        sample_grid = image_xy_to_normalized_grid(a_estimate, height, width)[None]
        image_b = F.grid_sample(
            image_a[None],
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0]
        return image_b, a_estimate, _inside(a_estimate, height, width)

    def _photometric_augment(
        self,
        image: Tensor,
        generator: torch.Generator,
        *,
        allow_occlusion: bool,
    ) -> tuple[Tensor, Tensor]:
        height, width = image.shape[-2:]
        gains = torch.empty(3).uniform_(
            self.config.channel_gain_min,
            self.config.channel_gain_max,
            generator=generator,
        )[:, None, None]
        brightness = _uniform(
            generator,
            -self.config.brightness_delta,
            self.config.brightness_delta,
        )
        contrast = _uniform(
            generator, self.config.contrast_min, self.config.contrast_max
        )
        output = ((image - 0.5) * contrast + 0.5) * gains + brightness
        if _uniform(generator, 0.0, 1.0) < self.config.blur_probability:
            output = F.avg_pool2d(output[None], 3, stride=1, padding=1)[0]
        if self.config.noise_std > 0.0:
            output = output + torch.randn(
                output.shape,
                generator=generator,
                dtype=output.dtype,
            ) * self.config.noise_std
        occlusion = torch.zeros(1, height, width, dtype=torch.bool)
        if (
            allow_occlusion
            and self.config.max_occlusions > 0
            and _uniform(generator, 0.0, 1.0) < self.config.occlusion_probability
        ):
            count = int(
                torch.randint(
                    1,
                    self.config.max_occlusions + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            for _ in range(count):
                box_w = int(_uniform(generator, 0.06, 0.18) * width)
                box_h = int(_uniform(generator, 0.06, 0.18) * height)
                x0 = int(torch.randint(max(width - box_w, 1), (1,), generator=generator).item())
                y0 = int(torch.randint(max(height - box_h, 1), (1,), generator=generator).item())
                output[:, y0 : y0 + box_h, x0 : x0 + box_w] = torch.rand(
                    3, 1, 1, generator=generator
                )
                occlusion[:, y0 : y0 + box_h, x0 : x0 + box_w] = True
        return output.clamp(0.0, 1.0), occlusion

    def _make_hard_negatives(
        self,
        dense_mapping: Tensor,
        overlap_a: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        height, width = overlap_a.shape
        count = self.config.hard_negative_count
        valid_yx = torch.nonzero(overlap_a, as_tuple=False)
        src = torch.zeros(count, 2)
        dst = torch.zeros(count, 2)
        negative_type = torch.arange(count, dtype=torch.long) % 4
        valid = torch.zeros(count, dtype=torch.bool)
        if valid_yx.numel() == 0:
            return src, dst, negative_type, valid
        chosen = torch.randint(valid_yx.shape[0], (count,), generator=generator)
        yx = valid_yx[chosen]
        src = torch.stack((yx[:, 1], yx[:, 0]), dim=-1).float()
        true_dst = dense_mapping[yx[:, 0], yx[:, 1]].clone()
        dst = true_dst.clone()
        periodic = negative_type == 0
        period = _uniform(generator, 8.0, 32.0)
        dst[periodic, 0] += period * torch.where(
            torch.arange(count)[periodic] % 2 == 0, 1.0, -1.0
        )
        swapped = negative_type == 1
        dst[swapped] = true_dst.roll(shifts=7, dims=0)[swapped]
        non_overlap = negative_type == 2
        dst[non_overlap, 0] = torch.empty(int(non_overlap.sum())).uniform_(
            0, max(width - 1, 1), generator=generator
        )
        dst[non_overlap, 1] = torch.empty(int(non_overlap.sum())).uniform_(
            0, max(height - 1, 1), generator=generator
        )
        perturbed = negative_type == 3
        perturbation = torch.randn(int(perturbed.sum()), 2, generator=generator)
        perturbation = perturbation / torch.linalg.vector_norm(
            perturbation, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        perturbation = perturbation * _uniform(generator, 5.0, 18.0)
        dst[perturbed] += perturbation
        dst[..., 0].clamp_(0.0, float(width - 1))
        dst[..., 1].clamp_(0.0, float(height - 1))
        valid[:] = True
        return src, dst, negative_type, valid


class GRMTargetDomainSyntheticDataset(SyntheticGRMDataset):
    """Use real target-domain images as texture with exact controlled geometry.

    This dataset is deliberately separate from :class:`GRMRealPairDataset`:
    training samples have exact synthetic geometry, while validation real pairs
    retain their original semantics and deterministic files.
    """

    def __init__(
        self,
        image_sources: Sequence[ImageSource],
        data_config: SyntheticGRMDataConfig,
        matcher_config: GRMMatcherConfig,
    ) -> None:
        super().__init__(image_sources, data_config)
        self.matcher_config = matcher_config
        bucket = f"{data_config.output_width}x{data_config.output_height}"
        self.bucket_to_indices = {bucket: list(range(len(self)))}

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = super().__getitem__(index)
        if self.matcher_config.input_range == "imagenet_normalized":
            mean = sample["image_a"].new_tensor(
                self.matcher_config.normalization_mean
            ).view(3, 1, 1)
            std = sample["image_a"].new_tensor(
                self.matcher_config.normalization_std
            ).view(3, 1, 1)
            sample["image_a"] = (sample["image_a"] - mean) / std
            sample["image_b"] = (sample["image_b"] - mean) / std
        return sample


class GRMControlledGeometryDataset(GRMTargetDomainSyntheticDataset):
    """Backward-compatible name for exact controlled-geometry replay."""


def _inside(points_xy: Tensor, height: int, width: int) -> Tensor:
    return (
        (points_xy[..., 0] >= 0.0)
        & (points_xy[..., 0] <= float(width - 1))
        & (points_xy[..., 1] >= 0.0)
        & (points_xy[..., 1] <= float(height - 1))
    )


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high, generator=generator).item())
