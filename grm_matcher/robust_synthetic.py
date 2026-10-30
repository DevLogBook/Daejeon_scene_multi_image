"""Online exact-label augmentation of existing synthetic homography pairs.

Target deformation is defined by y + u(y) = S H x. B is rendered using the
same inverse map S^-1(y+u(y)). This is controlled 2D deformation, not a 3D
visibility simulator; labels exclude failed inverse solves and occlusions.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .coordinates import image_xy_to_normalized_grid
from .stage25 import transform_points


@dataclass
class RobustAugmentConfig:
    seed: int = 2026
    # unchanged, weak, medium, strong, severe; counts are allocated per epoch.
    severity_fractions: tuple = (.10, .30, .30, .20, .10)
    severity_ranges: tuple = ((0.,0.),(.10,.25),(.30,.50),(.55,.75),(.80,1.))
    max_rotation_degrees: float = 30.
    min_scale: float = .75
    max_scale: float = 1.3
    max_log_stretch: float = .18
    max_shear: float = .10
    translation_fraction: float = .15
    perspective_at_512: float = .00015
    flow_probability: float = .65
    flow_pixels_at_512: float = 20.
    flow_gradient_limit: float = .30
    inverse_iterations: int = 20
    inverse_tolerance_pixels: float = .25
    brightness_delta: float = .15
    color_strength: float = .25
    noise_std: float = .01
    blur_probability: float = .25
    occlusion_probability: float = .15
    curriculum_epochs: int = 3

    def __post_init__(self):
        self.severity_fractions = tuple(float(v) for v in self.severity_fractions)
        self.severity_ranges = tuple(tuple(float(v) for v in pair) for pair in self.severity_ranges)
        if len(self.severity_fractions)!=5 or min(self.severity_fractions)<0 or abs(sum(self.severity_fractions)-1)>1e-6:
            raise ValueError('Five severity fractions must sum to 1')
        if len(self.severity_ranges)!=5 or self.severity_ranges[0]!=(0.,0.):
            raise ValueError('Unchanged range must be exactly (0,0)')
        if any(len(pair)!=2 or not 0<=pair[0]<=pair[1]<=1 for pair in self.severity_ranges):
            raise ValueError('Severity ranges must lie in [0,1]')
        for key in ('flow_probability', 'blur_probability', 'occlusion_probability'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'{key} must be in [0,1]')
        if not 0 < self.min_scale <= self.max_scale or not 0 < self.flow_gradient_limit < .5:
            raise ValueError('Invalid scale or flow gradient bound')
        if self.inverse_iterations < 1 or self.curriculum_epochs < 1 or self.inverse_tolerance_pixels <= 0:
            raise ValueError('Invalid inverse solver/curriculum')
        for key in ('max_rotation_degrees', 'max_log_stretch', 'max_shear', 'translation_fraction',
                    'perspective_at_512', 'flow_pixels_at_512', 'brightness_delta', 'noise_std'):
            if getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative')
        if not 0 <= self.color_strength < .8:
            raise ValueError('color_strength must be in [0,.8)')

    def to_dict(self):
        return asdict(self)


def pixel_grid(h, w):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    return torch.stack((xx, yy), -1).float()


def sample_field(field, xy, mode='bilinear', padding='zeros'):
    h, w = field.shape[-2:]
    grid = image_xy_to_normalized_grid(xy, h, w)[None]
    return F.grid_sample(field[None].float(), grid, mode=mode,
                         padding_mode=padding, align_corners=False)[0]


def project(xy, matrix):
    return transform_points(xy.reshape(1, -1, 2), matrix[None]).reshape_as(xy)


def inside(xy, h, w):
    return torch.isfinite(xy).all(-1) & (xy[..., 0] >= 0) & (xy[..., 0] <= w-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h-1)


def uniform(generator, lo, hi):
    return lo + (hi-lo)*torch.rand((), generator=generator).item()


def sample_transform(config, h, w, generator, strength):
    angle = math.radians(uniform(generator, -config.max_rotation_degrees, config.max_rotation_degrees)*strength)
    scale = math.exp(uniform(generator, math.log(config.min_scale), math.log(config.max_scale))*strength)
    stretch = uniform(generator, -config.max_log_stretch, config.max_log_stretch)*strength
    shear = uniform(generator, -config.max_shear, config.max_shear)*strength
    rotate = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    linear = rotate @ torch.tensor([[scale*math.exp(stretch), shear], [0., scale*math.exp(-stretch)]])
    center = torch.tensor([(w-1)/2, (h-1)/2])
    offset = torch.tensor([uniform(generator, -1, 1)*w, uniform(generator, -1, 1)*h])*config.translation_fraction*strength
    matrix = torch.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = center + offset - linear @ center
    # Centered perspective whose denominator remains positive on this canvas.
    bound = config.perspective_at_512 * 512/max(h, w)*strength
    p = torch.tensor([uniform(generator, -bound, bound), uniform(generator, -bound, bound)])
    perspective = torch.eye(3)
    perspective[2, :2], perspective[2, 2] = p, 1-p @ center
    result = matrix @ perspective
    return result/result[2, 2]


def sample_flow(config, h, w, generator, strength):
    grid = pixel_grid(h, w)
    flow = torch.zeros(2, h, w)
    if uniform(generator, 0, 1) >= config.flow_probability:
        return flow
    for _ in range(int(torch.randint(1, 4, (), generator=generator))):
        cx, cy = uniform(generator, .1*w, .9*w), uniform(generator, .1*h, .9*h)
        sigma = uniform(generator, .12, .28)*min(h, w)
        magnitude = uniform(generator, .25, 1)*config.flow_pixels_at_512*min(h, w)/512*strength
        angle = uniform(generator, -math.pi, math.pi)
        weight = torch.exp(-((grid[...,0]-cx)**2+(grid[...,1]-cy)**2)/(2*sigma*sigma))
        flow += torch.tensor([math.cos(angle), math.sin(angle)])[:, None, None]*magnitude*weight
    # Bound the Jacobian by a conservative sum of componentwise derivative maxima.
    # Also bounds the Lipschitz constant of bilinear interpolation inside cells.
    derivative_bound = (flow.diff(dim=1).abs().amax(dim=(1,2)).sum() +
                        flow.diff(dim=2).abs().amax(dim=(1,2)).sum())
    flow *= min(1., config.flow_gradient_limit/max(float(derivative_bound), 1e-8))
    maximum = config.flow_pixels_at_512*min(h,w)/512*strength
    flow *= min(1., maximum/max(float(flow.norm(dim=0).max()), 1e-8))
    return flow


def photometric(image, config, generator, strength):
    c = config.color_strength*strength
    gamma = uniform(generator, 1-c, 1+c)
    contrast = uniform(generator, 1-c, 1+c)
    saturation = uniform(generator, 1-c, 1+c)
    gains = torch.tensor([uniform(generator, 1-c, 1+c) for _ in range(3)])[:, None, None]
    image = ((image-.5)*contrast+.5)*gains + uniform(generator, -1, 1)*config.brightness_delta*strength
    gray = image.mean(0, keepdim=True)
    image = gray + saturation*(image-gray)
    image = image.clamp(0, 1).pow(gamma)
    if uniform(generator, 0, 1) < config.blur_probability*strength:
        image = F.avg_pool2d(image[None], 3, 1, 1)[0]
    if config.noise_std:
        image += torch.randn(image.shape, generator=generator)*config.noise_std*strength
    return image.clamp(0, 1)


def augment_pair(sample, config, generator, strength):
    a, b = sample['image_a'].clone(), sample['image_b'].clone()
    h, w = a.shape[-2:]
    grid = pixel_grid(h, w)
    matrix = sample_transform(config, h, w, generator, strength)
    deformation = sample_flow(config, h, w, generator, strength)
    original_h = sample['H_ab'].float()
    composed = matrix @ original_h
    global_target = project(grid, composed)
    # Solve y+u(y)=S H x. B rendering uses the exact same definition.
    forward = global_target.clone()
    for _ in range(config.inverse_iterations):
        forward = global_target-sample_field(deformation, forward, padding='border').permute(1,2,0)
    residual = (forward+sample_field(deformation, forward, padding='border').permute(1,2,0)-global_target).norm(dim=-1)
    inverse_b = project(grid+deformation.permute(1,2,0), torch.linalg.inv(matrix))
    backward = project(inverse_b, torch.linalg.inv(original_h))
    rendered = sample_field(b, inverse_b)
    valid_b = inside(inverse_b, h, w) & (sample_field(sample['valid_b'], inverse_b)[0] > .999)
    overlap_b = valid_b & inside(backward, h, w) & (sample_field(sample['gt_overlap_b'], inverse_b)[0] > .999)
    valid_a = sample['valid_a'][0].bool()
    overlap_a = (sample['gt_overlap_a'][0].bool() & valid_a & inside(forward,h,w) &
                 (residual <= config.inverse_tolerance_pixels))
    occluded = torch.zeros(h, w, dtype=torch.bool)
    if uniform(generator,0,1) < config.occlusion_probability*strength:
        bw, bh = max(1, int(uniform(generator,.06,.18)*w)), max(1,int(uniform(generator,.06,.18)*h))
        x, y = int(torch.randint(w-bw+1,(),generator=generator)), int(torch.randint(h-bh+1,(),generator=generator))
        rendered[:,y:y+bh,x:x+bw] = torch.rand(3,1,1,generator=generator)
        occluded[y:y+bh,x:x+bw] = True
    overlap_b &= ~occluded
    overlap_a &= sample_field(overlap_b[None], forward)[0] > .999
    # Reject raster-interpolation inconsistency as well as solver failures.
    returned = sample_field(backward.permute(2,0,1), forward).permute(1,2,0)
    overlap_a &= (returned-grid).norm(dim=-1) <= config.inverse_tolerance_pixels
    returned_b = sample_field(forward.permute(2,0,1), backward).permute(1,2,0)
    overlap_b &= (returned_b-grid).norm(dim=-1) <= config.inverse_tolerance_pixels
    overlap_b &= sample_field(overlap_a[None], backward)[0] > .999
    # Invalid target pixels remain masked after color/noise, including padding.
    image_a = photometric(a, config, generator, strength)*valid_a[None]
    image_b = photometric(rendered, config, generator, strength)*valid_b[None]
    return dict(image_a=image_a, image_b=image_b, H_ab=composed,
                gt_dense_mapping=forward.permute(2,0,1).contiguous(),
                gt_dense_mapping_ba=backward.permute(2,0,1).contiguous(),
                gt_overlap_a=overlap_a[None], gt_overlap_b=overlap_b[None],
                valid_a=valid_a[None], valid_b=valid_b[None],
                augmentation_strength=torch.tensor(strength),
                flow_active=torch.tensor(bool(deformation.abs().max() > 0)),
                inverse_residual_max=torch.tensor(float(residual[inside(forward,h,w)].max()) if inside(forward,h,w).any() else 0.))


class RobustSyntheticDataset(Dataset):
    """Preserve original train/val source split; train augmentation changes by epoch."""
    def __init__(self, base, matcher_config, augmentation=None, train=True):
        self.base, self.matcher_config = base, matcher_config
        self.augmentation = augmentation or RobustAugmentConfig()
        self.train, self.epoch = train, 0
        self.set_epoch(0)

    def set_epoch(self, epoch):
        if not self.train:
            epoch = 0
        self.epoch = epoch
        count = len(self.base)
        raw = [count*f for f in self.augmentation.severity_fractions]
        counts = [math.floor(v) for v in raw]
        for tier in sorted(range(5),key=lambda i:raw[i]-counts[i],reverse=True)[:count-sum(counts)]:
            counts[tier] += 1
        generator=torch.Generator().manual_seed(self.augmentation.seed+epoch*10000019+(0 if self.train else 701))
        order=torch.randperm(count,generator=generator)
        self.tiers=torch.empty(count,dtype=torch.long)
        offset=0
        for tier,size in enumerate(counts):
            self.tiers[order[offset:offset+size]]=tier
            offset+=size
        self.severity_counts=dict(zip(('unchanged','weak','medium','strong','severe'),counts))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        original_images = {key:sample[key] for key in ('image_a','image_b')}
        if self.matcher_config.input_range == 'imagenet_normalized':
            mean = torch.tensor(self.matcher_config.normalization_mean)[:,None,None]
            std = torch.tensor(self.matcher_config.normalization_std)[:,None,None]
            for key in ('image_a', 'image_b'):
                sample[key] = (sample[key]*std+mean).clamp(0,1)
        epoch = self.epoch if self.train else 0
        generator = torch.Generator().manual_seed(self.augmentation.seed+index*1009+epoch*10000019+(0 if self.train else 701))
        tier=int(self.tiers[index])
        curriculum = min(1., .4+.6*epoch/max(1,self.augmentation.curriculum_epochs-1)) if self.train else 1.
        difficulty = uniform(generator,*self.augmentation.severity_ranges[tier])
        strength = curriculum*difficulty
        if tier == 0:
            h,w=sample['image_a'].shape[-2:]
            result={key:sample[key] for key in ('image_a','image_b','H_ab','gt_dense_mapping',
                     'gt_overlap_a','gt_overlap_b','valid_a','valid_b')}
            result.update(gt_dense_mapping_ba=project(pixel_grid(h,w),torch.linalg.inv(sample['H_ab'])).permute(2,0,1).contiguous(),
                          augmentation_strength=torch.tensor(0.),flow_active=torch.tensor(False),
                          inverse_residual_max=torch.tensor(0.))
        else:
            result = augment_pair(sample, self.augmentation, generator, strength)
        if self.matcher_config.input_range == 'imagenet_normalized':
            for key in ('image_a', 'image_b'):
                result[key] = (result[key]-mean)/std
        if tier == 0:
            result.update(original_images)
        result['augmentation_tier']=torch.tensor(tier,dtype=torch.long)
        result['pair_id'] = sample.get('pair_id', str(index))
        return result
