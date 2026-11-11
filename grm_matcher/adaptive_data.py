"""Deterministic controlled stress cases; source split and exact maps are preserved.

These are synthetic diagnostic conditions, not annotations of real-world scenes.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .coordinates import image_xy_to_normalized_grid
from .stage25 import transform_points


SCENARIOS=('normal','repeated_texture','local_parallax','weak_texture','occlusion_low_overlap')


class ScenarioDataset(Dataset):
    def __init__(self,base,config,scenario):
        if scenario not in SCENARIOS:raise ValueError('Unknown scenario')
        self.base,self.config,self.scenario=base,config,scenario
        if scenario=='local_parallax':
            from .robust_synthetic import RobustSyntheticDataset,RobustAugmentConfig
            augmentation=RobustAugmentConfig(severity_fractions=(0,0,0,0,1), severity_ranges=((0,0),(.1,.25),(.3,.5),(.55,.75),(1,1)),
                max_rotation_degrees=0,min_scale=1,max_scale=1,max_log_stretch=0,max_shear=0,
                translation_fraction=0,perspective_at_512=0,flow_probability=1,flow_pixels_at_512=28,
                brightness_delta=0,color_strength=0,noise_std=0,blur_probability=0,occlusion_probability=0)
            self.base=RobustSyntheticDataset(base,config,augmentation=augmentation,train=False)

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        sample=self.base[index]
        scenario=self.scenario
        h,w=sample['image_a'].shape[-2:]
        yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
        if scenario=='weak_texture':
            for key in ('image_a','image_b'):
                value=sample[key][None]
                low=F.avg_pool2d(F.pad(value,(8,8,8,8),mode='reflect'),17,stride=1)
                sample[key]=(low*.15+value.mean((-2,-1),keepdim=True)*.85)[0]
        elif scenario=='repeated_texture':
            if self.config.input_range!='zero_one':
                raise ValueError('Repeated texture diagnostic requires zero_one image range')
            stripe=torch.stack((torch.sin(xx*2*math.pi/16),torch.cos(yy*2*math.pi/16),torch.sin((xx+yy)*2*math.pi/16)),0)*.35+.5
            sample['image_a']=stripe*.9+sample['image_a']*.1
            points=torch.stack((xx,yy),-1).float().reshape(1,-1,2)
            reverse=transform_points(points,torch.linalg.inv(sample['H_ab'])[None]).reshape(1,h,w,2)
            grid=image_xy_to_normalized_grid(reverse,h,w)
            sample['image_b']=F.grid_sample(sample['image_a'][None],grid,align_corners=False)[0]*sample['valid_b']
        elif scenario=='occlusion_low_overlap':
            mask=(xx >= int(w*.45)) & ~((xx > w*.65)&(xx < w*.85)&(yy > h*.3)&(yy < h*.7))
            sample['image_b']=sample['image_b']*mask
            # Occlusion is visibility, distinct from the image's padding validity.
            sample['gt_overlap_b']=sample['gt_overlap_b'] & mask[None]
            mapping=sample['gt_dense_mapping'].permute(1,2,0)
            grid=image_xy_to_normalized_grid(mapping,h,w)[None]
            visible=F.grid_sample(mask[None,None].float(),grid,align_corners=False)[0]>.999
            sample['gt_overlap_a']=sample['gt_overlap_a'] & visible
        sample['adaptive_scenario']=scenario
        return sample
