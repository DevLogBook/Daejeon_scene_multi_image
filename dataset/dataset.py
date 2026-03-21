from pathlib import Path
import random
from typing import List, Tuple, Dict

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 多尺度数据集
class MultiScaleDataset(Dataset):
    """
    多尺度数据集，支持 train/val 划分
    """

    def __init__(
            self,
            pairs_file: Path,
            is_train: bool = True,
            val_ratio: float = 0.1,
            split_seed: int = 42,
            return_split: str = 'train',
    ):
        self.is_train = is_train
        self.return_split = return_split

        all_items: List[Tuple[Path, Path]] = []
        base_dir = pairs_file.parent

        for line in pairs_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 2:
                continue
            a = Path(parts[0]) if Path(parts[0]).is_absolute() else (base_dir / parts[0]).resolve()
            b = Path(parts[1]) if Path(parts[1]).is_absolute() else (base_dir / parts[1]).resolve()
            all_items.append((a, b))

        rng = random.Random(split_seed)
        indices = list(range(len(all_items)))
        rng.shuffle(indices)

        val_size = int(len(all_items) * val_ratio)
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])

        if return_split == 'train':
            self.items = [all_items[i] for i in sorted(train_indices)]
        elif return_split == 'val':
            self.items = [all_items[i] for i in sorted(val_indices)]
        else:
            self.items = all_items

        print(f"[Dataset] split={return_split}, total={len(all_items)}, using={len(self.items)}")

        # 尺寸池
        self.pool_lowres = [(256, 256), (384, 384), (512, 512), (384, 512), (512, 384)]
        self.pool_highres = [(512, 768), (768, 512), (512, 512), (640, 480), (480, 640)]

        # 预构建 transforms
        self.transforms_lowres = {
            (h, w): self._build_transform(h, w, scale=(0.6, 1.0))
            for h, w in self.pool_lowres
        }
        self.transforms_highres = {
            (h, w): self._build_transform(h, w, scale=(0.4, 1.0))
            for h, w in self.pool_highres
        }
        self.color_norm_transform = A.Compose([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.val_color_norm = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # 验证集固定尺寸
        self.val_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'image_b': 'image'})

    def _build_transform(self, target_h: int, target_w: int, scale: Tuple[float, float]) -> A.Compose:
        if not self.is_train:
            return A.Compose([
                A.Resize(target_h, target_w),
            ], additional_targets={'image_b': 'image'})

        return A.Compose([
            A.RandomResizedCrop(
                size=(target_h, target_w),
                scale=scale,
                ratio=(0.75, 1.33),
                p=1.0
            ),
        ], additional_targets={'image_b': 'image'})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path_a, path_b = self.items[idx]

        img_a = cv2.cvtColor(cv2.imread(str(path_a)), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(str(path_b)), cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img_a.shape[:2]

        # 验证集固定尺寸
        if self.return_split == 'val':
            transformed = self.val_transform(image=img_a, image_b=img_b)
            return {"img_a": transformed["image"], "img_b": transformed["image_b"]}

        # 训练集动态选择
        if max(orig_h, orig_w) > 1000:
            target_h, target_w = random.choice(self.pool_highres)
            transform = self.transforms_highres[(target_h, target_w)]
        else:
            target_h, target_w = random.choice(self.pool_lowres)
            transform = self.transforms_lowres[(target_h, target_w)]

        geo_transformed = transform(image=img_a, image_b=img_b)
        color_trans = self.val_color_norm if self.return_split == 'val' else self.color_norm_transform
        tensor_a = color_trans(image=geo_transformed["image"])["image"]
        tensor_b = color_trans(image=geo_transformed["image_b"])["image"]

        return {"img_a": tensor_a, "img_b": tensor_b}