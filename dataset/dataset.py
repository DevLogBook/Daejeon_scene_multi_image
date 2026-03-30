import os
from pathlib import Path
import random
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import h5py  # type: ignore[import-not-found]
except ImportError:
    h5py = None

# 多尺度数据集
class MultiScaleDataset(Dataset):
    """
    多尺度数据集，支持 train/val 划分
    """

    def __init__(
            self,
            pairs_file: Path,
            val_ratio: float = 0.1,
            split_seed: int = 42,
            return_split: str = 'train',
    ):
        self.is_train = (return_split == 'train')
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
        self.pool_highres = [(512, 512), (640, 480), (480, 480)]

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
            A.ColorJitter(
                brightness=0.1,  # 亮度微调 ±10% (模拟云层遮挡或轻微的顺/逆光)
                contrast=0.1,  # 对比度微调 ±10% (模拟光照强弱造成的反差变化)
                saturation=0.1,  # 饱和度微调 ±10% (保持植被色彩不过于鲜艳或灰暗)
                hue=0.03,  # 色相极微调 ±3% (必须严格限制，防止植被颜色失真)
                p=0.3  # 维持 50% 的触发概率，保证一半的数据是原图分布
            ),
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
        random_num = random.randint(1, 11)
        if random_num % 4 == 0:
            tensor_a = color_trans(image=geo_transformed["image"])["image"]
            tensor_b = color_trans(image=geo_transformed["image_b"])["image"]
        else:
            tensor_a = self.color_norm_transform(image=geo_transformed["image"])["image"]
            tensor_b = self.color_norm_transform(image=geo_transformed["image_b"])["image"]

        return {"img_a": tensor_a, "img_b": tensor_b}


class CachedTeacherDataset(Dataset):
    def __init__(
            self,
            cache_dir: str,
            val_ratio: float = 0.1,
            split_seed: int = 42,
            return_split: str = 'train'
    ):
        """
        基于预计算缓存的 Dataset，支持稳定、确定性的 Train/Val 划分。
        """
        self.return_split = return_split
        self.is_train = (return_split == 'train')

        # 获取所有缓存文件并排序，确保在不同操作系统下顺序一致
        all_files = sorted(list(Path(cache_dir).glob("*.pt")))
        if len(all_files) == 0:
            raise ValueError(f"No cache files found in {cache_dir}")

        # 使用局部独立的随机数生成器进行打乱，不污染全局 Seed
        rng = random.Random(split_seed)
        rng.shuffle(all_files)

        val_size = max(1, int(len(all_files) * val_ratio))
        if self.is_train:
            self.cache_files = all_files[val_size:]
        else:
            self.cache_files = all_files[:val_size]

        print(f"[{return_split.upper()} Set] Loaded {len(self.cache_files)} cached items.")

        if self.is_train:
            # 训练集：包含温和的色彩抖动，提升光照鲁棒性
            self.online_transform = A.Compose([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'image_b': 'image'})
        else:
            # 验证集：绝对禁止色彩抖动，仅做标准化和张量化，确保验证指标稳定可靠
            self.online_transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'image_b': 'image'})

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        cache_data = torch.load(self.cache_files[idx], weights_only=False)

        # 执行线上 Transform (自动区分 Train/Val)
        aug_out = self.online_transform(
            image=cache_data['img_a'],
            image_b=cache_data['img_b']
        )

        return {
            "img_a": aug_out['image'],
            "img_b": aug_out['image_b'],
            "t_warp_AB": cache_data['warp_AB'].float(),
            "t_conf_AB": cache_data['confidence_AB'].float(),
            "t_feat_a": cache_data['feat_A'].float(),
            "t_feat_b": cache_data['feat_B'].float(),
        }


class BucketedH5TeacherDataset(Dataset):
    def __init__(
            self,
            h5_path: str,
            val_ratio: float = 0.1,
            split_seed: int = 42,
            return_split: str = 'train'
    ):
        if h5py is None:
            raise ImportError("h5py is required to read bucketed HDF5 teacher cache.")

        self.h5_path = Path(h5_path)
        self.return_split = return_split
        self.is_train = (return_split == 'train')
        self._h5_file: Optional["h5py.File"] = None

        with h5py.File(self.h5_path, 'r') as h5f:
            bucket_to_samples: Dict[str, List[Tuple[str, str]]] = {}
            for bucket_name in sorted(h5f.keys()):
                sample_keys = sorted(h5f[bucket_name].keys())
                bucket_to_samples[bucket_name] = [(bucket_name, sample_key) for sample_key in sample_keys]

        rng = random.Random(split_seed)
        self.items: List[Tuple[str, str]] = []
        self.bucket_to_indices: Dict[str, List[int]] = {}

        for bucket_name, bucket_items in bucket_to_samples.items():
            rng.shuffle(bucket_items)
            val_size = max(1, int(len(bucket_items) * val_ratio))
            selected_items = bucket_items[val_size:] if self.is_train else bucket_items[:val_size]

            start_idx = len(self.items)
            self.items.extend(selected_items)
            self.bucket_to_indices[bucket_name] = list(range(start_idx, start_idx + len(selected_items)))

        if self.is_train:
            self.online_transform = A.Compose([
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'image_b': 'image'})
        else:
            self.online_transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'image_b': 'image'})

        print(f"[{return_split.upper()} Set] Loaded {len(self.items)} bucketed HDF5 cached items from {self.h5_path}.")

    def __len__(self) -> int:
        return len(self.items)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _get_h5_file(self):
        current_pid = os.getpid()
        if self._h5_file is None or getattr(self, '_h5_pid', None) != current_pid:
            if self._h5_file is not None:
                try:
                    self._h5_file.close()
                except Exception:
                    pass
            self._h5_file = h5py.File(self.h5_path, 'r')
            self._h5_pid = current_pid
        return self._h5_file

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bucket_name, sample_key = self.items[idx]
        sample_group = self._get_h5_file()[bucket_name][sample_key]

        img_a = sample_group['img_a'][()]
        img_b = sample_group['img_b'][()]
        aug_out = self.online_transform(image=img_a, image_b=img_b)

        return {
            "img_a": aug_out['image'],
            "img_b": aug_out['image_b'],
            "t_warp_AB": torch.from_numpy(sample_group['warp_AB'][()]).float(),
            "t_conf_AB": torch.from_numpy(sample_group['confidence_AB'][()]).float(),
            "t_feat_a": torch.from_numpy(sample_group['feat_A'][()]).float(),
            "t_feat_b": torch.from_numpy(sample_group['feat_B'][()]).float(),
            "bucket_name": bucket_name,
        }

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


class BucketedBatchSampler(Sampler[List[int]]):
    def __init__(
            self,
            bucket_to_indices: Dict[str, List[int]],
            batch_size: int,
            shuffle: bool = True,
            drop_last: bool = False,
            seed: int = 42,
    ):
        self.bucket_to_indices = {k: list(v) for k, v in bucket_to_indices.items() if v}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []

        for indices in self.bucket_to_indices.values():
            bucket_indices = list(indices)
            if self.shuffle:
                rng.shuffle(bucket_indices)

            for start in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self.bucket_to_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total
