import json
import sys
import random
from pathlib import Path

from tqdm import tqdm
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent.parent))
from dense_match.train import make_teacher_inputs, extract_teacher_features_ds, teacher_overlap_map

REPO_ROOT = Path(__file__).resolve().parent
ROMA_SRC = REPO_ROOT / "RoMaV2" / "src"
if str(ROMA_SRC) not in sys.path:
    sys.path.insert(0, str(ROMA_SRC))
from romav2 import RoMaV2


def precompute(pairs_file, save_dir, teacher_setting='base', teacher_grid_size=32, variations=3):
    device = torch.device('cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    teacher = RoMaV2(RoMaV2.Cfg(setting=teacher_setting)).to(device).eval()
    teacher = torch.compile(teacher, mode="reduce-overhead")
    pairs = []
    for line in Path(pairs_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.replace(',', ' ').split()
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))



    # 仅用于 Teacher 推理前所需的纯归一化 (不含任何改变像素结构的 Jitter)
    tensor_norm = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'image_b': 'image'})

    print(
        f'Total original pairs: {len(pairs)} | Variations per pair: {variations} | Total Target: {len(pairs) * variations}')

    with torch.inference_mode():
        for idx, (path_a, path_b) in enumerate(tqdm(pairs)):
            # 定义多尺度池
            pool_sizes = [(256, 256), (384, 384), (512, 512), (384, 512), (512, 384)]
            selected_pool_sizes = random.sample(pool_sizes, variations)
            img_a = cv2.cvtColor(cv2.imread(path_a), cv2.COLOR_BGR2RGB)
            img_b = cv2.cvtColor(cv2.imread(path_b), cv2.COLOR_BGR2RGB)

            for i, size in enumerate(selected_pool_sizes):
                key = f'{idx:06d}_v{i}'
                save_path = save_dir / f'{key}.pt'
                if save_path.exists():
                    continue
                # 线下随机空间增强 (仅做 Resize 和 Crop)
                th, tw = size

                geo_transform = A.Compose([
                    # 将 height=th, width=tw 改为 size=(th, tw)
                    A.RandomResizedCrop(size=(th, tw), scale=(0.75, 1.0), p=1.0)
                ], additional_targets={'image_b': 'image'})

                geo_out = geo_transform(image=img_a, image_b=img_b)
                # 保存为 uint8 格式，极大地节省磁盘空间
                img_a_geo = geo_out['image']
                img_b_geo = geo_out['image_b']

                # 转换为 Tensor 喂给 Teacher
                t_out_norm = tensor_norm(image=img_a_geo, image_b=img_b_geo)
                img_a_t = t_out_norm['image'].unsqueeze(0).to(device)
                img_b_t = t_out_norm['image_b'].unsqueeze(0).to(device)

                img_a_lr, img_b_lr, img_a_hr, img_b_hr = make_teacher_inputs(img_a_t, img_b_t, teacher)
                t_out = teacher(img_a_lr, img_b_lr, img_a_hr, img_b_hr)
                t_feat_a, t_feat_b = extract_teacher_features_ds(teacher, img_a_lr, img_b_lr, teacher_grid_size)

                torch.save({
                    'img_a': img_a_geo,  # uint8 numpy array
                    'img_b': img_b_geo,  # uint8 numpy array
                    'warp_AB': t_out['warp_AB'].squeeze(0).half().cpu(),
                    'confidence_AB': teacher_overlap_map(t_out['confidence_AB']).squeeze(0).half().cpu(),
                    'feat_A': t_feat_a.squeeze(0).half().cpu(),
                    'feat_B': t_feat_b.squeeze(0).half().cpu(),
                }, save_path)


if __name__ == '__main__':
    precompute('pair.txt', 'teacher_cache/', teacher_setting='precise', variations=2)