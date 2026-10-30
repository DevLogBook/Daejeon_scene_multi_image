"""Isolated cache-v2 contracts; does not alter existing Student training heads."""
import csv
import hashlib
import os
import shlex
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CACHE_VERSION = 2


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(pairs_file, image_root=None):
    pairs_file = Path(pairs_file).resolve()
    root = Path(image_root).resolve() if image_root else pairs_file.parent
    pairs = []
    for number, line in enumerate(pairs_file.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if '\t' in line:
            fields = next(csv.reader([line], delimiter='\t'))
        elif ',' in line:
            fields = next(csv.reader([line]))
        else:
            fields = [part.strip('\"') for part in shlex.split(line, posix=False)]
        if len(fields) != 2:
            raise ValueError(f'{pairs_file}:{number}: expected exactly two image paths')
        paths = []
        for field in fields:
            path = Path(field.strip()).expanduser()
            path = path if path.is_absolute() else root/path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f'{pairs_file}:{number}: {path}')
            paths.append(str(path))
        pairs.append(tuple(paths))
    if not pairs:
        raise ValueError('Empty pairs file')
    return pairs


def rgb_tensor(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected HWC uint8 RGB')
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()/255


def make_teacher_inputs(a, b, teacher):
    def resize(image, height, width):
        return F.interpolate(image, (height, width), mode='bicubic',
                             align_corners=False, antialias=True).clamp(0, 1)
    low_a, low_b = resize(a, teacher.H_lr, teacher.W_lr), resize(b, teacher.H_lr, teacher.W_lr)
    if teacher.H_hr is None or teacher.W_hr is None:
        return low_a, low_b, None, None
    return low_a, low_b, resize(a, teacher.H_hr, teacher.W_hr), resize(b, teacher.H_hr, teacher.W_hr)


@torch.no_grad()
def extract_teacher_features_ds(teacher, a, b, grid_size):
    """Current RoMa Descriptor returns BHWD; channel count is never guessed."""
    def extract(image):
        layers = teacher.f(image)
        if not layers or any(x.ndim != 4 or x.shape[:3] != layers[0].shape[:3] for x in layers):
            raise ValueError('Expected aligned BHWD RoMa descriptor layers')
        feature = torch.cat([x.float() for x in layers], dim=-1).permute(0, 3, 1, 2)
        feature = F.interpolate(feature, (grid_size, grid_size), mode='bilinear', align_corners=False)
        return feature.flatten(2).transpose(1, 2).contiguous()
    return extract(a), extract(b)


def teacher_overlap_map(raw):
    if raw.ndim == 4:
        raw = raw[..., 0]
    if raw.ndim != 3 or not torch.isfinite(raw).all():
        raise ValueError('Expected finite batch overlap logits')
    return raw.float().sigmoid()


def validate_payload(payload):
    if payload.get('metadata', {}).get('cache_version') != CACHE_VERSION:
        raise ValueError('Legacy cache: use a NEW save directory; old labels must be regenerated')
    for name in ('img_a', 'img_b'):
        rgb_tensor(payload[name])
    for direction in ('AB', 'BA'):
        warp, prob, raw = (payload[key+direction] for key in ('warp_', 'confidence_', 'confidence_raw_'))
        if warp.ndim != 3 or warp.shape[-1] != 2 or prob.shape != warp.shape[:2]:
            raise ValueError(f'Bad {direction} warp/probability shape')
        if raw.shape[:2] != warp.shape[:2] or raw.ndim not in (2, 3):
            raise ValueError(f'Bad {direction} raw confidence shape')
        if not all(torch.isfinite(tensor).all() for tensor in (warp, prob, raw)):
            raise FloatingPointError(f'Non-finite {direction} Teacher output')
        if prob.min() < 0 or prob.max() > 1:
            raise ValueError('Overlap probability outside [0,1]')
    for name in ('feat_A', 'feat_B'):
        if name in payload and (payload[name].ndim != 2 or not torch.isfinite(payload[name]).all()):
            raise ValueError(f'Invalid {name}')


def save_one(path, payload):
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
