"""Fetch paired RGB/depth frames and cameras for a held-out 2025 subset.

Run --plan first, then --download. Re-running --download verifies and skips
completed files. No training data or original remote metadata is modified.
"""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
from pathlib import Path, PurePosixPath

import requests

BASE = 'https://huggingface.co'
REPOS = {'rgb': 'Link-Dev/UAV3DCrop', 'depth': 'Link-Dev/UAV3DCrop_depth'}
SCENES = ['Day007_Wheat', 'Day040_Wheat', 'Day033_Corn', 'Day052_Oat', 'Day047_soybean']
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'dataset' / 'UAV3DCrop_2025_validation'


def get(url, **kwargs):
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(20, 60), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def listing(repo, revision, directory):
    url = f'{BASE}/api/datasets/{repo}/tree/{revision}/{directory}?recursive=true&limit=1000'
    result = {}
    while url:
        response = get(url)
        result.update({item['path']: item for item in response.json() if item['type'] == 'file'})
        url = response.links.get('next', {}).get('url')
    return result


def local_path(root, entry):
    path = root / entry['kind'] / entry['path']
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Remote path leaves output root')
    return path


def valid_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    lfs_hash = entry.get('lfs', {}).get('oid')
    with path.open('rb') as stream:
        if lfs_hash:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            return actual == lfs_hash
        digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest() == entry['oid']


def fetch_file(root, entry, revisions):
    path = local_path(root, entry)
    if valid_file(path, entry):
        return entry['size'], True
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    url = f"{BASE}/datasets/{REPOS[entry['kind']]}/resolve/{revisions[entry['kind']]}/{entry['path']}"
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > entry['size']:
                offset = 0
            if offset == entry['size']:
                if valid_file(part, entry):
                    part.replace(path)
                    return entry['size'], False
                offset = 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with get(url, stream=True, headers=headers) as response:
                if offset and response.status_code == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Unexpected response range')
                    mode = 'ab'
                else:
                    mode = 'wb'
                with part.open(mode) as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if not valid_file(part, entry):
                # A complete invalid file must not be used as a Range prefix.
                part.unlink(missing_ok=True)
                raise RuntimeError('Downloaded size or hash mismatch')
            part.replace(path)
            return entry['size'], False
        except (requests.RequestException, RuntimeError, OSError):
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def plan(root, budget_gib):
    manifest_path = root / 'download_manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Plan already exists; use --download to continue this pinned selection')
    root.mkdir(parents=True, exist_ok=True)
    revisions = {kind: get(f'{BASE}/api/datasets/{repo}').json()['sha'] for kind, repo in REPOS.items()}
    selected, groups, frames_manifest = [], [], []
    for scene in SCENES:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rgb_job = pool.submit(listing, REPOS['rgb'], revisions['rgb'], f'2025/{scene}')
            depth_job = pool.submit(listing, REPOS['depth'], revisions['depth'], f'2025/{scene}_depth')
            rgb_files, depth_files = rgb_job.result(), depth_job.result()
        camera_path = f'2025/{scene}/transforms.json'
        camera_entry = dict(rgb_files[camera_path], kind='rgb')
        fetch_file(root, camera_entry, revisions)
        selected.append(camera_entry)
        metadata = json.loads(local_path(root, camera_entry).read_bytes())
        matched = []
        for frame in metadata['frames']:
            rel = PurePosixPath(frame['file_path'].removeprefix('./'))
            rgb_path = f'2025/{scene}/{rel}'
            depth_path = f'2025/{scene}_depth/{rel.stem}_depth.tif'
            if rgb_path in rgb_files and depth_path in depth_files:
                matched.append((frame, rgb_path, depth_path))
        matched.sort(key=lambda row: int(PurePosixPath(row[1]).stem))
        if len(matched) < 60:
            raise RuntimeError(f'{scene}: fewer than 60 paired frames')
        # Keep contiguous available camera views, rather than isolated random images.
        starts = [0, (len(matched) - 20) // 2, len(matched) - 20]
        subset, blocks = [], []
        for block_id, start in enumerate(starts):
            block = matched[start:start + 20]
            blocks.append([PurePosixPath(block[0][1]).stem, PurePosixPath(block[-1][1]).stem])
            for frame, rgb_path, depth_path in block:
                subset.append(frame)
                selected.extend([dict(rgb_files[rgb_path], kind='rgb'), dict(depth_files[depth_path], kind='depth')])
                frames_manifest.append({
                    'scene': scene, 'block': block_id, 'frame_id': PurePosixPath(rgb_path).stem,
                    'rgb': f'rgb/{rgb_path}', 'depth': f'depth/{depth_path}',
                    'camera': f'rgb/{camera_path}', 'colmap_im_id': frame.get('colmap_im_id'),
                })
        subset_metadata = dict(metadata, frames=subset)
        write_json(root / 'rgb' / '2025' / scene / 'transforms_subset.json', subset_metadata)
        group = {'scene': scene, 'metadata_frames': len(metadata['frames']), 'available_rgb_depth_frames': len(matched),
                 'selected_frames': len(subset), 'blocks_first_last_ids': blocks,
                 'bytes': sum(rgb_files[r]['size'] + depth_files[d]['size'] for _, r, d in
                              [row for start in starts for row in matched[start:start + 20]])}
        groups.append(group)
        print(json.dumps(group), flush=True)
    total = sum(entry['size'] for entry in selected)
    if total > budget_gib * 1024**3:
        raise RuntimeError(f'Selection {total / 1024**3:.3f} GiB exceeds budget {budget_gib}')
    if shutil.disk_usage(root).free < total + 5 * 1024**3:
        raise RuntimeError('Insufficient free disk space with 5 GiB reserve')
    write_json(root / 'frames.json', frames_manifest)
    write_json(manifest_path, {'recipe': 'paired_2025_validation_v1', 'repositories': REPOS, 'revisions': revisions,
                              'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'groups': groups,
                              'total_bytes': total, 'files': selected,
                              'split': 'validation_only', 'overlap_verified': False})
    (root / 'README.md').write_text(
        '# UAV3DCrop 2025 validation subset\n\n'
        'Validation only: 5 scenes, 60 RGB/depth frames each, 3 blocks of 20 available camera views per scene.\n'
        'Pinned source revisions and expected file hashes: `download_manifest.json`. Relative RGB/depth/camera '
        'paths and block membership: `frames.json`. Original full `transforms.json` is preserved; '
        '`transforms_subset.json` contains only the downloaded frames.\n\n'
        'RGB and depth retain their source encoding/resolution. Download verification does not establish '
        'geometric label accuracy. Candidate neighboring views still need visibility, depth consistency '
        'and cycle checks before constructing supervised pairs. Do not require a single homography for '
        'nonplanar crops. Camera model is OPENCV with distortion parameters; do not ignore distortion.\n\n'
        'The original camera file describes more images than this subset. Use `frames.json` or '
        '`transforms_subset.json`, not the full frame list, to enumerate available data.\n\n'
        'Held out by year relative to the current 2024 training cache; this does not prove geographic '
        'independence, since fields may be reused across years. Do not add this subset to training or '
        'use it as a final untouched test set after tuning.\n\n'
        'Continue/retry: `python tools/download_uav3dcrop_2025_validation.py --download`. '
        'Completed files are hash-checked and skipped; partial files use HTTP Range when supported.\n',
        encoding='utf-8')
    print(f'PLAN READY: {len(selected)} source files; {total / 1024**3:.3f} GiB; {root}', flush=True)


def download(root, workers):
    manifest_path = root / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries = manifest['files']
    remaining = sum(max(0, e['size'] - (local_path(root, e).stat().st_size if local_path(root, e).exists() else 0)) for e in entries)
    if shutil.disk_usage(root).free < remaining + 5 * 1024**3:
        raise RuntimeError('Insufficient free disk space with 5 GiB reserve')
    start, done, skipped, completed, errors = time.monotonic(), 0, 0, 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_file, root, e, manifest['revisions']): e for e in entries}
        for future in concurrent.futures.as_completed(futures):
            entry = futures[future]
            completed += 1
            try:
                size, was_skipped = future.result()
                done += size
                skipped += int(was_skipped)
            except Exception as exc:
                errors.append({'kind': entry['kind'], 'path': entry['path'], 'error': repr(exc)})
                print(f'FAILED {entry["path"]}: {exc}', flush=True)
            if completed % 20 == 0 or completed == len(entries):
                print(f'[{completed}/{len(entries)}] verified={done / 1024**3:.3f} GiB; '
                      f'skipped={skipped}; failures={len(errors)}; elapsed={(time.monotonic()-start)/60:.1f} min', flush=True)
    write_json(root / 'download_status.json', {'verified_files': len(entries) - len(errors),
               'verified_bytes': done, 'failures': errors, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    if errors:
        raise RuntimeError(f'{len(errors)} files failed; rerun --download to retry')
    print('ALL SOURCE FILES HASH VERIFIED', flush=True)


def inspect(root):
    import cv2
    import numpy as np
    import tifffile
    from PIL import Image, ImageDraw
    frames = json.loads((root / 'frames.json').read_text(encoding='utf-8'))
    samples, preview_rows = [], []
    # One image/depth pair from each of the three blocks per scene.
    seen = set()
    for frame in frames:
        key = (frame['scene'], frame['block'])
        if key in seen:
            continue
        seen.add(key)
        image = cv2.imread(str(root / frame['rgb']))
        depth = tifffile.imread(root / frame['depth'])
        camera = json.loads((root / frame['camera']).read_bytes())
        if image is None or depth.ndim != 2:
            raise RuntimeError(f'Cannot decode RGB/depth: {frame}')
        image_hw, depth_hw = list(image.shape[:2]), list(depth.shape)
        camera_hw = [int(camera['h']), int(camera['w'])]
        # RGB/depth may be released at different resolutions; report explicitly.
        positive = np.isfinite(depth) & (depth > 0)
        values = depth[positive]
        if not values.size:
            raise RuntimeError(f'No positive finite depth: {frame}')
        result = {'scene': frame['scene'], 'block': frame['block'], 'frame_id': frame['frame_id'],
                  'rgb_hw': image_hw, 'depth_hw': depth_hw, 'camera_hw': camera_hw,
                  'depth_dtype': str(depth.dtype), 'positive_finite_fraction': float(positive.mean()),
                  'positive_depth_p01_p50_p99': np.percentile(values, [1, 50, 99]).tolist(),
                  'rgb_matches_camera': image_hw == camera_hw, 'depth_matches_rgb': image_hw == depth_hw}
        samples.append(result)
        print(json.dumps(result), flush=True)
        if frame['block'] == 1:
            rgb_thumb = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
            depth_thumb = cv2.resize(depth.astype(np.float32), (480, 360), interpolation=cv2.INTER_NEAREST)
            low, high = np.percentile(values.astype(np.float32), [2, 98])
            scaled = np.clip((depth_thumb - low) / max(float(high - low), 1e-6), 0, 1)
            colors = cv2.applyColorMap(np.nan_to_num(scaled * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            colors[~np.isfinite(depth_thumb) | (depth_thumb <= 0)] = 0
            tile = Image.new('RGB', (960, 392), 'white')
            tile.paste(Image.fromarray(cv2.cvtColor(rgb_thumb, cv2.COLOR_BGR2RGB)), (0, 32))
            tile.paste(Image.fromarray(cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)), (480, 32))
            draw = ImageDraw.Draw(tile)
            draw.text((8, 8), f'{frame["scene"]} / {frame["frame_id"]} / RGB', fill='black')
            draw.text((488, 8), 'Depth (per-image 2-98% color range)', fill='black')
            preview_rows.append(tile)
    write_json(root / 'sanity_check.json', {'sampled_pairs': len(samples), 'samples': samples,
               'note': 'Decode/dimensions/value checks only; depth units and reprojection validity are not certified.'})
    preview = Image.new('RGB', (960, 392 * len(preview_rows)), 'white')
    for index, row in enumerate(preview_rows):
        preview.paste(row, (0, index * 392))
    preview.save(root / 'preview.jpg', quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--plan', action='store_true')
    actions.add_argument('--download', action='store_true')
    actions.add_argument('--inspect', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--budget-gib', type=float, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.budget_gib <= 0:
        parser.error('workers and budget must be positive')
    if args.plan:
        plan(args.root.resolve(), args.budget_gib)
    elif args.download:
        download(args.root.resolve(), args.workers)
    else:
        inspect(args.root.resolve())


if __name__ == '__main__':
    main()
