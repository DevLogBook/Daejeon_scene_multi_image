import os
import random
import shutil
from pathlib import Path


def clean_and_mv_dataset(
        dir1: str,
        dir2: str,
        out_dir1: str,
        out_dir2: str,
        output_txt: str = "sampled_pairs.txt",
        sample_size: int = 10000,
        sample_ratio: float = None,
        seed: int = 42
):
    """
    检查数据集，删除未对齐的孤立图片，随机抽样并将抽中的图片移动 (mv) 到新文件夹。
    """
    path1, path2 = Path(dir1), Path(dir2)
    out_path1, out_path2 = Path(out_dir1), Path(out_dir2)

    if not path1.exists() or not path2.exists():
        raise FileNotFoundError("❌ 输入文件夹不存在，请检查 dir1 和 dir2 路径。")

    out_path1.mkdir(parents=True, exist_ok=True)
    out_path2.mkdir(parents=True, exist_ok=True)

    # 1. 获取所有有效图片文件名
    files1 = {f.name for f in path1.iterdir() if f.is_file() and not f.name.startswith('.')}
    files2 = {f.name for f in path2.iterdir() if f.is_file() and not f.name.startswith('.')}

    # 2. 寻找并删除孤立文件
    orphans_in_1 = files1 - files2
    orphans_in_2 = files2 - files1

    print(
        f"🔍 检查完成: {path1.name} 中有 {len(orphans_in_1)} 个孤立文件, {path2.name} 中有 {len(orphans_in_2)} 个孤立文件。")

    for orphan in orphans_in_1:
        (path1 / orphan).unlink()
    for orphan in orphans_in_2:
        (path2 / orphan).unlink()

    if orphans_in_1 or orphans_in_2:
        print("🗑️ 孤立文件已全部删除。")

    # 3. 获取完美对齐的文件列表
    matched_files = sorted(list(files1 & files2))
    total_matched = len(matched_files)
    print(f"✅ 完美匹配的图像对总数: {total_matched}")

    if total_matched == 0:
        print("⚠️ 没有找到任何匹配的图像对，程序退出。")
        return

    # 4. 计算抽样数量
    random.seed(seed)
    if sample_ratio is not None:
        actual_sample_size = int(total_matched * sample_ratio)
    else:
        actual_sample_size = min(sample_size, total_matched)

    sampled_files = random.sample(matched_files, actual_sample_size)
    print(f"🎲 随机抽取了 {actual_sample_size} 对图像进行移动。")

    # 5. 执行移动 (mv) 并写入 pair 文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        for filename in sampled_files:
            src_p1 = path1 / filename
            src_p2 = path2 / filename

            dst_p1 = out_path1 / filename
            dst_p2 = out_path2 / filename

            # 使用 shutil.move 物理移动文件
            shutil.move(str(src_p1), str(dst_p1))
            shutil.move(str(src_p2), str(dst_p2))

            # 写入新位置的绝对路径
            f.write(f"{dst_p1.resolve()} {dst_p2.resolve()}\n")

    print(f"🚀 移动完成！抽样列表已保存至: {output_txt}")


if __name__ == "__main__":
    clean_and_mv_dataset(
        dir1=r"D:\Program\Stitching\stitch-wheat\training\input1",
        dir2=r"D:\Program\Stitching\stitch-wheat\training\input2",
        out_dir1="../training/output1",
        out_dir2="../training/output2",
        output_txt="uav_sampled_pairs.txt",
        sample_size=15000
    )