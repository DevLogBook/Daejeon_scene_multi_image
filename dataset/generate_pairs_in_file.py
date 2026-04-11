import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate the pair of dataset")
parser.add_argument("--pairs-folder", type=Path, required=True, help="Path to the dataset folder")
args = parser.parse_args()

base_folder = args.pairs_folder.resolve()

img_list = sorted([f for f in os.listdir(base_folder) if not f.startswith('.')])

count = 0

with open("pairs_2.txt", "w", encoding='utf-8') as f:
    for i, img_name in enumerate(img_list):
        if img_name.startswith('.') or img_list[i+1].startswith('.'):
            continue
        if i == len(img_list) - 2:
            break
        path_img1 = args.pairs_folder / img_list[i]
        path_img2 = args.pairs_folder / img_list[i+1]

        if not path_img1.exists() or not path_img2.exists():
            print(f"Warning: {img_name} is missing in one of the folders, skipping!")
        else:
            pair_txt = f"{path_img1} {path_img2}\n"
            f.write(pair_txt)
            count += 1

print("Done! pair.txt has been generated.")
print(f"Success! Generated pair.txt with {count} pairs.")
print(f"Sample path: {path_img1}")