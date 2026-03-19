import os
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description="Generate the pair of dataset")
parser.add_argument("--pairs-folder", type=Path, required=True, help="Path to the dataset folder")
args = parser.parse_args()

base_folder = args.pairs_folder.resolve()

input1_path = base_folder / 'input1'
input2_path = base_folder / 'input2'

if not input1_path.exists() or not input2_path.exists():
    raise f"Error: {input1_path} or {input2_path} does not exist!"

img_list = sorted([f for f in os.listdir(input1_path) if not f.startswith('.')])

count = 0

with open("pair.txt", "w", encoding='utf-8') as f:
    for img_name in img_list:
        if img_name.startswith('.'):
            continue
            
        path_img1 = args.pairs_folder / "input1" / img_name
        path_img2 = args.pairs_folder / "input2" / img_name

        if not path_img1.exists() or not path_img2.exists():
            print(f"Warning: {img_name} is missing in one of the folders, skipping!")
        else:
            pair_txt = f"{path_img1} {path_img2}\n"
            f.write(pair_txt)
            count += 1

print("Done! pair.txt has been generated.")
print(f"Success! Generated pair.txt with {count} pairs.")
print(f"Sample path: {path_img1}")
