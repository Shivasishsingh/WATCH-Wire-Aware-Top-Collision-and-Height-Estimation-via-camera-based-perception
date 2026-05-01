#!/usr/bin/env python3
"""
Split dataset of images + ground-truths + depths into train/val/test.
Usage:
    cd /Vaibhav/shivasish1/PDFNet/combined_dataset
    # --- MODIFIED ---
    python3 split_dataset.py --images images --gts gts --depths depths --train 0.7 --val 0.2 --test 0.1 --seed 42
"""
import os
import shutil
import argparse
import random
from pathlib import Path

def get_basenames(folder, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    """Finds all file basenames in a folder with given extensions."""
    files = []
    for p in Path(folder).iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p.stem)
    return set(files)

def copy_list(pairs, src_images, src_gts, src_depths, dst_base, split_name):
    """Copies image/gt/depth pairs to a destination split folder."""
    img_out = Path(dst_base) / split_name / "images"
    gt_out  = Path(dst_base) / split_name / "gts"
    depth_out = Path(dst_base) / split_name / "depths"
    
    img_out.mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    listfile = Path(dst_base) / f"{split_name}.txt"
    with listfile.open("w") as f:
        for stem in pairs:
            # find actual file names (prefer jpg then png)
            img_src = None
            gt_src  = None
            depth_src = None
            
            # Find image file
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                cand = Path(src_images) / (stem + ext)
                if cand.exists():
                    img_src = cand
                    break
            
            # Find ground-truth file
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                cand = Path(src_gts) / (stem + ext)
                if cand.exists():
                    gt_src = cand
                    break
            
            # Find depth file
            # (Assuming common depth formats like .png or .tif)
            for ext in [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"]:
                cand = Path(src_depths) / (stem + ext)
                if cand.exists():
                    depth_src = cand
                    break

            # Check if all three files were found
            if img_src is None or gt_src is None or depth_src is None:
                # skip if one of them missing
                print(f"Warning: Skipping {stem}, a file is missing (img: {img_src is not None}, gt: {gt_src is not None}, depth: {depth_src is not None}).")
                continue

            # Copy all three files
            shutil.copy2(img_src, img_out / img_src.name)
            shutil.copy2(gt_src,  gt_out  / gt_src.name)
            shutil.copy2(depth_src, depth_out / depth_src.name)
            
            # Write basename to the list file
            f.write(f"{stem}\n")

def main(args):
    src_images = Path(args.images)
    src_gts    = Path(args.gts)
    src_depths = Path(args.depths)
    
    # Assert all three paths exist
    assert src_images.exists() and src_gts.exists() and src_depths.exists(), \
        f"Paths must exist: {src_images}, {src_gts}, {src_depths}"

    # get basenames present in all three
    image_basenames = get_basenames(src_images)
    gt_basenames    = get_basenames(src_gts)
    depth_basenames = get_basenames(src_depths)

    # Find intersection of all three sets
    common = sorted(list(
        image_basenames.intersection(gt_basenames).intersection(depth_basenames)
    ))

    if len(common) == 0:
        raise SystemExit("No matching image–gt–depth basenames found. Check filenames and extensions.")

    print(f"Found {len(common)} matching triplets (image, gt, depth).")

    # deterministic shuffle
    random.seed(args.seed)
    random.shuffle(common)

    n = len(common)
    n_train = int(args.train_ratio * n)
    n_val   = int(args.val_ratio * n)
    # ensure total <= n, assign the rest to test
    n_test  = n - n_train - n_val
    if n_test < 0:
        raise SystemExit("Ratios sum to >1. Adjust them.")

    train_list = common[:n_train]
    val_list   = common[n_train:n_train + n_val]
    test_list  = common[n_train + n_val:]

    print(f"Split counts -> train: {len(train_list)}, val: {len(val_list)}, test: {len(test_list)}")

    # destination base (in same parent by default)
    dst_base = args.output if args.output else Path(src_images).parent

    # Copy files for each split
    copy_list(train_list, src_images, src_gts, src_depths, dst_base, "train")
    copy_list(val_list,   src_images, src_gts, src_depths, dst_base, "val")
    copy_list(test_list,  src_images, src_gts, src_depths, dst_base, "test")

    print("Done. Created directories (train/val/test) with images/, gts/, depths/ and .txt lists.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split image+gt+depth dataset into train/val/test keeping sets")
    parser.add_argument("--images", required=True, help="Path to images folder (e.g. images)")
    parser.add_argument("--gts", required=True, help="Path to ground-truth folder (e.g. gts)")
    parser.add_argument("--depths", required=True, help="Path to depth maps folder (e.g. depths)")
    
    parser.add_argument("--train", dest="train_ratio", type=float, default=0.7, help="Train ratio (default 0.7)")
    parser.add_argument("--val", dest="val_ratio", type=float, default=0.2, help="Val ratio (default 0.2)")
    parser.add_argument("--test", dest="test_ratio", type=float, default=0.1, help="Test ratio (default 0.1). If ratios don't sum to 1, remainder goes to test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output base folder (default: same parent of images/gts/depths)")
    args = parser.parse_args()

    # normalize: if user passed test ratio explicitly, override computed remainder
    s = args.train_ratio + args.val_ratio + args.test_ratio
    if s <= 1.0:
        # use as-is (test may be remainder)
        pass
    else:
        # normalize proportions
        print(f"Warning: Ratios sum to {s} > 1.0. Normalizing proportions.")
        args.train_ratio = args.train_ratio / s
        args.val_ratio   = args.val_ratio / s
        args.test_ratio  = args.test_ratio / s

    main(args)