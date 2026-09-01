"""
Train / Val / Test split generation from patch blueprint JSONs.

Supports:
- Target class filtering
- Optional crop exclusion (e.g., holding out crop234 for testing)
- Configurable split ratios
"""

import os
import sys
import json
import random
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def prepare_splits(
    blueprint_json_path: str,
    output_dir: str,
    target_classes: list,
    split_ratios: list = None,
    excluded_crop: str = None,
    class_mapping: dict = None,
    strip_suffixes: bool = True,
    seed: int = 42,
) -> dict:
    """
    Filter a blueprint JSON by target classes, optionally exclude a crop,
    then split into train/val/test JSONs.

    Parameters
    ----------
    blueprint_json_path : str
        Path to the full patch blueprint JSON.
    output_dir : str
        Directory where train.json, val.json, test.json will be written.
    target_classes : list of str
        Organelle class names to retain (e.g., ['mito', 'ves', ...]).
    split_ratios : list of float, optional
        [train_frac, val_frac, test_frac]. Default [0.80, 0.10, 0.10].
    excluded_crop : str, optional
        If set, patches from this crop are excluded entirely (held out).
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    dict
        Keys: 'train_path', 'val_path', 'test_path', 'stats' (dict with counts).
    """
    if split_ratios is None:
        split_ratios = [0.80, 0.10, 0.10]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load blueprint
    with open(blueprint_json_path, "r") as f:
        blueprint = json.load(f)

    target_set = set(target_classes)

    # 2. Filter by target classes and optionally exclude a crop
    filtered_patches = []
    test_patches = []
    excluded_datasets = set()

    for patch in blueprint:
        # 1. Get the exact class string from the JSON
        cls = patch.get("class", "")
        
        # 2. Strict filtering: Must match your 13 targets exactly
        if cls not in target_set:
            continue

        # 3. Route valid patches to Test OR Train/Val
        if excluded_crop and patch.get("crop") == excluded_crop:
            excluded_datasets.add(patch.get("dataset"))
            test_patches.append(patch)
        else:
            filtered_patches.append(patch)

    print(f"Total original patches: {len(blueprint)}")
    print(f"Total filtered (train/val) patches: {len(filtered_patches)}")
    if excluded_crop and excluded_datasets:
        print(f"Held out {excluded_crop} for test set from dataset(s): {', '.join(excluded_datasets)}")
        print(f"Total holdout test patches: {len(test_patches)}")

    # 3. Save targets only patches
    targets_only_path = os.path.join(output_dir, "targets_only.json")
    with open(targets_only_path, "w") as f:
        json.dump(filtered_patches + test_patches, f, indent=4)

    # 4. Shuffle and split
    random.seed(seed)
    random.shuffle(filtered_patches)

    total_train_val = len(filtered_patches)
    # If no test_patches, we do standard 3-way split, else 2-way split of the remaining
    if len(test_patches) > 0:
        # Normalize train/val ratios to sum to 1
        train_frac = split_ratios[0] / (split_ratios[0] + split_ratios[1])
        train_end = int(total_train_val * train_frac)
        train_patches = filtered_patches[:train_end]
        val_patches = filtered_patches[train_end:]
    else:
        train_end = int(total_train_val * split_ratios[0])
        val_end = int(total_train_val * (split_ratios[0] + split_ratios[1]))
        train_patches = filtered_patches[:train_end]
        val_patches = filtered_patches[train_end:val_end]
        test_patches = filtered_patches[val_end:]

    # 5. Write splits
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    test_path = os.path.join(output_dir, "test.json")

    with open(train_path, "w") as f:
        json.dump(train_patches, f, indent=4)
    with open(val_path, "w") as f:
        json.dump(val_patches, f, indent=4)
    with open(test_path, "w") as f:
        json.dump(test_patches, f, indent=4)

    total = len(train_patches) + len(val_patches) + len(test_patches)

    stats = {
        "total": total,
        "train": len(train_patches),
        "val": len(val_patches),
        "test": len(test_patches),
    }

    print(f"Total Patches: {total}")
    print(f"-> Train: {stats['train']}")
    print(f"-> Val  : {stats['val']}")
    print(f"-> Test : {stats['test']}")

    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "stats": stats,
    }



def split_handler(
    blueprint_json_path: str,
    output_dir: str,
    target_classes: list,
    )->dict:

    # json_path = "/kaggle/input/datasets/jetminds/all-centroids-across-all-crops/all_centroids_across_all_crops.json"

    json_path = blueprint_json_path
   
    # Define sets for rare classes
    rare_np = {'np', 'np_in', 'np_out'}
    rare_nuc = {'nuc', 'ne', 'ne_mem', 'ne_lum', 'chrom', 'echrom', 'hchrom', 'nhchrom', 'nechrom', 'nucpl', 'nucleo'}
    rare_perox = {'perox', 'perox_lum', 'perox_mem'}
    rare_ld = {'ld', 'ld_lum', 'ld_mem'}

    # 1. Load the master blueprint
    with open(json_path, 'r') as f:
        blueprint = json.load(f)

    # 2. Filter and Profile (Now tracking exact patch counts per class)
    crop_profiles = defaultdict(lambda: {"total": 0, "classes": set(), "class_counts": defaultdict(int)})
    valid_patches = []

    for patch in blueprint:
        crop_id = patch.get("crop")
        cls = patch.get("class")
        
        if cls in target_classes:
            valid_patches.append(patch)
            crop_profiles[crop_id]["total"] += 1
            crop_profiles[crop_id]["classes"].add(cls)
            crop_profiles[crop_id]["class_counts"][cls] += 1

    # 3. Sort Crops by Diversity (Descending)
    sorted_crop_ids = [
        crop for crop, data in sorted(
            crop_profiles.items(), 
            key=lambda x: (len(x[1]["classes"]), x[1]["total"]), 
            reverse=True
        )
    ]

    # 4. Identify Injection Candidates
    injection_candidates = []

    for crop in sorted_crop_ids:
        c_classes = crop_profiles[crop]["classes"]
        
        # Catch multi-rare crops
        if (c_classes.intersection(rare_np) and 
            c_classes.intersection(rare_nuc) and 
            c_classes.intersection(rare_perox)):
            injection_candidates.append(crop)

    # 5. Build Initial Validation and Test Sets with STRICT Injections
    val_crops = []
    test_crops = []

    # Inject exactly 2 multi-rare crops total (1 for val, 1 for test)
    if len(injection_candidates) >= 2:
        val_crops.append(injection_candidates[0])
        test_crops.append(injection_candidates[1])
    elif len(injection_candidates) == 1:
        val_crops.append(injection_candidates[0])
        
    # Inject exactly 1 LD crop into Validation that has >= 10 LD patches
    ld_injected = False
    for crop in sorted_crop_ids:
        if crop not in val_crops and crop not in test_crops:
            c_counts = crop_profiles[crop]["class_counts"]
            # Sum all sub-compartments of LD to get the total count for this crop
            ld_count = sum(c_counts.get(ld_cls, 0) for ld_cls in rare_ld)
            
            if ld_count >= 10:
                val_crops.append(crop)
                ld_injected = True
                break

    # Fallback: if no single crop has >= 10 LD patches, pick the one with the highest available
    if not ld_injected:
        best_ld_crop = None
        max_ld = 0
        for crop in sorted_crop_ids:
            if crop not in val_crops and crop not in test_crops:
                c_counts = crop_profiles[crop]["class_counts"]
                ld_count = sum(c_counts.get(ld_cls, 0) for ld_cls in rare_ld)
                if ld_count > max_ld:
                    max_ld = ld_count
                    best_ld_crop = crop
                    
        if best_ld_crop:
            val_crops.append(best_ld_crop)

    pool = [c for c in sorted_crop_ids if c not in (val_crops + test_crops)]

    # 6. Corrected Mathematical Slicing Order
    val_target = 15
    test_target = 15

    needed_val = max(0, val_target - len(val_crops))
    needed_test = max(0, test_target - len(test_crops))

    # Calculate Train allocations first
    train_total = len(pool) - needed_val - needed_test
    train_high_div = int(train_total * 0.70)
    train_low_div = train_total - train_high_div

    # Step A: Train takes the absolute top diversity crops FIRST
    train_crops = pool[:train_high_div]
    pool = pool[train_high_div:]

    # Step B: Val takes the next slice (medium diversity)
    val_crops.extend(pool[:needed_val])
    pool = pool[needed_val:]

    # Step C: Test takes the next slice (medium-low diversity)
    test_crops.extend(pool[:needed_test])
    pool = pool[needed_test:]

    # Step D: Train takes the remaining absolute bottom diversity
    train_crops.extend(pool)

    # 7. Route Patches
    train_patches, val_patches, test_patches = [], [], []

    for patch in valid_patches:
        crop_id = patch.get("crop")
        if crop_id in test_crops:
            test_patches.append(patch)
        elif crop_id in val_crops:
            val_patches.append(patch)
        elif crop_id in train_crops:
            train_patches.append(patch)

    # 8. Output Verification
    print("=" * 60)
    print(f"Total Available Target Crops: {len(sorted_crop_ids)}")
    print(f"Total Train Crops: {len(train_crops):<3} | Patches: {len(train_patches)}")
    print(f"Total Val Crops:   {len(val_crops):<3} | Patches: {len(val_patches)}")
    print(f"Total Test Crops:  {len(test_crops):<3} | Patches: {len(test_patches)}")
    print("=" * 60)

    # 9. Save Files
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    test_path = os.path.join(output_dir, "test.json")


    print(f"This is train path: {train_path}")
    with open(train_path, 'w') as f:
        json.dump(train_patches, f, indent=4)
    with open(val_path, 'w') as f:
        json.dump(val_patches, f, indent=4)
    with open(test_path, 'w') as f:
        json.dump(test_patches, f, indent=4)

    return {
        "train_path": train_path,
        "val_path":val_path,
        "test_path":test_path 
    }