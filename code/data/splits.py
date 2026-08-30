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
        cls = patch.get("class", "")
        
        # 1. Normalize class name first
        if class_mapping and cls in class_mapping:
            base_cls = class_mapping[cls]
        elif strip_suffixes and "_" in cls and cls.rsplit("_", 1)[-1] in ["lum", "mem", "in", "out"]:
            base_cls = cls.rsplit("_", 1)[0]
        else:
            base_cls = cls

        # 2. Filter out non-target classes immediately
        if base_cls not in target_set:
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
