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
    excluded_datasets = set()

    for patch in blueprint:
        if excluded_crop and patch.get("crop") == excluded_crop:
            excluded_datasets.add(patch.get("dataset"))
            continue
        if patch.get("class") in target_set:
            filtered_patches.append(patch)

    print(f"Total original patches: {len(blueprint)}")
    print(f"Total filtered patches: {len(filtered_patches)}")
    if excluded_crop and excluded_datasets:
        print(f"Excluded {excluded_crop} from dataset(s): {', '.join(excluded_datasets)}")

    # 3. Save filtered patches
    targets_only_path = os.path.join(output_dir, "targets_only.json")
    with open(targets_only_path, "w") as f:
        json.dump(filtered_patches, f, indent=4)

    # 4. Shuffle and split
    random.seed(seed)
    random.shuffle(filtered_patches)

    total = len(filtered_patches)
    train_end = int(total * split_ratios[0])
    val_end = int(total * (split_ratios[0] + split_ratios[1]))

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

    stats = {
        "total": total,
        "train": len(train_patches),
        "val": len(val_patches),
        "test": len(test_patches),
    }

    print(f"Total Patches: {total}")
    print(f"-> Train ({split_ratios[0]*100:.0f}%): {stats['train']}")
    print(f"-> Val   ({split_ratios[1]*100:.0f}%): {stats['val']}")
    print(f"-> Test  ({split_ratios[2]*100:.0f}%): {stats['test']}")

    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "stats": stats,
    }
