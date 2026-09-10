import os
import sys
import json
import zarr
import numpy as np
import argparse

# Ensure project root is on sys.path for cross-module imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.utils.config import load_config
from code.data.zarr_utils import build_zarr_map
from code.utils.paths import setup_run_directory
from code.data.splits import split_handler

def parse_args():
    parser = argparse.ArgumentParser(description="Compute RFS Weights")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    run_paths = setup_run_directory(config, config_path=args.config)
    print(f"Run directory: {run_paths['run_dir']}")

    # 3. Prepare data splits
    blueprint_path = os.path.join(config.paths.json_dir, config.data.blueprint_json)
    split_output_dir = os.path.join(run_paths["run_dir"], "splits")


    split_paths = split_handler(
            blueprint_json_path=blueprint_path,
            output_dir=split_output_dir,
    )



    print(split_paths["train_path"])
    sys.exit(0)
    data_dir = config.paths.data_dir
    print(f"Scanning for Zarr datasets in: {data_dir}")
    ZARR_MAP = build_zarr_map(data_dir)

    # 2. Define input and output paths
    train_json_path = split_paths["train_path"]
    output_json_path = "/kaggle/working/crop_rfs_weights.json"
    
    with open(train_json_path, 'r') as f:
        raw_train_crops = json.load(f)

    # EXPLICITLY REMOVE CROP247
    train_crops = [meta for meta in raw_train_crops if meta["crop"] != "crop247"]
    print(f"Removed crop247. Remaining training crops: {len(train_crops)}")

    # 3. Setup Mapping & Variables
    semantic_to_instance_map = {
        50: 1, 3: 1, 4: 1, 5: 1,
        41: 2, 8: 2, 9: 2,
        42: 3, 10: 3, 11: 3,
        43: 4, 12: 4, 13: 4,
        44: 5, 14: 5, 15: 5,
        37: 6, 52: 6, 20: 6, 21: 6, 65: 6, 54: 6, 26: 6, 24: 6, 25: 6, 27: 6, 28: 6, 29: 6,
        53: 7, 22: 7, 23: 7,
        55: 8, 30: 8, 36: 8,
        49: 9, 47: 9, 48: 9,
        40: 10, 6: 10, 7: 10,
        51: 11, 16: 11, 17: 11, 64: 11,
        46: 12, 18: 12, 19: 12,
        38: 0
    }
    
    macro_names = {
        0: "Background", 1: "Mito", 2: "Vesicles", 3: "Endosomes", 4: "Lysosomes",
        5: "Lipid Droplets", 6: "Nucleus", 7: "Nuclear Pores", 8: "Microtubules",
        9: "Peroxisomes", 10: "Golgi", 11: "ER", 12: "ERES"
    }

    label_lookup = np.zeros(256, dtype=np.int64)
    for semantic_id, instance_id in semantic_to_instance_map.items():
        label_lookup[semantic_id] = instance_id

    global_class_voxels = {i: 0 for i in range(1, 13)}
    crop_class_presence = {}

    # --- PHASE 1: LOOP AND COUNT ---
    for i, meta in enumerate(train_crops):
        print(f"\n--- Processing [{i+1}/{len(train_crops)}] ---")
        
        # Step 1: Select crop and print features
        dataset = meta["dataset"]
        crop_id = meta["crop"]
        lbl_lvl = str(meta["label_scale"])
        print(f"Step 1 | Crop: {crop_id} | Dataset: {dataset} | Label Scale: {lbl_lvl}")
        
        valid_lbl = None
        for zarr_path in ZARR_MAP.get(dataset, []):
            temp_lbl = os.path.join(zarr_path, "recon-1", "labels", "groundtruth", crop_id, "all", lbl_lvl)
            if os.path.exists(temp_lbl):
                valid_lbl = temp_lbl
                break
                
        if not valid_lbl:
            print(f"ERROR: Label path not found for {crop_id}. Skipping.")
            continue
            
        # Step 2: Load volume, print dim, number of raw classes, and raw IDs
        label_zarr = zarr.open(valid_lbl, mode='r')
        lbl_np = label_zarr[:]
        z, y, x = lbl_np.shape
        total_crop_voxels = z * y * x
        
        raw_unique = np.unique(lbl_np)
        print(f"Step 2 | Loaded RAM. Dim: {z}x{y}x{x} | Total Raw Classes: {len(raw_unique)} | Raw IDs: {raw_unique.tolist()}")
        
        # Step 3: Perform encoding
        remapped_lbl = label_lookup[lbl_np]
        encoded_unique, encoded_counts = np.unique(remapped_lbl, return_counts=True)
        print("Step 3 | Encoding Applied.")
        
        # Step 4: Calculate and verify voxels
        print("Step 4 | Voxel Calculation:")
        sum_encoded_voxels = 0
        present_targets = []
        
        for cls, count in zip(encoded_unique, encoded_counts):
            cls = int(cls)
            sum_encoded_voxels += count
            print(f"  -> Class: {macro_names[cls]:<14} (ID {cls:02d}): {count} voxels")
            
            if cls > 0: # Add to global target count
                global_class_voxels[cls] += count
                present_targets.append(cls)
                
        crop_class_presence[crop_id] = present_targets
        
        # Verification
        is_verified = (sum_encoded_voxels == total_crop_voxels)
        print(f"Verification | Total Encoded: {sum_encoded_voxels} == Crop Volume: {total_crop_voxels} -> {is_verified}")

    # --- PHASE 2: COMPUTING RFS VARIABLES ---
    # Note: This is now correctly outside the Phase 1 loop.
    print("\n" + "="*60)
    print("=== PHASE 2: COMPUTING RFS VARIABLES ===")

    total_annotated_voxels = sum(global_class_voxels.values())
    print(f"Total Target Voxels Across Dataset: {total_annotated_voxels:,}")

    if total_annotated_voxels == 0:
        print("ERROR: No target voxels found. Exiting.")
        return

    # 1. Frequencies
    print("\n[Frequencies (f_c)]")
    f_c = {}
    for cls in range(1, 13):
        count = global_class_voxels[cls]
        fraction = (count / total_annotated_voxels) if total_annotated_voxels > 0 else 0
        f_c[cls] = fraction
        print(f"  -> {macro_names[cls]:<14} (ID {cls:02d}): f_c = {fraction:.6f}")

    # 2. Threshold
    print("\n[Dynamic Threshold (t)]")
    valid_freqs = [freq for freq in f_c.values() if freq > 0]
    t = 0.5 * np.median(valid_freqs) if valid_freqs else 0
    print(f"  -> t = {t:.6f}")

    # 3. Repeat Factors
    print("\n[Class Repeat Factors (r_c)]")
    r_c = {}
    for cls in range(1, 13):
        freq = f_c[cls]
        factor = max(1.0, np.sqrt(t / freq)) if freq > 0 else 1.0
        r_c[cls] = factor
        print(f"  -> {macro_names[cls]:<14} (ID {cls:02d}): r_c = {factor:.4f}")

    # 4. Crop Weights
    print("\n[Crop Weights (r_i)]")
    r_i = {}
    for crop_id, classes_present in crop_class_presence.items():
        if classes_present:
            crop_weight = float(max([r_c[cls] for cls in classes_present]))
        else:
            crop_weight = 1.0 
        r_i[crop_id] = crop_weight

    with open(output_json_path, 'w') as f:
        json.dump(r_i, f, indent=4)

    print(f"\nSUCCESS: Pipeline complete. Saved {len(r_i)} crop weights to {output_json_path}")

if __name__ == "__main__":
    print("start...")
    main()