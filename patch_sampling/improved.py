import os
import json
import math
import gc
import zarr
import numpy as np
from scipy.ndimage import label

# --- Configuration & Constants ---
RECEPTIVE_FIELD = 128
V_PATCH = RECEPTIVE_FIELD ** 3

COMPRESSION_ALPHA = 0.33
VOLUME_WEIGHT = 1.0
MAX_PATCHES_PER_CLASS = 200
R_MIN = 32 
R_MAX = int(RECEPTIVE_FIELD * 1.0)
EXPANSION_RATE = 0.2
COVERAGE_THRESHOLD = 0.85
AVAILABLE_SCALES = ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]

INSTANCE_CLASSES = ["cell", "endo", "ld", "lyso", "mito", "mt", "np", "nuc", "perox", "ves", "vim"]
SEMANTIC_CLASSES = ['chrom', 'cyto', 'euchrom', 'ecs', 'endo_lum', 'endo_mem', 'er_lum', 'er_mem', 
                    'eres_lum', 'eres_mem', 'golgi_lum', 'golgi_mem', 'hchrom', 'ld_lum', 'ld_mem', 
                    'lyso_lum', 'lyso_mem', 'mito_lum', 'mito_mem', 'mito_ribo', 'mt_in', 'mt_out', 
                    'ne_lum', 'ne_mem', 'np_in', 'np_out', 'nucpl', 'perox_lum', 'perox_mem', 'pm', 
                    'ves_lum', 'ves_mem', 'nechrom', 'nhchrom', 'cent', 'ribo']
TARGET_CLASSES = INSTANCE_CLASSES + SEMANTIC_CLASSES

# --- Helper Functions ---
def get_scale_trans(path, level="s0"):
    scale = np.array([1.0, 1.0, 1.0])
    trans = np.array([0.0, 0.0, 0.0])
    
    if path is None:
        return scale, trans
        
    try:
        with open(f"{path}/.zattrs", 'r') as f:
            meta = json.load(f)
            multiscales = meta.get("multiscales", [{}])[0]
            
            # 1. Target specific pyramid level
            for ds in multiscales.get("datasets", []):
                if ds.get("path") == level:
                    for t in ds.get("coordinateTransformations", []):
                        if t.get("type") == "scale": scale = np.array(t["scale"])
                        if t.get("type") == "translation": trans = np.array(t["translation"])
                    return scale, trans
                    
            # 2. Fallback to global multiscale coordinates
            if "coordinateTransformations" in multiscales:
                for t in multiscales["coordinateTransformations"]:
                    if t.get("type") == "scale": scale = np.array(t["scale"])
                    if t.get("type") == "translation": trans = np.array(t["translation"])
    except Exception:
        pass
        
    return scale, trans

def find_matching_em_level(em_base, target_scale):
    if em_base is None:
        return None, None, None
        
    try:
        with open(f"{em_base}/.zattrs", 'r') as f:
            meta = json.load(f)
            multiscales = meta.get("multiscales", [{}])[0]
            
            for ds in multiscales.get("datasets", []):
                level = ds.get("path")
                scale = np.array([1.0, 1.0, 1.0])
                trans = np.array([0.0, 0.0, 0.0])
                
                for t in ds.get("coordinateTransformations", []):
                    if t.get("type") == "scale": scale = np.array(t["scale"])
                    if t.get("type") == "translation": trans = np.array(t["translation"])
                
                # atol=0.2 absorbs floating point metadata drift
                if np.allclose(scale, target_scale, atol=0.2):
                    return level, scale, trans
    except Exception as e:
        print(f"Warning: Failed to read EM metadata. Error: {e}")
        
    return None, None, None


# --- Core Extraction Pipeline ---
def patch_extractor(dataset, single_crop=None):
  # ========================================================================
    main_dir = "/mnt/voxelcell_vol1" # replace with project folder. 
    out_dir = f"{main_dir}/patch_json"
    dataset_dir = f"{main_dir}/raw_data"
  # ========================================================================
    
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{single_crop}" if single_crop else "_full_dataset"
    output_file = os.path.join(out_dir, f"{dataset}{suffix}_anchors.json")
    
    black_ship_crop = [
        # "crop282", "crop337", 
        # "crop357", "crop358", 
        "crop247"                      # jrc_cos7-1a
    ]

    def resolve_path(sub_path):
        p = f"{dataset_dir}/{dataset}/{dataset}.zarr/{sub_path}"
        if os.path.exists(p):
            return p
        return None

    try:
        all_crops = set()
        l_base = f"{dataset_dir}/{dataset}/{dataset}.zarr/recon-1/labels/groundtruth"
        if os.path.exists(l_base):
            all_crops.update([item for item in os.listdir(l_base) if item.startswith('crop')])
        all_crops = sorted(list(all_crops))
    except Exception as e:
        raise FileNotFoundError(f"Could not access local paths. Error: {e}")
        
    master_blueprint = []
    rng = np.random.default_rng(42)
    crops_to_process = [single_crop] if single_crop else all_crops
    
    print(f"=== Initialization: {dataset} | Total Crops: {len(crops_to_process)} ===")
    
    crop_count = 1
    for crop_id in crops_to_process:
        if crop_id in black_ship_crop: continue
        print(f"\nProcessing: {crop_count}. {crop_id}")
        crop_count += 1
        
        match_found = False
        lbl_lvl, all_scale_lbl, all_trans_lbl = None, None, None
        em_lvl, em_scale, em_trans = None, None, None
        
        em_base = resolve_path("recon-1/em/fibsem-uint8")
        
        # 1. Establish Multi-Resolution Mapping
        for potential_lbl_lvl in AVAILABLE_SCALES:
            lbl_path = resolve_path(f"recon-1/labels/groundtruth/{crop_id}/all/{potential_lbl_lvl}")
            
            if not lbl_path:
                continue
                
            scale_lbl, trans_lbl = get_scale_trans(resolve_path(f"recon-1/labels/groundtruth/{crop_id}/all"), potential_lbl_lvl)
            e_lvl, e_scale, e_trans = find_matching_em_level(em_base, scale_lbl)
            
            if e_lvl is not None:
                lbl_lvl = potential_lbl_lvl
                all_scale_lbl = scale_lbl
                all_trans_lbl = trans_lbl
                em_lvl = e_lvl
                em_scale = e_scale
                em_trans = e_trans
                match_found = True
                break
                
        if not match_found:
            print(f"  -> SKIPPING {crop_id}: Could not find a matching resolution between Label and EM.")
            continue
            
        print(f"  -> Match Locked: Label {lbl_lvl} == EM {em_lvl} at {all_scale_lbl.tolist()} nm")
        
        for cls in TARGET_CLASSES:
            cls_path = resolve_path(f"recon-1/labels/groundtruth/{crop_id}/{cls}/{lbl_lvl}")
            if not cls_path: continue
            
            cls_scale, cls_trans = get_scale_trans(resolve_path(f"recon-1/labels/groundtruth/{crop_id}/{cls}"), lbl_lvl)
            
            try:
                z_arr = zarr.open(cls_path, mode='r')
                tensor_np = np.array(z_arr)
                D_z, D_y, D_x = tensor_np.shape
                v_c = np.count_nonzero(tensor_np)
                if v_c == 0: continue
                
                # 2. Researched Patch Formulation (Budgeting)
                _, num_features = label(tensor_np > 0)
             
                
                # --------------------------------------------------------------------------------------
                V_CROP = D_z * D_y * D_x
                K = V_CROP / V_PATCH
                n_spatial = K * (1 - (1 - 1/K)**num_features) if K > 1 else (1.0 if num_features > 0 else 0.0)
                n_volumetric = VOLUME_WEIGHT * ((v_c / V_PATCH) ** COMPRESSION_ALPHA)
                
                n_c = int(round(n_spatial + n_volumetric))
                n_c = max(1, min(n_c, MAX_PATCHES_PER_CLASS))
                
                progress = 1.0 - math.exp(-EXPANSION_RATE * (n_c - 1))
                r_c = int(round(R_MIN + (R_MAX - R_MIN) * progress))
                
                # 3. Initialize Boolean Sieves
                valid_mask = (tensor_np > 0)
                covered_mask = np.zeros_like(tensor_np, dtype=bool)
                extracted_anchors = []
                
                # 4. High-Speed Random Boolean Sieve (Radius-Enforced)
                while len(extracted_anchors) < n_c:
                    if len(extracted_anchors) > 0:
                        if (np.count_nonzero(covered_mask) / v_c) >= COVERAGE_THRESHOLD: 
                            break
                            
                    valid_indices = np.argwhere(valid_mask)
                    if len(valid_indices) == 0: 
                        break 
                    
                    idx = rng.integers(0, len(valid_indices))
                    anchor = valid_indices[idx]
                    cz, cy, cx = anchor[0], anchor[1], anchor[2]
                    extracted_anchors.append([cz, cy, cx])
                    
                    # Spherical exclusion mask
                    z_r, y_r, x_r = np.ogrid[:D_z, :D_y, :D_x]
                    dist_mask = ((z_r - cz)**2 + (y_r - cy)**2 + (x_r - cx)**2) < (r_c**2)
                    valid_mask &= ~dist_mask
                    
                    # Cubic coverage mask
                    half_patch = RECEPTIVE_FIELD // 2
                    z_min, z_max = max(0, cz - half_patch), min(D_z, cz + half_patch)
                    y_min, y_max = max(0, cy - half_patch), min(D_y, cy + half_patch)
                    x_min, x_max = max(0, cx - half_patch), min(D_x, cx + half_patch)
                    
                    covered_mask[z_min:z_max, y_min:y_max, x_min:x_max] |= (
                        tensor_np[z_min:z_max, y_min:y_max, x_min:x_max] > 0
                    )
                
                print(f"  -> [{cls:<6}] Budget Allocation: {len(extracted_anchors)}/{n_c}")
                
                # 5. Physical Mapping to Centroids
                for coord in extracted_anchors:
                    raw_cls_center = np.array(coord)
                    
                    # Map to true physical nanometers
                    phys_center = (raw_cls_center * cls_scale) + cls_trans
                    
                    # Map physical nanometers back to unified Label and EM voxel spaces
                    all_center_lbl = np.round((phys_center - all_trans_lbl) / all_scale_lbl).astype(int)
                    all_center_em = np.round((phys_center - em_trans) / em_scale).astype(int)
                    
                    # Calculate equivalent EM patch shape based on scale ratios
                    e_shape = np.round(np.array([RECEPTIVE_FIELD, RECEPTIVE_FIELD, RECEPTIVE_FIELD]) * (all_scale_lbl / em_scale)).astype(int)
                    
                    master_blueprint.append({
                        "dataset": dataset, 
                        "crop": crop_id, 
                        "em_scale": em_lvl,     
                        "label_scale": lbl_lvl,
                        "resolution": all_scale_lbl.tolist(),  
                        "class": cls,
                        "center_z": int(coord[0]), 
                        "center_y": int(coord[1]), 
                        "center_x": int(coord[2]),
                        "l_center": all_center_lbl.tolist(),
                        "e_center": all_center_em.tolist(),
                        "e_shape": e_shape.tolist()
                    })
                
                del tensor_np, z_arr, valid_mask, covered_mask, extracted_anchors
            except Exception as e:
                print(f"  -> Error {cls}: {e}")
        
        gc.collect()
        
    with open(output_file, "w") as f:
        json.dump(master_blueprint, f, indent=4)
    print(f"  [Checkpoint] Saved {len(master_blueprint)} total patches to {output_file}")
        
    print(f"\nExtraction Complete. Final dataset saved to {output_file}")

datasets = [
    "jrc_cos7-1a",
    "jrc_cos7-1b",
    "jrc_ctl-id8-1",
    "jrc_fly-mb-1a",
    "jrc_fly-vnc-1",
    "jrc_hela-2",
    "jrc_hela-3",
    "jrc_jurkat-1",
    "jrc_macrophage-2",
    "jrc_mus-heart-1",
    "jrc_mus-kidney",
    "jrc_mus-kidney-3",
    "jrc_mus-kidney-glomerulus-2",
    "jrc_mus-liver",
    "jrc_mus-liver-3",
    "jrc_mus-liver-zon-1",
    "jrc_mus-liver-zon-2",
    "jrc_mus-nacc-1",
    "jrc_sum159-1",
    "jrc_sum159-4",
    "jrc_ut21-1413-003",
    "jrc_zf-cardiac-1"
]
for dataset in datasets:
    patch_extractor(dataset, single_crop=None)
