"""
PyTorch Dataset for loading 3D EM/label patches from zarr volumes.

Configurable for different jitter levels, class counts, and label remapping.
"""

import os
import sys
import json
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.data.zarr_utils import extract_safe
from code.data.augmentations import apply_augmentations


class PatchDataset(Dataset):
    """
    3D patch dataset that reads EM images and segmentation labels from zarr.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing patch metadata.
    zarr_map : dict
        Mapping from dataset names to lists of zarr paths.
    label_map : dict
        Mapping from raw semantic IDs to merged instance class IDs.
    patch_dim : int
        Patch size in voxels (isotropic cube). Default 128.
    max_jitter : int
        Maximum random spatial jitter in voxels. 0 = static sampling. Default 0.
    """

    def __init__(self, json_path, zarr_map, label_map, patch_dim=128, max_jitter=0,
                 augmentation_config=None, target_type="labels", num_classes=13,
                 scale_conditioned=False):
        self.patch_dim = patch_dim
        self.max_jitter = max_jitter
        self.zarr_map = zarr_map
        self.zarr_cache = {}
        self.augmentation_config = augmentation_config
        self.target_type = target_type
        self.num_classes = num_classes
        self.scale_conditioned = scale_conditioned

        with open(json_path, "r") as f:
            raw_patches = json.load(f)

        # Build fast label lookup array from the mapping dict
        self.label_lookup = np.zeros(256, dtype=np.int64)
        for semantic_id, instance_id in label_map.items():
            if semantic_id < 256:
                self.label_lookup[semantic_id] = instance_id

        # Validate patches against available zarr data
        self.patches = []
        missing_count = 0

        for patch in raw_patches:
            dataset = patch["dataset"]
            crop_id = patch["crop"]
            em_lvl = str(patch["em_scale"])
            lbl_lvl = str(patch["label_scale"])

            crop_found = False
            if dataset in self.zarr_map:
                for zarr_path in self.zarr_map[dataset]:
                    base_recon_path = os.path.join(zarr_path, "recon-1")
                    em_path = os.path.join(base_recon_path, "em", "fibsem-uint8", em_lvl)
                    label_path = os.path.join(
                        base_recon_path, "labels", "groundtruth", crop_id, "all", lbl_lvl
                    )
                    if os.path.exists(em_path) and os.path.exists(label_path):
                        crop_found = True
                        break

            if crop_found:
                self.patches.append(patch)
            else:
                missing_count += 1

        print(
            f"Dataset initialized. Retained {len(self.patches)} valid patches. "
            f"Pruned {missing_count} missing patches."
        )

    def __len__(self):
        return len(self.patches)

    def _get_zarr_handles(self, dataset, crop_id, em_scale, label_scale):
        """Fetch (or cache) opened zarr handles for a given crop."""
        import zarr

        cache_key = f"{dataset}_{crop_id}_{em_scale}_{label_scale}"
        if cache_key in self.zarr_cache:
            return self.zarr_cache[cache_key]

        if dataset not in self.zarr_map:
            raise FileNotFoundError(f"Dataset '{dataset}' not found in zarr map.")

        valid_em_path = None
        valid_label_path = None

        for zarr_path in self.zarr_map[dataset]:
            base_recon_path = os.path.join(zarr_path, "recon-1")
            temp_em = os.path.join(base_recon_path, "em", "fibsem-uint8", str(em_scale))
            temp_label = os.path.join(
                base_recon_path, "labels", "groundtruth", crop_id, "all", str(label_scale)
            )
            if os.path.exists(temp_label) and os.path.exists(temp_em):
                valid_em_path = temp_em
                valid_label_path = temp_label
                break

        if not valid_em_path:
            raise FileNotFoundError(
                f"Crop {crop_id} for dataset '{dataset}' could not be found."
            )

        em_zarr = zarr.open(valid_em_path, mode="r")
        label_zarr = zarr.open(valid_label_path, mode="r")

        self.zarr_cache[cache_key] = (em_zarr, label_zarr)
        return em_zarr, label_zarr

    def __getitem__(self, idx):
        patch = self.patches[idx]
        dataset = patch["dataset"]
        crop_id = patch["crop"]
        em_lvl = patch["em_scale"]
        lbl_lvl = patch["label_scale"]

        # 1. Load exact centers from the JSON
        l_center = np.array(patch["l_center"], dtype=float)
        e_center = np.array(patch["e_center"], dtype=float)
        e_shape = np.array(patch["e_shape"], dtype=float)
        
        # Scale e_shape if patch_dim differs from the blueprint's base 128
        if self.patch_dim != 128:
            ratio = self.patch_dim / 128.0
            e_shape = e_shape * ratio
        e_shape = np.round(e_shape).astype(int)

        # 2. Fetch Cached Zarr Handles
        em_zarr, label_zarr = self._get_zarr_handles(dataset, crop_id, em_lvl, lbl_lvl)

        # 3. Base Mathematical Centering
        base_l_start = np.floor(l_center - (self.patch_dim / 2.0)).astype(int)
        base_e_start = np.floor(e_center - (e_shape / 2.0)).astype(int)

        # 4. Generate Spatial Jitter
        if self.max_jitter > 0:
            jitter_vector = torch.randint(
                -self.max_jitter, self.max_jitter + 1, (3,)
            ).numpy()
        else:
            jitter_vector = np.zeros(3, dtype=int)

        # 5. Dynamic Boundary Clamping
        lbl_shape = np.array(label_zarr.shape)
        max_l_start = np.maximum(lbl_shape - self.patch_dim, 0)

        clamped_l_start = np.clip(base_l_start + jitter_vector, 0, max_l_start)
        effective_jitter = clamped_l_start - base_l_start
        clamped_e_start = base_e_start + effective_jitter

        # 6. Extraction
        lbl_np = extract_safe(
            label_zarr,
            clamped_l_start,
            [self.patch_dim, self.patch_dim, self.patch_dim],
            pad_value=0,
            out_dtype=np.int64,
        )
        em_np = extract_safe(em_zarr, clamped_e_start, e_shape.tolist(), pad_value=0)

        # 7. Semantic Remapping and Tensor Conversion
        remapped_lbl = self.label_lookup[lbl_np]

        # 8. Normalize EM to [0, 1] float32
        em_float = em_np.astype(np.float32) / 255.0

        # 9. Apply augmentations (only if config is provided and enabled)
        if self.augmentation_config is not None and self.augmentation_config.enabled:
            em_float, remapped_lbl = apply_augmentations(
                em_float, remapped_lbl, self.augmentation_config
            )

        # 10. Generate Targets
        if self.target_type == "sdt":
            import scipy.ndimage as ndi
            resolution = patch.get("resolution", [8.0, 8.0, 8.0])
            sdt_scale = 40.0 # nm
            
            sdt_tensor = np.zeros((self.num_classes, self.patch_dim, self.patch_dim, self.patch_dim), dtype=np.float32)
            for c in range(self.num_classes):
                mask = (remapped_lbl == c)
                if not np.any(mask):
                    # Class not in patch. Target is -1 (far outside).
                    sdt_tensor[c] = -1.0
                else:
                    in_dist = ndi.distance_transform_edt(mask, sampling=resolution)
                    out_dist = ndi.distance_transform_edt(~mask, sampling=resolution)
                    sdt = in_dist - out_dist
                    sdt_tensor[c] = np.tanh(sdt / sdt_scale)
            
            lbl_tensor = torch.from_numpy(sdt_tensor)
        else:
            lbl_tensor = torch.from_numpy(remapped_lbl)

        em_tensor = torch.from_numpy(em_float).unsqueeze(0)
        
        if getattr(self, "scale_conditioned", False):
            res = patch.get("resolution", [8.0, 8.0, 8.0])
            z_chan = torch.full_like(em_tensor, res[0])
            y_chan = torch.full_like(em_tensor, res[1])
            x_chan = torch.full_like(em_tensor, res[2])
            em_tensor = torch.cat([em_tensor, z_chan, y_chan, x_chan], dim=0)

        return em_tensor, lbl_tensor

    def get_patch_metadata(self, idx):
        """
        Return the raw metadata dict for a patch (dataset, crop, resolution, class, etc.).

        Used by evaluate_detailed.py to record per-patch provenance alongside metrics.
        """
        return self.patches[idx]



class DynamicCropDataset(Dataset):
    def __init__(self, crops_json_path, zarr_map, label_map, patch_dim=128):
        self.patch_dim = patch_dim
        self.zarr_map = zarr_map
        
        with open(crops_json_path, 'r') as f:
            self.crops = json.load(f)
            
        self.label_lookup = np.zeros(256, dtype=np.int64)
        for semantic_id, instance_id in label_map.items():
            if semantic_id < 256: 
                self.label_lookup[semantic_id] = instance_id
                
        self.zarr_cache = {}

    def get_scale_trans(self, path, level="s0"):
        scale = np.array([1.0, 1.0, 1.0])
        trans = np.array([0.0, 0.0, 0.0])
        if path is None: return scale, trans
        try:
            with open(f"{path}/.zattrs", 'r') as f:
                meta = json.load(f)
                multiscales = meta.get("multiscales", [{}])[0]
                for ds in multiscales.get("datasets", []):
                    if ds.get("path") == level:
                        for t in ds.get("coordinateTransformations", []):
                            if t.get("type") == "scale": scale = np.array(t["scale"])[-3:]
                            if t.get("type") == "translation": trans = np.array(t["translation"])[-3:]
                        return scale, trans
                if "coordinateTransformations" in multiscales:
                    for t in multiscales["coordinateTransformations"]:
                        if t.get("type") == "scale": scale = np.array(t["scale"])[-3:]
                        if t.get("type") == "translation": trans = np.array(t["translation"])[-3:]
        except Exception:
            pass
        return scale, trans

    def _get_zarr_handles(self, dataset, crop_id, em_scale, label_scale):
        cache_key = f"{dataset}_{crop_id}_{em_scale}_{label_scale}"
        if cache_key in self.zarr_cache:
            return self.zarr_cache[cache_key]
            
        valid_em_base = None
        valid_lbl_base = None
        for zarr_path in self.zarr_map.get(dataset, []):
            base_recon = os.path.join(zarr_path, "recon-1")
            temp_em = os.path.join(base_recon, "em", "fibsem-uint8")
            temp_lbl = os.path.join(base_recon, "labels", "groundtruth", crop_id, "all")
            if os.path.exists(temp_em) and os.path.exists(temp_lbl):
                valid_em_base = temp_em
                valid_lbl_base = temp_lbl
                break
                
        if not valid_em_base: raise FileNotFoundError(f"Missing Zarr paths for {crop_id}")
            
        em_zarr = zarr.open(os.path.join(valid_em_base, str(em_scale)), mode='r')
        label_zarr = zarr.open(os.path.join(valid_lbl_base, str(label_scale)), mode='r')
        
        all_scale_lbl, all_trans_lbl = self.get_scale_trans(valid_lbl_base, str(label_scale))
        em_scale_arr, em_trans_arr = self.get_scale_trans(valid_em_base, str(em_scale))
        
        self.zarr_cache[cache_key] = (em_zarr, label_zarr, all_scale_lbl, all_trans_lbl, em_scale_arr, em_trans_arr)
        return self.zarr_cache[cache_key]

    def __len__(self):
        return len(self.crops)
        
    def __getitem__(self, idx):
        crop_meta = self.crops[idx]
        dataset = crop_meta["dataset"]
        crop_id = crop_meta["crop"]
        em_lvl = crop_meta["em_scale"]
        lbl_lvl = crop_meta["label_scale"]
        
        em_zarr, label_zarr, all_scale_lbl, all_trans_lbl, em_scale, em_trans = self._get_zarr_handles(
            dataset, crop_id, em_lvl, lbl_lvl
        )
        
        lbl_shape = np.array(label_zarr.shape)
        half_patch = self.patch_dim // 2
        max_bounds = np.maximum(lbl_shape - half_patch, half_patch)
        
        # 1. Rejection Sampling Sieve
        max_retries = 5
        for attempt in range(max_retries):
            l_center = np.array([np.random.randint(half_patch, limit + 1) for limit in max_bounds])
            
            phys_center = (l_center * all_scale_lbl) + all_trans_lbl
            e_center = np.round((phys_center - em_trans) / em_scale).astype(int)
            e_shape = np.round(np.array([self.patch_dim]*3) * (all_scale_lbl / em_scale)).astype(int)
            
            l_start = np.floor(l_center - (self.patch_dim / 2.0)).astype(int)
            e_start = np.floor(e_center - (e_shape / 2.0)).astype(int)
            
            # Extract only the label first for the density check
            lbl_np = extract_safe(label_zarr, l_start.tolist(), [self.patch_dim]*3, pad_value=0, out_dtype=np.int64)
            remapped_lbl = self.label_lookup[lbl_np]
            
            bg_ratio = np.sum(remapped_lbl == 0) / (self.patch_dim ** 3)
            
            # Keep patch if it has at least 5% foreground, or if we run out of retries
            if bg_ratio < 0.95 or attempt == max_retries - 1:
                break
                
        # 2. Safe Extraction of EM (Executes only after a valid coordinate is found)
        em_np = extract_safe(em_zarr, e_start.tolist(), e_shape.tolist(), pad_value=0)
        
        em_tensor = torch.from_numpy(em_np.astype(np.float32) / 255.0).unsqueeze(0)
        lbl_tensor = torch.from_numpy(remapped_lbl)
        
        if em_tensor.shape[1:] != lbl_tensor.shape:
            em_tensor = torch.nn.functional.interpolate(
                em_tensor.unsqueeze(0), 
                size=lbl_tensor.shape, 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
            
        return em_tensor, lbl_tensor