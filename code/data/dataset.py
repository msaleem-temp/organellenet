"""
PyTorch Dataset for loading 3D EM/label patches from zarr volumes.

Configurable for different jitter levels, class counts, and label remapping.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.data.zarr_utils import extract_safe


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

    def __init__(self, json_path, zarr_map, label_map, patch_dim=128, max_jitter=0):
        self.patch_dim = patch_dim
        self.max_jitter = max_jitter
        self.zarr_map = zarr_map
        self.zarr_cache = {}

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
        e_shape = np.array(patch["e_shape"], dtype=int)

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

        em_tensor = torch.from_numpy(em_np.astype(np.float32) / 255.0).unsqueeze(0)
        lbl_tensor = torch.from_numpy(remapped_lbl)

        return em_tensor, lbl_tensor
