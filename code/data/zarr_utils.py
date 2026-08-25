"""
Zarr I/O utilities for OrganelleNet.

Provides safe boundary-clamped patch extraction, zarr directory scanning,
coordinate metadata reading, and full-crop aligned volume extraction.
"""

import os
import sys
import json
import glob
import numpy as np
import zarr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def extract_safe(zarr_arr, start_coords, patch_shape, pad_value=0, out_dtype=None):
    """
    Extract a 3D patch from a zarr array with safe boundary clamping.

    If the requested region extends beyond the array boundaries, the output
    patch is zero-padded (or filled with `pad_value`).

    Parameters
    ----------
    zarr_arr : zarr.Array
        Source zarr array (3D).
    start_coords : array-like of int, shape (3,)
        Starting ZYX coordinates for extraction.
    patch_shape : list/tuple of int, length 3
        Desired output patch shape [Z, Y, X].
    pad_value : scalar, optional
        Fill value for out-of-bounds regions. Default 0.
    out_dtype : dtype, optional
        Output dtype. If None, uses zarr_arr.dtype.

    Returns
    -------
    np.ndarray
        Extracted patch of shape `patch_shape`.
    """
    arr_shape = zarr_arr.shape
    z_min, z_max = max(0, start_coords[0]), min(arr_shape[0], start_coords[0] + patch_shape[0])
    y_min, y_max = max(0, start_coords[1]), min(arr_shape[1], start_coords[1] + patch_shape[1])
    x_min, x_max = max(0, start_coords[2]), min(arr_shape[2], start_coords[2] + patch_shape[2])

    target_dtype = out_dtype if out_dtype is not None else zarr_arr.dtype
    patch = np.full(patch_shape, fill_value=pad_value, dtype=target_dtype)

    pz_min = z_min - start_coords[0]
    py_min = y_min - start_coords[1]
    px_min = x_min - start_coords[2]

    if z_max > z_min and y_max > y_min and x_max > x_min:
        patch[
            pz_min:pz_min + (z_max - z_min),
            py_min:py_min + (y_max - y_min),
            px_min:px_min + (x_max - x_min),
        ] = zarr_arr[z_min:z_max, y_min:y_max, x_min:x_max]

    return patch


def build_zarr_map(data_root: str) -> dict:
    """
    Scan a data directory for zarr datasets and return a mapping.

    Searches for pattern: `data_root/*/dataset.zarr`

    Parameters
    ----------
    data_root : str
        Root directory containing dataset folders.

    Returns
    -------
    dict
        Mapping of dataset_name → list of zarr paths.
    """
    zarr_map = {}
    search_pattern = os.path.join(data_root, "*", "*.zarr")

    for zarr_path in glob.glob(search_pattern):
        dataset_name = os.path.basename(zarr_path).replace(".zarr", "")
        if dataset_name not in zarr_map:
            zarr_map[dataset_name] = []
        if zarr_path not in zarr_map[dataset_name]:
            zarr_map[dataset_name].append(zarr_path)

    return zarr_map


def get_scale_trans(base_path: str, level: str = "s0"):
    """
    Read scale and translation coordinate transforms from zarr .zattrs metadata.

    Parameters
    ----------
    base_path : str
        Path to the zarr group containing .zattrs.
    level : str
        Pyramid level to extract transforms for (e.g., "s0").

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (scale, translation) each of shape (3,).
    """
    scale = np.array([1.0, 1.0, 1.0])
    trans = np.array([0.0, 0.0, 0.0])

    try:
        zattrs_path = os.path.join(base_path, ".zattrs")
        with open(zattrs_path, "r") as f:
            meta = json.load(f)
            multiscales = meta.get("multiscales", [{}])[0]

            for ds in multiscales.get("datasets", []):
                if ds.get("path") == level:
                    for t in ds.get("coordinateTransformations", []):
                        if t.get("type") == "scale":
                            scale = np.array(t["scale"])
                        if t.get("type") == "translation":
                            trans = np.array(t["translation"])
                    return scale, trans

            # Fallback: global coordinateTransformations
            if "coordinateTransformations" in multiscales:
                for t in multiscales["coordinateTransformations"]:
                    if t.get("type") == "scale":
                        scale = np.array(t["scale"])
                    if t.get("type") == "translation":
                        trans = np.array(t["translation"])
    except Exception as e:
        print(f"Warning reading metadata at {base_path}: {e}")

    return scale, trans


def extract_aligned_volumes(dataset_base: str, crop_id: str = "crop234", em_scale: str = "s0"):
    """
    Extract spatially aligned EM and label 3D volumes for a crop.

    Uses zarr metadata (scale/translation) to compute the physical bounding
    box of the label crop, then extracts the corresponding EM sub-volume.

    Parameters
    ----------
    dataset_base : str
        Path to the dataset.zarr root (e.g., `.../jrc_cos7-1a.zarr`).
    crop_id : str
        Crop identifier (e.g., "crop234").
    em_scale : str
        The scale to load EM data from (e.g. "s0", "s1"). Default "s0".

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (em_volume, label_volume) — aligned 3D arrays.
    """
    print(f"--- Extracting 3D Volumes for {crop_id} at EM scale {em_scale} ---")

    lbl_base = os.path.join(dataset_base, "recon-1", "labels", "groundtruth", crop_id, "all")
    em_base = os.path.join(dataset_base, "recon-1", "em", "fibsem-uint8")

    # 1. Get Scales and Translations
    scale_lbl, trans_lbl = get_scale_trans(lbl_base, "s0")
    scale_em, trans_em = get_scale_trans(em_base, em_scale)

    # 2. Open Zarr Arrays (metadata only, no RAM used yet)
    lbl_zarr = zarr.open(os.path.join(lbl_base, "s0"), mode="r")
    em_zarr = zarr.open(os.path.join(em_base, em_scale), mode="r")

    lbl_shape = np.array(lbl_zarr.shape)

    # 3. Calculate Spatial Bounding Box for the EM Sub-volume
    phys_min = trans_lbl
    phys_max = (lbl_shape * scale_lbl) + trans_lbl

    e_min = np.round((phys_min - trans_em) / scale_em).astype(int)
    e_max = np.round((phys_max - trans_em) / scale_em).astype(int)

    print(f"Label Crop Shape: {lbl_shape}")
    print(f"Target EM Bounding Box: Z[{e_min[0]}:{e_max[0]}] Y[{e_min[1]}:{e_max[1]}] X[{e_min[2]}:{e_max[2]}]")

    # 4. Extract Data into RAM
    lbl_volume = np.array(lbl_zarr[:])
    em_volume = np.array(em_zarr[e_min[0]:e_max[0], e_min[1]:e_max[1], e_min[2]:e_max[2]])

    # 5. Fix any 1-voxel rounding disparities
    min_z = min(lbl_volume.shape[0], em_volume.shape[0])
    min_y = min(lbl_volume.shape[1], em_volume.shape[1])
    min_x = min(lbl_volume.shape[2], em_volume.shape[2])

    lbl_volume = lbl_volume[:min_z, :min_y, :min_x]
    em_volume = em_volume[:min_z, :min_y, :min_x]

    print(f"Extraction Complete. Final Aligned Shape: {em_volume.shape}")

    return em_volume, lbl_volume
