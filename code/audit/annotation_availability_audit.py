#!/usr/bin/env python3
"""Read-only audit of CellMap annotation availability and ROI support.

This script inspects existing Zarr annotation arrays and evaluation JSONL files.
It does not train models and does not modify data, checkpoints, configs, or
manuscript files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.config import CLASS_NAMES_14, LABEL_MAP_14CLS


SEMANTIC_CLASS_RAW_IDS = {
    "Mitochondria": [3, 4, 5],
    "Vesicles": [8, 9],
    "Endosomes": [10, 11],
    "Lysosomes": [12, 13],
    "Lipid Droplets": [14, 15],
    "Nucleus": [20, 21, 24, 25, 26, 27, 28, 29],
    "Nuclear Pores": [22, 23],
    "Microtubules": [30, 36],
    "Peroxisomes": [47, 48],
    "Golgi": [6, 7],
    "ER": [16, 17],
    "ERES": [18, 19],
    "Vimentin": [38],
}

SEMANTIC_CLASS_ARRAYS = {
    "Mitochondria": ["mito", "mito_lum", "mito_mem", "mito_ribo"],
    "Vesicles": ["ves", "ves_lum", "ves_mem"],
    "Endosomes": ["endo", "endo_lum", "endo_mem"],
    "Lysosomes": ["lyso", "lyso_lum", "lyso_mem"],
    "Lipid Droplets": ["ld", "ld_lum", "ld_mem"],
    "Nucleus": [
        "nuc",
        "nucpl",
        "chrom",
        "euchrom",
        "hchrom",
        "nechrom",
        "nhchrom",
        "ne_lum",
        "ne_mem",
    ],
    "Nuclear Pores": ["np", "np_in", "np_out"],
    "Microtubules": ["mt", "mt_in", "mt_out"],
    "Peroxisomes": ["perox", "perox_lum", "perox_mem"],
    "Golgi": ["golgi", "golgi_lum", "golgi_mem"],
    "ER": ["er", "er_lum", "er_mem"],
    "ERES": ["eres", "eres_lum", "eres_mem"],
    "Vimentin": ["vim"],
}


def build_lookup() -> np.ndarray:
    lookup = np.zeros(256, dtype=np.int64)
    for raw_id, class_id in LABEL_MAP_14CLS.items():
        if raw_id < lookup.size:
            lookup[raw_id] = class_id
    return lookup


def zarr_root(data_root: Path, dataset: str) -> Path:
    candidates = sorted(data_root.glob(f"*/{dataset}.zarr"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {dataset}.zarr below {data_root}")
    return candidates[0]


def first_existing_scale(base: Path) -> tuple[str | None, Path | None]:
    for scale in [f"s{i}" for i in range(7)]:
        path = base / scale
        if path.exists():
            return scale, path
    return None, None


def nonzero_count(path: Path) -> int | None:
    if path is None or not path.exists():
        return None
    arr = zarr.open(path, mode="r")
    return int(np.count_nonzero(arr[:]))


def crop_slices_from_record(record: dict, score_roi_dim: int) -> tuple[slice, slice, slice]:
    center = np.array(record["l_center"], dtype=float)
    start = np.floor(center - (score_roi_dim / 2.0)).astype(int)
    return tuple(slice(int(s), int(s + score_roi_dim)) for s in start)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def gt_positive_from_metrics(records: list[dict], class_name: str) -> bool:
    return any(
        int(record.get("per_class_tp", {}).get(class_name, 0))
        + int(record.get("per_class_fn", {}).get(class_name, 0))
        > 0
        for record in records
    )


def roi_positive_from_all(
    all_label_path: Path, test_records: list[dict], class_name: str, score_roi_dim: int
) -> tuple[bool, int]:
    arr = zarr.open(all_label_path, mode="r")
    lookup = build_lookup()
    class_id = {v: k for k, v in CLASS_NAMES_14.items()}[class_name]
    total = 0
    for record in test_records:
        patch = arr[crop_slices_from_record(record, score_roi_dim)]
        total += int(np.count_nonzero(lookup[patch] == class_id))
    return total > 0, total


def interpretation(available: bool, full_crop_positive: bool, roi_positive: bool) -> str:
    if not available:
        return "not annotated/available for crop"
    if roi_positive:
        return "positive in evaluated ROIs"
    if full_crop_positive:
        return "annotation available elsewhere in crop but not in evaluated ROIs"
    return "annotation available; zero positive support observed in crop"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="jrc_cos7-1a")
    parser.add_argument("--crop", default="crop234")
    parser.add_argument(
        "--test-json",
        default="runs/latest-unet-14cls-jitter48-p64/splits/test.json",
    )
    parser.add_argument(
        "--metrics-jsonl",
        default="runs/latest-unet-14cls-jitter48-p64/results/detailed_metrics_matched_roi.jsonl",
    )
    parser.add_argument("--score-roi-dim", type=int, default=64)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    root = zarr_root(Path(args.data_root), args.dataset)
    gt_root = root / "recon-1" / "labels" / "groundtruth" / args.crop
    all_scale, all_path = first_existing_scale(gt_root / "all")
    if all_path is None:
        raise FileNotFoundError(f"No all-label Zarr scale found under {gt_root / 'all'}")

    test_records = json.loads((ROOT / args.test_json).read_text())
    metrics_path = ROOT / args.metrics_jsonl
    metrics_records = load_jsonl(metrics_path) if metrics_path.exists() else []

    rows = []
    for semantic_class, array_names in SEMANTIC_CLASS_ARRAYS.items():
        available_arrays = []
        array_positive_counts = {}
        for array_name in array_names:
            scale, path = first_existing_scale(gt_root / array_name)
            if path is not None:
                available_arrays.append(f"{array_name}/{scale}")
                array_positive_counts[array_name] = nonzero_count(path)

        available = bool(available_arrays)
        full_crop_positive = any((count or 0) > 0 for count in array_positive_counts.values())
        if metrics_records:
            positive_in_roi = gt_positive_from_metrics(metrics_records, semantic_class)
            roi_positive_voxels = ""
        else:
            positive_in_roi, roi_positive_voxels = roi_positive_from_all(
                all_path, test_records, semantic_class, args.score_roi_dim
            )

        valid_negative = (
            "yes" if available and not positive_in_roi else "no"
        )
        rows.append(
            {
                "semantic_class": semantic_class,
                "annotation_available_in_crop": "yes" if available else "no",
                "positive_in_73_rois": "yes" if positive_in_roi else "no",
                "valid_negative_interpretation": valid_negative,
                "interpretation": interpretation(available, full_crop_positive, positive_in_roi),
                "available_arrays": ";".join(available_arrays),
                "full_crop_positive_voxels_by_array": json.dumps(array_positive_counts, sort_keys=True),
                "roi_positive_voxels_from_all_if_computed": roi_positive_voxels,
                "all_label_scale": all_scale,
            }
        )

    fieldnames = list(rows[0].keys())
    if args.output == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
