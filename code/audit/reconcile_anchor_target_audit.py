#!/usr/bin/env python3
"""Read-only reconciliation of stored anchor records against training targets.

This script inspects existing split JSONs and CellMap Zarr labels. It does not
train models and does not modify training code, configs, data, checkpoints,
manuscript files, or stored splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.data.zarr_utils import extract_safe
from code.utils.config import CLASS_NAMES_14, LABEL_MAP_14CLS


RAW_TO_SEMANTIC_IDS = {
    "mito": [3, 4, 5],
    "ves": [8, 9],
    "endo": [10, 11],
    "lyso": [12, 13],
    "ld": [14, 15],
    "nuc": [20, 21, 24, 25, 26, 27, 28, 29],
    "np": [22, 23],
    "mt": [30, 36],
    "perox": [47, 48],
    "golgi": [6, 7],
    "er": [16, 17],
    "eres": [18, 19],
    "vim": [38],
}


def build_lookup() -> np.ndarray:
    lookup = np.zeros(256, dtype=np.int64)
    for raw_id, class_id in LABEL_MAP_14CLS.items():
        if raw_id < lookup.size:
            lookup[raw_id] = class_id
    return lookup


def zarr_roots(data_root: Path) -> dict[str, list[Path]]:
    roots = defaultdict(list)
    for path in data_root.glob("*/*.zarr"):
        roots[path.name.replace(".zarr", "")].append(path)
    return dict(roots)


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def label_path(roots, record, label_name: str) -> Path:
    candidates = []
    for zroot in roots.get(record["dataset"], []):
        candidates.append(
            zroot
            / "recon-1"
            / "labels"
            / "groundtruth"
            / record["crop"]
            / label_name
            / str(record["label_scale"])
        )
    path = first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            f"Could not find {label_name}/{record['label_scale']} for "
            f"{record['dataset']} {record['crop']}"
        )
    return path


def value_at(arr, coord) -> int | None:
    coord = np.array(coord, dtype=int)
    if np.any(coord < 0) or np.any(coord >= np.array(arr.shape)):
        return None
    return int(arr[tuple(coord.tolist())])


def expected_mapped_class(raw_category: str) -> int | None:
    mapped = {
        LABEL_MAP_14CLS[raw_id]
        for raw_id in RAW_TO_SEMANTIC_IDS.get(raw_category, [])
        if raw_id in LABEL_MAP_14CLS
    }
    if len(mapped) != 1:
        return None
    return mapped.pop()


def classify_location(raw_category: str, all_raw_value: int | None, all_mapped_value: int | None) -> str:
    if all_raw_value is None:
        return "out_of_bounds"
    expected = expected_mapped_class(raw_category)
    if all_mapped_value == expected:
        return "same_semantic_class_as_anchor"
    if all_mapped_value and all_mapped_value > 0:
        return "different_modeled_foreground_class"
    if all_raw_value > 0 and all_mapped_value == 0:
        return "ignored_unmapped_raw_value"
    return "background"


def training_start(record: dict, label_shape, patch_dim: int, jitter) -> np.ndarray:
    center = np.array(record["l_center"], dtype=float)
    base = np.floor(center - (patch_dim / 2.0)).astype(int)
    max_start = np.maximum(np.array(label_shape) - patch_dim, 0)
    return np.clip(base + np.array(jitter, dtype=int), 0, max_start).astype(int)


def weighted_record_probabilities(records: list[dict]) -> np.ndarray:
    counts = Counter(record["class"] for record in records)
    weights = np.array([1.0 / counts[record["class"]] for record in records], dtype=float)
    return weights / weights.sum()


def summarize(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out = []
    for value, group in sorted(groups.items()):
        total = len(group)
        counts = Counter(row["classification"] for row in group)
        out.append(
            {
                key: value,
                "records": total,
                "source_center_positive_pct": 100.0
                * sum(row["source_center_positive"] for row in group)
                / total,
                "source_l_center_positive_pct": 100.0
                * sum(row["source_l_center_positive"] for row in group)
                / total,
                "same_semantic_pct": 100.0
                * counts["same_semantic_class_as_anchor"]
                / total,
                "different_foreground_pct": 100.0
                * counts["different_modeled_foreground_class"]
                / total,
                "background_pct": 100.0 * counts["background"] / total,
                "ignored_unmapped_pct": 100.0
                * counts["ignored_unmapped_raw_value"]
                / total,
            }
        )
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--train-json",
        default="runs/latest-unet-14cls-jitter48-dice/splits/train.json",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--background-sample-draws", type=int, default=20000)
    parser.add_argument("--background-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--output-dir",
        default="analysis/results_snapshot/anchor_target_reconciliation",
    )
    args = parser.parse_args()

    records = json.loads((ROOT / args.train_json).read_text())
    if args.max_records is not None:
        records = records[: args.max_records]
    roots = zarr_roots(Path(args.data_root))
    lookup = build_lookup()

    all_cache = {}
    src_cache = {}
    rows = []
    mismatches = []
    for record in records:
        key = (record["dataset"], record["crop"], record["label_scale"])
        if key not in all_cache:
            all_cache[key] = zarr.open(label_path(roots, record, "all"), mode="r")
        src_key = (*key, record["class"])
        if src_key not in src_cache:
            src_cache[src_key] = zarr.open(label_path(roots, record, record["class"]), mode="r")

        all_arr = all_cache[key]
        src_arr = src_cache[src_key]
        source_center = [record["center_z"], record["center_y"], record["center_x"]]
        l_center = record["l_center"]
        source_center_value = value_at(src_arr, source_center)
        source_l_center_value = value_at(src_arr, l_center)
        all_raw_value = value_at(all_arr, l_center)
        all_mapped_value = int(lookup[all_raw_value]) if all_raw_value is not None and all_raw_value < lookup.size else None
        classification = classify_location(record["class"], all_raw_value, all_mapped_value)
        row = {
            "dataset": record["dataset"],
            "crop": record["crop"],
            "raw_category": record["class"],
            "l_center": json.dumps(l_center),
            "source_center": json.dumps(source_center),
            "source_center_value": source_center_value,
            "source_l_center_value": source_l_center_value,
            "source_center_positive": bool(source_center_value and source_center_value > 0),
            "source_l_center_positive": bool(source_l_center_value and source_l_center_value > 0),
            "all_raw_value": all_raw_value,
            "all_mapped_value": all_mapped_value,
            "all_mapped_class": CLASS_NAMES_14.get(all_mapped_value, "None"),
            "expected_mapped_value": expected_mapped_class(record["class"]),
            "expected_mapped_class": CLASS_NAMES_14.get(expected_mapped_class(record["class"]), "None"),
            "classification": classification,
        }
        rows.append(row)
        if classification != "same_semantic_class_as_anchor":
            mismatches.append(
                {
                    "dataset": row["dataset"],
                    "crop": row["crop"],
                    "raw_category": row["raw_category"],
                    "l_center": row["l_center"],
                    "source_mask_value": source_l_center_value,
                    "source_mask_value_at_source_center": source_center_value,
                    "all_raw_value": all_raw_value,
                    "all_mapped_value": all_mapped_value,
                    "explanation": classification,
                }
            )

    output_dir = ROOT / args.output_dir
    write_rows(output_dir / "anchor_target_records.csv", rows)
    write_rows(output_dir / "anchor_target_global.csv", summarize(rows, "dataset"))
    write_rows(output_dir / "anchor_target_by_raw_category.csv", summarize(rows, "raw_category"))
    write_rows(output_dir / "anchor_target_mismatch_examples.csv", mismatches[: max(args.background_examples, 20)])

    rng = np.random.default_rng(args.seed)
    probs = weighted_record_probabilities(records)
    bg_examples = []
    for _ in range(args.background_sample_draws):
        idx = int(rng.choice(len(records), p=probs))
        record = records[idx]
        key = (record["dataset"], record["crop"], record["label_scale"])
        all_arr = all_cache.get(key)
        if all_arr is None:
            all_arr = zarr.open(label_path(roots, record, "all"), mode="r")
            all_cache[key] = all_arr
        jitter = rng.integers(-48, 49, size=3)
        start = training_start(record, all_arr.shape, 128, jitter)
        raw_patch = extract_safe(all_arr, start, [128, 128, 128], pad_value=0, out_dtype=np.int64)
        mapped_patch = lookup[raw_patch]
        if not np.any(mapped_patch > 0):
            center_value = value_at(all_arr, record["l_center"])
            mapped_center = int(lookup[center_value]) if center_value is not None and center_value < lookup.size else None
            bg_examples.append(
                {
                    "dataset": record["dataset"],
                    "crop": record["crop"],
                    "raw_category": record["class"],
                    "l_center": json.dumps(record["l_center"]),
                    "jitter": json.dumps(jitter.tolist()),
                    "patch_start": json.dumps(start.tolist()),
                    "all_raw_value_at_l_center": center_value,
                    "all_mapped_value_at_l_center": mapped_center,
                    "seed_classification": classify_location(record["class"], center_value, mapped_center),
                    "explanation": "128^3 crop contains the stored coordinate, but the merged/remapped all-label patch contains no modeled foreground",
                }
            )
            if len(bg_examples) >= args.background_examples:
                break
    write_rows(output_dir / "background_only_128_examples.csv", bg_examples)

    total = len(rows)
    same = sum(row["classification"] == "same_semantic_class_as_anchor" for row in rows)
    print(f"records={total}")
    print(f"stored_seed_positive_in_training_target_pct={100.0 * same / total:.6f}")
    print(f"mismatching_records={len(mismatches)}")
    print(f"background_only_128_examples={len(bg_examples)}")
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
