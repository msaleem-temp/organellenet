#!/usr/bin/env python3
"""Read-only audit of realized OrganelleNet training patches.

Run from the repository root on the server with CellMap Zarr data available.
This script does not train models and does not modify training code, configs,
data, checkpoints, or manuscript files. It writes CSV summaries under
analysis/results_snapshot/patch_sampling_audit by default.

The crop geometry mirrors code/data/dataset.py::PatchDataset.__getitem__ and
uses code/data/zarr_utils.py::extract_safe for label extraction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import os
import sys

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.data.zarr_utils import extract_safe
from code.utils.config import LABEL_MAP_14CLS, CLASS_NAMES_14


SUFFIXES = {"lum", "mem", "in", "out"}
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


def organelle_class_name(class_name: str) -> str:
    if "_" in class_name:
        base, suffix = class_name.rsplit("_", 1)
        if suffix in SUFFIXES:
            return base
    return class_name


def build_zarr_map(data_root: Path) -> dict[str, list[Path]]:
    zarr_map: dict[str, list[Path]] = defaultdict(list)
    for path in data_root.glob("*/*.zarr"):
        zarr_map[path.name.replace(".zarr", "")].append(path)
    return dict(zarr_map)


def find_label_path(zarr_map: dict[str, list[Path]], record: dict) -> Path:
    for zarr_path in zarr_map.get(record["dataset"], []):
        label_path = (
            zarr_path
            / "recon-1"
            / "labels"
            / "groundtruth"
            / record["crop"]
            / "all"
            / str(record["label_scale"])
        )
        if label_path.exists():
            return label_path
    raise FileNotFoundError(
        f"Could not find label zarr for {record['dataset']} {record['crop']} "
        f"scale {record['label_scale']}"
    )


def raw_ids_for(raw_class: str) -> list[int]:
    return RAW_TO_SEMANTIC_IDS.get(organelle_class_name(raw_class), [])


def mapped_class_for(raw_class: str) -> int | None:
    mapped = sorted(
        {
            LABEL_MAP_14CLS[semantic_id]
            for semantic_id in raw_ids_for(raw_class)
            if semantic_id in LABEL_MAP_14CLS
        }
    )
    if len(mapped) != 1:
        return None
    return mapped[0]


def label_lookup() -> np.ndarray:
    lookup = np.zeros(256, dtype=np.int64)
    for semantic_id, class_id in LABEL_MAP_14CLS.items():
        if semantic_id < len(lookup):
            lookup[semantic_id] = class_id
    return lookup


def training_window(record: dict, label_shape, patch_dim: int, jitter) -> dict:
    """Mirror PatchDataset.__getitem__ lines 140-178 for label coordinates."""
    l_center = np.array(record["l_center"], dtype=float)
    base_l_start = np.floor(l_center - (patch_dim / 2.0)).astype(int)
    max_l_start = np.maximum(np.array(label_shape) - patch_dim, 0)
    requested_l_start = base_l_start + np.array(jitter, dtype=int)
    clamped_l_start = np.clip(requested_l_start, 0, max_l_start).astype(int)
    return {
        "base_l_start": base_l_start,
        "requested_l_start": requested_l_start,
        "clamped_l_start": clamped_l_start,
        "effective_jitter": clamped_l_start - base_l_start,
        "boundary_clamped": bool(np.any(clamped_l_start != requested_l_start)),
    }


def contains_anchor(record: dict, start, patch_dim: int) -> bool:
    anchor = np.array(record["l_center"], dtype=int)
    start = np.array(start, dtype=int)
    return bool(np.all(anchor >= start) and np.all(anchor < start + patch_dim))


def weighted_record_probabilities(records: list[dict]) -> np.ndarray:
    raw_classes = [record.get("class", "unknown") for record in records]
    counts = Counter(raw_classes)
    weights = np.array([1.0 / counts[class_name] for class_name in raw_classes])
    return weights / weights.sum()


def weighted_quantile(values, weights, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / weights.sum()
    cdf = np.cumsum(weights)
    return float(values[np.searchsorted(cdf, quantile, side="left")])


def summarize(rows: list[dict]) -> dict:
    weights = np.ones(len(rows), dtype=float) / len(rows)

    def mean(key: str) -> float:
        return float(np.sum(weights * np.array([row[key] for row in rows], dtype=float)))

    foreground = np.array([row["foreground_fraction"] for row in rows], dtype=float)
    distinct = np.array([row["distinct_mapped_fg_classes"] for row in rows], dtype=float)
    return {
        "draws": len(rows),
        "anchor_voxel_pct": 100.0 * mean("contains_anchor_voxel"),
        "raw_anchor_category_pct": 100.0 * mean("contains_raw_anchor_category"),
        "mapped_anchor_class_pct": 100.0 * mean("contains_mapped_anchor_class"),
        "any_foreground_pct": 100.0 * mean("contains_any_foreground"),
        "all_background_pct": 100.0 * (1.0 - mean("contains_any_foreground")),
        "foreground_fraction_mean": mean("foreground_fraction"),
        "foreground_fraction_median": weighted_quantile(foreground, weights, 0.50),
        "foreground_fraction_p25": weighted_quantile(foreground, weights, 0.25),
        "foreground_fraction_p75": weighted_quantile(foreground, weights, 0.75),
        "distinct_mapped_fg_classes_mean": mean("distinct_mapped_fg_classes"),
        "distinct_mapped_fg_classes_median": weighted_quantile(distinct, weights, 0.50),
        "boundary_clamped_pct": 100.0 * mean("boundary_clamped"),
    }


def summarize_geometric(records: list[dict], zarr_map, patch_dim: int, draws_per_record: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    label_cache = {}
    rows = []
    for record in records:
        key = (record["dataset"], record["crop"], record["label_scale"])
        if key not in label_cache:
            label_cache[key] = zarr.open(find_label_path(zarr_map, record), mode="r")
        label_shape = label_cache[key].shape
        for _ in range(draws_per_record):
            jitter = rng.integers(-48, 49, size=3)
            win = training_window(record, label_shape, patch_dim, jitter)
            rows.append(
                {
                    "contains_anchor_voxel": contains_anchor(record, win["clamped_l_start"], patch_dim),
                    "boundary_clamped": win["boundary_clamped"],
                }
            )
    weights = np.ones(len(rows), dtype=float) / len(rows)
    return {
        "draws": len(rows),
        "anchor_voxel_pct": 100.0 * float(np.sum(weights * [r["contains_anchor_voxel"] for r in rows])),
        "boundary_clamped_pct": 100.0 * float(np.sum(weights * [r["boundary_clamped"] for r in rows])),
    }


def audit_draws(records, zarr_map, patch_dim: int, mode: str, draws: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    record_probs = weighted_record_probabilities(records)
    if mode == "uniform":
        record_indices = rng.integers(0, len(records), size=draws)
    elif mode == "weighted":
        record_indices = rng.choice(len(records), size=draws, replace=True, p=record_probs)
    else:
        raise ValueError(mode)

    lookup = label_lookup()
    label_cache = {}
    rows = []
    for record_idx in record_indices:
        record = records[int(record_idx)]
        key = (record["dataset"], record["crop"], record["label_scale"])
        if key not in label_cache:
            label_cache[key] = zarr.open(find_label_path(zarr_map, record), mode="r")
        label_zarr = label_cache[key]

        jitter = rng.integers(-48, 49, size=3)
        win = training_window(record, label_zarr.shape, patch_dim, jitter)
        raw_patch = extract_safe(
            label_zarr,
            win["clamped_l_start"],
            [patch_dim, patch_dim, patch_dim],
            pad_value=0,
            out_dtype=np.int64,
        )
        mapped_patch = lookup[raw_patch]
        raw_ids = raw_ids_for(record["class"])
        mapped_class_id = mapped_class_for(record["class"])
        foreground = mapped_patch > 0
        distinct_classes = sorted(set(np.unique(mapped_patch).tolist()) - {0})

        rows.append(
            {
                "raw_anchor_category": record["class"],
                "mapped_anchor_class": CLASS_NAMES_14.get(mapped_class_id, "unknown"),
                "contains_anchor_voxel": contains_anchor(record, win["clamped_l_start"], patch_dim),
                "contains_raw_anchor_category": bool(np.isin(raw_patch, raw_ids).any()) if raw_ids else False,
                "contains_mapped_anchor_class": bool((mapped_patch == mapped_class_id).any())
                if mapped_class_id is not None
                else False,
                "contains_any_foreground": bool(foreground.any()),
                "foreground_fraction": float(foreground.mean()),
                "distinct_mapped_fg_classes": len(distinct_classes),
                "boundary_clamped": win["boundary_clamped"],
            }
        )
    return rows


def write_one_row_csv(path: Path, row: dict) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def breakdown(rows: list[dict], key: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    output = []
    for value, group in sorted(grouped.items()):
        summary = summarize(group)
        output.append(
            {
                key: value,
                "draws": len(group),
                "anchor_voxel_pct": summary["anchor_voxel_pct"],
                "raw_anchor_category_pct": summary["raw_anchor_category_pct"],
                "mapped_anchor_class_pct": summary["mapped_anchor_class_pct"],
                "any_foreground_pct": summary["any_foreground_pct"],
                "boundary_clamped_pct": summary["boundary_clamped_pct"],
            }
        )
    return output


def record_count_table(records: list[dict]) -> list[dict]:
    counts = Counter((record["class"], CLASS_NAMES_14.get(mapped_class_for(record["class"]), "unknown")) for record in records)
    return [
        {"raw_anchor_category": raw, "mapped_anchor_class": mapped, "records": count}
        for (raw, mapped), count in sorted(counts.items())
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only realized patch audit.")
    parser.add_argument("--train-json", default="runs/latest-unet-14cls-jitter48-p64/splits/train.json")
    parser.add_argument("--data-root", default="/mnt/graid/codebases_other/cellmap-segmentation-challenge/data")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--geometric-draws-per-record", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-dir", default="analysis/results_snapshot/patch_sampling_audit")
    return parser.parse_args()


def main():
    args = parse_args()
    records = json.loads((ROOT / args.train_json).read_text())
    zarr_map = build_zarr_map(Path(args.data_root))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_rows(output_dir / "training_record_counts.csv", record_count_table(records))

    all_summaries = []
    for patch_dim in (64, 128):
        geom = summarize_geometric(
            records,
            zarr_map,
            patch_dim=patch_dim,
            draws_per_record=args.geometric_draws_per_record,
            seed=args.seed + patch_dim,
        )
        write_one_row_csv(
            output_dir / f"geometric_anchor_boundary_p{patch_dim}.csv",
            {"patch_dim": patch_dim, **geom},
        )

        for mode in ("uniform", "weighted"):
            rows = audit_draws(
                records,
                zarr_map,
                patch_dim=patch_dim,
                mode=mode,
                draws=args.draws,
                seed=args.seed + patch_dim + (0 if mode == "uniform" else 1000),
            )
            summary = {"patch_dim": patch_dim, "record_sampling": mode, **summarize(rows)}
            all_summaries.append(summary)
            write_one_row_csv(output_dir / f"summary_p{patch_dim}_{mode}.csv", summary)
            write_rows(
                output_dir / f"per_raw_retention_p{patch_dim}_{mode}.csv",
                breakdown(rows, "raw_anchor_category"),
            )
            write_rows(
                output_dir / f"per_mapped_retention_p{patch_dim}_{mode}.csv",
                breakdown(rows, "mapped_anchor_class"),
            )
            print(summary)

    write_rows(output_dir / "summary_all.csv", all_summaries)
    print(f"Wrote audit CSVs to {output_dir}")


if __name__ == "__main__":
    main()
