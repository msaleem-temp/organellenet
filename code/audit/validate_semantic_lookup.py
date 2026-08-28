#!/usr/bin/env python3
"""Validate CellMap raw-ID to semantic-class lookup for retained training data.

This audit is read-only with respect to data, splits, checkpoints, and training
code. It writes only analysis artifacts under the requested output directory.
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

from code.utils.config import CLASS_NAMES_13, CLASS_NAMES_14, LABEL_MAP_14CLS


AUTHORITATIVE_PARENT_OUTPUT = {
    50: 1,  # mito
    44: 5,  # ld
    37: 6,  # nuc
    49: 9,  # perox
}

RAW_CATEGORY_TO_OUTPUT = {
    "mito": 1,
    "ves": 2,
    "endo": 3,
    "lyso": 4,
    "ld": 5,
    "nuc": 6,
    "np": 7,
    "mt": 8,
    "perox": 9,
    "golgi": 10,
    "er": 11,
    "eres": 12,
    "vim": 13,
}


def read_classes_csv(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                raw_id = int(row[1])
            except ValueError:
                continue
            children = []
            if len(row) > 2 and row[2].strip():
                children = [int(v) for v in row[2].split(",") if v.strip()]
            rows[raw_id] = {"label_name": row[0], "children": children}
    return rows


def build_parent_name_by_raw_id(classes: dict[int, dict]) -> dict[int, str]:
    parent = {}
    for raw_id, row in classes.items():
        parent.setdefault(raw_id, row["label_name"])
        for child_id in row["children"]:
            parent[child_id] = row["label_name"]
    return parent


def build_expected_output_by_raw_id(classes: dict[int, dict]) -> dict[int, int]:
    expected = dict(LABEL_MAP_14CLS)
    expected.update(AUTHORITATIVE_PARENT_OUTPUT)
    for aggregate_id, output_id in AUTHORITATIVE_PARENT_OUTPUT.items():
        for child_id in classes.get(aggregate_id, {}).get("children", []):
            expected[child_id] = output_id
    return expected


def build_lookup() -> np.ndarray:
    lookup = np.zeros(1024, dtype=np.int64)
    for raw_id, output_id in LABEL_MAP_14CLS.items():
        if raw_id >= lookup.size:
            new = np.zeros(raw_id + 1, dtype=np.int64)
            new[: lookup.size] = lookup
            lookup = new
        lookup[raw_id] = output_id
    return lookup


def lookup_value(lookup: np.ndarray, raw_id: int) -> int:
    if raw_id < 0 or raw_id >= lookup.size:
        return 0
    return int(lookup[raw_id])


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


def unique_values_by_z(arr, z_step: int) -> Counter:
    counts = Counter()
    for z0 in range(0, arr.shape[0], z_step):
        z1 = min(z0 + z_step, arr.shape[0])
        block = np.asarray(arr[z0:z1, :, :])
        vals, block_counts = np.unique(block, return_counts=True)
        counts.update({int(v): int(c) for v, c in zip(vals, block_counts)})
    return counts


def summarize_seed_rows(rows: list[dict], group_key: str | None = None) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key] if group_key else "all"].append(row)
    out = []
    for key, group in sorted(groups.items()):
        total = len(group)
        modeled = sum(row["mapped_output"] > 0 for row in group)
        same = sum(row["mapped_output"] == row["expected_output"] for row in group)
        out.append(
            {
                (group_key or "group"): key,
                "records": total,
                "modeled_foreground_seed_pct": f"{100.0 * modeled / total:.6f}",
                "same_semantic_seed_pct": f"{100.0 * same / total:.6f}",
                "background_after_lookup_pct": f"{100.0 * (total - modeled) / total:.6f}",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--classes-csv", required=True)
    parser.add_argument(
        "--train-json",
        default="runs/latest-unet-14cls-jitter48-dice/splits/train.json",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/results_snapshot/corrected_label_lookup_audit",
    )
    parser.add_argument("--z-step", type=int, default=16)
    parser.add_argument(
        "--skip-volume-scan",
        action="store_true",
        help="Only audit seed coordinates; do not enumerate full all-label volumes.",
    )
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    records = json.loads((ROOT / args.train_json).read_text())
    classes = read_classes_csv(Path(args.classes_csv))
    parent_name = build_parent_name_by_raw_id(classes)
    expected_output = build_expected_output_by_raw_id(classes)
    lookup = build_lookup()
    roots = zarr_roots(Path(args.data_root))

    cache = {}
    seed_rows = []
    for record in records:
        key = (record["dataset"], record["crop"], record["label_scale"])
        if key not in cache:
            cache[key] = zarr.open(label_path(roots, record, "all"), mode="r")
        raw_id = value_at(cache[key], record["l_center"])
        mapped = lookup_value(lookup, raw_id) if raw_id is not None else 0
        exp = RAW_CATEGORY_TO_OUTPUT.get(record["class"], 0)
        seed_rows.append(
            {
                "dataset": record["dataset"],
                "crop": record["crop"],
                "raw_category": record["class"],
                "l_center": json.dumps(record["l_center"]),
                "all_raw_id_at_seed": raw_id,
                "classes_csv_name": classes.get(raw_id, {}).get("label_name", ""),
                "biological_parent": parent_name.get(raw_id, ""),
                "mapped_output": mapped,
                "mapped_class": CLASS_NAMES_14.get(mapped, "Background"),
                "expected_output": exp,
                "expected_class": CLASS_NAMES_14.get(exp, ""),
                "status": "same_semantic" if mapped == exp else ("modeled_foreground" if mapped > 0 else "background"),
            }
        )

    write_csv(output_dir / "seed_coordinate_lookup_records.csv", seed_rows)
    write_csv(output_dir / "seed_coordinate_lookup_global.csv", summarize_seed_rows(seed_rows))
    write_csv(output_dir / "seed_coordinate_lookup_by_raw_category.csv", summarize_seed_rows(seed_rows, "raw_category"))

    volume_rows = []
    remaining_unmapped_rows = []
    if not args.skip_volume_scan:
        raw_counts = Counter()
        scanned = sorted(cache.items())
        for (_, _, _), arr in scanned:
            raw_counts.update(unique_values_by_z(arr, args.z_step))
        for raw_id, count in sorted(raw_counts.items()):
            if raw_id == 0:
                continue
            mapped = lookup_value(lookup, raw_id)
            exp = expected_output.get(raw_id)
            row = {
                "raw_id": raw_id,
                "voxel_count": count,
                "classes_csv_name": classes.get(raw_id, {}).get("label_name", ""),
                "current_LABEL_MAP_14CLS_output": mapped,
                "mapped_or_unmapped": "mapped" if mapped > 0 else "unmapped",
                "current_mapped_class": CLASS_NAMES_14.get(mapped, "Background"),
                "expected_modeled_output_if_known": exp if exp is not None else "",
                "expected_modeled_class_if_known": CLASS_NAMES_14.get(exp, "") if exp is not None else "",
                "biological_parent_class": parent_name.get(raw_id, ""),
            }
            volume_rows.append(row)
            if mapped == 0:
                remaining_unmapped_rows.append(row)
        write_csv(output_dir / "merged_all_raw_id_vocabulary.csv", volume_rows)
        write_csv(output_dir / "remaining_nonzero_raw_ids_mapped_to_background.csv", remaining_unmapped_rows)

    summary = [
        "# Corrected Label Lookup Audit",
        "",
        f"training_records: {len(records)}",
        f"classes_csv: {Path(args.classes_csv)}",
        f"train_json: {ROOT / args.train_json}",
        "",
        "## Required Aggregate IDs",
        "",
        "raw_id,label_name,expected_output,expected_class,current_output,current_class",
    ]
    for raw_id in [37, 44, 49, 50]:
        out = lookup_value(lookup, raw_id)
        exp = AUTHORITATIVE_PARENT_OUTPUT[raw_id]
        summary.append(
            f"{raw_id},{classes.get(raw_id, {}).get('label_name', '')},{exp},"
            f"{CLASS_NAMES_14[exp]},{out},{CLASS_NAMES_14.get(out, 'Background')}"
        )
    summary.extend(["", "## Seed Summary"])
    for row in summarize_seed_rows(seed_rows):
        summary.append(",".join(str(v) for v in row.values()))
    summary.extend(["", "## Remaining Nonzero Raw IDs Mapped To Background"])
    if remaining_unmapped_rows:
        for row in remaining_unmapped_rows:
            summary.append(
                f"{row['raw_id']},{row['classes_csv_name']},{row['biological_parent_class']},"
                f"voxels={row['voxel_count']}"
            )
    elif args.skip_volume_scan:
        summary.append("not scanned; rerun without --skip-volume-scan")
    else:
        summary.append("none")
    (output_dir / "semantic_lookup_audit.md").write_text("\n".join(summary) + "\n")

    print(f"wrote={output_dir}")
    print((output_dir / "semantic_lookup_audit.md").read_text())


if __name__ == "__main__":
    main()
