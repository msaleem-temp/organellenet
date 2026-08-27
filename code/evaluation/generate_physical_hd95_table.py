"""
Generate physical-unit HD95 tables from detailed evaluation JSONL files.

The existing detailed JSONL stores HD95 in voxel units. This script converts
patch-level HD95 to nanometers using the largest physical voxel spacing in each
record's resolution tuple, then averages by model and class.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def spacing_nm(record):
    resolution = record.get("resolution") or []
    if not resolution:
        return None
    return float(max(resolution))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate physical HD95 tables")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.inputs) != len(args.labels):
        raise SystemExit("number of --inputs must match number of --labels")

    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []
    summary_rows = []

    for path, label in zip(args.inputs, args.labels):
        records = load_jsonl(path)
        by_class = {}
        all_values = []

        for record in records:
            scale = spacing_nm(record)
            if scale is None:
                continue
            for cls, hd95_vox in record.get("per_class_hd95", {}).items():
                hd95_nm = float(hd95_vox) * scale
                by_class.setdefault(cls, []).append(hd95_nm)
                all_values.append(hd95_nm)

        for cls in sorted(by_class):
            values = by_class[cls]
            all_rows.append([
                label,
                cls,
                len(values),
                f"{np.mean(values):.4f}",
                f"{np.median(values):.4f}",
                f"{np.std(values):.4f}",
            ])

        summary_rows.append([
            label,
            len(records),
            len(all_values),
            f"{np.mean(all_values):.4f}" if all_values else "N/A",
            f"{np.median(all_values):.4f}" if all_values else "N/A",
        ])

    per_class_path = os.path.join(args.output_dir, "physical_hd95_per_class.csv")
    with open(per_class_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Class", "N finite patches", "Mean HD95 nm", "Median HD95 nm", "Std HD95 nm"])
        writer.writerows(all_rows)

    summary_path = os.path.join(args.output_dir, "physical_hd95_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "N patches", "N finite class-patch HD95", "Mean HD95 nm", "Median HD95 nm"])
        writer.writerows(summary_rows)

    print(f"Wrote {per_class_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
