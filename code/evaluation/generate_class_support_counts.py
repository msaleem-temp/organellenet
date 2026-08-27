"""
Generate class-support counts from detailed evaluation JSONL records.

The table answers a simple paper-review question: which classes are actually
present in the held-out evaluation, and how much ground-truth support do they
have? This helps separate absent-class limitations from model failures.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.utils.config import load_config


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate held-out class support counts.")
    parser.add_argument("--input", required=True, help="Detailed metrics JSONL.")
    parser.add_argument("--config", required=True, help="Config used to define class names.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    records = load_records(args.input)
    config = load_config(args.config)
    class_names = {
        name: idx
        for idx, name in config.class_names.items()
        if idx != 0
    }

    support = {
        name: {
            "class_id": idx,
            "gt_patches": 0,
            "gt_voxels": 0,
            "pred_patches": 0,
            "pred_voxels": 0,
        }
        for name, idx in class_names.items()
    }

    datasets = defaultdict(int)
    crops = defaultdict(int)
    resolutions = defaultdict(int)

    for record in records:
        datasets[record.get("dataset", "unknown")] += 1
        crops[record.get("crop", "unknown")] += 1
        resolutions[tuple(record.get("resolution") or [])] += 1

        tp = record.get("per_class_tp", {})
        fp = record.get("per_class_fp", {})
        fn = record.get("per_class_fn", {})

        for name in support:
            gt_voxels = int(tp.get(name, 0)) + int(fn.get(name, 0))
            pred_voxels = int(tp.get(name, 0)) + int(fp.get(name, 0))

            support[name]["gt_voxels"] += gt_voxels
            support[name]["pred_voxels"] += pred_voxels
            if gt_voxels > 0:
                support[name]["gt_patches"] += 1
            if pred_voxels > 0:
                support[name]["pred_patches"] += 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class ID",
            "Class",
            "GT patches",
            "GT voxels",
            "Predicted patches",
            "Predicted voxels",
            "Support status",
        ])
        for name, row in sorted(support.items(), key=lambda item: item[1]["class_id"]):
            status = "present" if row["gt_voxels"] > 0 else "absent_in_ground_truth"
            writer.writerow([
                row["class_id"],
                name,
                row["gt_patches"],
                row["gt_voxels"],
                row["pred_patches"],
                row["pred_voxels"],
                status,
            ])

    print(f"Wrote {args.output}")
    print(f"Records: {len(records)}")
    print(f"Datasets: {dict(datasets)}")
    print(f"Crops: {dict(crops)}")
    print(f"Resolutions: {dict(resolutions)}")


if __name__ == "__main__":
    main()
