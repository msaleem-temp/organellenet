"""
OrganelleNet — Paper Table Generator

Reads JSONL files from evaluate_detailed.py and produces CSV tables
for the paper:

1. per_class_comparison.csv     — Per-class IoU/Dice across all models
2. per_dataset_comparison.csv   — Per-dataset mean IoU across all models
3. per_resolution_band.csv      — Per-resolution-band mean IoU across models
4. resolution_x_class.csv       — Class × resolution-band IoU heatmap data
5. summary.csv                  — Overall mean metrics per model

Usage:
    python code/evaluation/analyze_results.py \
        --inputs \
            runs/static-unet-13cls-nojitter/results/detailed_metrics.jsonl \
            runs/latest-unet-14cls-jitter48-dice/results/detailed_metrics.jsonl \
            runs/latest-unet-14cls-em-aug/results/detailed_metrics.jsonl \
            runs/latest-unet-14cls-res-aug/results/detailed_metrics.jsonl \
        --labels "Static (No Aug)" "Latest (Jitter)" "Latest+EM Aug" "Latest+Res Aug" \
        --output-dir results/paper_tables/
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Resolution band definitions ─────────────────────────────────────────
# Based on actual CellMap resolution distribution:
#   Fine:   ≤ 4 nm  (cell lines at 2-4nm, fly at 4nm, brain at 4nm)
#   Medium: 5-8 nm  (cell lines at 8nm, liver at 8nm)
#   Coarse: ≥ 16 nm (kidney at 16-64nm, heart at 32-64nm, liver-zon at 32nm)

def get_resolution_band(resolution):
    """
    Classify a resolution into fine/medium/coarse bands.

    Parameters
    ----------
    resolution : list of float
        [z_nm, y_nm, x_nm] voxel size in nanometers.

    Returns
    -------
    str
        One of 'fine', 'medium', 'coarse'.
    """
    # Use the maximum dimension as the effective resolution
    max_res = max(resolution) if resolution else 0
    if max_res <= 4.0:
        return "fine"
    elif max_res <= 8.0:
        return "medium"
    else:
        return "coarse"


def load_jsonl(path):
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-ready CSV tables from detailed evaluation JSONL files."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to detailed_metrics.jsonl files (one per model)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Human-readable labels for each model (same order as --inputs)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write CSV tables to")
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.inputs) != len(args.labels):
        print("Error: number of --inputs must match number of --labels")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    all_results = {}
    for path, label in zip(args.inputs, args.labels):
        print(f"Loading {label} from {path}")
        all_results[label] = load_jsonl(path)
        print(f"  → {len(all_results[label])} patches")

    # ── Table 1: Per-Class Comparison ───────────────────────────────────
    print("\nGenerating per_class_comparison.csv ...")
    all_classes = set()
    for label, records in all_results.items():
        for r in records:
            all_classes.update(r.get("per_class_iou", {}).keys())

    # Remove Background from paper tables
    all_classes.discard("Background")
    sorted_classes = sorted(all_classes)

    out_path = os.path.join(args.output_dir, "per_class_comparison.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Class"]
        for label in args.labels:
            header.extend([f"{label} IoU", f"{label} Dice"])
        writer.writerow(header)

        for cls in sorted_classes:
            row = [cls]
            for label in args.labels:
                ious = [r["per_class_iou"].get(cls)
                        for r in all_results[label]
                        if cls in r.get("per_class_iou", {})]
                dices = [r["per_class_dice"].get(cls)
                         for r in all_results[label]
                         if cls in r.get("per_class_dice", {})]
                row.append(f"{np.mean(ious):.4f}" if ious else "N/A")
                row.append(f"{np.mean(dices):.4f}" if dices else "N/A")
            writer.writerow(row)

    print(f"  → {out_path}")

    # ── Table 2: Per-Dataset Comparison ─────────────────────────────────
    print("Generating per_dataset_comparison.csv ...")
    all_datasets = set()
    for label, records in all_results.items():
        for r in records:
            if r.get("dataset"):
                all_datasets.add(r["dataset"])
    sorted_datasets = sorted(all_datasets)

    out_path = os.path.join(args.output_dir, "per_dataset_comparison.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Dataset", "Resolution"]
        for label in args.labels:
            header.extend([f"{label} mIoU", f"{label} mDice"])
        writer.writerow(header)

        for ds in sorted_datasets:
            # Get representative resolution for this dataset
            rep_res = None
            for label, records in all_results.items():
                for r in records:
                    if r.get("dataset") == ds and r.get("resolution"):
                        rep_res = r["resolution"]
                        break
                if rep_res:
                    break

            row = [ds, str(rep_res) if rep_res else "N/A"]
            for label in args.labels:
                mious = [r["mean_iou"] for r in all_results[label]
                         if r.get("dataset") == ds]
                mdices = [r["mean_dice"] for r in all_results[label]
                          if r.get("dataset") == ds]
                row.append(f"{np.mean(mious):.4f}" if mious else "N/A")
                row.append(f"{np.mean(mdices):.4f}" if mdices else "N/A")
            writer.writerow(row)

    print(f"  → {out_path}")

    # ── Table 3: Per-Resolution-Band Comparison ─────────────────────────
    print("Generating per_resolution_band.csv ...")
    bands = ["fine", "medium", "coarse"]
    band_labels_map = {
        "fine": "Fine (≤4 nm)",
        "medium": "Medium (5-8 nm)",
        "coarse": "Coarse (≥16 nm)",
    }

    out_path = os.path.join(args.output_dir, "per_resolution_band.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Resolution Band", "N patches"]
        for label in args.labels:
            header.extend([f"{label} mIoU", f"{label} mDice"])
        writer.writerow(header)

        for band in bands:
            row = [band_labels_map[band]]

            # Count patches in this band (use first model as reference)
            first_label = args.labels[0]
            band_patches = [r for r in all_results[first_label]
                           if get_resolution_band(r.get("resolution", [])) == band]
            row.append(str(len(band_patches)))

            for label in args.labels:
                band_records = [r for r in all_results[label]
                               if get_resolution_band(r.get("resolution", [])) == band]
                mious = [r["mean_iou"] for r in band_records]
                mdices = [r["mean_dice"] for r in band_records]
                row.append(f"{np.mean(mious):.4f}" if mious else "N/A")
                row.append(f"{np.mean(mdices):.4f}" if mdices else "N/A")
            writer.writerow(row)

    print(f"  → {out_path}")

    # ── Table 4: Resolution × Class Heatmap ─────────────────────────────
    # Uses the LAST model (assumed to be the proposed method) for the heatmap
    print("Generating resolution_x_class.csv ...")
    primary_label = args.labels[-1]
    primary_records = all_results[primary_label]

    out_path = os.path.join(args.output_dir, "resolution_x_class.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Model: {primary_label}", ""] + sorted_classes)

        for band in bands:
            band_records = [r for r in primary_records
                           if get_resolution_band(r.get("resolution", [])) == band]
            row = [band_labels_map[band], str(len(band_records))]
            for cls in sorted_classes:
                ious = [r["per_class_iou"].get(cls)
                        for r in band_records
                        if cls in r.get("per_class_iou", {})]
                row.append(f"{np.mean(ious):.4f}" if ious else "N/A")
            writer.writerow(row)

    # Also generate a comparison version: delta between first and last model
    if len(args.labels) >= 2:
        baseline_label = args.labels[0]
        proposed_label = args.labels[-1]
        baseline_records = all_results[baseline_label]

        out_path2 = os.path.join(args.output_dir, "resolution_x_class_delta.csv")
        with open(out_path2, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [f"Δ IoU ({proposed_label} - {baseline_label})", ""]
                + sorted_classes
            )
            for band in bands:
                row = [band_labels_map[band], ""]
                for cls in sorted_classes:
                    # Baseline
                    base_ious = [r["per_class_iou"].get(cls)
                                 for r in baseline_records
                                 if (get_resolution_band(r.get("resolution", [])) == band
                                     and cls in r.get("per_class_iou", {}))]
                    # Proposed
                    prop_ious = [r["per_class_iou"].get(cls)
                                 for r in all_results[proposed_label]
                                 if (get_resolution_band(r.get("resolution", [])) == band
                                     and cls in r.get("per_class_iou", {}))]
                    if base_ious and prop_ious:
                        delta = np.mean(prop_ious) - np.mean(base_ious)
                        row.append(f"{delta:+.4f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)
        print(f"  → {out_path2}")

    print(f"  → {out_path}")

    # ── Table 5: Overall Summary ────────────────────────────────────────
    print("Generating summary.csv ...")
    out_path = os.path.join(args.output_dir, "summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "N patches", "Mean IoU", "Mean Dice",
                          "Fine IoU", "Medium IoU", "Coarse IoU",
                          "Fine-Coarse Gap"])

        for label in args.labels:
            records = all_results[label]
            all_mious = [r["mean_iou"] for r in records]
            all_mdices = [r["mean_dice"] for r in records]

            band_ious = {}
            for band in bands:
                band_records = [r for r in records
                               if get_resolution_band(r.get("resolution", [])) == band]
                band_ious[band] = np.mean([r["mean_iou"] for r in band_records]) if band_records else float("nan")

            gap = band_ious.get("fine", 0) - band_ious.get("coarse", 0)

            writer.writerow([
                label,
                len(records),
                f"{np.mean(all_mious):.4f}" if all_mious else "N/A",
                f"{np.mean(all_mdices):.4f}" if all_mdices else "N/A",
                f"{band_ious['fine']:.4f}" if not np.isnan(band_ious.get('fine', float('nan'))) else "N/A",
                f"{band_ious['medium']:.4f}" if not np.isnan(band_ious.get('medium', float('nan'))) else "N/A",
                f"{band_ious['coarse']:.4f}" if not np.isnan(band_ious.get('coarse', float('nan'))) else "N/A",
                f"{gap:.4f}" if not np.isnan(gap) else "N/A",
            ])

    print(f"  → {out_path}")
    print(f"\nAll tables written to {args.output_dir}/")


if __name__ == "__main__":
    main()
