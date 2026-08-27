"""
OrganelleNet — Paper Table Generator (Updated for Global IoU)

Reads JSONL files from evaluate_detailed.py and produces CSV tables
for the paper:

1. per_class_comparison.csv     — Per-class Global IoU/Dice across all models
2. per_dataset_comparison.csv   — Per-dataset Global mIoU across all models
3. per_resolution_band.csv      — Per-resolution-band Global mIoU across models
4. resolution_x_class.csv       — Class × resolution-band Global IoU heatmap data
5. summary.csv                  — Overall Global mean metrics per model
"""

import os
import sys
import json
import argparse
import csv
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def get_resolution_band(resolution):
    max_res = max(resolution) if resolution else 0
    if max_res <= 4.0:
        return "fine"
    elif max_res <= 8.0:
        return "medium"
    else:
        return "coarse"

def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def compute_global_class_metrics(records, cls):
    """Computes global metrics for a specific class across a set of records."""
    eps = 1e-8
    has_cls = any(cls in r.get("per_class_tp", {}) for r in records)
    if not has_cls:
        # Fallback to mean patch IoU if tp is missing (for older jsonls temporarily)
        if any(cls in r.get("per_class_iou", {}) for r in records):
            ious = [r["per_class_iou"].get(cls) for r in records if cls in r["per_class_iou"]]
            dices = [r["per_class_dice"].get(cls) for r in records if cls in r["per_class_dice"]]
            precs = [r["per_class_precision"].get(cls) for r in records if cls in r["per_class_precision"]]
            recs = [r["per_class_recall"].get(cls) for r in records if cls in r["per_class_recall"]]
            f1s = [r["per_class_f1"].get(cls) for r in records if cls in r["per_class_f1"]]
            return np.mean(ious), np.mean(dices), np.mean(precs), np.mean(recs), np.mean(f1s)
        return None, None, None, None, None
        
    tp = sum(r.get("per_class_tp", {}).get(cls, 0) for r in records if cls in r.get("per_class_tp", {}))
    fp = sum(r.get("per_class_fp", {}).get(cls, 0) for r in records if cls in r.get("per_class_fp", {}))
    fn = sum(r.get("per_class_fn", {}).get(cls, 0) for r in records if cls in r.get("per_class_fn", {}))
    
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / ((2 * tp) + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    
    return float(iou), float(dice), float(prec), float(rec), float(dice)

def compute_global_dataset_metrics(records, classes):
    """Computes the mean of the global per-class metrics (mIoU)."""
    ious, dices, precs, recs, f1s = [], [], [], [], []
    for cls in classes:
        iou, dice, prec, rec, f1 = compute_global_class_metrics(records, cls)
        if iou is not None:
            ious.append(iou)
            dices.append(dice)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            
    return (np.mean(ious) if ious else 0.0, 
            np.mean(dices) if dices else 0.0, 
            np.mean(precs) if precs else 0.0, 
            np.mean(recs) if recs else 0.0, 
            np.mean(f1s) if f1s else 0.0)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CSV tables")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    if len(args.inputs) != len(args.labels):
        print("Error: number of --inputs must match number of --labels")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for path, label in zip(args.inputs, args.labels):
        print(f"Loading {label} from {path}")
        all_results[label] = load_jsonl(path)
        print(f"  → {len(all_results[label])} patches")

    # Get all classes
    all_classes = set()
    for label, records in all_results.items():
        for r in records:
            all_classes.update(r.get("per_class_iou", {}).keys())

    all_classes.discard("Background")
    sorted_classes = sorted(all_classes)

    # Table 1: Per-Class Comparison
    print("\nGenerating per_class_comparison.csv ...")
    out_path = os.path.join(args.output_dir, "per_class_comparison.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Class"]
        for label in args.labels:
            header.extend([f"{label} IoU", f"{label} Dice", f"{label} HD95", f"{label} Precision", f"{label} Recall", f"{label} F1"])
        writer.writerow(header)

        for cls in sorted_classes:
            row = [cls]
            for label in args.labels:
                iou, dice, prec, rec, f1 = compute_global_class_metrics(all_results[label], cls)
                
                # HD95 remains an average since we can't reconstruct volume here easily
                hd95s = [r.get("per_class_hd95", {}).get(cls) for r in all_results[label] if cls in r.get("per_class_hd95", {})]
                mean_hd95 = np.mean(hd95s) if hd95s else None
                
                row.append(f"{iou:.4f}" if iou is not None else "N/A")
                row.append(f"{dice:.4f}" if dice is not None else "N/A")
                row.append(f"{mean_hd95:.4f}" if mean_hd95 is not None else "N/A")
                row.append(f"{prec:.4f}" if prec is not None else "N/A")
                row.append(f"{rec:.4f}" if rec is not None else "N/A")
                row.append(f"{f1:.4f}" if f1 is not None else "N/A")
            writer.writerow(row)

    # Table 2: Per-Dataset Comparison
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
            rep_res = None
            for label, records in all_results.items():
                for r in records:
                    if r.get("dataset") == ds and r.get("resolution"):
                        rep_res = r["resolution"]
                        break
                if rep_res: break

            row = [ds, str(rep_res) if rep_res else "N/A"]
            for label in args.labels:
                ds_records = [r for r in all_results[label] if r.get("dataset") == ds]
                if not ds_records:
                    row.extend(["N/A", "N/A"])
                    continue
                miou, mdice, _, _, _ = compute_global_dataset_metrics(ds_records, sorted_classes)
                row.append(f"{miou:.4f}")
                row.append(f"{mdice:.4f}")
            writer.writerow(row)

    # Table 3: Per-Resolution-Band Comparison
    print("Generating per_resolution_band.csv ...")
    bands = ["fine", "medium", "coarse"]
    band_labels_map = {"fine": "Fine (≤4 nm)", "medium": "Medium (5-8 nm)", "coarse": "Coarse (≥16 nm)"}

    out_path = os.path.join(args.output_dir, "per_resolution_band.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Resolution Band", "N patches"]
        for label in args.labels:
            header.extend([f"{label} mIoU", f"{label} mDice"])
        writer.writerow(header)

        for band in bands:
            row = [band_labels_map[band]]
            first_label = args.labels[0]
            band_patches = [r for r in all_results[first_label] if get_resolution_band(r.get("resolution", [])) == band]
            row.append(str(len(band_patches)))

            for label in args.labels:
                band_records = [r for r in all_results[label] if get_resolution_band(r.get("resolution", [])) == band]
                if not band_records:
                    row.extend(["N/A", "N/A"])
                    continue
                miou, mdice, _, _, _ = compute_global_dataset_metrics(band_records, sorted_classes)
                row.append(f"{miou:.4f}")
                row.append(f"{mdice:.4f}")
            writer.writerow(row)

    # Table 4: Resolution × Class Heatmap
    print("Generating resolution_x_class.csv ...")
    primary_label = args.labels[-1]
    primary_records = all_results[primary_label]

    out_path = os.path.join(args.output_dir, "resolution_x_class.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Model: {primary_label}", ""] + sorted_classes)

        for band in bands:
            band_records = [r for r in primary_records if get_resolution_band(r.get("resolution", [])) == band]
            row = [band_labels_map[band], str(len(band_records))]
            for cls in sorted_classes:
                iou, _, _, _, _ = compute_global_class_metrics(band_records, cls)
                row.append(f"{iou:.4f}" if iou is not None else "N/A")
            writer.writerow(row)

    # Table 4b: Resolution × Class Heatmap Delta (Missing from rewrite)
    if len(args.labels) >= 2:
        baseline_label = args.labels[0]
        proposed_label = args.labels[-1]
        baseline_records = all_results[baseline_label]
        proposed_records = all_results[proposed_label]

        out_path2 = os.path.join(args.output_dir, "resolution_x_class_delta.csv")
        with open(out_path2, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"Δ IoU ({proposed_label} - {baseline_label})", ""] + sorted_classes)
            
            for band in bands:
                row = [band_labels_map[band], ""]
                base_band_records = [r for r in baseline_records if get_resolution_band(r.get("resolution", [])) == band]
                prop_band_records = [r for r in proposed_records if get_resolution_band(r.get("resolution", [])) == band]
                
                for cls in sorted_classes:
                    base_iou, _, _, _, _ = compute_global_class_metrics(base_band_records, cls)
                    prop_iou, _, _, _, _ = compute_global_class_metrics(prop_band_records, cls)
                    
                    if base_iou is not None and prop_iou is not None:
                        delta = prop_iou - base_iou
                        row.append(f"{delta:+.4f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)
        print(f"  → {out_path2}")

    # Table 5: Overall Summary
    print("Generating summary.csv ...")
    out_path = os.path.join(args.output_dir, "summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "N patches", "Mean IoU", "Mean Dice", 
                          "Mean HD95", "Mean Precision", "Mean Recall", "Mean F1",
                          "Fine IoU", "Medium IoU", "Coarse IoU", "Fine-Coarse Gap"])

        for label in args.labels:
            records = all_results[label]
            miou, mdice, mprec, mrec, mf1 = compute_global_dataset_metrics(records, sorted_classes)
            
            # HD95 is still average of patches
            all_hd95 = [r.get("mean_hd95", 0) for r in records if r.get("mean_hd95", 0) > 0]

            band_ious = {}
            for band in bands:
                band_records = [r for r in records if get_resolution_band(r.get("resolution", [])) == band]
                b_iou, _, _, _, _ = compute_global_dataset_metrics(band_records, sorted_classes)
                band_ious[band] = b_iou if band_records else float("nan")

            gap = band_ious.get("fine", 0) - band_ious.get("coarse", 0)

            writer.writerow([
                label,
                len(records),
                f"{miou:.4f}" if len(records) > 0 else "N/A",
                f"{mdice:.4f}" if len(records) > 0 else "N/A",
                f"{np.mean(all_hd95):.4f}" if all_hd95 else "N/A",
                f"{mprec:.4f}" if len(records) > 0 else "N/A",
                f"{mrec:.4f}" if len(records) > 0 else "N/A",
                f"{mf1:.4f}" if len(records) > 0 else "N/A",
                f"{band_ious['fine']:.4f}" if not np.isnan(band_ious.get('fine', float('nan'))) else "N/A",
                f"{band_ious['medium']:.4f}" if not np.isnan(band_ious.get('medium', float('nan'))) else "N/A",
                f"{band_ious['coarse']:.4f}" if not np.isnan(band_ious.get('coarse', float('nan'))) else "N/A",
                f"{gap:.4f}" if not np.isnan(gap) else "N/A",
            ])

    print(f"  → {out_path}")
    print(f"\nAll tables written to {args.output_dir}/")

if __name__ == "__main__":
    main()
