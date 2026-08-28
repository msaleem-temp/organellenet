#!/usr/bin/env python3
"""Summarize corrected-label NewInML matched-ROI runs.

Reads existing detailed_metrics_matched_roi.jsonl files and writes corrected
tables plus old-vs-corrected comparison artifacts. This script does not train,
evaluate, or modify checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SNAPSHOT_DIR = ROOT / "analysis" / "results_snapshot"

SUPPORTED_CLASSES = [
    "ER",
    "Endosomes",
    "Lysosomes",
    "Microtubules",
    "Mitochondria",
    "Vesicles",
]

ANNOTATED_NEGATIVE_CLASSES = [
    "Lipid Droplets",
    "Nucleus",
    "Nuclear Pores",
    "Peroxisomes",
    "Golgi",
    "ERES",
]

VIMENTIN_CLASS = "Vimentin"

RUN_PAIRS = [
    {
        "configuration": "Categorical, 64^3",
        "old_run": "latest-unet-14cls-jitter48-p64",
        "corrected_run": "corrected-labels-latest-unet-14cls-jitter48-p64",
    },
    {
        "configuration": "Categorical, 128^3",
        "old_run": "latest-unet-14cls-jitter48-dice",
        "corrected_run": "corrected-labels-latest-unet-14cls-jitter48-dice",
    },
    {
        "configuration": "EM augmentation, 64^3",
        "old_run": "latest-unet-14cls-em-aug-p64",
        "corrected_run": "corrected-labels-latest-unet-14cls-em-aug-p64",
    },
    {
        "configuration": "EM augmentation, 128^3",
        "old_run": "latest-unet-14cls-em-aug",
        "corrected_run": "corrected-labels-latest-unet-14cls-em-aug",
    },
    {
        "configuration": "SDT target, train 64^3",
        "old_run": "sdt-unet-14cls-baseline",
        "corrected_run": "corrected-labels-sdt-unet-14cls-baseline",
    },
]


def result_path(run: str, filename: str) -> Path:
    return ROOT / "runs" / run / "results" / filename


def load_jsonl(run: str, filename: str) -> list[dict]:
    path = result_path(run, filename)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def record_signature(record: dict) -> tuple:
    return (
        record.get("patch_idx"),
        record.get("dataset"),
        record.get("crop"),
        tuple(record.get("resolution") or []),
        record.get("patch_class"),
        record.get("em_scale"),
        record.get("label_scale"),
    )


def verify_common_records(records_by_run: dict[str, list[dict]]) -> None:
    first_run = next(iter(records_by_run))
    first = [record_signature(record) for record in records_by_run[first_run]]
    if len(first) != 73:
        raise RuntimeError(f"{first_run} has {len(first)} records, expected 73.")
    for run, records in records_by_run.items():
        signatures = [record_signature(record) for record in records]
        if signatures != first:
            raise RuntimeError(f"{run} does not match the common ordered records.")
        datasets = {record.get("dataset") for record in records}
        crops = {record.get("crop") for record in records}
        if datasets != {"jrc_cos7-1a"} or crops != {"crop234"}:
            raise RuntimeError(f"{run} is not exclusively jrc_cos7-1a/crop234.")


def class_counts(records: list[dict], class_name: str) -> tuple[int, int, int, float, float]:
    tp = sum(int(record.get("per_class_tp", {}).get(class_name, 0)) for record in records)
    fp = sum(int(record.get("per_class_fp", {}).get(class_name, 0)) for record in records)
    fn = sum(int(record.get("per_class_fn", {}).get(class_name, 0)) for record in records)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else math.nan
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan
    return tp, fp, fn, iou, dice


def total_evaluated_voxels(records: list[dict]) -> int:
    total = 0
    per_record = set()
    for record in records:
        classes = (
            set(record.get("per_class_tp", {}))
            | set(record.get("per_class_fp", {}))
            | set(record.get("per_class_fn", {}))
        )
        pred_total = sum(
            int(record.get("per_class_tp", {}).get(cls, 0))
            + int(record.get("per_class_fp", {}).get(cls, 0))
            for cls in classes
        )
        gt_total = sum(
            int(record.get("per_class_tp", {}).get(cls, 0))
            + int(record.get("per_class_fn", {}).get(cls, 0))
            for cls in classes
        )
        if pred_total != gt_total:
            raise RuntimeError(f"{record.get('patch_idx')} prediction/GT totals differ.")
        total += pred_total
        per_record.add(pred_total)
    if per_record != {64**3}:
        raise RuntimeError(f"Expected matched 64^3 ROIs, found per-record voxels {sorted(per_record)}")
    if total != 73 * 64**3:
        raise RuntimeError(f"Expected {73 * 64**3} total voxels, found {total}")
    return total


def aggregate(records: list[dict]) -> dict:
    per_class = {cls: class_counts(records, cls) for cls in SUPPORTED_CLASSES}
    total_voxels = total_evaluated_voxels(records)
    annotated_negative_fp = sum(
        sum(int(record.get("per_class_fp", {}).get(cls, 0)) for record in records)
        for cls in ANNOTATED_NEGATIVE_CLASSES
    )
    vimentin_fp = sum(
        int(record.get("per_class_fp", {}).get(VIMENTIN_CLASS, 0))
        for record in records
    )
    return {
        "per_class": per_class,
        "macro_iou": float(np.mean([counts[3] for counts in per_class.values()])),
        "macro_dice": float(np.mean([counts[4] for counts in per_class.values()])),
        "annotated_negative_fp": annotated_negative_fp,
        "annotated_negative_rate": annotated_negative_fp / total_voxels,
        "vimentin_fp": vimentin_fp,
        "vimentin_rate": vimentin_fp / total_voxels,
        "total_voxels": total_voxels,
    }


def write_corrected_tables(results: dict[str, dict]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with (SNAPSHOT_DIR / "newinml_corrected_matched_roi_main_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "configuration",
                "macro_iou_6cls",
                "macro_dice_6cls",
                "annotated_negative_fp_voxels",
                "annotated_negative_prediction_rate",
                "vimentin_fp_voxels",
                "vimentin_prediction_rate",
            ]
        )
        for pair in RUN_PAIRS:
            result = results[pair["corrected_run"]]
            writer.writerow(
                [
                    pair["configuration"],
                    f"{result['macro_iou']:.6f}",
                    f"{result['macro_dice']:.6f}",
                    result["annotated_negative_fp"],
                    f"{result['annotated_negative_rate']:.8f}",
                    result["vimentin_fp"],
                    f"{result['vimentin_rate']:.8f}",
                ]
            )

    with (SNAPSHOT_DIR / "newinml_corrected_matched_roi_per_class_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "class", "tp", "fp", "fn", "iou", "dice"])
        for pair in RUN_PAIRS:
            result = results[pair["corrected_run"]]
            for cls in SUPPORTED_CLASSES:
                tp, fp, fn, iou, dice = result["per_class"][cls]
                writer.writerow([pair["configuration"], cls, tp, fp, fn, f"{iou:.6f}", f"{dice:.6f}"])

    with (SNAPSHOT_DIR / "newinml_corrected_matched_roi_per_class_iou.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", *SUPPORTED_CLASSES])
        for pair in RUN_PAIRS:
            result = results[pair["corrected_run"]]
            writer.writerow(
                [
                    pair["configuration"],
                    *[f"{result['per_class'][cls][3]:.6f}" for cls in SUPPORTED_CLASSES],
                ]
            )


def write_comparison(old_results: dict[str, dict], corrected_results: dict[str, dict]) -> None:
    with (SNAPSHOT_DIR / "newinml_corrected_vs_old_comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "configuration",
                "old_mIoU6",
                "corrected_mIoU6",
                "old_mDice6",
                "corrected_mDice6",
                "old_annotated_negative_rate",
                "corrected_annotated_negative_rate",
            ]
        )
        for pair in RUN_PAIRS:
            old = old_results[pair["old_run"]]
            corrected = corrected_results[pair["corrected_run"]]
            writer.writerow(
                [
                    pair["configuration"],
                    f"{old['macro_iou']:.6f}",
                    f"{corrected['macro_iou']:.6f}",
                    f"{old['macro_dice']:.6f}",
                    f"{corrected['macro_dice']:.6f}",
                    f"{old['annotated_negative_rate']:.8f}",
                    f"{corrected['annotated_negative_rate']:.8f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-filename", default="detailed_metrics_matched_roi.jsonl")
    args = parser.parse_args()

    all_runs = [pair["old_run"] for pair in RUN_PAIRS] + [pair["corrected_run"] for pair in RUN_PAIRS]
    missing = [result_path(run, args.input_filename) for run in all_runs if not result_path(run, args.input_filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing matched-ROI JSONL files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    records_by_run = {run: load_jsonl(run, args.input_filename) for run in all_runs}
    verify_common_records(records_by_run)
    old_results = {pair["old_run"]: aggregate(records_by_run[pair["old_run"]]) for pair in RUN_PAIRS}
    corrected_results = {
        pair["corrected_run"]: aggregate(records_by_run[pair["corrected_run"]])
        for pair in RUN_PAIRS
    }
    write_corrected_tables(corrected_results)
    write_comparison(old_results, corrected_results)

    print("Wrote corrected NewInML tables:")
    print(SNAPSHOT_DIR / "newinml_corrected_matched_roi_main_results.csv")
    print(SNAPSHOT_DIR / "newinml_corrected_matched_roi_per_class_metrics.csv")
    print(SNAPSHOT_DIR / "newinml_corrected_matched_roi_per_class_iou.csv")
    print(SNAPSHOT_DIR / "newinml_corrected_vs_old_comparison.csv")


if __name__ == "__main__":
    main()
