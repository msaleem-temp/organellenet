"""Regenerate retained NewInML result tables and per-class IoU heatmap."""

from pathlib import Path
import argparse
import csv
import json
import math
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "analysis" / "results_snapshot"
FIGURE_DIR = ROOT / "figures"
CACHE_DIR = ROOT / ".matplotlib-cache"

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = [
    {
        "run": "latest-unet-14cls-jitter48-p64",
        "name": "Categorical, 64^3",
        "heatmap_name": "Categorical 64$^3$",
        "expected": (0.279227, 0.407549, 230286),
    },
    {
        "run": "latest-unet-14cls-jitter48-dice",
        "name": "Categorical, 128^3",
        "heatmap_name": "Categorical 128$^3$",
        "expected": (0.496438, 0.605558, 586723),
    },
    {
        "run": "latest-unet-14cls-em-aug-p64",
        "name": "EM augmentation, 64^3",
        "heatmap_name": "EM augmentation 64$^3$",
        "expected": (0.157996, 0.245581, 153414),
    },
    {
        "run": "latest-unet-14cls-em-aug",
        "name": "EM augmentation, 128^3",
        "heatmap_name": "EM augmentation 128$^3$",
        "expected": (0.390610, 0.529967, 708159),
    },
    {
        "run": "sdt-unet-14cls-baseline",
        "name": "SDT target, train 64^3",
        "heatmap_name": "SDT train 64$^3$",
        "expected": (0.288849, 0.431083, 157343),
    },
]

SUPPORTED_CLASSES = [
    "ER",
    "Endosomes",
    "Lysosomes",
    "Microtubules",
    "Mitochondria",
    "Vesicles",
]

SUPPORTED_PATCH_COUNTS = {
    "ER": 69,
    "Endosomes": 18,
    "Lysosomes": 6,
    "Microtubules": 42,
    "Mitochondria": 7,
    "Vesicles": 35,
}

UNSUPPORTED_CLASSES = [
    "Lipid Droplets",
    "Nucleus",
    "Nuclear Pores",
    "Peroxisomes",
    "Golgi",
    "ERES",
    "Vimentin",
]


def load_jsonl(run, filename):
    path = ROOT / "runs" / run / "results" / filename
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def record_signature(record):
    return (
        record.get("patch_idx"),
        record.get("dataset"),
        record.get("crop"),
        tuple(record.get("resolution") or []),
        record.get("patch_class"),
        record.get("em_scale"),
        record.get("label_scale"),
    )


def verify_common_records(records_by_run):
    first_run = RUNS[0]["run"]
    first_signatures = [record_signature(r) for r in records_by_run[first_run]]
    if len(first_signatures) != 73:
        raise RuntimeError(f"{first_run} has {len(first_signatures)} records, expected 73.")

    for run_info in RUNS:
        run = run_info["run"]
        records = records_by_run[run]
        signatures = [record_signature(r) for r in records]
        datasets = {r.get("dataset") for r in records}
        crops = {r.get("crop") for r in records}
        if signatures != first_signatures:
            raise RuntimeError(f"{run} does not match the common ordered record list.")
        if datasets != {"jrc_cos7-1a"} or crops != {"crop234"}:
            raise RuntimeError(f"{run} is not exclusively jrc_cos7-1a/crop234.")


def class_counts(records, class_name):
    tp = sum(int(r.get("per_class_tp", {}).get(class_name, 0)) for r in records)
    fp = sum(int(r.get("per_class_fp", {}).get(class_name, 0)) for r in records)
    fn = sum(int(r.get("per_class_fn", {}).get(class_name, 0)) for r in records)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else math.nan
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan
    return tp, fp, fn, iou, dice


def total_evaluated_voxels(records):
    total = 0
    per_record_voxels = set()
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
            raise RuntimeError(
                f"Prediction/GT voxel totals differ for record {record.get('patch_idx')}: "
                f"{pred_total} vs {gt_total}"
            )
        total += pred_total
        per_record_voxels.add(pred_total)
    if len(per_record_voxels) != 1:
        raise RuntimeError(f"Mixed evaluated patch volumes found: {sorted(per_record_voxels)}")
    return total, per_record_voxels.pop()


def patch_size_from_voxels(per_record_voxels):
    patch_size = round(per_record_voxels ** (1.0 / 3.0))
    if patch_size**3 != per_record_voxels:
        raise RuntimeError(f"Per-record voxel count is not cubic: {per_record_voxels}")
    return patch_size


def aggregate(records):
    per_class = {cls: class_counts(records, cls) for cls in SUPPORTED_CLASSES}
    macro_iou = float(np.mean([counts[3] for counts in per_class.values()]))
    macro_dice = float(np.mean([counts[4] for counts in per_class.values()]))
    unsupported_fp = sum(
        sum(int(r.get("per_class_fp", {}).get(cls, 0)) for r in records)
        for cls in UNSUPPORTED_CLASSES
    )
    total_voxels, per_record_voxels = total_evaluated_voxels(records)
    return {
        "per_class": per_class,
        "gt_voxels": {
            cls: per_class[cls][0] + per_class[cls][2]
            for cls in SUPPORTED_CLASSES
        },
        "patch_support": {
            cls: sum(
                int(r.get("per_class_tp", {}).get(cls, 0))
                + int(r.get("per_class_fn", {}).get(cls, 0))
                > 0
                for r in records
            )
            for cls in SUPPORTED_CLASSES
        },
        "macro_iou": macro_iou,
        "macro_dice": macro_dice,
        "unsupported_fp": unsupported_fp,
        "unsupported_fp_rate": unsupported_fp / total_voxels,
        "total_voxels": total_voxels,
        "per_record_voxels": per_record_voxels,
        "patch_size": patch_size_from_voxels(per_record_voxels),
    }


def native_patch_size(records):
    shapes = {
        tuple(record.get("native_patch_shape") or [])
        for record in records
        if record.get("native_patch_shape")
    }
    if not shapes:
        return None
    if len(shapes) != 1:
        raise RuntimeError(f"Mixed native patch shapes found: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 3 or len(set(shape)) != 1:
        raise RuntimeError(f"Unexpected native patch shape: {shape}")
    return shape[0]


def assert_matched_roi(results):
    expected_total = 73 * 64**3
    reference = None
    for run_info in RUNS:
        result = results[run_info["run"]]
        if result["total_voxels"] != expected_total:
            raise RuntimeError(
                f"{run_info['run']} scored {result['total_voxels']} voxels; "
                f"expected {expected_total} for 73 central 64^3 ROIs."
            )
        if result["patch_size"] != 64:
            raise RuntimeError(
                f"{run_info['run']} scored {result['patch_size']}^3, expected 64^3."
            )
        gt_voxels = result["gt_voxels"]
        if reference is None:
            reference = gt_voxels
        elif gt_voxels != reference:
            raise RuntimeError(
                f"{run_info['run']} has different TP+FN supported-class totals: "
                f"{gt_voxels} vs {reference}"
            )


def assert_expected(results):
    tolerance = 5e-7
    for run_info in RUNS:
        run = run_info["run"]
        expected_iou, expected_dice, expected_fp = run_info["expected"]
        actual = results[run]
        if abs(actual["macro_iou"] - expected_iou) > tolerance:
            raise RuntimeError(f"{run} macro IoU changed: {actual['macro_iou']} vs {expected_iou}")
        if abs(actual["macro_dice"] - expected_dice) > tolerance:
            raise RuntimeError(f"{run} macro Dice changed: {actual['macro_dice']} vs {expected_dice}")
        if actual["unsupported_fp"] != expected_fp:
            raise RuntimeError(f"{run} unsupported FP changed: {actual['unsupported_fp']} vs {expected_fp}")


def write_main_results(results, records_by_run, output_name):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / output_name
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "configuration",
                "native_eval_patch_size",
                "score_roi_size",
                "macro_iou_6cls",
                "macro_dice_6cls",
                "unsupported_fp_voxels",
                "unsupported_fp_rate",
            ]
        )
        for run_info in RUNS:
            run = run_info["run"]
            result = results[run]
            native_size = native_patch_size(records_by_run[run]) or result["patch_size"]
            writer.writerow(
                [
                    run_info["name"],
                    native_size,
                    result["patch_size"],
                    f"{result['macro_iou']:.6f}",
                    f"{result['macro_dice']:.6f}",
                    result["unsupported_fp"],
                    f"{result['unsupported_fp_rate']:.8f}",
                ]
            )
    print(f"Wrote {path}")


def write_per_class_iou(results, output_name):
    path = SNAPSHOT_DIR / output_name
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", *SUPPORTED_CLASSES])
        for run_info in RUNS:
            run = run_info["run"]
            writer.writerow(
                [
                    run_info["name"],
                    *[
                        f"{results[run]['per_class'][cls][3]:.6f}"
                        for cls in SUPPORTED_CLASSES
                    ],
                ]
            )
    print(f"Wrote {path}")


def write_per_class_metrics(results, output_name):
    path = SNAPSHOT_DIR / output_name
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "class", "tp", "fp", "fn", "iou", "dice"])
        for run_info in RUNS:
            run = run_info["run"]
            for cls in SUPPORTED_CLASSES:
                tp, fp, fn, iou, dice = results[run]["per_class"][cls]
                writer.writerow(
                    [
                        run_info["name"],
                        cls,
                        tp,
                        fp,
                        fn,
                        f"{iou:.6f}",
                        f"{dice:.6f}",
                    ]
                )
    print(f"Wrote {path}")


def write_unsupported_fp(records_by_run, output_name):
    path = SNAPSHOT_DIR / output_name
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", *UNSUPPORTED_CLASSES])
        for run_info in RUNS:
            run = run_info["run"]
            records = records_by_run[run]
            writer.writerow(
                [
                    run_info["name"],
                    *[
                        sum(int(r.get("per_class_fp", {}).get(cls, 0)) for r in records)
                        for cls in UNSUPPORTED_CLASSES
                    ],
                ]
            )
    print(f"Wrote {path}")


def plot_heatmap(results, figure_name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = np.array(
        [
            [results[run_info["run"]]["per_class"][cls][3] for cls in SUPPORTED_CLASSES]
            for run_info in RUNS
        ]
    )
    row_labels = [run_info["heatmap_name"] for run_info in RUNS]
    patch_support = results[RUNS[0]["run"]]["patch_support"]
    col_labels = [f"{cls}\n(n={patch_support[cls]})" for cls in SUPPORTED_CLASSES]

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.tick_params(axis="both", length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value < 0.55 else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("IoU", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.subplots_adjust(left=0.23, right=0.94, top=0.95, bottom=0.18)
    for ext in ("pdf", "svg", "png"):
        path = FIGURE_DIR / f"{figure_name}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(f"Wrote {path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate NewInML retained-run tables.")
    parser.add_argument(
        "--input-filename",
        default="detailed_metrics.jsonl",
        help="JSONL filename under each run's results directory.",
    )
    parser.add_argument(
        "--matched-roi",
        action="store_true",
        help="Use matched-ROI output filenames and assert common central 64^3 scoring.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records_by_run = {
        run_info["run"]: load_jsonl(run_info["run"], args.input_filename)
        for run_info in RUNS
    }
    verify_common_records(records_by_run)
    results = {run: aggregate(records) for run, records in records_by_run.items()}
    if args.matched_roi:
        assert_matched_roi(results)
        write_main_results(
            results,
            records_by_run,
            "newinml_matched_roi_main_results.csv",
        )
        write_per_class_iou(results, "newinml_matched_roi_per_class_iou.csv")
        write_per_class_metrics(results, "newinml_matched_roi_per_class_metrics.csv")
        write_unsupported_fp(records_by_run, "newinml_matched_roi_unsupported_fp.csv")
        plot_heatmap(results, "newinml_matched_roi_per_class_iou")
    else:
        assert_expected(results)
        write_main_results(results, records_by_run, "newinml_main_results.csv")
        write_per_class_iou(results, "newinml_per_class_iou.csv")
        write_per_class_metrics(results, "newinml_per_class_metrics.csv")
        write_unsupported_fp(records_by_run, "newinml_unsupported_fp.csv")
        plot_heatmap(results, "newinml_per_class_iou")

    print("\nEvaluated patch sizes from JSONL confusion totals:")
    for run_info in RUNS:
        run = run_info["run"]
        result = results[run]
        print(
            f"{run_info['name']}: {result['patch_size']}^3 "
            f"({result['per_record_voxels']} voxels per record)"
        )


if __name__ == "__main__":
    main()
