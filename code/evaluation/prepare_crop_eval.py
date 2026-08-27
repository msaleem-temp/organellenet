"""
Rank candidate crops and optionally export a crop-level evaluation JSON.

This is intended for quick paper triage. It can identify medium/coarse crops
with enough target-class support, and it can write a JSON file containing every
patch from a selected crop. Existing trained models can be evaluated on that
JSON, but the crop is only a clean held-out test if it was excluded during the
model's training run.
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


TARGET_CLASSES = {
    "endo", "ld", "lyso", "mito", "mt", "np", "nuc", "perox", "ves",
    "vim", "golgi", "er", "eres",
}
SUFFIXES = {"lum", "mem", "in", "out"}


def base_class(name):
    if "_" in name and name.rsplit("_", 1)[-1] in SUFFIXES:
        return name.rsplit("_", 1)[0]
    return name


def resolution_band(resolution):
    if not resolution:
        return "unknown"
    max_res = max(resolution)
    if max_res <= 4:
        return "fine"
    if max_res <= 8:
        return "medium"
    return "coarse"


def load_split_crops(split_dir):
    crops = set()
    for name in ("train", "val", "test"):
        path = os.path.join(split_dir, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            crops.update(record.get("crop") for record in json.load(f))
    return crops


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare crop-level evaluation JSON")
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--split-dir", default=None,
                        help="Optional split dir used to mark crops already seen")
    parser.add_argument("--band", choices=["fine", "medium", "coarse"], default=None)
    parser.add_argument("--min-target-patches", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--export-crop", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.blueprint, "r") as f:
        records = json.load(f)

    seen_crops = load_split_crops(args.split_dir) if args.split_dir else set()
    by_crop = {}
    for record in records:
        crop = record.get("crop")
        item = by_crop.setdefault(crop, {
            "records": [],
            "dataset": record.get("dataset"),
            "resolution": record.get("resolution") or [],
            "classes": {},
            "target_count": 0,
        })
        item["records"].append(record)
        cls = base_class(record.get("class", ""))
        item["classes"][cls] = item["classes"].get(cls, 0) + 1
        if cls in TARGET_CLASSES:
            item["target_count"] += 1

    if args.export_crop:
        if args.export_crop not in by_crop:
            raise SystemExit(f"crop not found: {args.export_crop}")
        if not args.output:
            raise SystemExit("--output is required with --export-crop")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(by_crop[args.export_crop]["records"], f, indent=4)
        print(f"Wrote {len(by_crop[args.export_crop]['records'])} records to {args.output}")
        return

    rows = []
    for crop, item in by_crop.items():
        band = resolution_band(item["resolution"])
        if args.band and band != args.band:
            continue
        if item["target_count"] < args.min_target_patches:
            continue
        top_classes = sorted(item["classes"].items(), key=lambda kv: -kv[1])[:6]
        rows.append((item["target_count"], crop, band, item["dataset"], item["resolution"], crop in seen_crops, top_classes))

    rows.sort(reverse=True)
    print("target_patches,crop,band,dataset,resolution,seen_in_split,top_classes")
    for row in rows[:args.top_k]:
        print(",".join([
            str(row[0]),
            row[1],
            row[2],
            row[3],
            str(row[4]),
            str(row[5]).lower(),
            str(row[6]),
        ]))


if __name__ == "__main__":
    main()
