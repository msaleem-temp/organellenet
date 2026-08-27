"""Generate a radar plot of per-class IoU for supported held-out classes."""

from pathlib import Path
import argparse
import csv
import math
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CLASSES = [
    "ER",
    "Endosomes",
    "Lysosomes",
    "Microtubules",
    "Mitochondria",
    "Vesicles",
]

DEFAULT_MODELS = [
    "Static (p128)",
    "Dynamic (p128)",
    "Latest (p128)",
    "Latest Fixed (p128)",
]


def to_float(value):
    if value in {"", "N/A", None}:
        return np.nan
    return float(value)


def load_values(path, classes, models, metric):
    rows = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rows[row["Class"]] = row

    values = {}
    for model in models:
        col = f"{model} {metric}"
        values[model] = [to_float(rows[cls].get(col)) for cls in classes]
    return values


def plot_radar(values, classes, output_dir, output_name, metric):
    n = len(classes)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.6, 4.2), subplot_kw={"polar": True})

    colors = ["#315c73", "#c77b32", "#6f8f3a", "#8d4f73", "#5b6fa8"]
    for idx, (model, vals) in enumerate(values.items()):
        closed = vals + vals[:1]
        ax.plot(angles, closed, linewidth=1.8, color=colors[idx % len(colors)], label=model)
        ax.fill(angles, closed, color=colors[idx % len(colors)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.grid(color="0.82", linewidth=0.7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{output_name}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Make per-class radar plot.")
    parser.add_argument(
        "--input",
        default="runs/paper_tables/per_class_comparison.csv",
        help="Per-class comparison CSV.",
    )
    parser.add_argument("--metric", default="IoU", choices=["IoU", "Dice"])
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--output-name", default="per_class_supported_radar_iou")
    return parser.parse_args()


def main():
    args = parse_args()
    values = load_values(ROOT / args.input, args.classes, args.models, args.metric)
    plot_radar(values, args.classes, ROOT / args.output_dir, args.output_name, args.metric)


if __name__ == "__main__":
    main()
