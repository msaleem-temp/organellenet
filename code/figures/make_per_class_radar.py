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
    {"source_name": "Static (p128)", "display_name": "13-class static"},
    {"source_name": "Dynamic (p128)", "display_name": "13-class jitter"},
    {"source_name": "Latest (p128)", "display_name": "14-class + jitter"},
    {
        "source_name": "Latest Fixed (p128)",
        "display_name": "14-class + ER/Golgi fix",
    },
]

AXIS_FONTSIZE = 20
RADIUS_FONTSIZE = 20
LEGEND_FONTSIZE = 20


def to_float(value):
    if value in {"", "N/A", None}:
        return np.nan
    return float(value)


def parse_model_spec(spec):
    if isinstance(spec, dict):
        return spec["source_name"], spec["display_name"]
    parts = spec.split("|", maxsplit=1)
    if len(parts) == 1:
        return spec, spec
    return parts[0], parts[1]


def load_values(path, classes, models, metric):
    rows = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rows[row["Class"]] = row

    values = {}
    for spec in models:
        csv_model, display_model = parse_model_spec(spec)
        col = f"{csv_model} {metric}"
        values[display_model] = [to_float(rows[cls].get(col)) for cls in classes]
    return values


def plot_radar(values, classes, output_dir, output_name, metric):
    n = len(classes)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10.5, 9.5), subplot_kw={"polar": True})

    colors = ["#315c73", "#c77b32", "#6f8f3a", "#8d4f73", "#5b6fa8"]
    for idx, (model, vals) in enumerate(values.items()):
        closed = vals + vals[:1]
        ax.plot(angles, closed, linewidth=1.8, color=colors[idx % len(colors)], label=model)
        ax.fill(angles, closed, color=colors[idx % len(colors)], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes, fontsize=AXIS_FONTSIZE)
    ax.tick_params(axis="x", pad=18)
    ax.tick_params(axis="y", pad=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=RADIUS_FONTSIZE)
    ax.grid(color="0.82", linewidth=0.7)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.subplots_adjust(left=0.14, right=0.86, top=0.90, bottom=0.28)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"{output_name}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.35)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.35)
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
