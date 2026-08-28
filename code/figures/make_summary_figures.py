"""Generate publication-style summary figures from current paper tables."""

from pathlib import Path
import csv
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / "runs" / "paper_tables" / "summary.csv"
OUT_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / ".matplotlib-cache"

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY_LABELS = {
    "Dynamic (p128)": "13-class jitter, 128-voxel patches",
    "Static (p128)": "13-class static, 128-voxel patches",
    "Static (p128, valdice)": "13-class static, 128-voxel validation-Dice checkpoint",
    "Latest (p128)": "14-class + jitter, 128-voxel patches",
    "Scale Conditioned": "14-class + EM aug. + scale channels, 128-voxel patches",
    "Latest Fixed (p128)": "14-class + corrected ER/Golgi sampling, 128-voxel patches",
    "Latest+EM (p128)": "14-class + EM aug., 128-voxel patches",
    "Latest (p64)": "14-class + jitter, 64-voxel patches",
    "Latest+Res (p128)": "14-class + EM/resolution aug., 128-voxel patches",
    "SDT Baseline": "14-class signed-distance target, 64-voxel patches",
    "Latest+EM (p64)": "14-class + EM aug., 64-voxel patches",
    "Latest+Res (p64)": "14-class + EM/resolution aug., 64-voxel patches",
    "Static (p64, valdice)": "13-class static, 64-voxel validation-Dice checkpoint",
    "Static (p64)": "13-class static, 64-voxel patches",
}


def _to_float(value):
    if value in {"", "N/A", None}:
        return np.nan
    return float(value)


def load_summary():
    with SUMMARY_CSV.open() as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if _to_float(r["Mean IoU"]) == _to_float(r["Mean IoU"])]


def plot_summary(rows):
    labels = [DISPLAY_LABELS.get(r["Model"], r["Model"]) for r in rows]
    miou = [_to_float(r["Mean IoU"]) for r in rows]
    mdice = [_to_float(r["Mean Dice"]) for r in rows]

    order = np.argsort(miou)[::-1]
    labels = [labels[i] for i in order]
    miou = [miou[i] for i in order]
    mdice = [mdice[i] for i in order]

    y = np.arange(len(labels))
    height = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    ax.barh(y + height / 2, miou, height, label="Mean IoU", color="#376b8c")
    ax.barh(y - height / 2, mdice, height, label="Mean Dice", color="#c27d38")
    ax.set_xlabel("Score")
    ax.set_xlim(0, max(max(miou), max(mdice)) * 1.18)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=20)
    ax.invert_yaxis()
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title("Held-out crop performance by model variant")
    fig.subplots_adjust(left=0.46, right=0.98, top=0.92, bottom=0.10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "summary_model_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / "summary_model_comparison.pdf")


def main():
    rows = load_summary()
    plot_summary(rows)


if __name__ == "__main__":
    main()
