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


def _to_float(value):
    if value in {"", "N/A", None}:
        return np.nan
    return float(value)


def load_summary():
    with SUMMARY_CSV.open() as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if _to_float(r["Mean IoU"]) == _to_float(r["Mean IoU"])]


def plot_summary(rows):
    labels = [r["Model"] for r in rows]
    miou = [_to_float(r["Mean IoU"]) for r in rows]
    mdice = [_to_float(r["Mean Dice"]) for r in rows]

    order = np.argsort(miou)[::-1]
    labels = [labels[i] for i in order]
    miou = [miou[i] for i in order]
    mdice = [mdice[i] for i in order]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, miou, width, label="Mean IoU", color="#376b8c")
    ax.bar(x + width / 2, mdice, width, label="Mean Dice", color="#c27d38")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(miou), max(mdice)) * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title("Held-out crop performance by model variant")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "summary_model_comparison.png", dpi=300)
    fig.savefig(OUT_DIR / "summary_model_comparison.pdf")


def main():
    rows = load_summary()
    plot_summary(rows)


if __name__ == "__main__":
    main()
