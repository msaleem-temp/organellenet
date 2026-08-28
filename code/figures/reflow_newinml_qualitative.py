"""Layout-only reflow for the NewInML qualitative figure.

This script uses the already-rendered qualitative PNG as input, crops the panel
rasters, and redraws only paper layout elements: titles, slice labels, and the
shared legend. It does not rerun inference or alter predictions/classes.
"""

from pathlib import Path
import argparse
import os

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PANEL_BOXES = [
    [(135, 145, 955, 965), (1093, 147, 1909, 962), (2049, 147, 2865, 962),
     (3005, 147, 3821, 962), (3961, 147, 4777, 962), (4917, 147, 5733, 962)],
    [(135, 1025, 955, 1846), (1093, 1027, 1909, 1843), (2049, 1027, 2865, 1843),
     (3005, 1027, 3821, 1843), (3961, 1027, 4777, 1843), (4917, 1027, 5733, 1843)],
]

TITLES = [
    "Raw EM",
    "Ground truth",
    r"Cat. $64^3$",
    r"Cat. $128^3$",
    r"EM aug. $128^3$",
    r"SDT $64^3$",
]

SLICE_LABELS = ["z=57", "z=399"]


def class_color(class_id):
    if class_id == 0:
        return (0.0, 0.0, 0.0, 1.0)
    return plt.get_cmap("tab20").colors[class_id - 1]


LEGEND_ITEMS = [
    ("Vesicles", class_color(2)),
    ("Endosomes", class_color(3)),
    ("Microtubules", class_color(8)),
    ("ER", class_color(11)),
    ("Mitochondria (predicted only)", class_color(1)),
    ("Unsupported class prediction", class_color(14)),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Reflow qualitative figure layout only.")
    parser.add_argument(
        "--input",
        default="paper/figures/newinml_qualitative.png",
        help="Existing qualitative PNG generated from inference.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/figures",
        help="Output directory for reflowed figure.",
    )
    parser.add_argument("--output-name", default="newinml_qualitative")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Image.open(ROOT / args.input).convert("RGBA")
    panels = [
        [source.crop(box) for box in row]
        for row in PANEL_BOXES
    ]

    fig, axes = plt.subplots(
        2,
        6,
        figsize=(15.8, 6.2),
        gridspec_kw={"wspace": 0.06, "hspace": 0.08},
    )

    for row in range(2):
        for col in range(6):
            ax = axes[row, col]
            ax.imshow(panels[row][col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            if row == 0:
                ax.set_title(TITLES[col], fontsize=20, pad=8)
            if col == 0:
                ax.set_ylabel(SLICE_LABELS[row], fontsize=18, rotation=90, labelpad=12)

    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in LEGEND_ITEMS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.8,
        handlelength=1.8,
    )

    fig.subplots_adjust(left=0.045, right=0.995, top=0.91, bottom=0.20)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        out_path = out_dir / f"{args.output_name}.{ext}"
        if ext == "png":
            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        else:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        print(f"Wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
