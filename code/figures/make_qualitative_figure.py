"""Create a multi-slice qualitative segmentation figure for the paper.

The figure runs full-crop sliding-window inference for each requested model and
plots shared slices as rows. Columns are raw EM, ground truth, and model
predictions. Outputs are saved as PNG, PDF, and SVG.
"""

from pathlib import Path
import argparse
import os
import sys

import numpy as np
try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - depends on server environment
    ndimage = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

from code.data.zarr_utils import extract_aligned_volumes, get_scale_trans
from code.evaluation.evaluate_detailed import decode_predictions
from code.evaluation.visualize import run_sliding_window_inference
from code.models.unet import build_model, get_raw_model
from code.utils.config import load_config


DEFAULT_MODELS = [
    "Categorical 64^3|configs/latest_unet.yaml|runs/latest-unet-14cls-jitter48-p64/ckpts/best_model.pth|64",
    "Categorical 128^3|configs/latest_unet.yaml|runs/latest-unet-14cls-jitter48-dice/ckpts/best_model.pth|128",
    "EM augmentation 128^3|configs/latest_unet_em_aug.yaml|runs/latest-unet-14cls-em-aug/ckpts/best_model.pth|128",
    "SDT 64^3|configs/sdt_unet.yaml|runs/sdt-unet-14cls-baseline/ckpts/best_model.pth|64",
]

PREFERRED_SLICES = [387, 399]
SUPPORTED_CLASS_IDS = {1, 2, 3, 4, 8, 11}
UNSUPPORTED_CLASS_IDS = {5, 6, 7, 9, 10, 12, 13}
UNSUPPORTED_DISPLAY_NAME = "Unsupported class prediction"

TITLE_FONTSIZE = 24
SLICE_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE = 20
LEGEND_TITLE_FONTSIZE = 22


class ConditionedResolutionWrapper(torch.nn.Module):
    """Append constant resolution channels for scale-conditioned inference."""

    def __init__(self, base_model, resolution):
        super().__init__()
        self.base_model = base_model
        self.resolution = resolution

    def forward(self, x):
        b, _, z, y, x_dim = x.shape
        channels = [
            torch.full((b, 1, z, y, x_dim), value, device=x.device, dtype=x.dtype)
            for value in self.resolution
        ]
        return self.base_model(torch.cat([x] + channels, dim=1))


def parse_model_spec(spec):
    parts = spec.split("|")
    if len(parts) not in {3, 4}:
        raise ValueError(
            "--model entries must be 'Label|config|checkpoint' or "
            "'Label|config|checkpoint|patch_dim'"
        )
    label, config, checkpoint = parts[:3]
    patch_dim = int(parts[3]) if len(parts) == 4 else None
    return {
        "label": label,
        "config": ROOT / config,
        "checkpoint": ROOT / checkpoint,
        "patch_dim": patch_dim,
    }


def build_color_map(max_classes):
    base_colors = list(plt.get_cmap("tab20").colors)
    custom_colors = [(0.0, 0.0, 0.0, 1.0)] + base_colors[: max_classes - 1]
    return mcolors.ListedColormap(custom_colors), custom_colors


def remap_labels(lbl_volume, config):
    max_raw_id = max(256, int(lbl_volume.max()) + 1)
    label_lookup = np.zeros(max_raw_id, dtype=np.int64)
    for semantic_id, instance_id in config.label_map.items():
        if semantic_id < max_raw_id:
            label_lookup[semantic_id] = instance_id
    return label_lookup[lbl_volume]


def choose_slices(lbl_volume, num_slices, min_separation):
    """Pick separated slices with many foreground classes and pixels."""
    scores = []
    for z in range(lbl_volume.shape[0]):
        vals, counts = np.unique(lbl_volume[z], return_counts=True)
        foreground = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != 0]
        if not foreground:
            continue
        class_count = len(foreground)
        fg_pixels = sum(c for _, c in foreground)
        scores.append((class_count, fg_pixels, z))

    selected = []
    for _, _, z in sorted(scores, reverse=True):
        if all(abs(z - kept) >= min_separation for kept in selected):
            selected.append(z)
        if len(selected) == num_slices:
            break

    if not selected:
        selected = [lbl_volume.shape[0] // 2]
    return sorted(selected)


def choose_display_window(lbl_volume, slices, display_size):
    """Select a deterministic GT-only y/x window for all panels."""
    if display_size <= 0:
        return 0, lbl_volume.shape[1], 0, lbl_volume.shape[2], "full field of view"

    y_max = max(lbl_volume.shape[1] - display_size, 0)
    x_max = max(lbl_volume.shape[2] - display_size, 0)
    if y_max == 0 and x_max == 0:
        return 0, lbl_volume.shape[1], 0, lbl_volume.shape[2], "full field of view"

    # Use a modest stride for speed, then include borders so the rule is stable.
    stride = max(display_size // 4, 1)
    y_candidates = sorted(set(list(range(0, y_max + 1, stride)) + [y_max]))
    x_candidates = sorted(set(list(range(0, x_max + 1, stride)) + [x_max]))

    best = None
    for y0 in y_candidates:
        for x0 in x_candidates:
            crop = lbl_volume[slices, y0 : y0 + display_size, x0 : x0 + display_size]
            vals, counts = np.unique(crop, return_counts=True)
            supported = [
                (int(v), int(c))
                for v, c in zip(vals, counts)
                if int(v) in SUPPORTED_CLASS_IDS
            ]
            class_count = len(supported)
            fg_pixels = sum(c for _, c in supported)
            score = (class_count, fg_pixels, -y0, -x0)
            if best is None or score > best[0]:
                best = (score, y0, x0)

    _, y0, x0 = best
    return (
        y0,
        min(y0 + display_size, lbl_volume.shape[1]),
        x0,
        min(x0 + display_size, lbl_volume.shape[2]),
        (
            "GT-only window maximizing visible supported classes across selected "
            "slices; ties use lower y then lower x"
        ),
    )


def remove_small_components(pred, min_component_size):
    """Remove tiny 3D connected components from each foreground class."""
    if min_component_size <= 0:
        return pred
    if ndimage is None:
        print("scipy is unavailable; skipping connected-component cleanup.")
        return pred

    cleaned = pred.copy()
    structure = np.ones((3, 3, 3), dtype=bool)
    for class_id in sorted(int(v) for v in np.unique(pred) if int(v) != 0):
        mask = pred == class_id
        labels, num_labels = ndimage.label(mask, structure=structure)
        if num_labels == 0:
            continue
        sizes = np.bincount(labels.ravel())
        remove = sizes < min_component_size
        remove[0] = False
        cleaned[remove[labels]] = 0
    return cleaned


def collapse_unsupported_predictions(pred, unsupported_display_id):
    display = pred.copy()
    mask = np.isin(display, list(UNSUPPORTED_CLASS_IDS))
    display[mask] = unsupported_display_id
    return display


def load_model_prediction(spec, em_volume, resolution, gpu, overlap):
    config = load_config(str(spec["config"]))
    if spec["patch_dim"] is not None:
        config.data.patch_dim = spec["patch_dim"]

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    model, device = build_model(config, multi_gpu=False)
    raw_model = get_raw_model(model)
    checkpoint = torch.load(spec["checkpoint"], map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raw_model.load_state_dict(checkpoint)
    model.eval()

    infer_model = model
    if getattr(config.model, "scale_conditioned", False):
        infer_model = ConditionedResolutionWrapper(model, resolution)

    if getattr(config.data, "target_type", "labels") == "sdt":
        em_normalized = em_volume.astype(np.float32) / 255.0
        em_tensor = torch.tensor(em_normalized).unsqueeze(0).unsqueeze(0).to(device)
        from monai.inferers import SlidingWindowInferer

        inferer = SlidingWindowInferer(
            roi_size=(config.data.patch_dim,) * 3,
            sw_batch_size=1,
            overlap=overlap,
        )
        with torch.no_grad():
            outputs = inferer(inputs=em_tensor, network=infer_model)
            pred = decode_predictions(outputs.squeeze(0).cpu(), config)
        return pred.numpy(), config

    pred = run_sliding_window_inference(
        model=infer_model,
        em_volume=em_volume,
        device=device,
        roi_size=(config.data.patch_dim,) * 3,
        sw_batch_size=1,
        overlap=overlap,
    )
    return pred, config


def normalize_em_slice(em_slice):
    lo, hi = np.percentile(em_slice, [1, 99])
    return np.clip((em_slice - lo) / max(hi - lo, 1e-6), 0, 1)


def make_figure(args):
    model_specs = [parse_model_spec(spec) for spec in args.model]
    base_config = load_config(str(model_specs[0]["config"]))

    dataset_base = (
        Path(base_config.paths.data_dir) / args.dataset / f"{args.dataset}.zarr"
    )
    em_volume, lbl_volume = extract_aligned_volumes(str(dataset_base), args.crop)
    em_base = dataset_base / "recon-1" / "em" / "fibsem-uint8"
    resolution, _ = get_scale_trans(str(em_base), "s0")

    lbl_remapped = remap_labels(lbl_volume, base_config)
    selection_rule = "user-specified z slices"
    if args.slices:
        slices = args.slices
    elif all(0 <= z < lbl_remapped.shape[0] for z in PREFERRED_SLICES[: args.num_slices]):
        slices = PREFERRED_SLICES[: args.num_slices]
        selection_rule = "pre-specified paper slices retained from previous figure"
    else:
        slices = choose_slices(
            lbl_remapped,
            num_slices=args.num_slices,
            min_separation=args.min_separation,
        )
        selection_rule = (
            "GT-only slice selection maximizing visible foreground classes and "
            "foreground pixels with deterministic separation"
        )
    y0, y1, x0, x1, window_rule = choose_display_window(
        lbl_remapped,
        slices,
        args.display_size,
    )

    predictions = []
    configs = []
    for spec in model_specs:
        print(f"Running inference for {spec['label']} ...")
        pred, config = load_model_prediction(
            spec,
            em_volume=em_volume,
            resolution=resolution,
            gpu=args.gpu,
            overlap=args.overlap,
        )
        pred = remove_small_components(pred, args.min_component_size)
        predictions.append((spec["label"], pred))
        configs.append(config)

    unsupported_display_id = int(
        max(
            [lbl_remapped.max()]
            + [pred.max() for _, pred in predictions]
            + [config.data.num_classes - 1 for config in configs + [base_config]]
        )
    ) + 1
    display_predictions = [
        (label, collapse_unsupported_predictions(pred, unsupported_display_id))
        for label, pred in predictions
    ]
    max_observed_label = unsupported_display_id
    max_classes = max_observed_label + 1
    cmap, colors = build_color_map(max_classes)
    class_names = configs[-1].class_names
    class_names[unsupported_display_id] = UNSUPPORTED_DISPLAY_NAME

    n_rows = len(slices)
    n_cols = 2 + len(predictions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.25 * n_cols, 3.25 * n_rows + 1.35),
        squeeze=False,
    )

    for row, z in enumerate(slices):
        axes[row, 0].imshow(normalize_em_slice(em_volume[z, y0:y1, x0:x1]), cmap="gray")
        axes[row, 0].set_ylabel(f"z={z}", fontsize=SLICE_LABEL_FONTSIZE)
        axes[row, 0].set_title(
            "Raw EM" if row == 0 else "",
            fontsize=TITLE_FONTSIZE,
            pad=10,
        )
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(
            lbl_remapped[z, y0:y1, x0:x1],
            cmap=cmap,
            vmin=0,
            vmax=max_classes - 1,
            interpolation="nearest",
        )
        axes[row, 1].set_title(
            "Ground truth" if row == 0 else "",
            fontsize=TITLE_FONTSIZE,
            pad=10,
        )
        axes[row, 1].axis("off")

        for col, (label, pred) in enumerate(display_predictions, start=2):
            axes[row, col].imshow(
                pred[z, y0:y1, x0:x1],
                cmap=cmap,
                vmin=0,
                vmax=max_classes - 1,
                interpolation="nearest",
            )
            axes[row, col].set_title(
                label if row == 0 else "",
                fontsize=TITLE_FONTSIZE,
                pad=10,
            )
            axes[row, col].axis("off")

    gt_visible_classes = sorted(
            int(v)
            for z in slices
            for v in np.unique(lbl_remapped[z, y0:y1, x0:x1])
            if int(v) != 0
    )
    gt_visible_classes = sorted(set(gt_visible_classes))
    predicted_visible_classes = sorted(
        {
            int(v)
            for z in slices
            for _, pred in display_predictions
            for v in np.unique(pred[z, y0:y1, x0:x1])
            if int(v) != 0
        }
    )
    predicted_only_classes = [
        class_id
        for class_id in predicted_visible_classes
        if class_id not in gt_visible_classes and class_id != unsupported_display_id
    ]
    legend_handles = [
        mpatches.Patch(
            color=colors[cid],
            label=f"{cid}: {class_names.get(cid, f'Class {cid}')}",
        )
        for cid in gt_visible_classes
        if cid < len(colors)
    ]
    legend_handles.extend(
        mpatches.Patch(
            color=colors[cid],
            label=f"{cid}: {class_names.get(cid, f'Class {cid}')} (predicted only)",
        )
        for cid in predicted_only_classes
        if cid < len(colors)
    )
    if unsupported_display_id in predicted_visible_classes:
        legend_handles.append(
            mpatches.Patch(
                color=colors[unsupported_display_id],
                label=UNSUPPORTED_DISPLAY_NAME,
            )
        )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            title="Classes visible in selected slices",
            title_fontsize=LEGEND_TITLE_FONTSIZE,
        )

    fig.subplots_adjust(
        left=0.02,
        right=0.995,
        top=0.92,
        bottom=0.20 if legend_handles else 0.03,
        wspace=0.035,
        hspace=0.08,
    )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_name
    for ext in ("png", "pdf", "svg"):
        output_path = out_dir / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(output_path, bbox_inches="tight")
        print(f"Wrote {output_path}")
    plt.close(fig)

    support_path = out_dir / f"{stem}_slice_support.csv"
    with support_path.open("w") as f:
        f.write("z_slice,source,class_id,class_name,pixels\n")
        for z in slices:
            vals, counts = np.unique(lbl_remapped[z, y0:y1, x0:x1], return_counts=True)
            for cid, count in zip(vals, counts):
                cid = int(cid)
                if cid == 0:
                    continue
                f.write(
                    f"{z},ground_truth,{cid},{class_names.get(cid, f'Class {cid}')},{int(count)}\n"
                )
            for label, pred in display_predictions:
                vals, counts = np.unique(pred[z, y0:y1, x0:x1], return_counts=True)
                for cid, count in zip(vals, counts):
                    cid = int(cid)
                    if cid == 0:
                        continue
                    f.write(
                        f"{z},{label},{cid},{class_names.get(cid, f'Class {cid}')},{int(count)}\n"
                    )
    print(f"Wrote {support_path}")
    metadata_path = out_dir / f"{stem}_selection_metadata.csv"
    with metadata_path.open("w") as f:
        f.write("dataset,crop,z_slices,y_start,y_end,x_start,x_end,slice_rule,window_rule\n")
        f.write(
            f"{args.dataset},{args.crop},{'|'.join(str(z) for z in slices)},"
            f"{y0},{y1},{x0},{x1},{selection_rule},{window_rule}\n"
        )
    print(f"Wrote {metadata_path}")
    print(f"Selected slices: {slices}; display window y={y0}:{y1}, x={x0}:{x1}")


def parse_args():
    parser = argparse.ArgumentParser(description="Make qualitative paper figure.")
    parser.add_argument("--dataset", default="jrc_cos7-1a")
    parser.add_argument("--crop", default="crop234")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--num-slices", type=int, default=2)
    parser.add_argument("--min-separation", type=int, default=64)
    parser.add_argument(
        "--display-size",
        type=int,
        default=64,
        help="Shared y/x field-of-view size. Use 0 for the full slice.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=128,
        help="Remove predicted 3D connected components smaller than this many voxels. Use 0 to disable.",
    )
    parser.add_argument("--slices", type=int, nargs="*", default=None)
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--output-name", default="newinml_qualitative")
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec: 'Label|config|checkpoint|patch_dim'. May be repeated.",
    )
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_MODELS
    return args


if __name__ == "__main__":
    make_figure(parse_args())
