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
    "Static p128|configs/static_unet.yaml|runs/static-unet-13cls-nojitter/ckpts/best_model.pth|128",
    "Dynamic p128|configs/dynamic_unet.yaml|runs/dynamic-unet-13cls-jitter32/ckpts/best_model.pth|128",
    "Latest p128|configs/latest_unet.yaml|runs/latest-unet-14cls-jitter48-dice/ckpts/best_model.pth|128",
    "Latest Fixed p128|configs/latest_unet_fixed.yaml|runs/latest-unet-14cls-fixed/ckpts/best_model.pth|128",
]


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
    slices = args.slices or choose_slices(
        lbl_remapped,
        num_slices=args.num_slices,
        min_separation=args.min_separation,
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

    max_observed_label = int(
        max(
            [lbl_remapped.max()]
            + [pred.max() for _, pred in predictions]
            + [config.data.num_classes - 1 for config in configs + [base_config]]
        )
    )
    max_classes = max_observed_label + 1
    cmap, colors = build_color_map(max_classes)
    class_names = configs[-1].class_names

    n_rows = len(slices)
    n_cols = 2 + len(predictions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.55 * n_cols, 2.55 * n_rows + 0.75),
        squeeze=False,
    )

    for row, z in enumerate(slices):
        axes[row, 0].imshow(normalize_em_slice(em_volume[z]), cmap="gray")
        axes[row, 0].set_ylabel(f"z={z}", fontsize=9)
        axes[row, 0].set_title("Raw EM" if row == 0 else "", fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(
            lbl_remapped[z],
            cmap=cmap,
            vmin=0,
            vmax=max_classes - 1,
            interpolation="nearest",
        )
        axes[row, 1].set_title("Ground truth" if row == 0 else "", fontsize=9)
        axes[row, 1].axis("off")

        for col, (label, pred) in enumerate(predictions, start=2):
            axes[row, col].imshow(
                pred[z],
                cmap=cmap,
                vmin=0,
                vmax=max_classes - 1,
                interpolation="nearest",
            )
            axes[row, col].set_title(label if row == 0 else "", fontsize=9)
            axes[row, col].axis("off")

    gt_visible_classes = sorted(
        int(v)
        for z in slices
        for v in np.unique(lbl_remapped[z])
        if int(v) != 0
    )
    gt_visible_classes = sorted(set(gt_visible_classes))
    predicted_visible_classes = sorted(
        {
            int(v)
            for z in slices
            for _, pred in predictions
            for v in np.unique(pred[z])
            if int(v) != 0
        }
    )
    predicted_only_classes = [
        class_id for class_id in predicted_visible_classes if class_id not in gt_visible_classes
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
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 5),
            frameon=False,
            fontsize=8,
            title="Classes visible in selected slices",
            title_fontsize=9,
        )

    fig.subplots_adjust(
        left=0.02,
        right=0.995,
        top=0.92,
        bottom=0.12 if legend_handles else 0.03,
        wspace=0.025,
        hspace=0.04,
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
            vals, counts = np.unique(lbl_remapped[z], return_counts=True)
            for cid, count in zip(vals, counts):
                cid = int(cid)
                if cid == 0:
                    continue
                f.write(
                    f"{z},ground_truth,{cid},{class_names.get(cid, f'Class {cid}')},{int(count)}\n"
                )
            for label, pred in predictions:
                vals, counts = np.unique(pred[z], return_counts=True)
                for cid, count in zip(vals, counts):
                    cid = int(cid)
                    if cid == 0:
                        continue
                    f.write(
                        f"{z},{label},{cid},{class_names.get(cid, f'Class {cid}')},{int(count)}\n"
                    )
    print(f"Wrote {support_path}")
    print(f"Selected slices: {slices}")


def parse_args():
    parser = argparse.ArgumentParser(description="Make qualitative paper figure.")
    parser.add_argument("--dataset", default="jrc_cos7-1a")
    parser.add_argument("--crop", default="crop234")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--num-slices", type=int, default=2)
    parser.add_argument("--min-separation", type=int, default=64)
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=128,
        help="Remove predicted 3D connected components smaller than this many voxels. Use 0 to disable.",
    )
    parser.add_argument("--slices", type=int, nargs="*", default=None)
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--output-name", default="qualitative_crop234_multislice")
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
