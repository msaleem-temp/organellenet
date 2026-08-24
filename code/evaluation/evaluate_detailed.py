"""
OrganelleNet — Detailed Per-Patch Evaluation with Metadata

Evaluates a trained model on the test set and produces a JSONL file
where each line contains the patch metadata (dataset, crop, resolution,
class) alongside per-class IoU and Dice scores for that patch.

This enables post-hoc slicing by dataset, resolution band, organelle
morphology, etc. — the raw data for all paper tables and figures.

Usage:
    python code/evaluation/evaluate_detailed.py \
        --config configs/latest_unet.yaml \
        --checkpoint runs/latest-unet-14cls-jitter48-dice/ckpts/best_model.pth \
        --output runs/latest-unet-14cls-jitter48-dice/results/detailed_metrics.jsonl \
        --gpu 0

    # Quick test on a subset:
    python code/evaluation/evaluate_detailed.py \
        --config configs/latest_unet.yaml \
        --checkpoint runs/latest-unet-14cls-jitter48-dice/ckpts/best_model.pth \
        --output /tmp/test.jsonl --max-patches 10 --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from code.utils.config import load_config
from code.utils.paths import setup_run_directory
from code.data.zarr_utils import build_zarr_map
from code.data.dataset import PatchDataset
from code.models.unet import build_model, get_raw_model


def compute_per_class_metrics(pred, target, num_classes):
    """
    Compute per-class IoU and Dice for a single 3D patch.

    Parameters
    ----------
    pred : np.ndarray, int64, shape [Z, Y, X]
        Predicted class indices.
    target : np.ndarray, int64, shape [Z, Y, X]
        Ground truth class indices.
    num_classes : int
        Total number of classes.

    Returns
    -------
    dict with keys 'iou' and 'dice', each mapping class_idx -> float.
    Only classes present in either pred or target are included.
    """
    iou = {}
    dice = {}

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()

        # Skip classes absent in both pred and target
        if pred_sum == 0 and target_sum == 0:
            continue

        eps = 1e-8
        iou[c] = float(intersection / (union + eps))
        dice[c] = float(2 * intersection / (pred_sum + target_sum + eps))

    return {"iou": iou, "dice": dice}


def parse_args():
    parser = argparse.ArgumentParser(
        description="OrganelleNet — Detailed Per-Patch Evaluation"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the JSONL results file")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU index")
    parser.add_argument("--test-json", type=str, default=None,
                        help="Override test JSON path")
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Evaluate only the first N patches (for debugging)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. Load config
    config = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"Detailed Evaluation: {config.experiment_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    # 2. Locate test JSON
    run_paths = setup_run_directory(config, config_path=args.config)
    test_json = args.test_json
    if test_json is None:
        test_json = os.path.join(run_paths["run_dir"], "splits", "test.json")

    if not os.path.exists(test_json):
        print(f"Error: Test JSON not found at {test_json}")
        print("Run training first or provide --test-json explicitly.")
        sys.exit(1)

    # 3. Build zarr map and test dataset (no augmentation, no jitter)
    zarr_map = build_zarr_map(config.paths.data_dir)

    test_dataset = PatchDataset(
        json_path=test_json,
        zarr_map=zarr_map,
        label_map=config.label_map,
        patch_dim=config.data.patch_dim,
        max_jitter=0,
        augmentation_config=None,  # Never augment during evaluation
    )

    # 4. Build model and load weights
    model, device = build_model(config, multi_gpu=False)
    raw_model = get_raw_model(model)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raw_model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 5. Evaluate each patch and write JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    num_patches = len(test_dataset)
    if args.max_patches is not None:
        num_patches = min(num_patches, args.max_patches)

    num_classes = config.data.num_classes
    class_names = config.class_names

    # Aggregate accumulators for summary statistics
    class_iou_accum = {c: [] for c in range(num_classes)}
    class_dice_accum = {c: [] for c in range(num_classes)}

    with open(args.output, "w") as f_out:
        with torch.no_grad():
            for idx in tqdm(range(num_patches), desc="Evaluating patches"):
                em_tensor, lbl_tensor = test_dataset[idx]
                metadata = test_dataset.get_patch_metadata(idx)

                # Forward pass
                em_batch = em_tensor.unsqueeze(0).to(device)
                output = model(em_batch)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                target = lbl_tensor.numpy()

                # Per-class metrics for this patch
                patch_metrics = compute_per_class_metrics(pred, target, num_classes)

                # Convert class indices to names for readability
                iou_named = {
                    class_names.get(c, f"class_{c}"): v
                    for c, v in patch_metrics["iou"].items()
                }
                dice_named = {
                    class_names.get(c, f"class_{c}"): v
                    for c, v in patch_metrics["dice"].items()
                }

                # Accumulate for summary (skip background)
                for c, v in patch_metrics["iou"].items():
                    if c > 0:
                        class_iou_accum[c].append(v)
                for c, v in patch_metrics["dice"].items():
                    if c > 0:
                        class_dice_accum[c].append(v)

                # Compute mean IoU/Dice for this patch (excl. background)
                fg_ious = [v for c, v in patch_metrics["iou"].items() if c > 0]
                fg_dices = [v for c, v in patch_metrics["dice"].items() if c > 0]

                record = {
                    "patch_idx": idx,
                    "dataset": metadata.get("dataset"),
                    "crop": metadata.get("crop"),
                    "resolution": metadata.get("resolution"),
                    "patch_class": metadata.get("class"),
                    "em_scale": metadata.get("em_scale"),
                    "label_scale": metadata.get("label_scale"),
                    "per_class_iou": iou_named,
                    "per_class_dice": dice_named,
                    "mean_iou": float(np.mean(fg_ious)) if fg_ious else 0.0,
                    "mean_dice": float(np.mean(fg_dices)) if fg_dices else 0.0,
                }
                f_out.write(json.dumps(record) + "\n")

    # 6. Print summary
    print(f"\n{'='*60}")
    print(f"Summary — {num_patches} patches evaluated")
    print(f"{'='*60}")
    print(f"{'Class':<20} | {'Mean IoU':<10} | {'Mean Dice':<10} | {'N patches':<10}")
    print("-" * 55)

    for c in range(1, num_classes):
        name = class_names.get(c, f"Class {c}")
        ious = class_iou_accum[c]
        dices = class_dice_accum[c]
        if ious:
            print(f"{name:<20} | {np.mean(ious):<10.4f} | {np.mean(dices):<10.4f} | {len(ious):<10}")
        else:
            print(f"{name:<20} | {'N/A':<10} | {'N/A':<10} | {0:<10}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
