"""
OrganelleNet — Stratified Evaluation across multiple resolutions.

Evaluates a model's robustness to physical scale variations by running
full-volume sliding window inference on a specific crop across different
EM scales (s0, s1, s2).

Outputs a CSV containing False Positive Rates (FPR), IoU, and Dice
across all tested scales, allowing for Robustness curves.
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from code.utils.config import load_config
from code.utils.paths import setup_run_directory
from code.data.zarr_utils import extract_aligned_volumes, get_scale_trans
from code.models.unet import build_model, get_raw_model
from code.evaluation.visualize import run_sliding_window_inference
from code.evaluation.evaluate_detailed import compute_per_class_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="jrc_cos7-1a")
    parser.add_argument("--crop", type=str, default="crop234")
    parser.add_argument("--scales", nargs="+", default=["s0", "s1", "s2"],
                        help="List of scales to evaluate (e.g. s0 s1 s2)")
    parser.add_argument("--gpu", type=str, default=None)
    return parser.parse_args()


def compute_stratified_metrics(pred, target, num_classes):
    """Computes IoU, Dice, and FPR per class."""
    metrics = {"iou": {}, "dice": {}, "fpr": {}, "precision": {}, "recall": {}}
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        tp = np.logical_and(pred_c, target_c).sum()
        fp = np.logical_and(pred_c, ~target_c).sum()
        fn = np.logical_and(~pred_c, target_c).sum()
        tn = np.logical_and(~pred_c, ~target_c).sum()

        eps = 1e-8
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        fpr = fp / (fp + tn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        metrics["iou"][c] = float(iou)
        metrics["dice"][c] = float(dice)
        metrics["fpr"][c] = float(fpr)
        metrics["precision"][c] = float(precision)
        metrics["recall"][c] = float(recall)
    return metrics


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    run_paths = setup_run_directory(config, config_path=args.config)
    dataset_base = os.path.join(config.paths.data_dir, args.dataset, f"{args.dataset}.zarr")

    model, device = build_model(config, multi_gpu=False)
    raw_model = get_raw_model(model)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raw_model.load_state_dict(checkpoint)
    model.eval()

    out_csv = os.path.join(run_paths["results_dir"], f"stratified_metrics_{args.dataset}_{args.crop}.csv")
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scale", "Resolution_nm", "Class", "IoU", "Dice", "FPR", "Precision", "Recall"])

        for scale in args.scales:
            print(f"\n--- Evaluating at scale: {scale} ---")
            em_volume, lbl_volume = extract_aligned_volumes(dataset_base, args.crop, em_scale=scale)

            # Get physical resolution for this scale
            em_base = os.path.join(dataset_base, "recon-1", "em", "fibsem-uint8")
            scale_em, _ = get_scale_trans(em_base, scale)
            print(f"Physical resolution: {scale_em}")

            # Remap labels
            max_raw_id = max(256, int(lbl_volume.max()) + 1)
            label_lookup = np.zeros(max_raw_id, dtype=np.int64)
            for semantic_id, instance_id in config.label_map.items():
                if semantic_id < max_raw_id:
                    label_lookup[semantic_id] = instance_id
            lbl_remapped = label_lookup[lbl_volume]

            # Model wrapper for scale conditioning
            if getattr(config.model, "scale_conditioned", False):
                class ConditionedWrapper(torch.nn.Module):
                    def __init__(self, base_model, res):
                        super().__init__()
                        self.base_model = base_model
                        self.res = res
                    def forward(self, x):
                        b, c, z, y, x_d = x.shape
                        z_c = torch.full((b, 1, z, y, x_d), self.res[0], device=x.device, dtype=x.dtype)
                        y_c = torch.full((b, 1, z, y, x_d), self.res[1], device=x.device, dtype=x.dtype)
                        x_c = torch.full((b, 1, z, y, x_d), self.res[2], device=x.device, dtype=x.dtype)
                        return self.base_model(torch.cat([x, z_c, y_c, x_c], dim=1))
                infer_model = ConditionedWrapper(model, scale_em)
            else:
                infer_model = model

            # Run inference
            pred_volume = run_sliding_window_inference(
                model=infer_model,
                em_volume=em_volume,
                device=device,
                roi_size=(config.data.patch_dim,) * 3,
                sw_batch_size=1,
                overlap=0.5,
            )

            # Upsample prediction to s0 label shape
            print(f"Resampling prediction {pred_volume.shape} to {lbl_remapped.shape}...")
            zoom_factors = [l / p for l, p in zip(lbl_remapped.shape, pred_volume.shape)]
            pred_upsampled = ndi.zoom(pred_volume, zoom_factors, order=0)

            # Compute metrics
            metrics = compute_stratified_metrics(pred_upsampled, lbl_remapped, config.data.num_classes)

            # Write to CSV
            for c in range(1, config.data.num_classes):
                name = config.class_names.get(c, f"Class_{c}")
                writer.writerow([
                    scale, 
                    str(list(scale_em)), 
                    name, 
                    f"{metrics['iou'][c]:.4f}", 
                    f"{metrics['dice'][c]:.4f}", 
                    f"{metrics['fpr'][c]:.6f}", 
                    f"{metrics['precision'][c]:.4f}", 
                    f"{metrics['recall'][c]:.4f}"
                ])
                f.flush()

    print(f"\nStratified evaluation complete. Results saved to {out_csv}")

if __name__ == "__main__":
    main()
