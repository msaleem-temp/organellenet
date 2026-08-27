"""
OrganelleNet — Full-Crop Sliding Window Inference Entry Point

Usage:
    python code/infer.py --config configs/static_unet.yaml \
        --checkpoint runs/static-unet-13cls-nojitter/ckpts/best_model.pth \
        --dataset jrc_cos7-1a --crop crop234 --z-slice 70
"""

import os
import sys
import argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from code.utils.config import load_config
from code.utils.paths import setup_run_directory
from code.data.zarr_utils import extract_aligned_volumes, get_scale_trans
from code.models.unet import build_model, get_raw_model
from code.evaluation.visualize import run_sliding_window_inference, plot_inference_results


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


def parse_args():
    parser = argparse.ArgumentParser(description="OrganelleNet Sliding Window Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--dataset", type=str, default="jrc_cos7-1a", help="Dataset name")
    parser.add_argument("--crop", type=str, default="crop234", help="Crop ID")
    parser.add_argument("--z-slice", type=int, default=70, help="Z-slice for visualization")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap")
    parser.add_argument("--name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--patch-dim", type=int, default=None, help="Override patch dim")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. Load config
    config = load_config(args.config)
    
    if args.name:
        config.experiment_name = args.name
    if args.patch_dim:
        config.data.patch_dim = args.patch_dim
    print(f"\n{'='*60}")
    print(f"Inference: {config.experiment_name}")
    print(f"Dataset: {args.dataset} | Crop: {args.crop}")
    print(f"{'='*60}")

    run_paths = setup_run_directory(config, config_path=args.config)

    # 2. Build model and load weights
    model, device = build_model(config, multi_gpu=False)
    raw_model = get_raw_model(model)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raw_model.load_state_dict(checkpoint)
    model.eval()

    # 3. Extract aligned volumes
    dataset_base = os.path.join(
        config.paths.data_dir, args.dataset, f"{args.dataset}.zarr"
    )
    em_volume, lbl_volume = extract_aligned_volumes(dataset_base, args.crop)
    em_base = os.path.join(dataset_base, "recon-1", "em", "fibsem-uint8")
    scale_em, _ = get_scale_trans(em_base, "s0")

    # 4. Remap labels
    max_raw_id = max(256, int(lbl_volume.max()) + 1)
    label_lookup = np.zeros(max_raw_id, dtype=np.int64)
    for semantic_id, instance_id in config.label_map.items():
        if semantic_id < max_raw_id:
            label_lookup[semantic_id] = instance_id
    lbl_remapped = label_lookup[lbl_volume]

    print(f"Unique classes after remapping: {np.unique(lbl_remapped)}")

    # 5. Run sliding window inference
    infer_model = model
    if getattr(config.model, "scale_conditioned", False):
        infer_model = ConditionedResolutionWrapper(model, scale_em)

    pred_volume = run_sliding_window_inference(
        model=infer_model,
        em_volume=em_volume,
        device=device,
        roi_size=(config.data.patch_dim,) * 3,
        sw_batch_size=1,
        overlap=args.overlap,
    )

    # 6. Visualize
    output_path = os.path.join(
        run_paths["plots_dir"],
        f"inference_{args.dataset}_{args.crop}_z{args.z_slice}.png",
    )
    plot_inference_results(
        em_volume=em_volume,
        lbl_volume=lbl_remapped,
        pred_volume=pred_volume,
        class_names=config.class_names,
        z_slice=args.z_slice,
        num_classes=config.data.num_classes,
        output_path=output_path,
    )

    print(f"\nInference complete. Results saved to: {run_paths['plots_dir']}")


if __name__ == "__main__":
    main()
