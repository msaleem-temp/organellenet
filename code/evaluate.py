"""
OrganelleNet — Evaluation Entry Point

Usage:
    python code/evaluate.py --config configs/static_unet.yaml \
        --checkpoint runs/static-unet-13cls-nojitter/ckpts/best_model.pth
"""

import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from code.utils.config import load_config
from code.utils.paths import setup_run_directory
from code.data.zarr_utils import build_zarr_map
from code.data.dataset import PatchDataset
from code.models.unet import build_model, get_raw_model
from code.evaluation.evaluator import run_evaluation, save_metrics_to_file


def parse_args():
    parser = argparse.ArgumentParser(description="OrganelleNet Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index")
    parser.add_argument("--test-json", type=str, default=None, help="Override test JSON path")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. Load config
    config = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"Evaluation: {config.experiment_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}")

    # 2. Setup run directory (to locate output paths)
    run_paths = setup_run_directory(config, config_path=args.config)

    # 3. Build zarr map and test dataset
    zarr_map = build_zarr_map(config.paths.data_dir)

    test_json = args.test_json
    if test_json is None:
        # Default: look in the run's splits directory
        test_json = os.path.join(run_paths["run_dir"], "splits", "test.json")

    if not os.path.exists(test_json):
        print(f"Error: Test JSON not found at {test_json}")
        print("Run training first or provide --test-json explicitly.")
        sys.exit(1)

    test_dataset = PatchDataset(
        json_path=test_json,
        zarr_map=zarr_map,
        label_map=config.label_map,
        patch_dim=config.data.patch_dim,
        max_jitter=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=config.training.num_workers
    )

    # 4. Build model and load weights
    model, device = build_model(config, multi_gpu=False)
    raw_model = get_raw_model(model)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Handle both raw state_dict and full checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raw_model.load_state_dict(checkpoint)
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint}")

    # 5. Run evaluation
    metrics = run_evaluation(
        model=model,
        test_loader=test_loader,
        num_classes=config.data.num_classes,
        device=device,
        class_names=config.class_names,
    )

    # 6. Save results
    output_path = os.path.join(run_paths["results_dir"], "test_metrics.txt")
    save_metrics_to_file(metrics, config.class_names, output_path)


if __name__ == "__main__":
    main()
