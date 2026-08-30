"""
OrganelleNet — Training Entry Point

Usage:
    python code/train.py --config configs/static_unet.yaml
    python code/train.py --config configs/static_unet.yaml --gpu 0
    python code/train.py --config configs/static_unet.yaml --dry-run
"""

import os
import sys
import argparse

# Ensure project root is on sys.path for cross-module imports
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
from code.data.sampler import create_balanced_sampler
from code.data.splits import prepare_splits
from code.models.unet import build_model
from code.training.losses import build_loss
from code.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="OrganelleNet Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index (e.g., '0' or '0,1')")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit without training")
    parser.add_argument("--name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--patch-dim", type=int, default=None, help="Override patch dimension")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    # GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 1. Load config
    config = load_config(args.config)
    
    # CLI Overrides
    if args.name is not None:
        config.experiment_name = args.name
    if args.patch_dim is not None:
        config.data.patch_dim = args.patch_dim
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    print(f"\n{'='*60}")
    print(f"Experiment: {config.experiment_name}")
    print(f"{'='*60}")

    # 2. Setup run directory
    run_paths = setup_run_directory(config, config_path=args.config)
    print(f"Run directory: {run_paths['run_dir']}")

    # 3. Prepare data splits
    blueprint_path = os.path.join(config.paths.json_dir, config.data.blueprint_json)
    split_output_dir = os.path.join(run_paths["run_dir"], "splits")

    split_paths = prepare_splits(
        blueprint_json_path=blueprint_path,
        output_dir=split_output_dir,
        target_classes=config.data.target_classes,
        split_ratios=config.data.split_ratios,
        excluded_crop=config.data.excluded_crop,
        seed=config.data.seed,
    )
    print("\n[TEST] Splits prepared successfully. Halting execution for inspection.")
    sys.exit(0)

    # 4. Build zarr map
    zarr_map = build_zarr_map(config.paths.data_dir)
    print(f"Zarr map built: {len(zarr_map)} datasets found")

    # 5. Create datasets
    train_dataset = PatchDataset(
        json_path=split_paths["train_path"],
        zarr_map=zarr_map,
        label_map=config.label_map,
        patch_dim=config.data.patch_dim,
        max_jitter=config.data.max_jitter,
        augmentation_config=config.augmentation,
        target_type=config.data.target_type,
        num_classes=config.data.num_classes,
        scale_conditioned=config.model.scale_conditioned,
    )
    val_dataset = PatchDataset(
        json_path=split_paths["val_path"],
        zarr_map=zarr_map,
        label_map=config.label_map,
        patch_dim=config.data.patch_dim,
        max_jitter=0,  # Always static for validation
        target_type=config.data.target_type,
        num_classes=config.data.num_classes,
        scale_conditioned=config.model.scale_conditioned,
    )

    # 6. Create data loaders
    train_sampler = create_balanced_sampler(
        train_dataset,
        balance_level=config.training.sampler_balance_level,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 7. Build model and loss
    model, device = build_model(config)
    criterion = build_loss(config, device)

    # 7.1. Log model parameters
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {model_params:,}")
    params_log_path = os.path.join(run_paths["results_dir"], "model_parameters.txt")
    with open(params_log_path, "w") as f:
        f.write(f"Experiment: {config.experiment_name}\n")
        f.write(f"Trainable Parameters: {model_params:,}\n")

    if args.dry_run:
        print("\n[DRY RUN] Config parsed, directories created, model instantiated.")
        print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print("[DRY RUN] Exiting without training.")
        return

    # 8. Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        config=config,
        run_paths=run_paths,
        device=device,
    )
    trainer.train(train_loader, val_loader)

    # 9. Plot training curves
    from code.evaluation.visualize import plot_training_curves

    plot_output = os.path.join(run_paths["plots_dir"], "loss_curve.png")
    try:
        plot_training_curves(
            log_csv_path=run_paths["log_csv_path"],
            output_path=plot_output,
            title=f"{config.experiment_name}: Training & Validation Loss",
        )
    except Exception as e:
        print(f"Warning: Could not generate training plot: {e}")

    print(f"\nTraining session complete. All outputs saved to: {run_paths['run_dir']}")


if __name__ == "__main__":
    main()
