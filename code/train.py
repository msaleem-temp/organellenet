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
from code.data.sampler import create_balanced_sampler, sampler_test
from code.data.splits import prepare_splits
from code.data.splits import split_handler
from code.models.unet import build_model
from code.training.losses import build_loss
from code.training.trainer import Trainer

from code.utils.plot import plot_slice


def parse_args():
    parser = argparse.ArgumentParser(description="OrganelleNet Training")
    
    # Existing arguments
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index")
    parser.add_argument("--dry-run", action="store_true", help="Validate and exit")
    parser.add_argument("--name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--patch-dim", type=int, default=None, help="Override patch dimension")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--resume-log", type=str, default=None, help="Path to historical CSV")
    
    # New Standardized CLI Overrides
    parser.add_argument("--model", type=str, default=None, help="Model architecture (e.g., unet_2d)")
    parser.add_argument("--loss", type=str, default=None, help="Loss function (e.g., dice_ce, tvbce)")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Optimizer weight decay")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load config
    config = load_config(args.config)
    
    # 2. Map CLI Overrides safely to the config object
    if args.name is not None:
        config.experiment_name = args.name
    if args.patch_dim is not None:
        config.data.patch_dim = args.patch_dim
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.model is not None:
        config.model.name = args.model
    if args.loss is not None:
        config.training.loss = args.loss
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.workers is not None:
        config.training.num_workers = args.workers


    print(f"Name: {args.name}")

    sys.exit(0)
    print(f"\n{'='*60}")
    print(f"Experiment: {config.experiment_name}")
    print(f"{'='*60}")

    # 2. Setup run directory
    run_paths = setup_run_directory(config, config_path=args.config)
    print(f"Run directory: {run_paths['run_dir']}")

    # 3. Prepare data splits
    blueprint_path = os.path.join(config.paths.json_dir, config.data.blueprint_json)
    split_output_dir = os.path.join(run_paths["run_dir"], "splits")


    split_paths = split_handler(
            blueprint_json_path=blueprint_path,
            output_dir=split_output_dir,
    )


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
    print("main function start from here...")
    main()
