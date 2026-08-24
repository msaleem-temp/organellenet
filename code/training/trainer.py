"""
Unified training loop for OrganelleNet.

Supports:
- Mixed precision (AMP) with GradScaler
- Gradient accumulation
- Early stopping on val_loss or val_dice
- MONAI DiceMetric for Dice-based validation
- Checkpoint resume (full optimizer + scheduler + scaler state)
- CSV logging per epoch
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from monai.metrics import DiceMetric
from tqdm import tqdm
from monai.transforms import AsDiscrete

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from code.models.unet import get_raw_model


class Trainer:
    """
    Config-driven training engine for 3D segmentation models.

    Parameters
    ----------
    model : nn.Module
        The segmentation model (potentially wrapped in DataParallel).
    criterion : nn.Module
        Loss function.
    config : ExperimentConfig
        Full experiment configuration.
    run_paths : dict
        Output path dictionary from setup_run_directory().
    device : torch.device
        Target device.
    """

    def __init__(self, model, criterion, config, run_paths, device):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.run_paths = run_paths
        self.device = device

        tc = config.training

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay
        )
        self.scaler = GradScaler()

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=tc.scheduler_mode,
            factor=tc.scheduler_factor,
            patience=tc.scheduler_patience,
        )

        # Early stopping
        self.early_stop_metric = tc.early_stop_metric
        self.early_stop_patience = tc.early_stop_patience
        self.early_stop_counter = 0

        if self.early_stop_metric == "val_dice":
            self.best_metric = -1.0   # maximize
        else:
            self.best_metric = float("inf")   # minimize

        # Gradient accumulation
        self.accumulation_steps = tc.accumulation_steps

        # Dice metric (used when early_stop_metric == "val_dice")
        self.use_dice_metric = (self.early_stop_metric == "val_dice")
        if self.use_dice_metric:
            self.dice_metric = DiceMetric(include_background=False, reduction="mean")
            self.post_pred = AsDiscrete(argmax=True, to_onehot=config.model.out_channels)
            self.post_label = AsDiscrete(to_onehot=config.model.out_channels)

        # Epoch tracking
        self.start_epoch = 0
        self.num_epochs = tc.num_epochs
        self.print_freq = tc.print_freq

        # Paths
        self.log_csv_path = run_paths["log_csv_path"]
        self.best_model_path = run_paths["best_model_path"]
        self.resume_ckpt_path = run_paths["resume_ckpt_path"]

    def try_resume(self):
        """Attempt to resume from a checkpoint if it exists."""
        if os.path.exists(self.resume_ckpt_path):
            print(f"Resuming from checkpoint: {self.resume_ckpt_path}")
            checkpoint = torch.load(self.resume_ckpt_path, map_location=self.device)

            raw_model = get_raw_model(self.model)
            raw_model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.start_epoch = checkpoint.get("epoch", 0) + 1
            self.best_metric = checkpoint.get("best_metric", self.best_metric)
            self.early_stop_counter = checkpoint.get("early_stop_counter", 0)

            print(f"Resumed at epoch {self.start_epoch} | Best metric: {self.best_metric:.4f}")
        else:
            print("No checkpoint found. Starting from scratch.")
            # Initialize the CSV log header
            header = "epoch,lr,train_loss,val_loss"
            if self.use_dice_metric:
                header += ",val_dice"
            header += "\n"
            with open(self.log_csv_path, "w") as f:
                f.write(header)

    def _save_resume_checkpoint(self, epoch):
        """Save full training state for resumption."""
        raw_model = get_raw_model(self.model)
        state = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "early_stop_counter": self.early_stop_counter,
        }
        torch.save(state, self.resume_ckpt_path)

    def _save_best_model(self):
        """Save only the model weights for the best epoch."""
        raw_model = get_raw_model(self.model)
        torch.save(raw_model.state_dict(), self.best_model_path)

    def _is_improvement(self, current_metric):
        """Check if current metric is an improvement over the best."""
        if self.early_stop_metric == "val_dice":
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric

    def train(self, train_loader, val_loader):
        """
        Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        """
        self.try_resume()

        for epoch in range(self.start_epoch, self.num_epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n=== Epoch [{epoch + 1}/{self.num_epochs}] | LR: {current_lr:.2e} ===")

            # --- PHASE 1: TRAINING ---
            avg_train_loss = self._train_one_epoch(train_loader, epoch)

            # --- PHASE 2: VALIDATION ---
            avg_val_loss, mean_dice = self._validate_one_epoch(val_loader, epoch)

            # --- Scheduler step ---
            if self.use_dice_metric:
                self.scheduler.step(mean_dice)
                metric_value = mean_dice
                metric_name = "Val Dice"
            else:
                self.scheduler.step(avg_val_loss)
                metric_value = avg_val_loss
                metric_name = "Val Loss"

            # --- Epoch summary ---
            summary = (
                f"Epoch [{epoch + 1}/{self.num_epochs}] Summary | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )
            if self.use_dice_metric:
                summary += f" | Val Dice: {mean_dice:.4f}"
            print(summary)

            # --- CSV logging ---
            log_line = f"{epoch + 1},{current_lr:.2e},{avg_train_loss:.4f},{avg_val_loss:.4f}"
            if self.use_dice_metric:
                log_line += f",{mean_dice:.4f}"
            log_line += "\n"
            with open(self.log_csv_path, "a") as f:
                f.write(log_line)

            # --- PHASE 3: CHECKPOINTING & EARLY STOPPING ---
            self._save_resume_checkpoint(epoch)

            if self._is_improvement(metric_value):
                print(
                    f">>> {metric_name} improved from {self.best_metric:.4f} "
                    f"to {metric_value:.4f}. Saving best model."
                )
                self.best_metric = metric_value
                self._save_best_model()
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                print(
                    f">>> {metric_name} did not improve. "
                    f"Early stopping counter: {self.early_stop_counter}/{self.early_stop_patience}"
                )

            if self.early_stop_counter >= self.early_stop_patience:
                print(
                    f"\nEarly stopping triggered. {metric_name} has not improved "
                    f"for {self.early_stop_patience} consecutive epochs."
                )
                break

        print("\nTraining complete.")

    def _train_one_epoch(self, train_loader, epoch):
        """Run one training epoch."""
        self.model.train()
        train_epoch_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{self.num_epochs}", leave=False)
        for step, (em_batch, lbl_batch) in enumerate(pbar):
            em_batch = em_batch.to(self.device)
            if lbl_batch.dim() == 4:
                lbl_batch = lbl_batch.unsqueeze(1)
            lbl_batch = lbl_batch.to(self.device, dtype=torch.long)

            with autocast():
                outputs = self.model(em_batch)
                loss = self.criterion(outputs, lbl_batch)
                if self.accumulation_steps > 1:
                    loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            actual_loss = loss.item() * (self.accumulation_steps if self.accumulation_steps > 1 else 1)
            train_epoch_loss += actual_loss

            if (step + 1) % self.print_freq == 0:
                pbar.set_postfix(loss=f"{actual_loss:.4f}")

        return train_epoch_loss / len(train_loader)

    def _validate_one_epoch(self, val_loader, epoch):
        """Run one validation epoch."""
        self.model.eval()
        val_epoch_loss = 0.0
        mean_dice = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}/{self.num_epochs}", leave=False)
            for step, (em_batch, lbl_batch) in enumerate(pbar):
                em_batch = em_batch.to(self.device)
                if lbl_batch.dim() == 4:
                    lbl_batch = lbl_batch.unsqueeze(1)
                lbl_batch = lbl_batch.to(self.device, dtype=torch.long)

                with autocast():
                    outputs = self.model(em_batch)
                    val_loss = self.criterion(outputs, lbl_batch)

                val_epoch_loss += val_loss.item()

                # Dice metric accumulation (if enabled)
                if self.use_dice_metric:
                    val_outputs = [self.post_pred(i) for i in outputs]
                    val_labels = [self.post_label(i) for i in lbl_batch]
                    self.dice_metric(y_pred=val_outputs, y=val_labels)

                if (step + 1) % self.print_freq == 0:
                    pbar.set_postfix(loss=f"{val_loss.item():.4f}")

        avg_val_loss = val_epoch_loss / len(val_loader)

        if self.use_dice_metric:
            mean_dice = self.dice_metric.aggregate().item()
            self.dice_metric.reset()

        return avg_val_loss, mean_dice
