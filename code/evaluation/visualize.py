"""
Visualization utilities for OrganelleNet.

Provides training curve plotting, sliding window inference, and
3-panel EM/GT/Prediction visualization with dynamic colormaps.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
from monai.inferers import SlidingWindowInferer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def plot_training_curves(log_csv_path, output_path=None, title=None):
    """
    Plot training and validation loss curves from a CSV log file.

    Parameters
    ----------
    log_csv_path : str
        Path to the training_log.csv file.
    output_path : str, optional
        If provided, saves the plot to this path instead of showing.
    title : str, optional
        Plot title. Defaults to 'Training & Validation Loss Dynamics'.
    """
    if not os.path.exists(log_csv_path):
        raise FileNotFoundError(f"Log file not found: {log_csv_path}")

    df = pd.read_csv(log_csv_path)

    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor="white")

    # Plot losses
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue", linewidth=2)
    ax1.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="orange", linewidth=2)

    # Plot Dice if present
    if "val_dice" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df["epoch"], df["val_dice"], label="Val Dice", color="green", linewidth=2, linestyle="--")
        ax2.set_ylabel("Dice Score", fontsize=12, color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.legend(loc="upper left", fontsize=10)

    # LR decay markers
    lr_changes = df[df["lr"].diff() < 0]
    for _, row in lr_changes.iterrows():
        ax1.axvline(
            x=row["epoch"],
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"LR Decay (-> {row['lr']:.2e})",
        )

    if title is None:
        title = "Training & Validation Loss Dynamics"
    ax1.set_title(title, fontsize=14, pad=15)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (DiceCE)", fontsize=12)

    max_epoch = int(df["epoch"].max())
    ax1.set_xticks(range(0, max_epoch + 1, max(1, max_epoch // 10)))

    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Training curve saved to {output_path}")
        plt.close()
    else:
        plt.show()


def run_sliding_window_inference(
    model, em_volume, device, roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.5
):
    """
    Run sliding window inference on a full 3D EM volume.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    em_volume : np.ndarray
        3D EM volume (uint8).
    device : torch.device
        Target device.
    roi_size : tuple
        Patch size for sliding window.
    sw_batch_size : int
        Batch size within the sliding window.
    overlap : float
        Overlap fraction between windows.

    Returns
    -------
    np.ndarray
        Predicted 3D segmentation map.
    """
    # Normalize to [0, 1]
    em_normalized = em_volume.astype(np.float32) / 255.0

    # Add batch + channel dims: [1, 1, Z, Y, X]
    em_tensor = torch.tensor(em_normalized).unsqueeze(0).unsqueeze(0).to(device)

    inferer = SlidingWindowInferer(
        roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap
    )

    print("Executing sliding window inference...")
    model.eval()
    with torch.no_grad():
        logits = inferer(inputs=em_tensor, network=model)
        predicted = torch.argmax(logits, dim=1)

    pred_3d = predicted.squeeze(0).cpu().numpy()
    print(f"Inference complete. Output shape: {pred_3d.shape}")

    return pred_3d


def plot_inference_results(
    em_volume,
    lbl_volume,
    pred_volume,
    class_names,
    z_slice=70,
    num_classes=13,
    output_path=None,
):
    """
    Plot a 3-panel comparison: EM image, ground truth, model prediction.

    Parameters
    ----------
    em_volume : np.ndarray
        3D EM volume.
    lbl_volume : np.ndarray
        3D remapped ground truth labels.
    pred_volume : np.ndarray
        3D predicted labels.
    class_names : dict
        Mapping from class index to name.
    z_slice : int
        Z-slice to visualize.
    num_classes : int
        Total number of classes.
    output_path : str, optional
        If provided, saves the plot instead of showing.
    """
    em_slice = em_volume[z_slice, :, :]
    lbl_slice = lbl_volume[z_slice, :, :]
    pred_slice = pred_volume[z_slice, :, :]

    # Build custom colormap
    base_colors = plt.get_cmap("tab20").colors
    foreground_colors = list(base_colors[: num_classes - 1])
    custom_colors = [(0.0, 0.0, 0.0, 1.0)] + foreground_colors
    custom_cmap = mcolors.ListedColormap(custom_colors)

    vmin = 0
    vmax = num_classes - 1

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(em_slice, cmap="gray")
    axes[0].set_title(f"EM Image (Z={z_slice})", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(lbl_slice, cmap=custom_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title("Ground Truth", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(pred_slice, cmap=custom_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[2].set_title("Model Prediction", fontsize=14)
    axes[2].axis("off")

    # Dynamic legend
    unique_classes = np.unique(np.concatenate([lbl_volume.flatten(), pred_volume.flatten()]))
    legend_patches = []
    for cid in unique_classes:
        cid = int(cid)
        if cid < len(custom_colors):
            color = custom_colors[cid]
            label_text = class_names.get(cid, f"Class {cid}")
            legend_patches.append(mpatches.Patch(color=color, label=f"{cid}: {label_text}"))

    fig.legend(
        handles=legend_patches,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=7,
        title="Classes Present in Full 3D Crop",
        fontsize=11,
        title_fontsize=13,
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Inference plot saved to {output_path}")
        plt.close()
    else:
        plt.show()
