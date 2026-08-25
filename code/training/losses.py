"""
Loss function factory for OrganelleNet.
"""

import sys
import os
import torch
import torch.nn as nn
from monai.losses import DiceCELoss

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class BoundaryWeightedMSELoss(nn.Module):
    def __init__(self, boundary_weight=10.0, threshold=0.95):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.threshold = threshold

    def forward(self, pred, target):
        mse = torch.nn.functional.mse_loss(pred, target, reduction='none')
        weight_mask = torch.ones_like(target)
        weight_mask[torch.abs(target) < self.threshold] = self.boundary_weight
        return (mse * weight_mask).mean()


def build_loss(config, device=None):
    """
    Build a Loss function from an ExperimentConfig.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    device : torch.device, optional
        Device to place the class weight tensor on.

    Returns
    -------
    nn.Module
        The loss function.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_type = getattr(config.training, "loss_type", "dice_ce")

    if loss_type == "mse":
        criterion = BoundaryWeightedMSELoss(boundary_weight=10.0, threshold=0.95)
        print(f"Loss: BoundaryWeightedMSELoss (Weight: 10.0 | for SDT Regression)")
        return criterion
    elif loss_type == "smooth_l1":
        criterion = torch.nn.SmoothL1Loss()
        print(f"Loss: SmoothL1Loss (for Regression)")
        return criterion
    else:
        weights = torch.tensor(config.training.class_weights, dtype=torch.float32).to(device)

        criterion = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            weight=weights,
        )

        print(f"Loss: DiceCELoss | Class weights: {len(config.training.class_weights)} classes")
        return criterion
