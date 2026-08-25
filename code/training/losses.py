"""
Loss function factory for OrganelleNet.
"""

import sys
import os
import torch
from monai.losses import DiceCELoss

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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
        criterion = torch.nn.MSELoss()
        print(f"Loss: MSELoss (for Regression)")
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
