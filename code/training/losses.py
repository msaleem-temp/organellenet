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
    Build a DiceCELoss from an ExperimentConfig.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    device : torch.device, optional
        Device to place the class weight tensor on.

    Returns
    -------
    DiceCELoss
        The loss function with class weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = torch.tensor(config.training.class_weights, dtype=torch.float32).to(device)

    criterion = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        weight=weights,
    )

    print(f"Loss: DiceCELoss | Class weights: {len(config.training.class_weights)} classes")
    return criterion
