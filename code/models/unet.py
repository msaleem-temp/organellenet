"""
Model factory for OrganelleNet.

Builds MONAI UNet models from an ExperimentConfig, with optional
multi-GPU DataParallel wrapping.
"""

import sys
import os
import torch
import torch.nn as nn
from monai.networks.nets import UNet

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def build_model(config, device=None, multi_gpu=True):
    """
    Build a MONAI 3D UNet from an ExperimentConfig.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing model hyperparameters.
    device : torch.device, optional
        Target device. If None, auto-detects CUDA.
    multi_gpu : bool
        If True and multiple GPUs are available, wraps with DataParallel.

    Returns
    -------
    tuple of (nn.Module, torch.device)
        The model and the device it was moved to.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mc = config.model

    model = UNet(
        spatial_dims=mc.spatial_dims,
        in_channels=mc.in_channels,
        out_channels=mc.out_channels,
        channels=tuple(mc.channels),
        strides=tuple(mc.strides),
        kernel_size=mc.kernel_size,
        up_kernel_size=mc.up_kernel_size,
        num_res_units=mc.num_res_units,
        act=mc.act,
        norm=mc.norm,
    ).to(device)

    num_gpus = torch.cuda.device_count()
    print(f"Model built: {mc.out_channels}-class UNet | Device: {device} | GPUs: {num_gpus}")

    if multi_gpu and num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Wrapped model with DataParallel across {num_gpus} GPUs.")

    return model, device


def get_raw_model(model):
    """Unwrap DataParallel if present to access the raw model."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model
