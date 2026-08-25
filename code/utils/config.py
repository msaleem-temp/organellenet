"""
Configuration system for OrganelleNet experiments.

Loads YAML config files with inheritance support (via 'inherits' key).
Merges child config on top of parent defaults to produce a flat ExperimentConfig.
"""

import os
import sys
import copy
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Project root resolution — allows imports from anywhere in the repo
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Default semantic → instance label mapping (shared across all baselines)
# ---------------------------------------------------------------------------
LABEL_MAP_13CLS = {
    3: 1, 4: 1, 5: 1,                                          # 1. Mitochondria
    8: 2, 9: 2,                                                 # 2. Vesicles
    10: 3, 11: 3,                                               # 3. Endosomes
    12: 4, 13: 4,                                               # 4. Lysosomes
    14: 5, 15: 5,                                               # 5. Lipid Droplets
    20: 6, 21: 6, 24: 6, 25: 6, 26: 6, 27: 6, 28: 6, 29: 6,   # 6. Nucleus
    22: 7, 23: 7,                                               # 7. Nuclear Pores
    30: 8, 36: 8,                                               # 8. Microtubules
    47: 9, 48: 9,                                               # 9. Peroxisomes
    6: 10, 7: 10,                                               # 10. Golgi Apparatus
    16: 11, 17: 11,                                             # 11. Endoplasmic Reticulum
    18: 12, 19: 12,                                             # 12. ER Exit Sites
}

LABEL_MAP_14CLS = {
    **LABEL_MAP_13CLS,
    38: 13,                                                     # 13. Vimentin
}

CLASS_NAMES_13 = {
    0: "Background", 1: "Mitochondria", 2: "Vesicles", 3: "Endosomes",
    4: "Lysosomes", 5: "Lipid Droplets", 6: "Nucleus", 7: "Nuclear Pores",
    8: "Microtubules", 9: "Peroxisomes", 10: "Golgi", 11: "ER", 12: "ERES",
}

CLASS_NAMES_14 = {
    **CLASS_NAMES_13,
    13: "Vimentin",
}

# Default class weights for the DiceCE loss (background-inclusive vector)
DEFAULT_WEIGHTS_13 = [
    0.8419, 1.0428, 0.8653, 0.6660, 1.4594,
    0.3458, 0.6495, 1.6585, 0.7208, 0.1888,
    0.7276, 1.5408, 1.5923,
]

DEFAULT_WEIGHTS_14 = DEFAULT_WEIGHTS_13 + [1.7003]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    main_dir: str = "/mnt/graid/codebases_other/cellmap-segmentation-challenge/saleem/organellenet"
    data_dir: str = "/mnt/graid/codebases_other/cellmap-segmentation-challenge/data"
    json_dir: str = ""   # auto-resolved

    def __post_init__(self):
        if not self.json_dir:
            self.json_dir = os.path.join(self.main_dir, "all_jsons")


@dataclass
class DataConfig:
    patch_dim: int = 128
    max_jitter: int = 0
    seed: int = 42
    num_classes: int = 13
    target_type: str = "labels"
    target_classes: List[str] = field(default_factory=lambda: [
        "endo", "ld", "lyso", "mito", "mt", "np", "nuc", "perox", "ves",
        "golgi", "er", "eres",
    ])
    blueprint_json: str = "static_dynamic_baseline.json"
    split_ratios: List[float] = field(default_factory=lambda: [0.80, 0.10, 0.10])
    excluded_crop: Optional[str] = None


@dataclass
class ModelConfig:
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 13
    scale_conditioned: bool = False
    channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    kernel_size: int = 3
    up_kernel_size: int = 3
    num_res_units: int = 2
    act: str = "PRELU"
    norm: str = "INSTANCE"


@dataclass
class TrainingConfig:
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_mode: str = "min"     # 'min' for loss, 'max' for dice
    early_stop_patience: int = 12
    early_stop_metric: str = "val_loss"   # 'val_loss' or 'val_dice'
    loss_type: str = "dice_ce"
    accumulation_steps: int = 1
    print_freq: int = 1
    num_workers: int = 2
    class_weights: Optional[List[float]] = None   # if None, uses defaults


@dataclass
class AugmentationConfig:
    """Configuration for the 3D EM augmentation pipeline."""
    enabled: bool = False
    # Geometric
    flip: bool = True
    rotate90: bool = True
    # Intensity (EM only)
    intensity_scale_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    intensity_shift_range: List[float] = field(default_factory=lambda: [-0.1, 0.1])
    gaussian_noise_std: float = 0.03
    gamma_range: List[float] = field(default_factory=lambda: [0.7, 1.5])
    # Elastic deformation
    elastic_enabled: bool = True
    elastic_alpha: float = 80.0
    elastic_sigma: float = 10.0
    elastic_prob: float = 0.3
    # Resolution augmentation (novel)
    resolution_aug_enabled: bool = False
    resolution_scale_range: List[float] = field(default_factory=lambda: [0.25, 1.0])
    resolution_aug_prob: float = 0.4


@dataclass
class ExperimentConfig:
    experiment_name: str = "unnamed-experiment"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Derived helpers (populated after load)
    label_map: Dict[int, int] = field(default_factory=dict, repr=False)
    class_names: Dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Auto-select label map and class names based on num_classes
        if not self.label_map:
            self.label_map = LABEL_MAP_14CLS if self.data.num_classes == 14 else LABEL_MAP_13CLS
        if not self.class_names:
            self.class_names = CLASS_NAMES_14 if self.data.num_classes == 14 else CLASS_NAMES_13
        # Auto-select class weights if not provided
        if self.training.class_weights is None:
            self.training.class_weights = (
                DEFAULT_WEIGHTS_14 if self.data.num_classes == 14 else DEFAULT_WEIGHTS_13
            )
        # Auto-set scheduler mode from early_stop_metric
        if self.training.early_stop_metric == "val_dice":
            self.training.scheduler_mode = "max"
        else:
            self.training.scheduler_mode = "min"


# ---------------------------------------------------------------------------
# YAML Loading with Inheritance
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _dict_to_config(raw: dict) -> ExperimentConfig:
    """Convert a flat merged dict into an ExperimentConfig dataclass."""
    cfg = ExperimentConfig(
        experiment_name=raw.get("experiment_name", "unnamed-experiment"),
        paths=PathsConfig(**raw.get("paths", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        augmentation=AugmentationConfig(**raw.get("augmentation", {})),
    )
    return cfg


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load a YAML config, resolving inheritance via the 'inherits' key.

    If the config contains `inherits: base.yaml`, the base config is loaded
    first and the child config is merged on top.
    """
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Resolve inheritance
    parent_file = raw.pop("inherits", None)
    if parent_file:
        parent_path = os.path.join(config_dir, parent_file)
        if not os.path.isabs(parent_file):
            parent_path = os.path.join(config_dir, parent_file)
        with open(parent_path, "r") as f:
            parent_raw = yaml.safe_load(f) or {}
        # Remove inherits from parent (if any) to prevent recursive chains for now
        parent_raw.pop("inherits", None)
        raw = _deep_merge(parent_raw, raw)

    return _dict_to_config(raw)
