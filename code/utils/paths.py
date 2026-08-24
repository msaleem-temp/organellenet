"""
Run directory management for OrganelleNet experiments.

Creates self-contained experiment output directories under runs/ with
subdirectories for checkpoints, logs, plots, and results.
"""

import os
import sys
import shutil
import yaml
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def setup_run_directory(config, config_path: str = None) -> dict:
    """
    Create the experiment output directory and return a dict of paths.

    Directory structure:
        runs/<experiment_name>/
            ├── config.yaml          # Frozen copy of the config used
            ├── ckpts/
            │   ├── best_model.pth
            │   └── resume_checkpoint.pth
            ├── logs/
            │   └── training_log.csv
            ├── plots/
            └── results/
                └── test_metrics.txt

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration.
    config_path : str, optional
        Path to the original YAML config file (will be copied into the run dir).

    Returns
    -------
    dict
        Dictionary with keys: 'run_dir', 'ckpts_dir', 'logs_dir', 'plots_dir',
        'results_dir', 'best_model_path', 'resume_ckpt_path', 'log_csv_path'.
    """
    runs_root = os.path.join(config.paths.main_dir, "runs")
    run_dir = os.path.join(runs_root, config.experiment_name)

    ckpts_dir = os.path.join(run_dir, "ckpts")
    logs_dir = os.path.join(run_dir, "logs")
    plots_dir = os.path.join(run_dir, "plots")
    results_dir = os.path.join(run_dir, "results")

    for d in [ckpts_dir, logs_dir, plots_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    # Freeze a copy of the config into the run directory
    if config_path and os.path.exists(config_path):
        frozen_config = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(frozen_config):
            shutil.copy2(config_path, frozen_config)

    # Write a run metadata file
    meta_path = os.path.join(run_dir, "run_meta.txt")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            f.write(f"experiment: {config.experiment_name}\n")
            f.write(f"created: {datetime.now().isoformat()}\n")
            f.write(f"num_classes: {config.data.num_classes}\n")
            f.write(f"model_out_channels: {config.model.out_channels}\n")
            f.write(f"strides: {config.model.strides}\n")
            f.write(f"max_jitter: {config.data.max_jitter}\n")
            f.write(f"early_stop_metric: {config.training.early_stop_metric}\n")
            f.write(f"accumulation_steps: {config.training.accumulation_steps}\n")

    return {
        "run_dir": run_dir,
        "ckpts_dir": ckpts_dir,
        "logs_dir": logs_dir,
        "plots_dir": plots_dir,
        "results_dir": results_dir,
        "best_model_path": os.path.join(ckpts_dir, "best_model.pth"),
        "resume_ckpt_path": os.path.join(ckpts_dir, "resume_checkpoint.pth"),
        "log_csv_path": os.path.join(logs_dir, "training_log.csv"),
    }
