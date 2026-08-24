# Changelog

All notable changes to the OrganelleNet codebase.

---

## [2.0.0] — 2026-08-24

### Major Refactoring: Monolithic → Modular

**BREAKING**: The 3 original monolithic scripts have been replaced by a modular, config-driven architecture.

### Added

- **`code/` directory** — New modular codebase with 5 sub-packages:
  - `code/data/` — `PatchDataset`, `create_balanced_sampler`, `prepare_splits`, zarr utilities
  - `code/models/` — Model factory (`build_model`) with multi-GPU DataParallel support
  - `code/training/` — Unified `Trainer` class with AMP, gradient accumulation, early stopping (loss or Dice), checkpoint resume
  - `code/evaluation/` — `SegmentationEvaluator` with HD95, training curve plotter, sliding window inference
  - `code/utils/` — YAML config loader with inheritance, run directory manager

- **`configs/` directory** — YAML experiment configs:
  - `base.yaml` — Shared defaults
  - `static_unet.yaml` — 13 classes, no jitter, loss-based early stopping
  - `dynamic_unet.yaml` — 13 classes, jitter=32, loss-based early stopping
  - `latest_unet.yaml` — 14 classes (Vimentin), jitter=48, Dice-based early stopping, crop234 exclusion

- **Entry point scripts**:
  - `code/train.py` — Training with `--config`, `--gpu`, `--dry-run` flags
  - `code/evaluate.py` — Test set evaluation with per-class metrics + HD95
  - `code/infer.py` — Full-crop sliding window inference with visualization

- **Self-contained run directories** under `runs/<experiment_name>/`:
  - `config.yaml` — Frozen config copy
  - `ckpts/` — `best_model.pth` + `resume_checkpoint.pth`
  - `logs/` — `training_log.csv`
  - `plots/` — Loss curves, inference visualizations
  - `results/` — `test_metrics.txt`

- **`commands.md`** — GPU server deployment guide with parallel training instructions
- **`changelog.md`** — This file

### Changed

- `requirements.txt` moved from `organellenet/` to project root
- `README.md` updated with new directory structure and quickstart

### Deprecated

- Original scripts moved to `legacy/`:
  - `legacy/static_baseline.py`
  - `legacy/dynamic_baseline.py`
  - `legacy/latest_baseline.py`
  - `legacy/static_dynamic_baseline.json`
  - `legacy/latest_baseline_centroids.json`

### Design Decisions

1. **Config inheritance** — Child configs only override what differs from `base.yaml`, keeping configs DRY
2. **Trainer class** — Single training engine handles both loss-based and Dice-based early stopping via config
3. **sys.path manipulation** — Every module adds the project root to `sys.path` so imports work regardless of working directory
4. **Balanced sampling** — Inverse-frequency weighting preserved from original code (critical for rare organelle classes)
5. **Legacy preservation** — Original scripts kept in `legacy/` for reference; they can still be run independently

---

## [1.0.0] — Pre-refactoring

### Original codebase

- 3 monolithic scripts (~900 lines each):
  - `organellenet/static_baseline.py` — Static sampling UNet (13 classes)
  - `organellenet/dynamic_baseline.py` — Dynamic jitter UNet (13 classes)
  - `organellenet/latest_baseline.py` — Latest UNet with Vimentin (14 classes), gradient accumulation, Dice tracking
- `patch_sampling/improved.py` — Patch extraction pipeline
- Hardcoded paths to `/mnt/voxelcell_vol1`
- No config files, no run directories, no separation of concerns
