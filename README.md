# OrganelleNet

3D organelle segmentation for the [CellMap Segmentation Challenge](https://github.com/janelia-cellmap/cellmap-segmentation-challenge), targeting [NewInML 2026 @ NeurIPS](https://newinml.github.io/NewInML2026NeurIPS/).

## Project Structure

```text
organellenet/
├── configs/                     # YAML experiment configs
│   ├── base.yaml                #   Shared defaults
│   ├── static_unet.yaml         #   13-class, no jitter
│   ├── dynamic_unet.yaml        #   13-class, jitter=32
│   └── latest_unet.yaml         #   14-class, jitter=48, Dice-based
├── code/
│   ├── data/                    # Data loading pipeline
│   │   ├── dataset.py           #   PatchDataset (zarr → tensors)
│   │   ├── sampler.py           #   Balanced class sampling
│   │   ├── splits.py            #   Train/val/test split generation
│   │   └── zarr_utils.py        #   Zarr I/O, metadata, extraction
│   ├── models/
│   │   └── unet.py              #   Model factory (MONAI UNet)
│   ├── training/
│   │   ├── trainer.py           #   Training loop (AMP, early stopping, resume)
│   │   └── losses.py            #   Loss factory (DiceCELoss)
│   ├── evaluation/
│   │   ├── evaluator.py         #   Metrics (Dice, IoU, HD95, confusion matrix)
│   │   └── visualize.py         #   Training curves, inference visualization
│   ├── utils/
│   │   ├── config.py            #   YAML config loader with inheritance
│   │   └── paths.py             #   Run directory management
│   ├── train.py                 #   Training entry point
│   ├── evaluate.py              #   Evaluation entry point
│   └── infer.py                 #   Sliding window inference entry point
├── patch_sampling/
│   └── improved.py              # Patch extraction from raw zarr volumes
├── legacy/                      # Original monolithic scripts (preserved)
├── runs/                        # Auto-created experiment output directories
├── configs/                     # YAML experiment configurations
├── commands.md                  # GPU server deployment guide
├── changelog.md                 # Detailed change log
├── requirements.txt             # Python dependencies
└── plan.md                      # Research plan and instructions
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install pyyaml
```

### 2. Train a model

```bash
# Static UNet on GPU 0
python code/train.py --config configs/static_unet.yaml --gpu 0

# Dry run (validate setup without training)
python code/train.py --config configs/static_unet.yaml --dry-run
```

### 3. Evaluate

```bash
python code/evaluate.py \
    --config configs/static_unet.yaml \
    --checkpoint runs/static-unet-13cls-nojitter/ckpts/best_model.pth
```

### 4. Full-crop inference

```bash
python code/infer.py \
    --config configs/static_unet.yaml \
    --checkpoint runs/static-unet-13cls-nojitter/ckpts/best_model.pth \
    --dataset jrc_cos7-1a --crop crop234
```

## Configuration

All behavior is driven by YAML configs. Configs use inheritance:

```yaml
# configs/static_unet.yaml
inherits: base.yaml
experiment_name: static-unet-13cls-nojitter
data:
  max_jitter: 0
model:
  out_channels: 13
```

Only parameters that differ from `base.yaml` need to be specified. See [commands.md](commands.md) for full deployment instructions.

## Model Variants

| Config | Classes | Jitter | Strides | Early Stop | Grad Accum |
|--------|---------|--------|---------|------------|------------|
| `static_unet.yaml` | 13 | 0 | (2,2,2,2) | val_loss | 1 |
| `dynamic_unet.yaml` | 13 | 32 | (2,2,2,2) | val_loss | 1 |
| `latest_unet.yaml` | 14 | 48 | (1,2,2,2) | val_dice | 2 |

## Data

Data is expected in zarr format following the CellMap challenge directory structure at the path configured in `configs/base.yaml`:

```
data_dir/
├── jrc_cos7-1a/
│   └── jrc_cos7-1a.zarr/
│       └── recon-1/
│           ├── em/fibsem-uint8/s0/
│           └── labels/groundtruth/crop*/all/s0/
├── jrc_hela-2/
│   └── ...
└── ... (22 datasets)
```
