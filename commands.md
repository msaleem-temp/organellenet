# Commands — GPU Server Deployment Guide

All commands should be run from the project root on the **2x H200** server:
```
/mnt/graid/codebases_other/cellmap-segmentation-challenge/saleem/organellenet
```

Data resides at:
```
/mnt/graid/codebases_other/cellmap-segmentation-challenge/data
```

---

## 1. Environment Setup

```bash
# Create and activate environment
conda create -n organellenet python=3.10 -y
conda activate organellenet

# Install dependencies
pip install -r requirements.txt
pip install pyyaml
```

---

## 2. Training All Models in Parallel

Each model can be pinned to a specific GPU using `CUDA_VISIBLE_DEVICES`.
With 2x H200 GPUs, you can run up to 2 models truly in parallel, or use
DataParallel for a single model across both GPUs.

### Option A: 2 models in parallel (1 GPU each)

**Terminal 1 — Static UNet on GPU 0:**
```bash
python code/train.py --config configs/static_unet.yaml --gpu 0
```

**Terminal 2 — Dynamic UNet on GPU 1:**
```bash
python code/train.py --config configs/dynamic_unet.yaml --gpu 1
```

Then after those finish:

**Terminal 1 — Latest UNet on both GPUs (DataParallel):**
```bash
python code/train.py --config configs/latest_unet.yaml
```

### Option B: Run all 3 sequentially with nohup

```bash
nohup python code/train.py --config configs/static_unet.yaml --gpu 0 \
    > logs/static_train.log 2>&1 &

nohup python code/train.py --config configs/dynamic_unet.yaml --gpu 1 \
    > logs/dynamic_train.log 2>&1 &

# Wait for the above to finish, then:
nohup python code/train.py --config configs/latest_unet.yaml \
    > logs/latest_train.log 2>&1 &
```

### Option C: tmux sessions (recommended)

```bash
# Session 1
tmux new -s static
python code/train.py --config configs/static_unet.yaml --gpu 0

# Session 2
tmux new -s dynamic
python code/train.py --config configs/dynamic_unet.yaml --gpu 1

# Session 3 (after both finish)
tmux new -s latest
python code/train.py --config configs/latest_unet.yaml
```

---

## 3. Dry Run (Validate Setup Without Training)

```bash
python code/train.py --config configs/static_unet.yaml --dry-run
```

This validates config parsing, directory creation, and model instantiation without launching training.

---

## 4. Evaluation

After training completes, evaluate on the test set:

```bash
# Static UNet
python code/evaluate.py \
    --config configs/static_unet.yaml \
    --checkpoint runs/static-unet-13cls-nojitter/ckpts/best_model.pth \
    --gpu 0

# Dynamic UNet
python code/evaluate.py \
    --config configs/dynamic_unet.yaml \
    --checkpoint runs/dynamic-unet-13cls-jitter32/ckpts/best_model.pth \
    --gpu 0

# Latest UNet
python code/evaluate.py \
    --config configs/latest_unet.yaml \
    --checkpoint runs/latest-unet-14cls-jitter48-dice/ckpts/best_model.pth \
    --gpu 0
```

Results are saved to `runs/<experiment_name>/results/test_metrics.txt`.

---

## 5. Full-Crop Sliding Window Inference

```bash
# Static UNet on crop234
python code/infer.py \
    --config configs/static_unet.yaml \
    --checkpoint runs/static-unet-13cls-nojitter/ckpts/best_model.pth \
    --dataset jrc_cos7-1a --crop crop234 --z-slice 70 --gpu 0

# Latest UNet on crop234
python code/infer.py \
    --config configs/latest_unet.yaml \
    --checkpoint runs/latest-unet-14cls-jitter48-dice/ckpts/best_model.pth \
    --dataset jrc_cos7-1a --crop crop234 --z-slice 70 --gpu 0
```

Visualization is saved to `runs/<experiment_name>/plots/`.

---

## 6. Patch Sampling (Preprocessing)

If you need to regenerate the patch blueprint JSONs from raw zarr data:

```bash
python patch_sampling/improved.py
```

This produces `*_anchors.json` files in the configured output directory.

---

## 7. Monitoring Training

```bash
# Watch the training log in real-time
tail -f runs/static-unet-13cls-nojitter/logs/training_log.csv

# Check GPU utilization
watch -n 1 nvidia-smi
```
