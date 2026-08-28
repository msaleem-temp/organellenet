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


---

## 2b. Training at 64 Voxel Patch Size (Faster Training)

By overriding the patch dimension and experiment name, you can train 64-voxel variants of your models without altering the original 128-voxel configurations. We also increase the batch size from 4 to 16 to utilize the GPU efficiently since the patches are 8x smaller.

**Static UNet at 64³ (GPU 0):**
```bash
python code/train.py \
    --config configs/static_unet.yaml \
    --name static-unet-13cls-nojitter-p64 \
    --patch-dim 64 \
    --gpu 0
```
- Val Dice for Static UNet patch 64
```bash
python code/train.py \
    --config configs/static_unet.yaml \
    --name static-unet-13cls-nojitter-p64-valdice \
    --patch-dim 64 \
    --gpu 0
```

- Val Dice for Static UNet patch 128
```bash
python code/train.py \
    --config configs/static_unet.yaml \
    --name static-unet-13cls-nojitter-p128-valdice \
    --patch-dim 128 \
    --gpu 0
```

**Dynamic UNet at 64³ (GPU 1):**
```bash
python code/train.py \
    --config configs/dynamic_unet.yaml \
    --name dynamic-unet-13cls-jitter32-p64 \
    --patch-dim 64 \
    --gpu 1
```

**Latest UNet at 64³:**
```bash
python code/train.py \
    --config configs/latest_unet.yaml \
    --name latest-unet-14cls-jitter48-p64 \
    --patch-dim 64 --gpu 0
```

**Latest UNet + EM Aug at 64³:**
```bash
python code/train.py \
    --config configs/latest_unet_em_aug.yaml \
    --name latest-unet-14cls-em-aug-p64 \
    --patch-dim 64 --gpu 0
```

**Latest UNet + Res Aug at 64³:**
```bash
python code/train.py \
    --config configs/latest_unet_res_aug.yaml \
    --name latest-unet-14cls-res-aug-p64 \
    --patch-dim 64 --gpu 0
```

**Scale Conditioned UNet at 64³:**
```bash
python code/train.py \
    --config configs/scale_conditioned_unet.yaml \
    --name scale-conditioned-unet-14cls-p64 \
    --patch-dim 64
```

You can apply these `--patch-dim` and `--name` overrides to any training command in this guide to run the 64-voxel equivalent!


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

# Dynamic UNet on crop234
python code/infer.py \
    --config configs/dynamic_unet.yaml \
    --checkpoint runs/dynamic-unet-13cls-jitter32/ckpts/best_model.pth \
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

---

## 8. New Augmented Model Training (Paper Experiments)

These two models add augmentation on top of the latest_unet configuration.
They can run in parallel on separate GPUs if I/O bandwidth allows.

**Terminal 3 — Latest UNet + Standard EM Augmentation (GPU 0 or 1):**
```bash
# Standard EM augmentation: flips, rotations, intensity, elastic deform
python code/train.py --config configs/latest_unet_em_aug.yaml --gpu 0
```

**Terminal 4 — Latest UNet + Resolution Augmentation (GPU 0 or 1):**
```bash
# EM augmentation + novel resolution augmentation
python code/train.py --config configs/latest_unet_res_aug.yaml --gpu 1
```

### With nohup (if running unattended):
```bash
nohup python code/train.py --config configs/latest_unet_em_aug.yaml --gpu 0 \
    > logs/em_aug_train.log 2>&1 &

nohup python code/train.py --config configs/latest_unet_res_aug.yaml --gpu 1 \
    > logs/res_aug_train.log 2>&1 &

nohup python code/train.py --config configs/scale_conditioned_unet.yaml --gpu 0 \
    > logs/scale_cond_train.log 2>&1 &
```

### Dry run (validate the new configs):
```bash
python code/train.py --config configs/latest_unet_em_aug.yaml --dry-run
python code/train.py --config configs/latest_unet_res_aug.yaml --dry-run
```

---

## 9. Detailed Evaluation (Per-Patch with Metadata)

After training completes, run the detailed evaluator on each model.
This produces JSONL files with per-patch metrics + dataset/resolution metadata.

```bash
# Static UNet (baseline)
python code/evaluation/evaluate_detailed.py \
    --config configs/static_unet.yaml \
    --checkpoint runs/static-unet-13cls-nojitter/ckpts/best_model.pth \
    --output runs/static-unet-13cls-nojitter/results/detailed_metrics.jsonl \
    --gpu 0

# Latest UNet (jitter only, no real augmentation)
python code/evaluation/evaluate_detailed.py \
    --config configs/latest_unet.yaml \
    --checkpoint runs/latest-unet-14cls-jitter48-dice/ckpts/best_model.pth \
    --output runs/latest-unet-14cls-jitter48-dice/results/detailed_metrics.jsonl \
    --gpu 0

# Latest UNet + EM Augmentation
python code/evaluation/evaluate_detailed.py \
    --config configs/latest_unet_em_aug.yaml \
    --checkpoint runs/latest-unet-14cls-em-aug/ckpts/best_model.pth \
    --output runs/latest-unet-14cls-em-aug/results/detailed_metrics.jsonl \
    --gpu 0

# Latest UNet + Resolution Augmentation (proposed method)
python code/evaluation/evaluate_detailed.py \
    --config configs/latest_unet_res_aug.yaml \
    --checkpoint runs/latest-unet-14cls-res-aug/ckpts/best_model.pth \
    --output runs/latest-unet-14cls-res-aug/results/detailed_metrics.jsonl \
    --gpu 0

# Scale Conditioned UNet (proposed architectural method)
python code/evaluation/evaluate_detailed.py \
    --config configs/scale_conditioned_unet.yaml \
    --checkpoint runs/scale-conditioned-unet-14cls/ckpts/best_model.pth \
    --output runs/scale-conditioned-unet-14cls/results/detailed_metrics.jsonl \
    --gpu 0
```

---

## 10. Generate Paper Tables

After all detailed evaluations are complete, generate CSV tables:

```bash
python code/evaluation/analyze_results.py \
    --inputs \
        runs/static-unet-13cls-nojitter/results/detailed_metrics.jsonl \
        runs/latest-unet-14cls-jitter48-dice/results/detailed_metrics.jsonl \
        runs/latest-unet-14cls-em-aug/results/detailed_metrics.jsonl \
        runs/latest-unet-14cls-res-aug/results/detailed_metrics.jsonl \
        runs/scale-conditioned-unet-14cls/results/detailed_metrics.jsonl \
    --labels \
        "Static (No Aug)" \
        "Latest (Jitter)" \
        "Latest+EM Aug" \
        "Latest+Res Aug" \
        "Scale Cond" \
    --output-dir results/paper_tables/
```

This produces:
- `per_class_comparison.csv` — Per-class IoU/Dice across all models
- `per_dataset_comparison.csv` — Per-CellMap-dataset mean metrics
- `per_resolution_band.csv` — Fine/Medium/Coarse resolution band metrics
- `resolution_x_class.csv` — Class × resolution band heatmap
- `resolution_x_class_delta.csv` — Improvement over baseline per band × class
- `summary.csv` — Overall metrics + fine-coarse gap

---

## 11. Train SDT Baseline (Distance Transform)

The SDT model predicts physical distance transforms (in nanometers) instead of raw masks.

**Train the SDT Baseline:**
```bash
python code/train.py --config configs/sdt_unet.yaml --gpu 0
```

**Dry-run to validate setup:**
```bash
python code/train.py --config configs/sdt_unet.yaml --dry-run
```

---

## 12. Stratified Robustness Evaluation (Scale Robustness)

Evaluates how a model degrades (hallucinates false positives) across different physical scales on the exact same crop.

**Run the Stratified Evaluation on a trained model (e.g., Dynamic UNet) against crop234:**
```bash
python code/evaluation/evaluate_stratified.py \
    --config configs/dynamic_unet.yaml \
    --checkpoint runs/dynamic-unet-13cls-jitter32/ckpts/best_model.pth \
    --dataset jrc_cos7-1a --crop crop234 \
    --scales s0 s1 s2 s3
```
This produces a `stratified_metrics_jrc_cos7-1a_crop234.csv` in the run's `results/` directory containing FPR and Dice across varying scales.

---

## 13. Copy Results to Local Machine (for paper writing)

```bash
# From the GPU server, tar up everything needed:
tar czf organellenet_results.tar.gz \
    runs/*/logs/training_log.csv \
    runs/*/results/detailed_metrics.jsonl \
    runs/*/results/test_metrics.txt \
    runs/*/results/stratified_metrics_*.csv \
    results/paper_tables/

# Then scp to your local machine:
scp organellenet_results.tar.gz your_local_machine:~/
```

---

## 14. NewInML Matched Anchor 64³ ROI Evaluation

Use these commands for the final five-run NewInML paper tables. They evaluate
the same 73 `jrc_cos7-1a/crop234` anchor records from `runs/global_splits/test.json`.
The 64³ models are scored on their native 64³ patches. The 128³ models run
with native 128³ patches, then metrics are computed only on the exact 64³
label-coordinate ROI that the native 64³ patch would use for the same anchor.
This matters for anchors near crop boundaries, where the central 64³ crop of a
clamped 128³ patch is not necessarily the same physical region.

```bash
python code/evaluation/evaluate_detailed.py \
    --config configs/latest_unet.yaml \
    --checkpoint runs/latest-unet-14cls-jitter48-p64/ckpts/best_model.pth \
    --output runs/latest-unet-14cls-jitter48-p64/results/detailed_metrics_matched_roi.jsonl \
    --test-json runs/global_splits/test.json \
    --name latest-unet-14cls-jitter48-p64 \
    --patch-dim 64 \
    --score-roi-dim 64 \
    --gpu 0
```

```bash
python code/evaluation/evaluate_detailed.py \
    --config configs/latest_unet.yaml \
    --checkpoint runs/latest-unet-14cls-jitter48-dice/ckpts/best_model.pth \
    --output runs/latest-unet-14cls-jitter48-dice/results/detailed_metrics_matched_roi.jsonl \
    --test-json runs/global_splits/test.json \
    --name latest-unet-14cls-jitter48-dice \
    --patch-dim 128 \
    --score-roi-dim 64 \
    --gpu 0
```

```bash
python code/evaluation/evaluate_detailed.py \
    --config configs/latest_unet_em_aug.yaml \
    --checkpoint runs/latest-unet-14cls-em-aug-p64/ckpts/best_model.pth \
    --output runs/latest-unet-14cls-em-aug-p64/results/detailed_metrics_matched_roi.jsonl \
    --test-json runs/global_splits/test.json \
    --name latest-unet-14cls-em-aug-p64 \
    --patch-dim 64 \
    --score-roi-dim 64 \
    --gpu 0
```

```bash
python code/evaluation/evaluate_detailed.py \
    --config configs/latest_unet_em_aug.yaml \
    --checkpoint runs/latest-unet-14cls-em-aug/ckpts/best_model.pth \
    --output runs/latest-unet-14cls-em-aug/results/detailed_metrics_matched_roi.jsonl \
    --test-json runs/global_splits/test.json \
    --name latest-unet-14cls-em-aug \
    --patch-dim 128 \
    --score-roi-dim 64 \
    --gpu 0
```

The SDT model was trained with `configs/sdt_unet.yaml`, which sets
`data.patch_dim: 64`. Re-evaluate it with the same native 64³ patch size:

```bash
python code/evaluation/evaluate_detailed.py \
    --config configs/sdt_unet.yaml \
    --checkpoint runs/sdt-unet-14cls-baseline/ckpts/best_model.pth \
    --output runs/sdt-unet-14cls-baseline/results/detailed_metrics_matched_roi.jsonl \
    --test-json runs/global_splits/test.json \
    --name sdt-unet-14cls-baseline \
    --patch-dim 64 \
    --score-roi-dim 64 \
    --gpu 0
```

After all five JSONLs exist, regenerate the matched-ROI paper tables and heatmap:

```bash
python code/figures/make_newinml_results.py \
    --input-filename detailed_metrics_matched_roi.jsonl \
    --matched-roi
```
