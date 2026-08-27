#!/bin/bash
# run_eval_pipeline.sh
# Run this script on the GPU server from the project root directory.

# Array format: "RUN_DIR CONFIG_FILE 'FRIENDLY_LABEL'"
declare -a MODELS=(
    "runs/static-unet-13cls-nojitter-p64-valdice configs/static_unet.yaml 'Static (p64, valdice)'"
    "runs/static-unet-13cls-nojitter-p64 configs/static_unet.yaml 'Static (p64)'"
    "runs/static-unet-13cls-nojitter configs/static_unet.yaml 'Static (p128)'"
    "runs/dynamic-unet-13cls-jitter32 configs/dynamic_unet.yaml 'Dynamic (p128)'"
    "runs/latest-unet-14cls-jitter48-dice configs/latest_unet.yaml 'Latest (p128)'"
    "runs/latest-unet-14cls-jitter48-p64 configs/latest_unet.yaml 'Latest (p64)'"
    "runs/latest-unet-14cls-em-aug configs/latest_unet_em_aug.yaml 'Latest+EM (p128)'"
    "runs/latest-unet-14cls-em-aug-p64 configs/latest_unet_em_aug.yaml 'Latest+EM (p64)'"
    "runs/latest-unet-14cls-res-aug configs/latest_unet_res_aug.yaml 'Latest+Res (p128)'"
    "runs/latest-unet-14cls-res-aug-p64 configs/latest_unet_res_aug.yaml 'Latest+Res (p64)'"
    "runs/sdt-unet-14cls-baseline configs/sdt_unet.yaml 'SDT Baseline'"
)

# Note: Excluded because they are currently running:
# - runs/scale-conditioned-unet-14cls
# - runs/static-unet-13cls-nojitter-p128-valdice

echo "=========================================================="
echo "Generating Pristine Global Holdout Test Set (crop234)..."
echo "=========================================================="
mkdir -p results/global_splits
python3 -c "
import sys
sys.path.append('.')
from code.data.splits import prepare_splits
prepare_splits(
    blueprint_json_path='all_jsons/latest_baseline_centroids.json',
    output_dir='results/global_splits',
    target_classes=['endo', 'ld', 'lyso', 'mito', 'mt', 'np', 'nuc', 'perox', 'ves', 'vim', 'golgi', 'er', 'eres'],
    split_ratios=[0.85, 0.09, 0.06],
    excluded_crop='crop234'
)
"

MAX_JOBS=4
job_idx=0

echo "Starting parallel evaluation (max $MAX_JOBS concurrent jobs)..."

for item in "${MODELS[@]}"; do
    # Parse the space-separated fields
    run_dir=$(echo $item | awk '{print $1}')
    config=$(echo $item | awk '{print $2}')
    
    # Distribute load evenly across your 2 GPUs (0 and 1)
    GPU=$((job_idx % 2))
    
    (
        echo "--> [GPU $GPU] Evaluating: $run_dir"
        mkdir -p "${run_dir}/results"
        
        run_name=$(basename "$run_dir")
        patch_dim=128
        if [[ "$run_name" == *"-p64"* ]]; then
            patch_dim=64
        fi

        # 1. Detailed Evaluation (Per-patch metrics)
        python code/evaluation/evaluate_detailed.py \
            --config "$config" \
            --checkpoint "${run_dir}/ckpts/best_model.pth" \
            --output "${run_dir}/results/detailed_metrics.jsonl" \
            --test-json "results/global_splits/test.json" \
            --name "$run_name" \
            --patch-dim $patch_dim \
            --gpu $GPU > "${run_dir}/results/eval_stdout.log" 2>&1
            
        # 2. Inference (Qualitative Plots on a standard test crop)
        python code/infer.py \
            --config "$config" \
            --checkpoint "${run_dir}/ckpts/best_model.pth" \
            --dataset jrc_cos7-1a \
            --crop crop234 \
            --z-slice 70 \
            --name "$run_name" \
            --patch-dim $patch_dim \
            --gpu $GPU >> "${run_dir}/results/eval_stdout.log" 2>&1
            
        echo "--> [GPU $GPU] Finished: $run_dir"
    ) &
    
    ((job_idx++))
    
    # Throttle: Wait for any background job to finish before spawning more if we hit MAX_JOBS
    if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
        wait -n
    fi
done

# Wait for all remaining background jobs to finish
wait

echo "=========================================================="
echo "Consolidating Results for all completed models..."
echo "=========================================================="

INPUTS=()
LABELS=()
for item in "${MODELS[@]}"; do
    run_dir=$(echo $item | awk '{print $1}')
    label=$(echo "$item" | cut -d"'" -f2) 
    
    if [ -f "${run_dir}/results/detailed_metrics.jsonl" ]; then
        INPUTS+=("${run_dir}/results/detailed_metrics.jsonl")
        LABELS+=("$label")
    else
        echo "Warning: detailed_metrics.jsonl not found for ${run_dir}. Skipping in consolidation."
    fi
done

# 3. Analyze and consolidate
python code/evaluation/analyze_results.py \
    --inputs "${INPUTS[@]}" \
    --labels "${LABELS[@]}" \
    --output-dir results/paper_tables/

echo "=========================================================="
echo "Done! All results consolidated to results/paper_tables/"
echo "Zip these up to copy to your local machine:"
echo "tar czf paper_tables.tar.gz results/paper_tables/"
echo "=========================================================="
