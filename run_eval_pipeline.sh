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


for item in "${MODELS[@]}"; do
    # Parse the space-separated fields
    run_dir=$(echo $item | awk '{print $1}')
    config=$(echo $item | awk '{print $2}')
    
    echo "=========================================================="
    echo "Evaluating: $run_dir"
    echo "=========================================================="
    
    # Ensure results directory exists
    mkdir -p "${run_dir}/results"
    
    # 1. Detailed Evaluation (Per-patch metrics)
    echo "Running Detailed Eval..."
    python code/evaluation/evaluate_detailed.py \
        --config "$config" \
        --checkpoint "${run_dir}/ckpts/best_model.pth" \
        --output "${run_dir}/results/detailed_metrics.jsonl" \
        --gpu 0
        
    # 2. Inference (Qualitative Plots on a standard test crop)
    echo "Running Inference (Qualitative Plotting)..."
    python code/infer.py \
        --config "$config" \
        --checkpoint "${run_dir}/ckpts/best_model.pth" \
        --dataset jrc_cos7-1a \
        --crop crop234 \
        --z-slice 70 \
        --gpu 0
done


echo "=========================================================="
echo "Consolidating Results for all completed models..."
echo "=========================================================="

INPUTS=()
LABELS=()
for item in "${MODELS[@]}"; do
    run_dir=$(echo $item | awk '{print $1}')
    # Extract the string between single quotes for the label
    label=$(echo "$item" | cut -d"'" -f2) 
    
    # Only include if the JSONL was successfully created
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
