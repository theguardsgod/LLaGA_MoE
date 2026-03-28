#!/bin/bash
# Evaluate all MoE-LLaGA models across all NC datasets
# Usage: bash scripts/eval_moe_all.sh [gpu_id]

GPU=${1:-0}
OUTPUT_BASE="/localnvme/llaga/eval_output"
CACHE_DIR="../../checkpoint"
DATASETS=("arxiv" "products" "pubmed" "cora")

mkdir -p "${OUTPUT_BASE}"

for ds in "${DATASETS[@]}"; do
    MODEL_PATH="./checkpoints/${ds}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc"
    OUTPUT_FILE="${OUTPUT_BASE}/moe_nc_${ds}.jsonl"

    if [ ! -d "${MODEL_PATH}" ]; then
        echo "=== SKIP ${ds}: checkpoint not found ==="
        continue
    fi

    echo "=== Evaluating MoE on ${ds} (GPU ${GPU}) ==="
    CUDA_VISIBLE_DEVICES=${GPU} python eval/eval_moe.py \
        --model_path "${MODEL_PATH}" \
        --conv_mode v1 \
        --dataset "${ds}" \
        --pretrained_embedding_type simteg \
        --use_hop 2 \
        --sample_neighbor_size 10 \
        --answers_file "${OUTPUT_FILE}" \
        --task nc \
        --cache_dir ${CACHE_DIR} \
        --template "ND"

    echo "=== Scoring MoE on ${ds} ==="
    python eval/eval_res.py --dataset "${ds}" --task nc --res_path "${OUTPUT_FILE}"
    echo ""
done

echo "=== All MoE evaluations complete ==="
