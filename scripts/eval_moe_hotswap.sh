#!/bin/bash
# Hot-swap MoE evaluation: load LLM once, swap MoE projectors per dataset
# Usage: bash scripts/eval_moe_hotswap.sh [gpu_id] [batch_size] [datasets...]
# Example: bash scripts/eval_moe_hotswap.sh 0           # bs=1, all datasets
# Example: bash scripts/eval_moe_hotswap.sh 0 8          # bs=8, all datasets
# Example: bash scripts/eval_moe_hotswap.sh 0 8 arxiv cora

GPU=${1:-0}
BS=${2:-1}
shift 2 2>/dev/null
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("arxiv" "products" "pubmed" "cora")
fi

OUTPUT_BASE="/localnvme/llaga/eval_output"
CACHE_DIR="../../checkpoint"

mkdir -p "${OUTPUT_BASE}"

echo "GPU=${GPU}, batch_size=${BS}, datasets=${DATASETS[@]}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=${GPU} python eval/eval_moe_hotswap.py \
    --datasets "${DATASETS[@]}" \
    --model_base "lmsys/vicuna-7b-v1.5-16k" \
    --checkpoint_pattern "./checkpoints/{dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc" \
    --output_dir "${OUTPUT_BASE}" \
    --conv_mode v1 \
    --pretrained_embedding_type simteg \
    --use_hop 2 \
    --sample_neighbor_size 10 \
    --task nc \
    --cache_dir ${CACHE_DIR} \
    --template "ND" \
    --batch_size ${BS} \
    --max_new_tokens 32
