#!/bin/bash
# Quantized hot-swap MoE evaluation: load quantized LLM once, swap MoE projectors per dataset
# Usage: bash scripts/eval_moe_hotswap_quantized.sh [gpu_id] [datasets...]

GPU=${1:-0}
shift
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("arxiv" "products" "pubmed" "cora")
fi

OUTPUT_BASE="/localnvme/llaga/eval_output"
CACHE_DIR="../../checkpoint"

mkdir -p "${OUTPUT_BASE}"

CUDA_VISIBLE_DEVICES=${GPU} python eval/eval_moe_hotswap_quantized.py \
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
    --load_4bit
