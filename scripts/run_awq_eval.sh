#!/bin/bash

set -euo pipefail

DATASET="${1:?Usage: $0 <dataset> <gpu_id> <awq_model_path>}"
GPU_ID="${2:?Usage: $0 <dataset> <gpu_id> <awq_model_path>}"
AWQ_MODEL_PATH="${3:?Usage: $0 <dataset> <gpu_id> <awq_model_path>}"

OUTPUT_DIR="/localnvme/llaga/eval_output/quant_experiments"
mkdir -p "$OUTPUT_DIR"

VICUNA_BASE="lmsys/vicuna-7b-v1.5-16k"
CKPT_PATH="./checkpoints/${DATASET}/llaga-vicuna-7b-simteg-2-10-linear-projector_nc_v2"
OUTPUT_FILE="${OUTPUT_DIR}/baseline_awq_nc_${DATASET}.jsonl"

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n llaga \
    python -u eval/eval_pretrain_awq.py \
        --model_path "$CKPT_PATH" \
        --model_base "$VICUNA_BASE" \
        --awq_model_path "$AWQ_MODEL_PATH" \
        --pretrained_embedding_type simteg \
        --use_hop 2 \
        --sample_neighbor_size 10 \
        --conv_mode v1 \
        --task nc \
        --dataset "$DATASET" \
        --template ND \
        --answers_file "$OUTPUT_FILE" \
        --temperature 0.2 \
        --num_beams 1 \
    2>&1 | tee "${OUTPUT_DIR}/baseline_awq_${DATASET}.log"

conda run --no-capture-output -n llaga \
    python eval/eval_res.py \
        --dataset "$DATASET" \
        --task nc \
        --res_path "$OUTPUT_FILE"
