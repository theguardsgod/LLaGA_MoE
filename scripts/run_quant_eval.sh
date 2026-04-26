#!/bin/bash
# =============================================================================
# Quantization Evaluation Script for LLaGA
# Runs post-training quantization (PTQ) evaluations on baseline and MoE models
# using bitsandbytes 4-bit NF4 and 8-bit INT8 quantization.
#
# Usage:
#   bash scripts/run_quant_eval.sh <model_type> <quant_mode> <dataset> <gpu_id>
#
# Arguments:
#   model_type: "baseline" or "moe"
#   quant_mode: "fp16", "4bit", or "8bit"
#   dataset:    "cora", "pubmed", "arxiv", or "products"
#   gpu_id:     GPU device ID (0, 2, or 3 — avoid GPU 1)
#
# Examples:
#   bash scripts/run_quant_eval.sh baseline 4bit cora 0
#   bash scripts/run_quant_eval.sh moe 8bit pubmed 2
# =============================================================================

set -euo pipefail

MODEL_TYPE="${1:?Usage: $0 <baseline|moe> <fp16|4bit|8bit> <dataset> <gpu_id>}"
QUANT_MODE="${2:?Usage: $0 <baseline|moe> <fp16|4bit|8bit> <dataset> <gpu_id>}"
DATASET="${3:?Usage: $0 <baseline|moe> <fp16|4bit|8bit> <dataset> <gpu_id>}"
GPU_ID="${4:?Usage: $0 <baseline|moe> <fp16|4bit|8bit> <dataset> <gpu_id>}"

OUTPUT_DIR="/localnvme/llaga/eval_output/quant_experiments"
mkdir -p "$OUTPUT_DIR"

VICUNA_BASE="lmsys/vicuna-7b-v1.5-16k"

# Build output filename
OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_TYPE}_${QUANT_MODE}_nc_${DATASET}.jsonl"

# Build quantization flags
QUANT_FLAGS=""
if [ "$QUANT_MODE" = "4bit" ]; then
    QUANT_FLAGS="--load_4bit"
elif [ "$QUANT_MODE" = "8bit" ]; then
    QUANT_FLAGS="--load_8bit"
fi

echo "======================================================================"
echo "Quantization Eval: ${MODEL_TYPE} | ${QUANT_MODE} | ${DATASET} | GPU ${GPU_ID}"
echo "Output: ${OUTPUT_FILE}"
echo "======================================================================"

if [ "$MODEL_TYPE" = "baseline" ]; then
    CKPT_PATH="./checkpoints/${DATASET}/llaga-vicuna-7b-simteg-2-10-linear-projector_nc_v2"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "ERROR: Checkpoint not found at $CKPT_PATH"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n llaga \
        python -u eval/eval_pretrain_quantized.py \
            --model_path "$CKPT_PATH" \
            --model_base "$VICUNA_BASE" \
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
            $QUANT_FLAGS \
        2>&1 | tee "${OUTPUT_DIR}/${MODEL_TYPE}_${QUANT_MODE}_${DATASET}.log"

elif [ "$MODEL_TYPE" = "moe" ]; then
    CKPT_PATH="/localnvme/llaga/checkpoints/${DATASET}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc_v2"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "ERROR: Checkpoint not found at $CKPT_PATH"
        exit 1
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n llaga \
        python -u eval/eval_moe_hotswap_quantized.py \
            --datasets "$DATASET" \
            --model_base "$VICUNA_BASE" \
            --checkpoint_pattern "/localnvme/llaga/checkpoints/{dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc_v2" \
            --pretrained_embedding_type simteg \
            --use_hop 2 \
            --sample_neighbor_size 10 \
            --conv_mode v1 \
            --task nc \
            --template ND \
            --output_dir "$OUTPUT_DIR" \
            --temperature 0.2 \
            --num_beams 1 \
            $QUANT_FLAGS \
        2>&1 | tee "${OUTPUT_DIR}/${MODEL_TYPE}_${QUANT_MODE}_${DATASET}.log"
else
    echo "ERROR: model_type must be 'baseline' or 'moe'"
    exit 1
fi

echo ""
echo "======================================================================"
echo "SCORING: ${MODEL_TYPE} | ${QUANT_MODE} | ${DATASET}"
echo "======================================================================"

# For MoE hotswap, output file naming is different
if [ "$MODEL_TYPE" = "moe" ]; then
    SCORE_FILE="${OUTPUT_DIR}/moe_nc_${DATASET}.jsonl"
else
    SCORE_FILE="$OUTPUT_FILE"
fi

if [ -f "$SCORE_FILE" ]; then
    conda run --no-capture-output -n llaga \
        python eval/eval_res.py \
            --dataset "$DATASET" \
            --task nc \
            --res_path "$SCORE_FILE"
else
    echo "WARNING: Output file $SCORE_FILE not found, skipping scoring"
fi

echo ""
echo "Done: ${MODEL_TYPE} | ${QUANT_MODE} | ${DATASET}"
