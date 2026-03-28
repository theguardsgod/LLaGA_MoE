#!/bin/bash
# Parallel MoE evaluation: split test samples across GPUs
# Usage: bash scripts/eval_moe_parallel.sh <dataset> <gpu_list>
# Example: bash scripts/eval_moe_parallel.sh arxiv 1,2,3

DATASET=${1:-"arxiv"}
GPU_LIST=${2:-"1,2,3"}
OUTPUT_BASE="/localnvme/llaga/eval_output"
CACHE_DIR="../../checkpoint"
MODEL_PATH="./checkpoints/${DATASET}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc"

mkdir -p "${OUTPUT_BASE}"

# Parse GPU list
IFS=',' read -ra GPUS <<< "${GPU_LIST}"
NUM_GPUS=${#GPUS[@]}

# Determine dataset dir and test file
case ${DATASET} in
    arxiv)   DATA_DIR="/localnvme/llaga/dataset/ogbn-arxiv" ;;
    products) DATA_DIR="/localnvme/llaga/dataset/ogbn-products" ;;
    pubmed)  DATA_DIR="/localnvme/llaga/dataset/pubmed" ;;
    cora)    DATA_DIR="/localnvme/llaga/dataset/cora" ;;
    *)       echo "Unknown dataset: ${DATASET}"; exit 1 ;;
esac

TEST_FILE="${DATA_DIR}/sampled_2_10_test.jsonl"
TOTAL=$(wc -l < "${TEST_FILE}")
CHUNK_SIZE=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "=== Parallel MoE Eval: ${DATASET} ==="
echo "  Total samples: ${TOTAL}, GPUs: ${NUM_GPUS}, Chunk size: ~${CHUNK_SIZE}"
echo ""

# Launch parallel eval on each GPU
PIDS=()
for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    START=$(( i * CHUNK_SIZE ))
    END=$(( START + CHUNK_SIZE ))
    if [ ${END} -gt ${TOTAL} ]; then
        END=${TOTAL}
    fi
    if [ ${START} -ge ${TOTAL} ]; then
        continue
    fi

    CHUNK_FILE="${OUTPUT_BASE}/moe_nc_${DATASET}_chunk${i}.jsonl"
    # Remove stale chunk file
    rm -f "${CHUNK_FILE}"

    echo "  GPU ${GPU}: samples ${START}-${END} -> ${CHUNK_FILE}"
    CUDA_VISIBLE_DEVICES=${GPU} python eval/eval_moe.py \
        --model_path "${MODEL_PATH}" \
        --conv_mode v1 \
        --dataset "${DATASET}" \
        --pretrained_embedding_type simteg \
        --use_hop 2 \
        --sample_neighbor_size 10 \
        --answers_file "${CHUNK_FILE}" \
        --task nc \
        --cache_dir ${CACHE_DIR} \
        --template "ND" \
        --start ${START} \
        --end ${END} &
    PIDS+=($!)
done

echo ""
echo "  Waiting for ${#PIDS[@]} processes..."

# Wait for all to finish
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "  PID ${pid} done (exit: $?)"
done

# Merge chunk files
MERGED_FILE="${OUTPUT_BASE}/moe_nc_${DATASET}.jsonl"
rm -f "${MERGED_FILE}"
for i in "${!GPUS[@]}"; do
    CHUNK_FILE="${OUTPUT_BASE}/moe_nc_${DATASET}_chunk${i}.jsonl"
    if [ -f "${CHUNK_FILE}" ]; then
        cat "${CHUNK_FILE}" >> "${MERGED_FILE}"
    fi
done

MERGED_COUNT=$(wc -l < "${MERGED_FILE}")
echo ""
echo "  Merged: ${MERGED_COUNT}/${TOTAL} samples -> ${MERGED_FILE}"

# Score
echo "=== Scoring MoE on ${DATASET} ==="
python eval/eval_res.py --dataset "${DATASET}" --task nc --res_path "${MERGED_FILE}"
echo ""
