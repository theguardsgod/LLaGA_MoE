#!/bin/bash
# Evaluate all trained NC models across all datasets
# Usage: bash scripts/eval_all.sh

OUTPUT_BASE="/localnvme/llaga/eval_output"
mkdir -p "${OUTPUT_BASE}"
MODEL_BASE="lmsys/vicuna-7b-v1.5-16k"
CACHE_DIR="../../checkpoint"
DATASETS=("arxiv" "products" "pubmed" "cora")

run_eval() {
    local gpu=$1
    local model_path=$2
    local model_base=$3
    local dataset=$4
    local use_hop=$5
    local template=$6
    local tag=$7

    local output_file="${OUTPUT_BASE}/${tag}_nc_${dataset}.jsonl"
    echo "=== Evaluating ${tag} on ${dataset} (GPU ${gpu}) ==="

    local base_arg=""
    if [ "${model_base}" != "NONE" ]; then
        base_arg="--model_base ${model_base}"
    fi

    CUDA_VISIBLE_DEVICES=${gpu} python eval/eval_pretrain.py \
        --model_path "${model_path}" \
        ${base_arg} \
        --conv_mode v1 \
        --dataset "${dataset}" \
        --pretrained_embedding_type simteg \
        --use_hop ${use_hop} \
        --sample_neighbor_size 10 \
        --answers_file "${output_file}" \
        --task nc \
        --cache_dir ${CACHE_DIR} \
        --template "${template}"

    echo "=== Scoring ${tag} on ${dataset} ==="
    python eval/eval_res.py --dataset "${dataset}" --task nc --res_path "${output_file}"
}

# --- Projector-only (ND, hop=2) ---
for ds in "${DATASETS[@]}"; do
    run_eval 0 "./checkpoints/${ds}/llaga-vicuna-7b-simteg-2-10-linear-projector_nc" "${MODEL_BASE}" "${ds}" 2 "ND" "projector"
done

# --- Full finetune (ND, hop=2) ---
for ds in "${DATASETS[@]}"; do
    run_eval 0 "./checkpoints/${ds}/llaga-vicuna-7b-simteg-2-10-linear-projector-ft_nc" "NONE" "${ds}" 2 "ND" "ft"
done

# --- Node-only (HO, hop=0) ---
for ds in "${DATASETS[@]}"; do
    run_eval 0 "./checkpoints/${ds}/llaga-vicuna-7b-simteg-nodeonly-linear-projector_nc" "${MODEL_BASE}" "${ds}" 0 "HO" "nodeonly"
done

echo "=== All evaluations complete ==="
