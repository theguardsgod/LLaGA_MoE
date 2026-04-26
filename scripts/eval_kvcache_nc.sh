#!/bin/bash

model_path=${1}
model_base=${2:-"lmsys/vicuna-7b-v1.5-16k"}
dataset=${3:-"arxiv"}
batch_size=${4:-32}
emb=${5:-"simteg"}
use_hop=${6:-2}
sample_size=${7:-10}
conv_mode=${8:-"v1"}

output_dir="./results"
mkdir -p ${output_dir}

model_name=$(basename ${model_path})
answers_file="${output_dir}/${model_name}_kvcache_nc_${dataset}.jsonl"

echo "=== KV-Cache Accelerated NC Evaluation ==="
echo "Model: ${model_path}"
echo "Dataset: ${dataset}"
echo "Batch size: ${batch_size}"
echo "Output: ${answers_file}"

python eval/eval_pretrain_kvcache.py \
    --model_path ${model_path} \
    --model_base ${model_base} \
    --dataset ${dataset} \
    --pretrained_embedding_type ${emb} \
    --use_hop ${use_hop} \
    --sample_neighbor_size ${sample_size} \
    --batch_size ${batch_size} \
    --conv_mode ${conv_mode} \
    --answers_file ${answers_file} \
    --temperature 0.2 \
    --cache_dir ../../checkpoint

echo "Evaluating results..."
python eval/eval_res.py --res_path ${answers_file} --task nc --dataset ${dataset}
