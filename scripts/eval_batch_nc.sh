#!/bin/bash

model_path=${1}
model_base=${2:-"lmsys/vicuna-7b-v1.5-16k"}
dataset=${3:-"arxiv"}
nc_batch_size=${4:-20}
emb=${5:-"simteg"}
use_hop=${6:-2}
sample_size=${7:-10}
conv_mode=${8:-"v1"}
template=${9:-"ND"}
use_hop_default=${6:-2}
gpu_id=${GPU_ID:-0}

output_dir="./results"
mkdir -p ${output_dir}

model_name=$(basename ${model_path})

if [ "${template}" = "HO" ]; then
    use_hop=0
    answers_file="${output_dir}/${model_name}_batch${nc_batch_size}_nodeonly_nc_${dataset}.jsonl"
else
    use_hop=${use_hop_default}
    answers_file="${output_dir}/${model_name}_batch${nc_batch_size}_nc_${dataset}.jsonl"
fi

echo "Model: ${model_path}"
echo "Dataset: ${dataset}"
echo "Batch size: ${nc_batch_size}"
echo "Template: ${template}"
echo "GPU: ${gpu_id}"
echo "Output: ${answers_file}"

CUDA_VISIBLE_DEVICES=${gpu_id} python eval/eval_pretrain_batch_nc.py \
    --model_path ${model_path} \
    --model_base ${model_base} \
    --dataset ${dataset} \
    --template ${template} \
    --pretrained_embedding_type ${emb} \
    --use_hop ${use_hop} \
    --sample_neighbor_size ${sample_size} \
    --nc_batch_size ${nc_batch_size} \
    --conv_mode ${conv_mode} \
    --answers_file ${answers_file} \
    --temperature 0.2

echo "Evaluating results..."
python eval/eval_res.py --res_path ${answers_file} --task nc --dataset ${dataset}
