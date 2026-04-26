#!/bin/bash

model_path=${1}
dataset=${2:-"arxiv"}
emb=${3:-"simteg"}
template=${4:-"ND"}
gpu_id=${GPU_ID:-0}
conv_mode=${5:-"v1"}

if [ "${template}" = "HO" ]; then
  use_hop=0
else
  use_hop=2
fi

output_dir="./results"
mkdir -p ${output_dir}

model_name=$(basename ${model_path})
answers_file="${output_dir}/${model_name}_classifier_nc_${dataset}.jsonl"

echo "Model: ${model_path}"
echo "Dataset: ${dataset}"
echo "Template: ${template}"
echo "GPU: ${gpu_id}"
echo "Output: ${answers_file}"

CUDA_VISIBLE_DEVICES=${gpu_id} python eval/eval_pretrain_classifier.py \
  --model_path ${model_path} \
  --dataset ${dataset} \
  --pretrained_embedding_type ${emb} \
  --template ${template} \
  --use_hop ${use_hop} \
  --sample_neighbor_size 10 \
  --conv_mode ${conv_mode} \
  --answers_file ${answers_file}
