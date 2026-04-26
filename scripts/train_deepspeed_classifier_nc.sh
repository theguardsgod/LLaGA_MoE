#!/bin/bash

max_len=4096
sample_size=10
gpu_list=${GPU_LIST:-0}

model=${1:-"vicuna"}
dataset=${2:-"arxiv"}
bs=${3:-4}
emb=${4:-"simteg"}
template=${5:-"ND"}
freeze_backbone=${FREEZE_BACKBONE:-True}
classifier_pooling=${CLASSIFIER_POOLING:-graph_mean}

if [ "${model}" = "vicuna" ]; then
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
  prefix=llaga-vicuna-7b-${emb}-${template,,}-classifier
elif [ "${model}" = "llama" ]; then
  model_base=meta-llama/Llama-2-7b-hf
  mode="llaga_llama_2"
  prefix=llaga-llama-2-7b-hf-${emb}-${template,,}-classifier
else
  echo "Unsupported model: ${model}"
  exit 1
fi

if [ "${freeze_backbone}" = "True" ] || [ "${freeze_backbone}" = "true" ]; then
  prefix=${prefix}-frozen
fi

prefix=${prefix}-${classifier_pooling}

if [ "${template}" = "HO" ]; then
  use_hop=0
else
  use_hop=2
fi

echo "PREFIX: ${prefix}"
echo "GPUS: ${gpu_list}"
echo "FREEZE_BACKBONE: ${freeze_backbone}"
echo "CLASSIFIER_POOLING: ${classifier_pooling}"

deepspeed --include localhost:${gpu_list} --master_port 61010 train/train_mem_classifier.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path ${model_base} \
  --version ${mode} \
  --cache_dir ../../checkpoint \
  --pretrained_embedding_type ${emb} \
  --bf16 True \
  --output_dir /localnvme/llaga/checkpoints/${dataset}/${prefix} \
  --num_train_epochs 10 \
  --per_device_train_batch_size ${bs} \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length ${max_len} \
  --gradient_checkpointing True \
  --freeze_backbone ${freeze_backbone} \
  --classifier_pooling ${classifier_pooling} \
  --dataset ${dataset} \
  --pretrained_embedding_type ${emb} \
  --use_hop ${use_hop} \
  --sample_neighbor_size ${sample_size} \
  --template ${template}
