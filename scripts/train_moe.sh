#!/bin/bash
# Train MoE-LLaGA model
# Usage: bash scripts/train_moe.sh [task] [dataset] [num_experts] [top_k]

task=${1:-"nc"}
dataset=${2:-"arxiv"}
num_experts=${3:-4}
top_k=${4:-2}
bs=${5:-16}
emb="simteg"

use_hop=2
template="ND"
projector_type="linear"
model_base=lmsys/vicuna-7b-v1.5-16k
mode="v1"
max_len=4096
sample_size=10

prefix=moe-llaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-E${num_experts}-K${top_k}

echo "PREFIX: ${prefix}"
echo "Experts: ${num_experts}, Top-K: ${top_k}"

wandb online

deepspeed --include localhost:1,2,3 --master_port 61000 train/train_moe_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir ../../checkpoint \
--pretrained_embedding_type ${emb} \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--output_dir ./checkpoints/${dataset}/${prefix}_${task} \
--num_train_epochs 1 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length ${max_len} \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to wandb \
--run_name "${prefix}_${task}_${dataset}" \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--mm_projector_type ${projector_type} \
--use_task ${task} \
--use_dataset ${dataset} \
--template ${template} \
--num_experts ${num_experts} \
--top_k ${top_k} \
--aux_loss_weight 0.01 \
--noise_std 1.0
