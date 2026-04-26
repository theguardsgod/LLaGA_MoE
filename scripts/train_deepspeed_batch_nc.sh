#!/bin/bash

max_len=4096
sample_size=10

model=${1:-"vicuna"}
dataset=${2:-"arxiv"}
nc_batch_size=${3:-20}
bs=${4:-4}
emb=${5:-"simteg"}

task="nc_batch"

if [ ${model} = "vicuna" ]; then
  use_hop=2
  template="ND"
  projector_type="linear"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector-batch${nc_batch_size}
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "vicuna_2layer" ]; then
  use_hop=2
  template="ND"
  projector_type="2-layer-mlp"
  prefix=llaga-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector-batch${nc_batch_size}
  model_base=lmsys/vicuna-7b-v1.5-16k
  mode="v1"
elif [ ${model} = "llama" ]; then
  use_hop=2
  template="ND"
  projector_type="linear"
  prefix=llaga-llama-2-7b-hf-${emb}-${use_hop}-${sample_size}-${projector_type}-projector-batch${nc_batch_size}
  model_base=meta-llama/Llama-2-7b-hf
  mode="llaga_llama_2"
fi

echo "PREFIX:  ${prefix}"
echo "NC_BATCH_SIZE: ${nc_batch_size}"

export BNB_CUDA_VERSION=121

# Step 1: Check batch NC data exists (generate with graph_gpt env if needed)
# python scripts/prepare_batch_nc_data.py --dataset ${dataset} --batch_size ${nc_batch_size} --use_hop ${use_hop} --sample_neighbor_size ${sample_size}

wandb online
deepspeed --include localhost:0,1,2,3 --master_port 61000 train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir ../../checkpoint \
--pretrained_embedding_type ${emb} \
--tune_mm_mlp_adapter True \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--bf16 True \
--output_dir /localnvme/llaga/checkpoints/${dataset}/${prefix}_${task} \
--num_train_epochs 10 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--save_total_limit 2 \
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
--nc_batch_size ${nc_batch_size}
