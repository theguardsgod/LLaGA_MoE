#!/bin/bash

model_path="/home/23131884r/code/LLaGA/checkpoints/arxiv-products-pubmed-cora.3/llaga-vicuna-7b-simteg-2-10-linear-projector_nc-lp/"
model_base="lmsys/vicuna-7b-v1.5-16k" #meta-llama/Llama-2-7b-hf
mode="v1" # use 'llaga_llama_2' for llama and "v1" for others
dataset="arxiv" #test dataset
task="nc" #test task
emb="simteg"
use_hop=2 # 2 for ND and 4 for HO
sample_size=10
template="ND"
output_path="/localnvme/llaga/output"

CUDA_VISIBLE_DEVICES=0  python eval/eval_pretrain.py \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file ${output_path} \
--task ${task} \
--cache_dir ../../checkpoint \
--template ${template}