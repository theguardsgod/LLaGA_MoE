# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LLaGA (Large Language and Graph Assistant) — an ICML 2024 paper that integrates graph neural network representations with large language models for graph understanding tasks. It projects graph structure embeddings into the LLM token space via learned projectors.

## Environment Setup

```bash
conda create -n llaga python=3.10
conda activate llaga
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.3.2 --no-build-isolation
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements.txt
```

Key pinned versions: `deepspeed==0.11.1`, `transformers==4.31.0`, `peft==0.5.0`, `sentence_transformers==2.2.2`.

## Commands

### Training

Single GPU:
```bash
bash scripts/train.sh <model_type> <task> <dataset> [batch_size] [embedding_type]
```

Multi-GPU (DeepSpeed Zero2):
```bash
bash scripts/train_deepspeed.sh <model_type> <task> <dataset> [batch_size] [embedding_type]
```

QLoRA fine-tuning:
```bash
bash scripts/train_deepspeed_qlora.sh <model_type> <task> <dataset> [batch_size] [embedding_type]
```

**Arguments:**
- `model_type`: `vicuna` (ND/2-hop), `vicuna_4hop` (HO/4-hop), `vicuna_2layer` (2-layer MLP projector), `llama` (LLaMA-2-7B), `opt_2.7b`
- `task`: `nc` (node classification), `lp` (link prediction), `nd` (node description)
- `dataset`: `arxiv`, `products`, `pubmed`, `cora` (append `_x3`/`_x5` for repetition)
- `batch_size`: default 16
- `embedding_type`: `simteg` (default), `sbert`, `roberta`

### Evaluation

```bash
bash scripts/eval.sh
```

Core eval script: `python eval/eval_pretrain.py` with args for model path, graph data, and task.

## Architecture

### Data Flow

1. **Graph embeddings** (SimTEG/SBERT/RoBERTa/E5) are loaded from pre-computed `.pt` files in `/localnvme/llaga/dataset/`
2. **Multi-hop sampling** (`utils/data_process.py`) extracts k-hop subgraph sequences around each node with fixed neighbor counts per hop
3. **Projector** (`model/llaga_arch.py`) maps graph embeddings to LLM hidden dimension via linear or 2-layer MLP
4. **LLM** generates text conditioned on projected graph tokens inserted at `<graph>` token positions

### Model Layer (`model/`)

- `llaga_arch.py` — `LlagaMetaModel` (projector init, graph token handling) and `LlagaMetaForCausalLM` (encoding interface)
- `language_model/llaga_llama.py`, `llaga_opt.py`, `llaga_mpt.py` — LLM-specific wrappers (`LlagaLlamaForCausalLM`, `LlagaOPTForCausalLM`, `LlagaMPTForCausalLM`)
- `builder.py` — Model/tokenizer loading with LoRA and quantization setup

### Training Layer (`train/`)

- `train.py` — Main entry point; uses HuggingFace `HfArgumentParser` with `ModelArguments`, `DataArguments`, `TrainingArguments`; defines `LagaDataset` with lazy preprocessing
- `train_qlora.py` — QLoRA variant with 4-bit NF4 quantization
- `llaga_trainer.py` — Custom `Trainer` subclass handling DeepSpeed state saving

### Key Constants (`utils/constants.py`)

- Graph pad token value: `-500`
- Graph token index: `-200`
- Special tokens: `<graph>`, `<pad>`

### Conversation Templates (`utils/conversation.py`)

Templates: `SINGLE`, `TWO`, `MPT`, `PLAIN`, `LLAMA_2` — each defines system message, role separators, and stop tokens for the corresponding model family.

### Graph Processing (`utils/data_process.py`)

- `get_fix_shape_subgraph_sequence_fast()` — Fixed-size k-hop neighbor extraction
- `generate_multi_hop_x()` — Generates multi-hop embedding tensors
- `load_pretrain_embedding_hop()` — Loads and caches pre-computed embeddings with hop structure

## Data

Training data and pre-computed embeddings are stored under `dataset/`. Download from the Box link in README.md. Each dataset has graph structure files and embedding `.pt` files per embedding type.

## DeepSpeed Configs

Located in `scripts/`: `zero2.json`, `zero3.json`, `zero2_offload.json`, `zero3_offload.json`. Training scripts default to Zero2.
