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

## MoE (Mixture of Experts) Extension

Replaces the single `mm_projector` with multiple expert projectors routed by a learned Top-K gating network. All MoE code is in separate files — no existing LLaGA files were modified. Full documentation: `docs/MoE_Implementation.md`.

### How It Works

1. **Offline graph partitioning**: Louvain community detection splits the graph into communities. Mean-pooled SimTEG embeddings (dim=2432) per community serve as routing features.
2. **TopKRouter**: A linear gate over community features → softmax → top-K selection. Produces per-sample expert weights and indices, plus a load-balancing auxiliary loss.
3. **Expert projectors**: `num_experts` (default 4) independent projectors, each with the same architecture as the original `mm_projector` (Linear 2543→4096 for ND template).
4. **DDP-safe forward**: All experts compute on all inputs unconditionally, then `torch.gather` selects top-K outputs and weights them. This avoids NCCL deadlock from mismatched gradient reductions across ranks.
5. **Loss**: `total_loss = lm_loss + aux_loss_weight * aux_loss` (default `aux_loss_weight=0.01`).

### MoE Files

| File | Purpose |
|------|---------|
| `model/moe_llaga.py` | TopKRouter, GraphProjectorExpert, MoEGraphProjector |
| `model/language_model/moe_llaga_llama.py` | MoELlagaConfig, MoELlagaLlamaForCausalLM |
| `train/train_moe.py` | MoE dataset (with routing_features), collator, training loop |
| `train/train_moe_mem.py` | FlashAttn wrapper for MoE training |
| `eval/eval_moe.py` | MoE evaluation with explicit MoE projector loading |
| `utils/graph_partition.py` | LouvainGraphPartitioner |
| `scripts/partition_graph.py` | Offline Louvain preprocessing script |
| `scripts/train_moe.sh` | DeepSpeed multi-GPU training launcher |
| `scripts/eval_moe_all.sh` | Sequential eval across all datasets |
| `scripts/eval_moe_parallel.sh` | Parallel eval splitting samples across GPUs |

### MoE Commands

Partition graphs (run once):
```bash
python scripts/partition_graph.py --dataset arxiv products pubmed cora
```

Train:
```bash
bash scripts/train_moe.sh [task] [dataset] [num_experts] [top_k] [batch_size]
# Example: bash scripts/train_moe.sh nc arxiv 4 2 16
```

Evaluate (parallel across GPUs):
```bash
bash scripts/eval_moe_parallel.sh <dataset> <gpu_list>
# Example: bash scripts/eval_moe_parallel.sh arxiv 1,2,3
```

### MoE Eval Loading Gotcha

`MoELlagaLlamaForCausalLM.__init__` sets `self.moe_projector = None`, so `from_pretrained()` silently ignores MoE weights from the checkpoint. The eval script must explicitly create the MoE module from config and load weights from `moe_projector.bin` after `from_pretrained`.

## Data

Training data and pre-computed embeddings are stored under `dataset/`. Download from the Box link in README.md. Each dataset has graph structure files and embedding `.pt` files per embedding type.

## DeepSpeed Configs

Located in `scripts/`: `zero2.json`, `zero3.json`, `zero2_offload.json`, `zero3_offload.json`. Training scripts default to Zero2.
