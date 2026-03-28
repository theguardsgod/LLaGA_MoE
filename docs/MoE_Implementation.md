# MoE-LLaGA: Mixture of Experts Implementation

## Overview

This document describes the implementation of a **Subgraph Partitioning + Mixture of Experts (MoE)** framework built on top of the LLaGA (Large Language and Graph Assistant) codebase. The core idea is to replace LLaGA's single graph-to-LLM projector (`mm_projector`) with multiple expert projectors, each specializing in different graph community structures, routed by a learned Top-K gating network.

**Design constraint**: All MoE code lives in newly created files. No existing LLaGA files were modified.

### Architecture Diagram

```
                        +-----------------------+
                        |  Graph Embeddings     |
                        |  [B, S, 2543]         |
                        +-----------+-----------+
                                    |
                    +---------------+---------------+
                    |                               |
          +---------v----------+          +---------v-----------+
          | Community Features |          |   MoE Projector     |
          | [B, 2432]         |          |                     |
          +--------+-----------+          |  Expert 0 (Linear)  |
                   |                      |  Expert 1 (Linear)  |
          +--------v-----------+          |  Expert 2 (Linear)  |
          |   TopKRouter       |          |  Expert 3 (Linear)  |
          |   (Linear gate +   |          +----+----+----+------+
          |    softmax + topK) |               |    |    |
          +--------+-----------+               |    |    |
                   |                           |    |    |
           weights & indices                   |    |    |
                   |           +---------------+    |    |
                   +---------->| Gather Top-K       |    |
                               | + Weighted Sum     |<---+
                               +---------+----------+
                                         |
                               +---------v----------+
                               | Projected Tokens    |
                               | [B, S, 4096]        |
                               +---------+----------+
                                         |
                               +---------v----------+
                               |  Vicuna-7B LLM     |
                               |  (frozen backbone)  |
                               +--------------------+
```

## Step 1: Subgraph Partitioning (Louvain Community Detection)

### Purpose

Partition the graph into communities so that each node can be assigned a community-level feature vector. These features serve as the routing signal for the MoE gating network.

### Implementation

**File**: `utils/graph_partition.py` -- `LouvainGraphPartitioner` class

1. **Convert** PyG `edge_index` to NetworkX graph (undirected)
2. **Run** `community_louvain.best_partition()` from the `python-louvain` package
3. **Post-process**: merge communities smaller than `min_size` into their most-connected neighbor, then re-index contiguously
4. **Compute community features**: mean-pool SimTEG node embeddings (dim=2432) per community

**File**: `scripts/partition_graph.py` -- offline preprocessing script

```bash
# Partition all datasets
python scripts/partition_graph.py --dataset arxiv products pubmed cora
```

**Outputs** (saved to each dataset directory):
- `node_to_community.pt` -- `[N]` int64 tensor mapping each node to a community ID
- `community_features.pt` -- `[C, 2432]` float tensor of mean-pooled SimTEG embeddings per community
- `partition_info.pt` -- metadata dict with community sizes and parameters

### Partitioning Results

| Dataset  | Nodes   | Edges     | Communities | Avg Size |
|----------|---------|-----------|-------------|----------|
| arxiv    | 169,343 | 1,166,243 | 102         | 1,660    |
| products | 2,449,029 | 61,859,140 | 190       | 12,890   |
| pubmed   | 19,717  | 44,338    | 36          | 548      |
| cora     | 2,708   | 5,278     | 28          | 97       |

## Step 2: MoE Forward Propagation

### Core MoE Components

**File**: `model/moe_llaga.py`

#### TopKRouter

```python
class TopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, noise_std=1.0):
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
```

- **Input**: Community features `[B, 2432]`
- **Process**: Linear gate -> (optional training noise) -> softmax -> top-K selection
- **Output**: `weights [B, K]`, `indices [B, K]`, `aux_loss` (scalar)
- **Load-balancing loss** (Switch Transformer style):
  ```
  L_balance = num_experts * sum_i(f_i * P_i)
  ```
  where `f_i` = fraction of tokens routed to expert `i`, `P_i` = mean routing probability for expert `i`

#### GraphProjectorExpert

A single expert wrapping a projector with the same architecture as the original `mm_projector`:

```python
class GraphProjectorExpert(nn.Module):
    def __init__(self, mm_hidden_size, llm_hidden_size, projector_type='linear'):
        self.projector = nn.Linear(mm_hidden_size, llm_hidden_size)  # 2543 -> 4096
```

#### MoEGraphProjector

Full MoE module combining router + experts + dispatch/combine:

```python
class MoEGraphProjector(nn.Module):
    def __init__(self, mm_hidden_size, llm_hidden_size, num_experts=4, top_k=2, ...):
        self.router = TopKRouter(routing_dim, num_experts, top_k, noise_std)
        self.experts = nn.ModuleList([
            GraphProjectorExpert(mm_hidden_size, llm_hidden_size, projector_type)
            for _ in range(num_experts)
        ])
```

### DDP-Safe Forward Pass (Critical Design Decision)

The forward pass computes **all experts on all inputs unconditionally**, then uses `torch.gather` to select and weight the top-K outputs. This is essential for DeepSpeed/DDP compatibility.

**Why**: In distributed training, all ranks must execute identical parameter operations so that gradient reduction (allreduce) stays synchronized. If experts were conditionally skipped based on routing decisions (which differ per rank due to different data), NCCL operations would become mismatched across ranks, causing a deadlock.

```python
def forward(self, graph_emb, routing_features, graph_mask=None):
    B, S, D = graph_emb.shape

    # Step 1: Route
    weights, indices, aux_loss = self.router(routing_features)  # [B,K], [B,K]

    # Step 2: Compute ALL experts (DDP-safe)
    all_expert_outputs = []
    for expert_id in range(self.num_experts):
        all_expert_outputs.append(self.experts[expert_id](graph_emb))
    all_expert_outputs = torch.stack(all_expert_outputs, dim=1)  # [B, E, S, H]

    # Step 3: Gather top-K and weight
    idx_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(B, K, S, H)
    selected = torch.gather(all_expert_outputs, dim=1, index=idx_expanded)
    w = weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
    combined = (w * selected).sum(dim=1)     # [B, S, H]

    return combined, aux_loss
```

### Model Integration

**File**: `model/language_model/moe_llaga_llama.py`

- `MoELlagaConfig(LlamaConfig)` -- adds `num_experts`, `top_k`, `routing_dim`, `aux_loss_weight`, `noise_std`
- `MoELlagaLlamaForCausalLM(LlamaForCausalLM, LlagaMetaForCausalLM)`:
  - `initialize_moe_modules()` -- creates MoEGraphProjector from model_args
  - `encode_graphs()` -- overrides parent, routes through MoE projector, returns aux_loss
  - `prepare_inputs_labels_for_multimodal_moe()` -- same token-replacement logic as parent, but propagates aux_loss
  - `forward()` -- total loss = `lm_loss + aux_loss_weight * aux_loss`
  - `prepare_inputs_for_generation()` -- passes `routing_features` through to generation

### Loss Function

```python
loss = lm_loss + self.aux_loss_weight * aux_loss
```

- `lm_loss`: standard cross-entropy on next-token prediction
- `aux_loss`: load-balancing auxiliary loss (encourages uniform expert utilization)
- `aux_loss_weight`: default 0.01

## Training

### Files

| File | Purpose |
|------|---------|
| `train/train_moe.py` | Main training script with MoE dataset, collator, and training loop |
| `train/train_moe_mem.py` | Wrapper that applies FlashAttention monkey patch before training |
| `scripts/train_moe.sh` | Shell script for DeepSpeed multi-GPU training |

### Data Pipeline

`MoELazySupervisedGraphDataset` extends the original dataset by:
1. Loading community partition data (`node_to_community.pt`, `community_features.pt`)
2. For each sample, looking up the center node's community ID
3. Returning the community's mean-pooled feature vector as `routing_features`

`MoEDataCollatorForSupervisedDataset` collates `routing_features` alongside `graph`, `graph_emb`, `input_ids`, and `labels`.

### Training Command

```bash
bash scripts/train_moe.sh [task] [dataset] [num_experts] [top_k] [batch_size]
# Example:
bash scripts/train_moe.sh nc arxiv 4 2 16
```

Runs DeepSpeed Zero2 on GPUs 1,2,3:
```bash
deepspeed --include localhost:1,2,3 --master_port 61000 train/train_moe_mem.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path lmsys/vicuna-7b-v1.5-16k \
  --tune_mm_mlp_adapter True \
  --bf16 True \
  --num_experts 4 --top_k 2 \
  --aux_loss_weight 0.01 --noise_std 1.0 \
  ...
```

### What Gets Trained

With `--tune_mm_mlp_adapter True`:
- LLM backbone (Vicuna-7B): **frozen**
- MoE projector (router + 4 experts): **trainable**
- Base mm_projector: **trainable** (kept as fallback)

### Training Results

| Dataset  | Steps | Duration | Final Train Loss |
|----------|-------|----------|-----------------|
| arxiv    | 1,895 | 3h 23m   | 0.105           |
| products | 4,097 | 5h 43m   | 0.086           |
| pubmed   | 247   | 15m      | 0.085           |
| cora     | 34    | 3m       | 0.837           |

Checkpoints saved to: `checkpoints/{dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc/`

Each checkpoint contains:
- Full model weights (`pytorch_model-*.bin`)
- MoE projector weights (`moe_projector.bin`, saved separately for easy loading)
- Config (`config.json` with MoE parameters)

## Evaluation

### Files

| File | Purpose |
|------|---------|
| `eval/eval_moe.py` | Single-GPU MoE evaluation script |
| `scripts/eval_moe_all.sh` | Sequential evaluation across all datasets |
| `scripts/eval_moe_parallel.sh` | Parallel evaluation splitting samples across GPUs |

### Model Loading

A critical implementation detail: `MoELlagaLlamaForCausalLM.__init__` sets `self.moe_projector = None`. When `from_pretrained()` loads the checkpoint, the MoE weights have no corresponding module and are silently ignored. The eval script explicitly reconstructs the MoE projector from config and loads weights from `moe_projector.bin`:

```python
if model.moe_projector is None:
    cfg = model.config
    model.moe_projector = MoEGraphProjector(
        mm_hidden_size=cfg.mm_hidden_size,
        llm_hidden_size=cfg.hidden_size,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        ...
    )
    moe_weights = torch.load(os.path.join(model_path, 'moe_projector.bin'))
    model.moe_projector.load_state_dict(clean_weights)
```

### Parallel Evaluation

`scripts/eval_moe_parallel.sh` splits test samples by index ranges across GPUs:

```bash
bash scripts/eval_moe_parallel.sh arxiv 1,2,3
```

1. Counts total test samples, divides into chunks
2. Launches parallel `eval_moe.py` processes with `--start` and `--end` flags
3. Waits for all processes to complete
4. Merges chunk files into a single output
5. Scores using `eval/eval_res.py`

### Evaluation Results (Node Classification)

| Model | arxiv |
|-------|-------|
| Baseline projector-only | 75.09% |
| MoE (E=4, K=2, projector-only) | 75.14% |
| Full fine-tune (LLM + projector) | 75.64% |

## File Summary

```
LLaGA/
+-- model/
|   +-- moe_llaga.py                         # TopKRouter, GraphProjectorExpert, MoEGraphProjector
|   +-- language_model/
|       +-- moe_llaga_llama.py                # MoELlagaConfig, MoELlagaLlamaForCausalLM
+-- train/
|   +-- train_moe.py                          # MoE dataset, collator, training loop
|   +-- train_moe_mem.py                      # FlashAttn wrapper
+-- eval/
|   +-- eval_moe.py                           # MoE evaluation with model loading fix
+-- utils/
|   +-- graph_partition.py                    # LouvainGraphPartitioner
+-- scripts/
|   +-- partition_graph.py                    # Offline Louvain preprocessing
|   +-- train_moe.sh                          # DeepSpeed training launcher
|   +-- eval_moe_all.sh                       # Sequential eval launcher
|   +-- eval_moe_parallel.sh                  # Parallel eval launcher
```

## Key Dimensions Reference

| Component | Dimension | Notes |
|-----------|-----------|-------|
| SimTEG embedding | 2432 | SBERT(384) + RoBERTa(1024) + E5(1024) |
| Laplacian PE (ND template) | 111 | `(10^3 - 1) / (10 - 1)` for hop=2, sample=10 |
| mm_hidden_size (ND) | 2543 | 2432 + 111 |
| mm_hidden_size (HO) | 2432 | No Laplacian PE |
| LLM hidden size | 4096 | Vicuna-7B |
| routing_dim | 2432 | Community features (SimTEG mean-pooled) |
| num_experts | 4 | Default |
| top_k | 2 | Default |

## Known Issues and Lessons Learned

1. **NCCL Deadlock with Conditional Expert Execution**: The initial implementation skipped experts with no routed samples (`if mask.sum() == 0: continue`). This caused NCCL timeout (30min) because different ranks had different routing decisions, leading to mismatched gradient reduction operations. **Fix**: Compute all experts unconditionally, use `torch.gather` for selection.

2. **Silent Weight Dropping in `from_pretrained`**: Setting `self.moe_projector = None` in `__init__` causes `from_pretrained` to silently discard MoE checkpoint weights. The model falls back to the untrained base projector, producing 0% accuracy. **Fix**: Explicitly reconstruct MoE module and load from `moe_projector.bin` after `from_pretrained`.

3. **Import Path Mismatch with DeepSpeed**: DeepSpeed runs from the project root, but `train_moe.py` initially used `from train.llaga_trainer import LLaGATrainer` while the original `train.py` uses `from llaga_trainer import LLaGATrainer`. **Fix**: Use `sys.path.append(".")` and relative imports matching the original pattern.

4. **Missing `group_by_modality_length`**: The custom `LLaGATrainer._get_train_sampler()` checks for `self.args.group_by_modality_length`, which wasn't in MoE's `TrainingArguments`. **Fix**: Added the field with `default=False`.
