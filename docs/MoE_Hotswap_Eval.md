# MoE Hot-Swap Evaluation

在单个 GPU 上高效评估多个 MoE checkpoint，只加载一次 base LLM，热切换 MoE projector 权重。

## 动机

原始的 `eval_moe_all.sh` 对每个数据集启动独立的 Python 进程，每次都重新加载完整的 Vicuna-7B 模型（~14GB FP16）。但所有 MoE checkpoint 共享同一个 base LLM，只有 MoE projector 权重不同（~80MB）。

| 方式 | 模型加载次数 | 每次加载量 | 4 数据集总加载量 |
|------|-------------|-----------|-----------------|
| `eval_moe_all.sh`（原始） | 4 次 | ~14GB | ~56GB |
| `eval_moe_hotswap.sh`（本方案） | 1 次 | ~14GB + 3×80MB | ~14.2GB |

## 使用方式

```bash
# 评估所有数据集（arxiv, products, pubmed, cora）
bash scripts/eval_moe_hotswap.sh [gpu_id]
bash scripts/eval_moe_hotswap.sh 0

# 只评估指定数据集
bash scripts/eval_moe_hotswap.sh [gpu_id] [dataset1] [dataset2] ...
bash scripts/eval_moe_hotswap.sh 0 arxiv cora
```

也可以直接调用 Python 脚本：

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_moe_hotswap.py \
    --datasets arxiv products pubmed cora \
    --model_base "lmsys/vicuna-7b-v1.5-16k" \
    --checkpoint_pattern "./checkpoints/{dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc" \
    --output_dir "/localnvme/llaga/eval_output" \
    --conv_mode v1 \
    --pretrained_embedding_type simteg \
    --use_hop 2 \
    --sample_neighbor_size 10 \
    --task nc \
    --cache_dir "../../checkpoint" \
    --template "ND"
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | `arxiv products pubmed cora` | 要评估的数据集列表 |
| `--model_base` | `None` | Base LLM 路径，推荐设置以避免从 checkpoint 加载完整模型 |
| `--checkpoint_pattern` | `./checkpoints/{dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc` | Checkpoint 路径模板，`{dataset}` 会被替换 |
| `--output_dir` | `/localnvme/llaga/eval_output` | 输出目录 |
| `--task` | `nc` | 任务类型：`nc`（节点分类）、`nd`（节点描述）、`lp`（链接预测） |
| `--template` | `ND` | 图模板：`ND`（邻居描述）或 `HO`（多跳） |
| `--conv_mode` | `v1` | 对话模板 |
| `--pretrained_embedding_type` | `simteg` | 嵌入类型 |
| `--use_hop` | `2` | 跳数 |
| `--sample_neighbor_size` | `10` | 每跳采样邻居数 |
| `--temperature` | `0.2` | 生成温度 |
| `--start` / `--end` | `-1` | 评估样本范围（用于调试） |
| `--cache_dir` | `../../checkpoint` | HuggingFace 模型缓存目录 |

## 工作原理

### 执行流程

```
┌─────────────────────────────────────────────────┐
│  1. load_base_model()                           │
│     - 加载 Vicuna-7B LLM（一次，~14GB）            │
│     - 从第一个 checkpoint config 初始化空            │
│       MoEGraphProjector                          │
│     - model.to(fp16).cuda().eval()               │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  for ds in datasets:    │
          │                         │
          │  2. swap_moe_projector()│◄── 只替换 ~80MB
          │     load_state_dict()   │    权重，<1 秒
          │                         │
          │  3. load_dataset_data() │◄── 加载该数据集的
          │     embeddings          │    嵌入和社区数据
          │     community features  │
          │     test prompts        │
          │                         │
          │  4. run_eval()          │◄── 逐样本推理
          │     model.generate()    │
          │                         │
          │  5. score results       │◄── subprocess 调用
          │     eval_res.py         │    eval_res.py
          │                         │
          │  6. 释放数据集数据        │◄── del + gc.collect()
          │     gc + empty_cache    │    + empty_cache()
          └────────────┬────────────┘
                       │
                       ▼
              All evaluations done
```

### 核心函数

#### `load_base_model(first_checkpoint_path, model_base, cache_dir)`

加载 base LLM 并初始化空的 MoE projector。

- 如果提供 `model_base`：从 HuggingFace 加载 base LLM，从第一个 checkpoint 读取 config
- 如果未提供 `model_base`：从第一个 checkpoint 加载完整模型
- 创建 `MoEGraphProjector`（不加载权重，留给 `swap_moe_projector`）

#### `swap_moe_projector(model, checkpoint_path)`

热切换 MoE projector 权重。

```python
# 1. 加载权重到 CPU
moe_weights = torch.load(moe_path, map_location='cpu')

# 2. 去除前缀，转 FP16
clean = {k.replace("moe_projector.", ""): v.half() for k, v in moe_weights.items()}

# 3. 原地替换参数（load_state_dict 替换所有 parameter tensor）
model.moe_projector.load_state_dict(clean)
model.moe_projector.to(device=model.device)

# 4. 释放 CPU 副本
del moe_weights, clean
gc.collect()
torch.cuda.empty_cache()
```

**Config 兼容性检查**：如果不同 checkpoint 的 MoE config 不同（如 `num_experts` 不同），自动重建 `MoEGraphProjector` 模块再加载权重。

#### `load_dataset_data(dataset, args)`

加载数据集特定的数据，返回一个 dict：

- `data`：PyG Data 对象（`processed_data.pt`）
- `questions`：测试样本列表
- `pretrained_emb`：SimTEG/SBERT/RoBERTa 嵌入
- `structure_emb`：Laplacian 结构嵌入（ND 模板）
- `node_to_community` / `community_features`：社区数据（MoE 路由用）
- `index`：节点索引映射（HO 模板）

切换数据集时整体 `del` 这个 dict 释放内存。

#### `run_eval_single_dataset(model, tokenizer, dataset_data, dataset, output_file, args)`

推理循环，与 `eval_moe.py` 逻辑一致：

- 支持 resume（检查 output file 已有行数，跳过已完成的样本）
- 逐样本构建 graph embedding + routing features
- 调用 `model.generate()` 生成文本
- 写入 JSONL 结果文件

## 文件结构

```
eval/
├── eval_moe.py            # 单 checkpoint 评估（不修改）
├── eval_moe_hotswap.py    # 多 checkpoint 热切换评估（新增）
└── eval_res.py            # 评分脚本（不修改）

scripts/
├── eval_moe_all.sh        # 原始顺序评估（不修改）
├── eval_moe_parallel.sh   # 多 GPU 并行评估（不修改）
└── eval_moe_hotswap.sh    # 热切换评估启动脚本（新增）
```

## 为什么不需要切换 mm_projector

`MoELlagaLlamaForCausalLM.encode_graphs()`（`moe_llaga_llama.py:130`）的逻辑：

```python
if self.moe_projector is not None and routing_features is not None:
    # MoE 路径 — 使用 moe_projector
    graph_features, aux_loss = self.moe_projector(graph_emb, routing_features, graph_mask)
else:
    # 回退路径 — 使用 mm_projector
    graph_features = self.get_model().mm_projector(graph_emb)
```

在 hot-swap 评估中，`moe_projector` 始终不为 `None`，`routing_features` 始终提供，所以 `mm_projector` 永远不会被调用。
