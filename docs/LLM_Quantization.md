# LLM Quantization

为 LLaGA 推理阶段增加 4-bit / 8-bit 量化支持，降低 LLM backbone 显存占用。

当前实现采用“新增入口文件，不改旧入口文件”的方式落地。原有 `eval/eval_pretrain.py`、`eval/eval_moe.py`、`eval/eval_moe_hotswap.py` 保持不变；量化能力通过新的 `*_quantized.py` 入口提供。

## 背景

| 配置 | LLM backbone 显存 | 含 projector 总显存 |
|------|-------------------|-------------------|
| FP16 | ~14 GB | ~14.1 GB |
| INT8 (8-bit) | ~7 GB | ~7.1 GB |
| NF4 + double quant (4-bit) | ~3.5 GB | ~3.6 GB |

核心思路是只量化 LLM backbone，graph projector 仍保持 FP16：

```text
Graph embeddings (FP16)
  -> mm_projector / MoE projector (FP16)
  -> quantized LLM backbone (4-bit / 8-bit)
```

量化基于 `bitsandbytes` 和 Hugging Face `BitsAndBytesConfig`。量化模型不能再调用 `model.to(torch.float16).cuda()`，因此运行时改为：

1. 通过 `device_map="auto"` 加载量化 LLM
2. 显式把 MoE projector 搬到实际运行设备
3. 按 runtime device 放置 `input_ids`、`graph_emb`、`routing_features`

## 已实现文件

### 1. `eval/quantization_utils.py`

共享量化辅助模块，提供：

- `add_quantization_args(parser)`：统一注入 `--load_4bit` / `--load_8bit`
- `validate_quantization_args(args)`：禁止同时启用 4-bit 和 8-bit
- `build_quantization_kwargs(...)`：生成 bitsandbytes 加载参数（量化路径包含 `torch_dtype=torch.float16`，确保非量化层不会回退到 float32）
- `get_input_device(model)`：解析量化模型的输入设备
- `get_graph_device(model)`：解析 projector 的运行设备
- `move_moe_projector_to_runtime_device(model)`：显式搬运 MoE projector
- `restore_fp16_mm_projector(...)`：在 baseline 量化路径中重建并恢复 FP16 `mm_projector`
- `prepare_model_for_inference(...)`：统一处理 FP16 与量化两种加载模式
- `build_moe_projector_from_config(cfg)`：从 model config 构建 MoEGraphProjector（共享，避免各入口重复定义）
- `load_clean_projector_weights(path, prefix)`：加载 projector 权重，去前缀并转 FP16（共享，避免各入口重复定义）

实现位置：`eval/quantization_utils.py`

### 2. `eval/eval_pretrain_quantized.py`

标准 LLaGA 量化评估入口。

特点：

- 复用 `model/builder.py` 已有的 `load_pretrained_model(...)`
- 透传 `load_4bit` / `load_8bit`
- FP16 模式下仍走 `model.to(torch.float16).cuda()`
- 量化模式下不再调用 `.cuda()` 迁移整模，而是根据 runtime device 放置输入和 graph tensor
- baseline 量化模式下会把被 bitsandbytes 替换过的 `mm_projector` 重建回 FP16 模块，再加载 `mm_projector.bin`

实现位置：`eval/eval_pretrain_quantized.py`

### 3. `eval/eval_moe_quantized.py`

MoE 量化评估入口。

相比原始 `eval/eval_moe.py`，这个新入口额外处理了两类问题：

- 增加 4-bit / 8-bit 量化加载逻辑
- 在 `model_base is not None` 的路径上，直接根据 config 构建并加载 `moe_projector`，避免依赖不存在的 `_build_moe_from_config(...)`

运行时逻辑：

- 量化 LLM 通过 `device_map="auto"` 加载
- `mm_projector` / `moe_projector` 权重保持 FP16
- `routing_features`、`graph_emb`、`graph` 被移动到 projector 实际所在设备

实现位置：`eval/eval_moe_quantized.py`

### 4. `eval/eval_moe_hotswap_quantized.py`

多 checkpoint 热切换量化评估入口。

特点：

- 基座 LLM 只加载一次
- 每个 checkpoint 只热切换 `moe_projector.bin`
- 支持 4-bit / 8-bit 基座加载
- `swap_moe_projector(...)` 后会再次把 projector 放到 runtime device，避免 `model.device` 在 `device_map="auto"` 下不可靠的问题

实现位置：`eval/eval_moe_hotswap_quantized.py`

### 5. `scripts/eval_moe_hotswap_quantized.sh`

新的 shell wrapper，默认启用 `--load_4bit`。

实现位置：`scripts/eval_moe_hotswap_quantized.sh`

### 6. `eval/benchmark_baseline_quant_runtime.py`

baseline runtime benchmark 入口，用于在同一协议下对比：

- `FP16`
- `bitsandbytes 4bit`
- `bitsandbytes 8bit`
- `AWQ`

该脚本当前只覆盖：

- baseline
- `task=nc`
- `template=ND`

并输出以下运行时指标：

- `load_memory_mb`
- `torch_peak_mb`
- `total_time_s`
- `avg_time_per_sample_s`
- `tokens_per_second`

实现位置：`eval/benchmark_baseline_quant_runtime.py`

## 为什么 projector 不量化

`bitsandbytes` 只会量化 `from_pretrained(...)` 过程中进入量化加载路径的 LLM 线性层。这里的 graph projector 权重是后加载的：

- 标准 LLaGA：`mm_projector.bin`
- MoE：`moe_projector.bin`

这些 projector 仍以 FP16 参数存在，因此：

- projector 参数 dtype 保持 FP16
- graph embedding 到 projector 的计算保持 FP16
- projector 输出再喂给量化 LLM

## 使用方式

### 标准 LLaGA 4-bit 推理

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_pretrain_quantized.py \
    --model_path ./checkpoints/arxiv/llaga-vicuna-7b-... \
    --model_base lmsys/vicuna-7b-v1.5-16k \
    --dataset arxiv \
    --task nc \
    --template ND \
    --load_4bit
```

### 标准 LLaGA 8-bit 推理

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_pretrain_quantized.py \
    --model_path ./checkpoints/arxiv/llaga-vicuna-7b-... \
    --model_base lmsys/vicuna-7b-v1.5-16k \
    --dataset arxiv \
    --task nc \
    --template ND \
    --load_8bit
```

### MoE 4-bit 推理

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_moe_quantized.py \
    --model_path ./checkpoints/arxiv/moe-llaga-... \
    --model_base lmsys/vicuna-7b-v1.5-16k \
    --dataset arxiv \
    --task nc \
    --template ND \
    --load_4bit
```

### MoE 8-bit 推理

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_moe_quantized.py \
    --model_path ./checkpoints/arxiv/moe-llaga-... \
    --model_base lmsys/vicuna-7b-v1.5-16k \
    --dataset arxiv \
    --task nc \
    --template ND \
    --load_8bit
```

### MoE 热切换 4-bit 推理

```bash
bash scripts/eval_moe_hotswap_quantized.sh 0
```

### MoE 热切换 8-bit 推理

直接调用 Python 入口：

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_moe_hotswap_quantized.py \
    --datasets arxiv products pubmed cora \
    --model_base lmsys/vicuna-7b-v1.5-16k \
    --checkpoint_pattern ./checkpoints/{dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc \
    --output_dir /localnvme/llaga/eval_output \
    --pretrained_embedding_type simteg \
    --use_hop 2 \
    --sample_neighbor_size 10 \
    --task nc \
    --template ND \
    --load_8bit
```

## 参数说明

所有新的量化入口都支持以下参数：

```bash
--load_4bit
--load_8bit
```

约束：

- 二者互斥，不能同时传入
- 两者都不传时，行为等价于 FP16 推理

## 与旧脚本的关系

当前量化实现不会改动旧文件，因此：

- 旧入口继续保持原有行为
- 新功能统一从 `*_quantized.py` 入口进入
- 如果已有外部脚本依赖旧入口，不会被这次改动破坏

## 依赖

需要可用的 `bitsandbytes` 环境：

```bash
pip install bitsandbytes>=0.39.0
```

注意：当前仓库的 `requirements.txt` 没有显式声明 `bitsandbytes`，因此需要手动安装或在运行环境中预装。

## 验证

建议至少做以下检查。

### 1. 语法检查

已对以下新文件做过语法级验证：

```bash
python -m py_compile \
    eval/quantization_utils.py \
    eval/eval_pretrain_quantized.py \
    eval/eval_moe_quantized.py \
    eval/eval_moe_hotswap_quantized.py

bash -n scripts/eval_moe_hotswap_quantized.sh
```

### 2. 功能验证

- 对比 FP16 和 4-bit / 8-bit 输出是否在可接受误差范围内
- 用 `nvidia-smi` 确认显存下降是否符合预期
- 确认量化模式下不再触发 `model.to(...).cuda()` 相关错误

### 3. dtype 验证

不要使用 `model.moe_projector.experts[0].projector[0]` 这种写法，因为默认 `linear` projector 不是 `nn.Sequential`。

推荐使用：

```python
next(model.moe_projector.parameters()).dtype
next(model.get_model().mm_projector.parameters()).dtype
```

期望结果：projector 参数 dtype 为 `torch.float16`。

## 实测结果

以下结果为 2026-04-05 在同一组 baseline 配置上得到的实测对照：

- dataset: `cora`
- task: `nc`
- test samples: `542`
- model_path: `./checkpoints/cora/llaga-vicuna-7b-simteg-2-10-linear-projector_nc_v2`
- model_base: `lmsys/vicuna-7b-v1.5-16k`

| Mode | Entry | Output | Time | Acc |
|------|------|------|------|------|
| FP16 | `eval/eval_pretrain.py` | `/localnvme/llaga/eval_output/quant_experiments/baseline_fp16_nc_cora_fresh.jsonl` | `2m02s` | `0.8303` |
| 4bit | `eval/eval_pretrain_quantized.py` | `/localnvme/llaga/eval_output/quant_experiments/baseline_4bit_nc_cora_fresh.jsonl` | `5m07s` | `0.7915` |
| 8bit | `eval/eval_pretrain_quantized.py` | `/localnvme/llaga/eval_output/quant_experiments/baseline_8bit_nc_cora_fresh.jsonl` | `7m37s` | `0.8247` |

这组结果里的结论是：

- 精度排序为 `FP16 > 8bit > 4bit`
- 相对 `FP16`，`8bit` 精度下降 `0.0056`
- 相对 `FP16`，`4bit` 精度下降 `0.0388`
- 在当前环境和这组任务上，`8bit` 与 `4bit` 都没有带来更快的端到端生成时间

### 运行时显存 / 速度对比

以下结果为 2026-04-11 在同一组 baseline 配置上得到的 runtime benchmark：

- dataset: `cora`
- task: `nc`
- template: `ND`
- test samples: `64`
- GPU: `GPU 2`
- benchmark entry: `eval/benchmark_baseline_quant_runtime.py`
- summary: `/tmp/quant_runtime_compare_gpu2_20260411_093648/summary.json`

| Mode | `nvidia-smi` Peak | Torch Peak | Model Time | Avg / Sample | Wall Time |
|------|-------------------|------------|------------|--------------|-----------|
| FP16 | `13566 MB` | `13033.6 MB` | `11.88 s` | `0.186 s` | `24.96 s` |
| 4bit bnb | `7046 MB` | `4352.2 MB` | `67.81 s` | `1.060 s` | `94.12 s` |
| 8bit bnb | `7732 MB` | `6874.2 MB` | `41.41 s` | `0.647 s` | `66.26 s` |
| AWQ | `4630 MB` | `3969.7 MB` | `12.53 s` | `0.196 s` | `28.81 s` |

这组结果里的结论是：

- 速度排序为 `FP16 ≈ AWQ >> 8bit bnb > 4bit bnb`
- 显存排序为 `FP16 > 8bit bnb > 4bit bnb > AWQ`
- 相对 `FP16`，`AWQ` 的 `nvidia-smi` 峰值显存下降 `65.9%`
- 相对 `FP16`，`4bit bnb` 的 `nvidia-smi` 峰值显存下降 `48.1%`
- 相对 `FP16`，`8bit bnb` 的 `nvidia-smi` 峰值显存下降 `43.0%`
- 相对 `FP16`，`AWQ` 的纯推理时间只慢 `5.5%`
- 在当前环境和这组任务上，`bitsandbytes 4bit/8bit` 的主要收益是降显存，不是提速
- 在当前环境和这组任务上，`AWQ` 同时取得了最低显存占用和接近 `FP16` 的推理速度

说明：

- `Model Time` 指脚本内部用 `torch.cuda.synchronize()` 包住生成阶段得到的纯推理时间
- `Wall Time` 包含模型加载和外部调度开销
- `AWQ` 这次的 `tokens_per_second` 不作为主要比较指标，因为 `autoawq` 路径下生成输出长度统计与其余三种模式不完全一致

## 已知限制

- 当前只实现了新的量化入口，没有把量化参数回灌到旧入口文件
- 当前只实测了 baseline 路径；MoE / hotswap 量化入口还没有补完整的端到端实测结果
- `bitsandbytes` 是否可用仍取决于实际 CUDA / 驱动 / PyTorch 环境
