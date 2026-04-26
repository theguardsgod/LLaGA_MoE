# TODO

## 推理加速

### Quantization Latency 优化

目标：只测试 quantization 本身还能否继续降低 baseline latency，不切到新的 serving/runtime 框架。

**优先级顺序**

- `AWQ group_size=64` vs `AWQ group_size=128`
- `AWQ GEMV` vs `AWQ GEMM`
- `AWQ` vs `GPTQ`
- 如果环境允许，再看 `FP8 / W8A8`

**原因**

- 这些对比仍然属于 quantization 策略优化，结果更容易归因
- 不需要把 baseline 迁到 vLLM / TensorRT-LLM
- 已有结果已经说明 `bitsandbytes 4bit/8bit` 不是 latency 优化方向，优先级应下调

**不作为当前优先项**

- `fuse_awq_layers`
  - 当前和 LLaGA baseline 的 `inputs_embeds` 路径不兼容，不能直接作为稳定 benchmark 路线
- `vLLM / TensorRT-LLM`
  - 这是 runtime 优化，不是单纯 quantization 优化
- 继续重复 `bnb 4bit/8bit`
  - 现有实测已经足够说明问题

**下一步执行项**

- 导出一个 `AWQ group_size=64` checkpoint
- 用现有 `eval/benchmark_baseline_quant_runtime.py` 对比 `group_size=64` 和 `group_size=128` 的 latency / memory / accuracy
- 如果 `group_size=64` 没有收益，再转到 `GPTQ` 路线

### 接入 vLLM / SGLang continuous batching

用推理框架（vLLM / SGLang / TensorRT-LLM）替代 HF `model.generate`，在单卡上用一份权重服务多个并发请求。

**核心机制**

- **PagedAttention**：把每个 sequence 的 KV cache 切成固定大小的 page，物理上不连续也能做 attention。sequence 之间长度完全无关，没有 padding 浪费。
- **Continuous batching**：每个 decode step 都重新组 batch。某个 sequence 生成完了立刻让位，新请求立刻插进来，GPU 永远不空转。
- **Prefix caching**：相同前缀的 KV cache 算一次复用。LLaGA-NC 的固定类标签列表（arxiv 40 类 / products 47 类）占 prompt ~70%，对所有测试样本完全相同，这是一块免费午餐。

**集成难点**

vLLM 不原生支持 LLaGA 在 input embedding 里 inject graph token 的输入方式。两条路径：

1. Fork vLLM，加一个自定义 input encoder，把 graph projector 的输出 inject 进 token embedding。
2. 离线把 graph projector 的输出预先算好缓存到磁盘，eval 时让 vLLM 只跑纯文本 + 预拼好的 input embeddings。**工作量更小，推荐先试这条。**

**预期收益**

相比 HF `generate`，单卡吞吐 5-10×。叠加 `scripts/eval_moe_parallel.sh` 的多卡数据并行，整体可到一个数量级以上。
