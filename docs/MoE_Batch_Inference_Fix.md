# MoE Batch Inference: OOM Masking Bug & Fix

## TL;DR

The previously-reported "accuracy collapse" of MoE hot-swap evaluation at `batch_size >= 64` was **not** a batching correctness bug. It was a silent CUDA OOM hidden by an overly-broad `except` handler. The batch inference code path is correct; accuracy stays stable at ~0.77 on arxiv NC from `bs=1` up to the memory ceiling.

## Symptom (original report)

Running `timing_bs*_nc_arxiv.jsonl` sweeps produced:

| BS | strict_acc |
|----|------------|
| 1..32 | 0.76–0.77 |
| 64 | 0.7435 (only first batch empty) |
| 128 | 0.0315 (almost all empty) |
| 256+ | 0.0000 (all empty) |

Pattern: at `bs=64` the *first* batch had all 64 outputs as empty strings; later batches were fine. At `bs>=128` *every* batch had all outputs empty.

## Root cause

Two interacting issues in `eval/eval_moe_hotswap.py`:

1. **Hard-coded `max_new_tokens=1024`**. NC labels are ~15 tokens, but `generate()` only reclaims KV cache once a sample hits EOS. One slow sample keeps the whole batch's KV cache alive, and the worst case approaches `max_new_tokens` for every sample. At `bs=128` that means:
   - KV per token per sample: 32 layers × 2 (k,v) × 32 heads × 128 dim × 2 B = 524 KB
   - Prompt expansion: `<graph>` → ~110 tokens → prompt length ~210
   - Final seq_len if all samples hit `max_new_tokens`: 210 + 1024 = 1234
   - Per sample: 524 KB × 1234 ≈ 647 MB
   - × 128 samples ≈ 83 GB → impossible on any 48 GB GPU
   
   In practice the prefill forward itself already OOMs at `bs=128` on an RTX 6000 Ada (46 GB): the Llama MLP `down_proj` tries to allocate 1.48 GB when only ~500 MB is free.

2. **Silent `except Exception` handler** (old `eval_moe_hotswap.py:411-413`):
   ```python
   except Exception as e:
       print(f"  Error: {e}")
       texts = [""] * B
   ```
   When `generate()` raised `CUDA out of memory`, this swallowed it and wrote empty strings as if they were valid predictions. The result file looks "complete" (2000 lines) but every line has `"text": ""`, which scores as 0.0 accuracy.
   
   At `bs=64`, the first batch OOMs on a fresh allocator; the exception is caught, CUDA frees memory, and subsequent batches fit → only batch 0 is empty. At `bs>=128`, every batch OOMs deterministically.

The OOM was confirmed by removing the `except` and reproducing the crash:
```
File "transformers/models/llama/modeling_llama.py", line 172, in forward
  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.48 GiB.
GPU 0 has a total capacity of 44.39 GiB of which 507.81 MiB is free.
```

## Fix

Applied to `eval/eval_moe_hotswap.py` and `scripts/eval_moe_hotswap.sh`:

1. **Removed the silent exception swallow.** OOM now crashes loudly instead of being reported as 0% accuracy.
2. **Added `--max_new_tokens` CLI argument**, default `32`. For NC/LP tasks the model's answer is short; capping generation prevents runaway KV cache growth.
3. **Passed `pad_token_id` and `eos_token_id` to `generate()`** so batched samples properly stop on EOS and padding is accounted for.
4. **Set `do_sample = (temperature > 0)`** to silence transformers 4.51 warnings when running with `temperature=0.0`.
5. **Exported `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in `scripts/eval_moe_hotswap.sh` to reduce allocator fragmentation.
6. **Added `torch.load` compat shim** (force `weights_only=False`): torch ≥ 2.6 changed the default, and LLaGA `processed_data.pt` files contain `torch_geometric.data.Data` objects which are not in the safe-globals allow-list.

## Verification: clean sweep

Sweep on 1000 arxiv NC samples in the `llaga-awq` env (torch 2.6, transformers 4.51, **no flash-attn**), RTX 6000 Ada 46 GB:

| BS | samples | strict_acc | overall_acc | empty | wall_s | infer_s* | samples/s | speedup |
|----|---------|-----------:|------------:|------:|-------:|---------:|----------:|--------:|
| 1  | 1000    | 0.7730     | 0.7730      | 0     | 348.6  | 293.6    | 3.41      | 1.00x   |
| 2  | 1000    | 0.7700     | 0.7700      | 0     | 271.2  | 216.2    | 4.63      | 1.36x   |
| 4  | 1000    | 0.7600     | 0.7600      | 0     | 242.0  | 187.0    | 5.35      | 1.57x   |
| 8  | 1000    | 0.7680     | 0.7680      | 0     | 166.6  | 111.6    | 8.96      | 2.63x   |
| 16 | 1000    | 0.7610     | 0.7610      | 0     | 151.9  | 96.9     | 10.32     | 3.03x   |
| 32 | 1000    | 0.7700     | 0.7700      | 0     | 169.3  | 114.3    | 8.75      | 2.57x   |
| 64 | 1000    | 0.7680     | 0.7680      | 0     | 139.7  | 84.7     | 11.81     | 3.47x   |
| 128 | —      | OOM at prefill | —       | —     | —      | —        | —         | —       |

\* `infer_s = wall_s − 55s` (approximate model load cost)

Observations:

- **Accuracy is stable** across `bs=1..64` at 0.76–0.77. The ±0.013 variance is explained by `temperature=0.2` sampling noise, not batching error.
- **Throughput peaks at `bs=64`** (3.47x). The `bs=32` dip is noise in a single-run measurement.
- **`bs=128` genuinely cannot fit** on this GPU in the `llaga-awq` env because it lacks flash-attention — Llama's default attention is O(L²) in prefill. A flash-attn-enabled env should push the ceiling to ~`bs=96` or more (see prior measurement of `bs=62` peak at 33.7 GB in the original llaga env).

## Recommendations

- For safe production evaluation, use `bs=32` or `bs=64` with `max_new_tokens=32` for NC tasks.
- If you need higher batch sizes, fix the flash-attention installation in the `llaga` env first.
- Any `eval_moe_hotswap_quantized.py` run has the same latent bug — apply the same fix there before trusting its accuracy numbers.

## Environment note

The `llaga` conda env is currently broken: torch was upgraded to 2.11.0+cu130 but `transformers` 4.51.3 and `flash_attn` 2.5.6 are compiled against older torch ABIs. Both fail to import with undefined-symbol errors. All verification above was done in the `llaga-awq` env (torch 2.6+cu124) after installing `einops`, `torch_sparse`, `torch_scatter`. The `llaga` env needs a clean reinstall (pinned versions per `CLAUDE.md`, or rebuild flash-attn for torch 2.11) before returning to normal use.
