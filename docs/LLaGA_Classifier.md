# LLaGA Embedding Classifier

This document records a non-generative LLaGA variant for node classification.
Instead of generating the class name token by token, the model produces a final
hidden embedding and uses a classifier head to predict the class id.

## Goal

Original LLaGA NC inference is generative:

```text
prompt + graph embedding -> LLM hidden states -> lm_head -> generated text label
```

The classifier path changes the final stage to:

```text
prompt + graph embedding -> LLM hidden states -> pooled embedding -> classifier -> class id
```

The graph injection and `mm_projector` logic are reused from the original
LLaGA architecture. The old generative path is untouched.

## New Files

- `model/language_model/llaga_llama_classifier.py`
  - Defines `LlagaLlamaForSequenceClassification`.
  - Reuses `LlagaMetaForCausalLM.prepare_inputs_labels_for_multimodal(...)`.
  - Pools the final valid token embedding.
  - Applies a linear classifier over the pooled embedding.
  - Can optionally return `pooled_embeddings` during evaluation.

- `train/train_classifier.py`
  - Trains the classifier model for NC.
  - Loads labels directly from `processed_data.pt`.
  - Supports `ND` and `HO`.
  - Supports optional `--pretrain_mm_mlp_adapter` to initialize the projector.

- `train/train_mem_classifier.py`
  - Flash-attention wrapper for classifier training.

- `eval/eval_pretrain_classifier.py`
  - Evaluates the classifier model.
  - Writes one jsonl record per node with predicted class id, class text, ground truth, and optional embedding.

- `scripts/train_deepspeed_classifier_nc.sh`
  - Deepspeed training wrapper.

- `scripts/eval_classifier_nc.sh`
  - Evaluation wrapper.

## Model Behavior

Forward pass:

1. Tokenize the original NC prompt with `<graph>`.
2. Build `graph` and `graph_emb` exactly like the baseline evaluation path.
3. Use the existing `mm_projector` to inject graph features into the text embedding sequence.
4. Run the LLaMA backbone.
5. Pool the last valid hidden state according to `attention_mask`.
6. Apply `classifier: hidden_size -> num_labels`.
7. Train with cross entropy over integer class ids.

The returned output includes:

- `loss`
- `logits`
- `pooled_embeddings`

## Training

ND training:

```bash
GPU_LIST=0,1,2,3 bash scripts/train_deepspeed_classifier_nc.sh \
  vicuna arxiv 4 simteg ND
```

HO / node-only training:

```bash
GPU_LIST=0,1,2,3 bash scripts/train_deepspeed_classifier_nc.sh \
  vicuna arxiv 4 simteg HO
```

The default output path is:

```text
/localnvme/llaga/checkpoints/<dataset>/llaga-vicuna-7b-<emb>-<template>-classifier
```

## Evaluation

```bash
GPU_ID=0 bash scripts/eval_classifier_nc.sh \
  /localnvme/llaga/checkpoints/arxiv/llaga-vicuna-7b-simteg-nd-classifier \
  arxiv simteg ND
```

To save pooled embeddings, call the Python entry directly:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_pretrain_classifier.py \
  --model_path /localnvme/llaga/checkpoints/arxiv/llaga-vicuna-7b-simteg-nd-classifier \
  --dataset arxiv \
  --pretrained_embedding_type simteg \
  --template ND \
  --use_hop 2 \
  --sample_neighbor_size 10 \
  --answers_file results/classifier_nc_arxiv_with_embeddings.jsonl \
  --save_embeddings
```

## Current Status

Implemented and checked:

- `python -m py_compile model/language_model/llaga_llama_classifier.py train/train_classifier.py train/train_mem_classifier.py eval/eval_pretrain_classifier.py`
- `bash -n scripts/train_deepspeed_classifier_nc.sh`
- `bash -n scripts/eval_classifier_nc.sh`
- Import check passed in the `llaga` environment.

Not yet completed:

- No full classifier training result has been produced yet.
- No classifier evaluation accuracy has been recorded yet.

## Previous Results

### Node-only Baseline

These are the existing single-node prompt `Node-only` results recorded in the
repo and/or re-scored from existing jsonl outputs.

| Dataset | Samples | Metric |
|---|---:|---:|
| arxiv | 48,603 | `strict_acc=0.7416`, `overall_acc=0.7416` |
| products | recorded in `docs/current_experiment_results.tex` | `0.0850` |
| pubmed | recorded in `docs/current_experiment_results.tex` | `0.0893` |
| cora | recorded in `docs/current_experiment_results.tex` | `0.8670` |

The full arxiv result file is:

```text
/localnvme/llaga/eval_output/nodeonly_nc_arxiv.jsonl
```

### 20-node Prompt Node-only Training

The 20-node prompt node-only training run has completed.

Configuration:

- Dataset: `arxiv`
- Template: `HO`
- Task: `nc_batch`
- Prompt size: `20 nodes / prompt`
- Base model: `lmsys/vicuna-7b-v1.5-16k`
- Embedding: `simteg`
- GPUs: `0,1,2,3`
- Epochs: `10`
- Global batch size: `4 GPUs * per_device_train_batch_size 4 = 16`
- Total steps: `2850`

Output checkpoint:

```text
/localnvme/llaga/checkpoints/arxiv/llaga-vicuna-7b-simteg-nodeonly-linear-projector-batch20_nc_batch
```

Training summary from `trainer_state.json`:

| Item | Value |
|---|---:|
| `global_step` | `2850` |
| `epoch` | `10.0` |
| `train_loss` | `0.040176` |
| `train_runtime` | `4920.50 s` |
| `train_samples_per_second` | `9.243` |
| `train_steps_per_second` | `0.579` |

Existing 200-sample batch20 eval files:

| Result file | Samples | Accuracy |
|---|---:|---:|
| `results/batch20_nc_arxiv.jsonl` | 200 | `0.3100` |
| `results/batch20_nc_arxiv_10ep.jsonl` | 200 | `0.4950` |

These are partial 200-sample checks, not full arxiv evaluation.

### Quantization Runtime Results

For reference, the previous quantization latency/memory benchmark on
`baseline + cora + 64 samples` found:

| Mode | `nvidia-smi` peak | Generate time | Total time |
|---|---:|---:|---:|
| FP16 | `13568 MB` | `10.64 s` | `12.84 s` |
| AWQ | `4630 MB` | `11.95 s` | `14.60 s` |

The best AWQ latency setting found later was:

```text
AWQ group_size=128 + GEMV
```

On the same 64-sample benchmark:

| AWQ mode | Peak memory | Generate time | Total time |
|---|---:|---:|---:|
| GEMM | `4630 MB` | `11.94 s` | `14.15 s` |
| GEMV | `4630 MB` | `9.86 s` | `12.26 s` |

GPTQ was tested but was slower than AWQ on the current LLaGA `inputs_embeds`
path:

| Mode | Generate time | Total time |
|---|---:|---:|
| AWQ `g128+GEMV` | `9.86 s` | `12.26 s` |
| GPTQ plain | `31.80 s` | `33.75 s` |
| GPTQ + ExLlama | `19.46 s` | `22.35 s` |
| GPTQ + ExLlamaV2 | `35.18 s` | `37.84 s` |

## Notes

- The classifier path is meant for NC only.
- It does not replace the original generative LLaGA model.
- It is best used when labels are known fixed classes and text generation is unnecessary.
- The current implementation pools the final valid token hidden state. If needed, a graph-token pooling variant can be added later.
