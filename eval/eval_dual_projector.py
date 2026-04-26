"""
Dual-projector evaluation with logit-level merging.
Two projectors (Linear + MLP) process the same graph through the same LLM,
logits are merged at each decoding step for better predictions.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/eval_dual_projector.py \
        --pruned_model_path <path> \
        --recovered_model_path <path> \
        --proj_a_path <dir_with_mm_projector.bin> \
        --proj_b_path <dir_with_mm_projector.bin> \
        --dataset arxiv --end 200
"""
import sys
sys.path.insert(0, "/home/23131884r/code/LLM-Pruner")
sys.path.append("./")
sys.path.append("./utils")

import argparse
import torch
import torch.nn as nn
import os
import json
import time
from tqdm import tqdm
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID
from utils.conversation import conv_templates, SeparatorStyle
from utils.utils import disable_torch_init, tokenizer_graph_token

from eval.eval_pretrain_kvcache import (
    DATASET_PATHS, build_nc_prompt_restructured,
    load_pretrain_embedding_graph, build_graph_embedding_single,
)


def load_projector(proj_dir, mm_hidden_size, hidden_size, proj_type="linear"):
    """Load projector from directory."""
    proj_path = os.path.join(proj_dir, "mm_projector.bin")
    state = torch.load(proj_path, map_location="cpu")

    if proj_type == "mlp":
        proj = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
    else:
        proj = nn.Linear(mm_hidden_size, hidden_size)

    # Handle different state_dict key formats
    clean = {}
    for k, v in state.items():
        k = k.replace("model.mm_projector.", "").replace("mm_projector.", "")
        clean[k] = v.float()
    proj.load_state_dict(clean)
    proj.half().cuda().eval()
    return proj


def build_suffix_embeds(suffix_ids, graph_proj, embed_tokens):
    """Build embeddings for suffix (graph tokens + text), replacing GRAPH_TOKEN_INDEX."""
    parts = []
    prev = 0
    graph_idx = 0
    for i, tid in enumerate(suffix_ids):
        if tid == GRAPH_TOKEN_INDEX:
            if i > prev:
                text_emb = embed_tokens(suffix_ids[prev:i].unsqueeze(0)).squeeze(0)
                parts.append(text_emb)
            parts.append(graph_proj[graph_idx])
            graph_idx += 1
            prev = i + 1
    if prev < len(suffix_ids):
        text_emb = embed_tokens(suffix_ids[prev:].unsqueeze(0)).squeeze(0)
        parts.append(text_emb)
    return torch.cat(parts, dim=0)


@torch.inference_mode()
def generate_dual(model, prefix_kv, prefix_len, emb_a, emb_b,
                   tokenizer, alpha=0.5, temperature=0.0, max_new_tokens=128):
    """Autoregressive generation with logit merging from two projector views."""
    device = emb_a.device

    # Clone prefix KV for both paths
    def clone_kv(kv):
        return tuple(tuple(t.clone() for t in layer) for layer in kv)

    kv_a = clone_kv(prefix_kv)
    kv_b = clone_kv(prefix_kv)

    # Process suffix (graph + text) for each path
    out_a = model.model(inputs_embeds=emb_a.unsqueeze(0), past_key_values=kv_a,
                        use_cache=True, return_dict=True)
    out_b = model.model(inputs_embeds=emb_b.unsqueeze(0), past_key_values=kv_b,
                        use_cache=True, return_dict=True)

    kv_a = out_a.past_key_values
    kv_b = out_b.past_key_values

    # Get merged logits for first token
    logits_a = model.lm_head(out_a.last_hidden_state[:, -1:, :])
    logits_b = model.lm_head(out_b.last_hidden_state[:, -1:, :])
    merged = alpha * logits_a + (1 - alpha) * logits_b

    if temperature > 0:
        probs = torch.softmax(merged / temperature, dim=-1)
        next_token = torch.multinomial(probs.squeeze(0), num_samples=1)
    else:
        next_token = merged.argmax(dim=-1)

    generated = [next_token.squeeze()]

    # Autoregressive loop
    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break

        token_emb = model.model.embed_tokens(next_token).half()

        out_a = model.model(inputs_embeds=token_emb, past_key_values=kv_a,
                            use_cache=True, return_dict=True)
        out_b = model.model(inputs_embeds=token_emb, past_key_values=kv_b,
                            use_cache=True, return_dict=True)

        kv_a = out_a.past_key_values
        kv_b = out_b.past_key_values

        logits_a = model.lm_head(out_a.last_hidden_state[:, -1:, :])
        logits_b = model.lm_head(out_b.last_hidden_state[:, -1:, :])
        merged = alpha * logits_a + (1 - alpha) * logits_b

        if temperature > 0:
            probs = torch.softmax(merged / temperature, dim=-1)
            next_token = torch.multinomial(probs.squeeze(0), num_samples=1)
        else:
            next_token = merged.argmax(dim=-1)

        generated.append(next_token.squeeze())

    token_ids = torch.stack(generated)
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text


def eval_model(args):
    disable_torch_init()

    # Load model
    print(f"Loading pruned model from {args.pruned_model_path}")
    pruned = torch.load(args.pruned_model_path, map_location="cpu")
    model = pruned["model"]
    tokenizer = pruned["tokenizer"]

    if args.recovered_model_path:
        print(f"Loading recovered weights from {args.recovered_model_path}")
        recovered = torch.load(args.recovered_model_path, map_location="cpu")
        model.load_state_dict(recovered["state_dict"])

    model = model.half().cuda().eval()
    hidden_size = model.config.hidden_size
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    # Load both projectors
    mm_hidden_size = 2543  # simteg(2432) + structure(111)
    print(f"Loading Proj_A ({args.proj_a_type}) from {args.proj_a_path}")
    proj_a = load_projector(args.proj_a_path, mm_hidden_size, hidden_size, args.proj_a_type)
    print(f"Loading Proj_B ({args.proj_b_type}) from {args.proj_b_path}")
    proj_b = load_projector(args.proj_b_path, mm_hidden_size, hidden_size, args.proj_b_type)

    # Load data
    data_dir = DATASET_PATHS[args.dataset]
    prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
    lines = open(prompt_file).readlines()
    if args.end > 0:
        lines = lines[:args.end]
    questions = [json.loads(q) for q in lines]
    print(f"Test samples: {len(questions)}")

    pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
    structure_emb = torch.load(
        f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
    )

    # Build shared prompt
    qs = build_nc_prompt_restructured(args.dataset, 1)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    first_graph_pos = full_prompt.index(DEFAULT_GRAPH_TOKEN)
    prefix_text = full_prompt[:first_graph_pos]
    suffix_text = full_prompt[first_graph_pos:]

    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
    suffix_ids = tokenizer_graph_token(
        suffix_text, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').cuda()

    # Compute prefix KV cache
    print("Computing prefix KV cache...")
    prefix_embeds = model.model.embed_tokens(prefix_ids).half()
    with torch.inference_mode():
        prefix_out = model.model(inputs_embeds=prefix_embeds, use_cache=True, return_dict=True)
    prefix_kv = prefix_out.past_key_values
    print(f"Prefix cached: {prefix_ids.shape[1]} tokens")

    embed_tokens = model.model.embed_tokens
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    alpha = args.alpha
    print(f"\nDual-projector eval: alpha={alpha}")
    print(f"  Proj_A ({args.proj_a_type}): weight={alpha}")
    print(f"  Proj_B ({args.proj_b_type}): weight={1-alpha}\n")

    t_start = time.time()
    for idx, node in enumerate(tqdm(questions, desc="Dual-proj eval")):
        graph_emb = build_graph_embedding_single(node, pretrained_emb, structure_emb)
        graph_emb_f = graph_emb.half().cuda()

        # Project through both projectors
        with torch.inference_mode():
            proj_graph_a = proj_a(graph_emb_f).squeeze(0)  # [111, 4096]
            proj_graph_b = proj_b(graph_emb_f).squeeze(0)

        # Build suffix embeddings for each path
        emb_a = build_suffix_embeds(suffix_ids, proj_graph_a.unsqueeze(0), embed_tokens)
        emb_b = build_suffix_embeds(suffix_ids, proj_graph_b.unsqueeze(0), embed_tokens)

        # Generate with merged logits
        text = generate_dual(
            model, prefix_kv, prefix_ids.shape[1], emb_a, emb_b,
            tokenizer, alpha=alpha, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        if text.endswith(stop_str):
            text = text[:-len(stop_str)].strip()

        gt = node['conversations'][1]['value']
        ans_file.write(json.dumps({
            "question_id": node['id'],
            "prompt": f"[dual-proj-a{alpha}]",
            "text": text,
            "gt": gt,
            "answer_id": shortuuid.uuid(),
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    total_time = time.time() - t_start
    n = len(questions)
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nDone! {n} samples in {total_time:.1f}s")
    print(f"Throughput: {n/total_time:.1f} samples/s, {total_time/n*1000:.0f} ms/sample")
    print(f"Peak GPU memory: {mem_gb:.1f} GB")
    print(f"Results: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned_model_path", type=str, required=True)
    parser.add_argument("--recovered_model_path", type=str, default=None)
    parser.add_argument("--proj_a_path", type=str, required=True)
    parser.add_argument("--proj_a_type", type=str, default="linear")
    parser.add_argument("--proj_b_path", type=str, required=True)
    parser.add_argument("--proj_b_type", type=str, default="mlp")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="results/dual_proj_nc.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="arxiv")
    args = parser.parse_args()
    eval_model(args)
