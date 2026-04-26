"""
Node-structure fusion evaluation.
Path A: node-only projector (center node embedding only, no graph structure)
Path B: structure projector (full multi-hop + Laplacian structure)
Both paths share the same LLM. Hidden states are fused at each decoding step.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/eval_node_structure_fusion.py \
        --pruned_model_path <path> \
        --recovered_model_path <path> \
        --node_proj_path <dir_with_mm_projector.bin> \
        --struct_proj_path <dir_with_mm_projector.bin> \
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

    clean = {}
    for k, v in state.items():
        k = k.replace("model.mm_projector.", "").replace("mm_projector.", "")
        clean[k] = v.float()
    proj.load_state_dict(clean)
    proj.half().cuda().eval()
    return proj


def build_suffix_embeds(suffix_ids, graph_proj, embed_tokens):
    """Build embeddings for suffix, replacing GRAPH_TOKEN_INDEX with projected graph."""
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
def generate_fused(model, prefix_kv, prefix_len, emb_node, emb_struct,
                   tokenizer, alpha=0.5, temperature=0.0, max_new_tokens=128):
    """Autoregressive generation with hidden-state fusion from node-only and structure paths."""

    def clone_kv(kv):
        return tuple(tuple(t.clone() for t in layer) for layer in kv)

    kv_node = clone_kv(prefix_kv)
    kv_struct = clone_kv(prefix_kv)

    # Process suffix for each path
    out_node = model.model(inputs_embeds=emb_node.unsqueeze(0), past_key_values=kv_node,
                           use_cache=True, return_dict=True)
    out_struct = model.model(inputs_embeds=emb_struct.unsqueeze(0), past_key_values=kv_struct,
                             use_cache=True, return_dict=True)

    kv_node = out_node.past_key_values
    kv_struct = out_struct.past_key_values

    # Fuse hidden states then decode
    h_node = out_node.last_hidden_state[:, -1:, :]
    h_struct = out_struct.last_hidden_state[:, -1:, :]
    h_fused = alpha * h_node + (1 - alpha) * h_struct
    logits = model.lm_head(h_fused)

    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs.squeeze(0), num_samples=1)
    else:
        next_token = logits.argmax(dim=-1)

    generated = [next_token.squeeze()]

    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break

        token_emb = model.model.embed_tokens(next_token).half()

        out_node = model.model(inputs_embeds=token_emb, past_key_values=kv_node,
                               use_cache=True, return_dict=True)
        out_struct = model.model(inputs_embeds=token_emb, past_key_values=kv_struct,
                                 use_cache=True, return_dict=True)

        kv_node = out_node.past_key_values
        kv_struct = out_struct.past_key_values

        h_node = out_node.last_hidden_state[:, -1:, :]
        h_struct = out_struct.last_hidden_state[:, -1:, :]
        h_fused = alpha * h_node + (1 - alpha) * h_struct
        logits = model.lm_head(h_fused)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs.squeeze(0), num_samples=1)
        else:
            next_token = logits.argmax(dim=-1)

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

    # Load node-only projector (input: pretrained emb only, no structure)
    data_dir = DATASET_PATHS[args.dataset]
    pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
    node_emb_dim = pretrained_emb.shape[1]  # 2432 for simteg
    print(f"Loading node projector ({args.node_proj_type}, input={node_emb_dim}) from {args.node_proj_path}")
    node_proj = load_projector(args.node_proj_path, node_emb_dim, hidden_size, args.node_proj_type)

    # Load structure projector (input: pretrained + Laplacian)
    structure_emb = torch.load(
        f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
    )
    struct_emb_dim = node_emb_dim + structure_emb.shape[1]  # 2432 + 111 = 2543
    print(f"Loading struct projector ({args.struct_proj_type}, input={struct_emb_dim}) from {args.struct_proj_path}")
    struct_proj = load_projector(args.struct_proj_path, struct_emb_dim, hidden_size, args.struct_proj_type)

    # Load data
    prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
    lines = open(prompt_file).readlines()
    if args.end > 0:
        lines = lines[:args.end]
    questions = [json.loads(q) for q in lines]
    print(f"Test samples: {len(questions)}")

    # Build shared prompt template
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

    # Compute prefix KV cache (shared by both paths)
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
    print(f"\nNode-structure fusion eval: alpha={alpha}")
    print(f"  Node path (node-only, {args.node_proj_type}): weight={alpha}")
    print(f"  Struct path (full graph, {args.struct_proj_type}): weight={1-alpha}\n")

    t_start = time.time()
    for idx, node in enumerate(tqdm(questions, desc="Node-struct fusion")):
        # Path A: node-only (just center node embedding)
        g = node["graph"]
        center_idx = g[0] if not isinstance(g[0], list) else g[0][0]
        node_emb = pretrained_emb[center_idx].unsqueeze(0).unsqueeze(0).half().cuda()  # [1, 1, 2432]
        with torch.inference_mode():
            proj_node = node_proj(node_emb).squeeze(0)  # [1, 4096]

        # Path B: full structure (multi-hop + Laplacian)
        graph_emb = build_graph_embedding_single(node, pretrained_emb, structure_emb)
        with torch.inference_mode():
            proj_struct = struct_proj(graph_emb.half().cuda()).squeeze(0)  # [111, 4096]

        # Build suffix embeddings for each path
        emb_node = build_suffix_embeds(suffix_ids, proj_node.unsqueeze(0), embed_tokens)
        emb_struct = build_suffix_embeds(suffix_ids, proj_struct.unsqueeze(0), embed_tokens)

        # Generate with fused hidden states
        text = generate_fused(
            model, prefix_kv, prefix_ids.shape[1], emb_node, emb_struct,
            tokenizer, alpha=alpha, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        if text.endswith(stop_str):
            text = text[:-len(stop_str)].strip()

        gt = node['conversations'][1]['value']
        ans_file.write(json.dumps({
            "question_id": node['id'],
            "prompt": f"[node-struct-fusion-a{alpha}]",
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
    parser.add_argument("--node_proj_path", type=str, required=True,
                        help="Dir with mm_projector.bin for node-only projector")
    parser.add_argument("--node_proj_type", type=str, default="linear")
    parser.add_argument("--struct_proj_path", type=str, required=True,
                        help="Dir with mm_projector.bin for structure projector")
    parser.add_argument("--struct_proj_type", type=str, default="linear")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for node path (1-alpha for structure path)")
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="results/node_struct_fusion_nc.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="arxiv")
    args = parser.parse_args()
    eval_model(args)
