"""
Node-structure fusion with 2-GPU parallelism and batched generation.
GPU 0: node-only path (center node embedding, no graph structure)
GPU 1: structure path (full multi-hop + Laplacian)
Both GPUs process batches in parallel, hidden states fused each step.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python eval/eval_node_structure_fusion_2gpu.py \
        --pruned_model_path <path> --recovered_model_path <path> \
        --node_proj_path <dir> --struct_proj_path <dir> \
        --dataset arxiv --end 200 --batch_size 32
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
import threading
from tqdm import tqdm
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID
from utils.conversation import conv_templates, SeparatorStyle
from utils.utils import disable_torch_init, tokenizer_graph_token

from eval.eval_pretrain_kvcache import (
    DATASET_PATHS, build_nc_prompt_restructured,
    load_pretrain_embedding_graph, build_graph_embedding_single,
)


def load_projector(proj_dir, mm_hidden_size, hidden_size, proj_type, device):
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
    proj.half().to(device).eval()
    return proj


def build_suffix_embeds(suffix_ids, graph_proj, embed_tokens, device):
    """Build suffix embeddings replacing GRAPH_TOKEN_INDEX with projected graph."""
    suffix_ids = suffix_ids.to(device)
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
def generate_batch_fused_2gpu(model_node, model_struct,
                               prefix_kv_node, prefix_kv_struct,
                               batch_emb_node, batch_emb_struct,
                               tokenizer, dev_node, dev_struct,
                               alpha=0.3, temperature=0.0, max_new_tokens=128):
    """Batched autoregressive generation with 2-GPU parallel forward + fusion.

    batch_emb_node: [B, seq_len_node, H] on dev_node
    batch_emb_struct: [B, seq_len_struct, H] on dev_struct
    """
    B = batch_emb_node.shape[0]

    def expand_kv(kv, B):
        return tuple(
            tuple(t.expand(B, -1, -1, -1) for t in layer)
            for layer in kv
        )

    kv_n = expand_kv(prefix_kv_node, B)
    kv_s = expand_kv(prefix_kv_struct, B)

    results = [None, None]

    def run_node():
        results[0] = model_node.model(inputs_embeds=batch_emb_node,
                                       past_key_values=kv_n,
                                       use_cache=True, return_dict=True)

    def run_struct():
        results[1] = model_struct.model(inputs_embeds=batch_emb_struct,
                                         past_key_values=kv_s,
                                         use_cache=True, return_dict=True)

    t1 = threading.Thread(target=run_node)
    t2 = threading.Thread(target=run_struct)
    t1.start(); t2.start()
    t1.join(); t2.join()

    out_n, out_s = results
    kv_n = out_n.past_key_values
    kv_s = out_s.past_key_values

    # Fuse hidden states [B, 1, H]
    h_n = out_n.last_hidden_state[:, -1:, :]
    h_s = out_s.last_hidden_state[:, -1:, :].to(dev_node)
    h_fused = alpha * h_n + (1 - alpha) * h_s
    logits = model_node.lm_head(h_fused)

    if temperature > 0:
        probs = torch.softmax(logits.squeeze(1) / temperature, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)  # [B, 1]
    else:
        next_tokens = logits.squeeze(1).argmax(dim=-1, keepdim=True)  # [B, 1]

    # Track generated tokens and finished status
    all_tokens = [next_tokens.cpu()]  # list of [B, 1]
    finished = (next_tokens.squeeze(-1).cpu() == tokenizer.eos_token_id)  # [B] on cpu

    for _ in range(max_new_tokens - 1):
        if finished.all():
            break

        tok_n = model_node.model.embed_tokens(next_tokens.to(dev_node)).half()
        tok_s = model_struct.model.embed_tokens(next_tokens.to(dev_struct)).half()

        def step_n():
            results[0] = model_node.model(inputs_embeds=tok_n,
                                           past_key_values=kv_n,
                                           use_cache=True, return_dict=True)
        def step_s():
            results[1] = model_struct.model(inputs_embeds=tok_s,
                                             past_key_values=kv_s,
                                             use_cache=True, return_dict=True)

        t1 = threading.Thread(target=step_n)
        t2 = threading.Thread(target=step_s)
        t1.start(); t2.start()
        t1.join(); t2.join()

        out_n, out_s = results
        kv_n = out_n.past_key_values
        kv_s = out_s.past_key_values

        h_n = out_n.last_hidden_state[:, -1:, :]
        h_s = out_s.last_hidden_state[:, -1:, :].to(dev_node)
        h_fused = alpha * h_n + (1 - alpha) * h_s
        logits = model_node.lm_head(h_fused)

        if temperature > 0:
            probs = torch.softmax(logits.squeeze(1) / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = logits.squeeze(1).argmax(dim=-1, keepdim=True)

        # Mask finished sequences to EOS
        next_tokens[finished.to(next_tokens.device)] = tokenizer.eos_token_id
        finished = finished | (next_tokens.squeeze(-1).cpu() == tokenizer.eos_token_id)
        all_tokens.append(next_tokens.cpu())

    # Decode each sequence
    token_ids = torch.cat(all_tokens, dim=1)  # [B, T]
    texts = []
    for i in range(B):
        seq = token_ids[i]
        # Trim at first EOS
        eos_mask = (seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_mask) > 0:
            seq = seq[:eos_mask[0]]
        texts.append(tokenizer.decode(seq, skip_special_tokens=True))
    return texts


def eval_model(args):
    disable_torch_init()

    dev_node = torch.device("cuda:0")
    dev_struct = torch.device("cuda:1")

    # Load model on both GPUs
    print(f"Loading pruned model from {args.pruned_model_path}")
    pruned = torch.load(args.pruned_model_path, map_location="cpu")
    tokenizer = pruned["tokenizer"]

    import copy
    base_model = pruned["model"]
    if args.recovered_model_path:
        print(f"Loading recovered weights from {args.recovered_model_path}")
        recovered = torch.load(args.recovered_model_path, map_location="cpu")
        base_model.load_state_dict(recovered["state_dict"])

    print("Loading model to GPU 0 (node path)...")
    model_node = copy.deepcopy(base_model).half().to(dev_node).eval()
    print("Loading model to GPU 1 (struct path)...")
    model_struct = base_model.half().to(dev_struct).eval()
    del pruned, base_model
    if args.recovered_model_path:
        del recovered

    hidden_size = model_node.config.hidden_size
    params_b = sum(p.numel() for p in model_node.parameters()) / 1e9
    print(f"Model: {params_b:.2f}B params x 2 GPUs")

    # Load projectors
    data_dir = DATASET_PATHS[args.dataset]
    pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
    node_emb_dim = pretrained_emb.shape[1]

    print(f"Loading node projector ({args.node_proj_type}) on GPU 0")
    node_proj = load_projector(args.node_proj_path, node_emb_dim, hidden_size,
                               args.node_proj_type, dev_node)

    structure_emb = torch.load(
        f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
    )
    struct_emb_dim = node_emb_dim + structure_emb.shape[1]
    print(f"Loading struct projector ({args.struct_proj_type}) on GPU 1")
    struct_proj = load_projector(args.struct_proj_path, struct_emb_dim, hidden_size,
                                 args.struct_proj_type, dev_struct)

    # Load data
    prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
    lines = open(prompt_file).readlines()
    if args.end > 0:
        lines = lines[:args.end]
    questions = [json.loads(q) for q in lines]
    print(f"Test samples: {len(questions)}")

    # Build prompt
    qs = build_nc_prompt_restructured(args.dataset, 1)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    first_graph_pos = full_prompt.index(DEFAULT_GRAPH_TOKEN)
    prefix_text = full_prompt[:first_graph_pos]
    suffix_text = full_prompt[first_graph_pos:]

    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True).input_ids
    suffix_ids = tokenizer_graph_token(suffix_text, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt')

    # Compute prefix KV cache on both GPUs in parallel
    print("Computing prefix KV caches on both GPUs...")
    prefix_kv = [None, None]

    def compute_prefix_kv_node():
        emb = model_node.model.embed_tokens(prefix_ids.to(dev_node)).half()
        out = model_node.model(inputs_embeds=emb, use_cache=True, return_dict=True)
        prefix_kv[0] = out.past_key_values

    def compute_prefix_kv_struct():
        emb = model_struct.model.embed_tokens(prefix_ids.to(dev_struct)).half()
        out = model_struct.model(inputs_embeds=emb, use_cache=True, return_dict=True)
        prefix_kv[1] = out.past_key_values

    with torch.inference_mode():
        t1 = threading.Thread(target=compute_prefix_kv_node)
        t2 = threading.Thread(target=compute_prefix_kv_struct)
        t1.start(); t2.start()
        t1.join(); t2.join()

    prefix_kv_node = prefix_kv[0]
    prefix_kv_struct = prefix_kv[1]
    print(f"Prefix cached: {prefix_ids.shape[1]} tokens on both GPUs")

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    alpha = args.alpha
    B = args.batch_size
    print(f"\n2-GPU Batched Node-structure fusion: alpha={alpha}, batch_size={B}")
    print(f"  GPU 0 - Node path: weight={alpha}")
    print(f"  GPU 1 - Struct path: weight={1-alpha}\n")

    embed_node = model_node.model.embed_tokens
    embed_struct = model_struct.model.embed_tokens

    t_start = time.time()
    for batch_start in tqdm(range(0, len(questions), B), desc=f"2GPU-B{B}"):
        batch_nodes = questions[batch_start:batch_start + B]
        actual_B = len(batch_nodes)

        # Build embeddings for the batch
        node_embeds_list = []
        struct_embeds_list = []

        for node in batch_nodes:
            # Node-only: center node embedding
            g = node["graph"]
            center_idx = g[0] if not isinstance(g[0], list) else g[0][0]
            n_emb = pretrained_emb[center_idx].unsqueeze(0).unsqueeze(0).half().to(dev_node)
            with torch.inference_mode():
                proj_n = node_proj(n_emb).squeeze(0)  # [1, H]
            suffix_emb_n = build_suffix_embeds(suffix_ids, proj_n.unsqueeze(0), embed_node, dev_node)
            node_embeds_list.append(suffix_emb_n)

            # Structure: full multi-hop
            graph_emb = build_graph_embedding_single(node, pretrained_emb, structure_emb)
            with torch.inference_mode():
                proj_s = struct_proj(graph_emb.half().to(dev_struct)).squeeze(0)  # [111, H]
            suffix_emb_s = build_suffix_embeds(suffix_ids, proj_s.unsqueeze(0), embed_struct, dev_struct)
            struct_embeds_list.append(suffix_emb_s)

        # Stack into batches (suffix lengths are fixed per path)
        batch_emb_node = torch.stack(node_embeds_list)      # [B, seq_n, H]
        batch_emb_struct = torch.stack(struct_embeds_list)   # [B, seq_s, H]

        with torch.inference_mode():
            texts = generate_batch_fused_2gpu(
                model_node, model_struct,
                prefix_kv_node, prefix_kv_struct,
                batch_emb_node, batch_emb_struct,
                tokenizer, dev_node, dev_struct,
                alpha=alpha, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

        for i, node in enumerate(batch_nodes):
            text = texts[i]
            if text.endswith(stop_str):
                text = text[:-len(stop_str)].strip()
            gt = node['conversations'][1]['value']
            ans_file.write(json.dumps({
                "question_id": node['id'],
                "prompt": f"[2gpu-batch-a{alpha}]",
                "text": text,
                "gt": gt,
                "answer_id": shortuuid.uuid(),
            }) + "\n")
        ans_file.flush()

    ans_file.close()
    total_time = time.time() - t_start
    n = len(questions)
    mem0 = torch.cuda.max_memory_allocated(dev_node) / 1e9
    mem1 = torch.cuda.max_memory_allocated(dev_struct) / 1e9
    print(f"\nDone! {n} samples in {total_time:.1f}s")
    print(f"Throughput: {n/total_time:.1f} samples/s, {total_time/n*1000:.0f} ms/sample")
    print(f"Peak GPU memory: GPU0={mem0:.1f}GB, GPU1={mem1:.1f}GB")
    print(f"Results: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned_model_path", type=str, required=True)
    parser.add_argument("--recovered_model_path", type=str, default=None)
    parser.add_argument("--node_proj_path", type=str, required=True)
    parser.add_argument("--node_proj_type", type=str, default="linear")
    parser.add_argument("--struct_proj_path", type=str, required=True)
    parser.add_argument("--struct_proj_type", type=str, default="linear")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="results/node_struct_fusion_2gpu.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="arxiv")
    args = parser.parse_args()
    eval_model(args)
