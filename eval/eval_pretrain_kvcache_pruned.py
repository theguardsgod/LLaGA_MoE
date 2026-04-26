"""
KV-cache NC eval using a pruned LLM loaded via torch.load.
The pruned model's hidden_size is still 4096, so existing mm_projector works.
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
import re
import time
from tqdm import tqdm
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID
from utils.conversation import conv_templates, SeparatorStyle
from utils.utils import disable_torch_init, tokenizer_graph_token

# Import everything we need from the kvcache eval
from eval.eval_pretrain_kvcache import (
    DATASET_PATHS, NC_CLASSES, EDGE_DESC,
    build_nc_prompt_restructured,
    load_pretrain_embedding_graph,
    build_graph_embedding_single,
    build_multi_graph_embeds,
    parse_batch_output,
    generate_batched,
)


def load_pruned_model(pruned_ckpt_path, recovered_path=None):
    """Load pruned model, optionally with LoRA-recovered weights."""
    pruned = torch.load(pruned_ckpt_path, map_location="cpu")
    tokenizer, model = pruned["tokenizer"], pruned["model"]
    if recovered_path:
        recovered = torch.load(recovered_path, map_location="cpu")
        model.load_state_dict(recovered["state_dict"])
    return tokenizer, model


def load_mm_projector(projector_path, mm_hidden_size, hidden_size):
    """Load mm_projector weights."""
    projector = nn.Linear(mm_hidden_size, hidden_size)
    weights = torch.load(projector_path, map_location="cpu")
    # Normalize key names
    clean = {}
    for k, v in weights.items():
        k = k.replace("model.mm_projector.", "").replace("mm_projector.", "")
        clean[k] = v.to(torch.float16)
    projector.load_state_dict(clean)
    projector.half().cuda().eval()
    return projector


def eval_model(args):
    disable_torch_init()

    # Load pruned model (optionally with recovered weights)
    recovered = getattr(args, 'recovered_model_path', None)
    print(f"Loading pruned model from {args.pruned_model_path}")
    if recovered:
        print(f"  with recovered weights from {recovered}")
    tokenizer, model = load_pruned_model(args.pruned_model_path, recovered)
    model = model.to(torch.float16).cuda()
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Pruned model: {params:.2f}B params")

    # Load mm_projector
    projector_path = os.path.join(args.projector_path, "mm_projector.bin")
    print(f"Loading mm_projector from {projector_path}")
    mm_projector = load_mm_projector(projector_path, 2543, 4096)

    data_dir = DATASET_PATHS[args.dataset]
    prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
    print(f"Loading test data from {prompt_file}")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    questions = [json.loads(q) for q in lines]
    print(f"Total test samples: {len(questions)}")

    pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
    structure_emb = torch.load(
        f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt")

    N = args.nodes_per_prompt
    qs = build_nc_prompt_restructured(args.dataset, N)
    mode_str = f"pruned-restructured-N{N}"

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    first_graph_pos = full_prompt.index(DEFAULT_GRAPH_TOKEN)
    prefix_text = full_prompt[:first_graph_pos]
    suffix_text_with_graphs = full_prompt[first_graph_pos:]

    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
    prefix_len = prefix_ids.shape[1]

    suffix_token_ids = tokenizer_graph_token(
        suffix_text_with_graphs, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').cuda()
    n_graphs_in_suffix = (suffix_token_ids == GRAPH_TOKEN_INDEX).sum().item()
    n_text_tokens_in_suffix = len(suffix_token_ids) - n_graphs_in_suffix
    graph_tokens_total = N * 111

    print(f"Mode: {mode_str}")
    print(f"Prefix tokens: {prefix_len} (cached)")
    print(f"Per-prompt: {N} graphs x 111 = {graph_tokens_total} graph + {n_text_tokens_in_suffix} text")

    # Compute prefix KV cache
    print("Computing prefix KV cache...")
    prefix_embeds = model.model.embed_tokens(prefix_ids).half()
    with torch.inference_mode():
        prefix_out = model.model(inputs_embeds=prefix_embeds, use_cache=True, return_dict=True)
    prefix_kv = prefix_out.past_key_values
    print(f"Prefix KV cached: {prefix_len} tokens")

    embed_tokens = model.model.embed_tokens
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    B = args.batch_size
    total_gen_time = 0
    t_start = time.time()
    nodes_per_batch = N * B

    for batch_start in tqdm(range(0, len(questions), nodes_per_batch), desc=f"Pruned KV N{N}"):
        batch_nodes = questions[batch_start:batch_start + nodes_per_batch]

        groups = []
        for g_start in range(0, len(batch_nodes), N):
            groups.append(batch_nodes[g_start:g_start + N])
        actual_B = len(groups)

        prompt_embeds_list = []
        group_ids = []
        group_gts = []

        for group in groups:
            actual_N = len(group)
            projected_graphs = []
            ids = []
            gts = []
            for node in group:
                graph_emb = build_graph_embedding_single(node, pretrained_emb, structure_emb)
                projected = mm_projector(graph_emb.half().cuda()).squeeze(0)
                projected_graphs.append(projected)
                ids.append(node['id'])
                gts.append(node['conversations'][1]['value'])
            group_ids.append(ids)
            group_gts.append(gts)

            if N == 1:
                suffix_emb = embed_tokens(
                    suffix_token_ids[suffix_token_ids != GRAPH_TOKEN_INDEX].unsqueeze(0)
                ).squeeze(0).half()
                prompt_emb = torch.cat([projected_graphs[0], suffix_emb], dim=0)
            else:
                if actual_N < N:
                    qs_last = build_nc_prompt_restructured(args.dataset, actual_N)
                    conv_last = conv_templates[args.conv_mode].copy()
                    conv_last.append_message(conv_last.roles[0], qs_last)
                    conv_last.append_message(conv_last.roles[1], None)
                    last_prompt = conv_last.get_prompt()
                    last_pos = last_prompt.index(DEFAULT_GRAPH_TOKEN)
                    last_suffix = last_prompt[last_pos:]
                    last_suffix_ids = tokenizer_graph_token(
                        last_suffix, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').cuda()
                    prompt_emb = build_multi_graph_embeds(
                        last_suffix_ids, projected_graphs, embed_tokens)
                else:
                    prompt_emb = build_multi_graph_embeds(
                        suffix_token_ids, projected_graphs, embed_tokens)
            prompt_embeds_list.append(prompt_emb)

        max_len = max(e.shape[0] for e in prompt_embeds_list)
        padded = []
        for e in prompt_embeds_list:
            if e.shape[0] < max_len:
                pad = torch.zeros(max_len - e.shape[0], e.shape[1], dtype=e.dtype, device=e.device)
                e = torch.cat([e, pad], dim=0)
            padded.append(e)
        per_prompt_embeds = torch.stack(padded)

        t0 = time.time()
        try:
            max_tokens = args.max_new_tokens if N == 1 else N * 15 + 20
            texts = generate_batched(
                model, prefix_kv, prefix_len, per_prompt_embeds,
                tokenizer, temperature=args.temperature,
                top_p=args.top_p, max_new_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Error in batch: {e}")
            texts = [""] * actual_B
        total_gen_time += time.time() - t0

        for g_idx in range(actual_B):
            text = texts[g_idx]
            if text.endswith(stop_str):
                text = text[:-len(stop_str)].strip()

            ids = group_ids[g_idx]
            gts = group_gts[g_idx]
            actual_N = len(ids)

            if N == 1:
                ans_file.write(json.dumps({
                    "question_id": ids[0],
                    "prompt": f"[pruned-kvcache-{mode_str}]",
                    "text": text,
                    "gt": gts[0],
                    "answer_id": shortuuid.uuid()
                }) + "\n")
            else:
                predictions = parse_batch_output(text, actual_N)
                for i in range(actual_N):
                    ans_file.write(json.dumps({
                        "question_id": ids[i],
                        "prompt": f"[pruned-kvcache-{mode_str}, node {i+1}/{actual_N}]",
                        "text": predictions[i],
                        "gt": gts[i],
                        "answer_id": shortuuid.uuid()
                    }) + "\n")
        ans_file.flush()

    ans_file.close()
    total_time = time.time() - t_start
    n = len(questions)
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nDone! {n} samples in {total_time:.1f}s ({total_gen_time:.1f}s generation)")
    print(f"Throughput: {n/total_time:.1f} samples/s, {total_time/n*1000:.0f} ms/sample")
    print(f"Peak GPU memory: {mem_gb:.1f} GB")
    print(f"Results: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned_model_path", type=str, required=True,
                        help="Path to LLM-Pruner pytorch_model.bin checkpoint")
    parser.add_argument("--projector_path", type=str, required=True,
                        help="Path to LLaGA checkpoint dir with mm_projector.bin")
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="results/pruned_kvcache_nc.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nodes_per_prompt", type=int, default=1)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--recovered_model_path", type=str, default=None,
                        help="Path to LoRA-recovered pytorch_model.bin (optional)")
    args = parser.parse_args()
    eval_model(args)
