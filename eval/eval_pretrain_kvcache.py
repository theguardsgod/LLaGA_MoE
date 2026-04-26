"""
KV-cache accelerated NC evaluation for LLaGA.

Supports both single-node and multi-node (N nodes per prompt) modes.
Restructures the prompt to put the class list BEFORE <graph> tokens, then:
  1. Cache prefix KV (class list text, ~435 tokens) — computed ONCE
  2. For each batch of B prompts (each with N nodes), forward graph+suffix tokens
  3. Batch autoregressive generation

Usage:
    # Single node per prompt (N=1), B=32 parallel prompts
    python eval/eval_pretrain_kvcache.py --model_path <path> --restructure_prompt --batch_size 32

    # 5 nodes per prompt, B=8 parallel prompts = 40 nodes per forward
    python eval/eval_pretrain_kvcache.py --model_path <path> --restructure_prompt --nodes_per_prompt 5 --batch_size 8
"""
import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import os
import json
import re
import time
from tqdm import tqdm
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, get_model_name_from_path, tokenizer_graph_token


DATASET_PATHS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}

NC_CLASSES = {
    "arxiv": "40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics)",
    "products": "47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D\u00e9cor, #508510",
    "pubmed": "3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2",
    "cora": "7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory",
}

EDGE_DESC = {
    "arxiv": "",
    "products": ", where nodes represent products sold in Amazon, and edges between products indicate they are purchased together",
    "pubmed": "",
    "cora": "",
}


def build_nc_prompt_restructured(dataset, nodes_per_prompt=1):
    """Prompt with class list BEFORE <graph> to maximize cacheable prefix."""
    classes = NC_CLASSES[dataset]
    edge_desc = EDGE_DESC[dataset]
    if nodes_per_prompt == 1:
        return (
            f"We need to classify the center node into {classes}. "
            f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}{edge_desc}, "
            f"please tell me which class the center node belongs to? "
            f"Answer with only the class name, e.g., cs.AI(Artificial Intelligence)."
        )
    else:
        graph_tokens = ", ".join([DEFAULT_GRAPH_TOKEN] * nodes_per_prompt)
        return (
            f"We need to classify each center node into {classes}. "
            f"Given {nodes_per_prompt} node-centered graphs: {graph_tokens}{edge_desc}, "
            f"for each node (1 to {nodes_per_prompt}), tell me which class the center node belongs to? "
            f"Answer format: one class per line, e.g.:\n"
            f"1. <class>\n2. <class>\n..."
        )


def build_nc_prompt_original(dataset, line):
    """Original prompt from test data."""
    if dataset == "products":
        return f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into {NC_CLASSES['products']}, please tell me which class the center node belongs to?"
    return line["conversations"][0]['value']


def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    else:
        pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
    return pretrained_emb


def build_graph_embedding_single(line, pretrained_emb, structure_emb):
    """Build raw graph embedding [1, 111, 2543] for one node."""
    g = line['graph']
    if not isinstance(g[0], list):
        g = [g]
    graph = torch.LongTensor(g)
    mask = graph != DEFAULT_GRAPH_PAD_ID
    masked_emb = pretrained_emb[graph[mask]]
    s, n, d = graph.shape[0], graph.shape[1], masked_emb.shape[1]
    graph_emb = torch.zeros((s, n, d))
    graph_emb[mask] = masked_emb
    if structure_emb is not None:
        graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
    return graph_emb  # [1, 111, 2543]


def build_multi_graph_embeds(suffix_token_ids, graph_projected_list, embed_tokens):
    """
    Build embedding sequence for suffix containing multiple <graph> tokens.
    Replace each GRAPH_TOKEN_INDEX with the corresponding projected graph embedding.

    Args:
        suffix_token_ids: 1D tensor, may contain multiple GRAPH_TOKEN_INDEX
        graph_projected_list: list of [111, hidden_dim] projected graph embeddings
        embed_tokens: model's token embedding layer
    Returns:
        [total_len, hidden_dim] tensor
    """
    parts = []
    graph_idx = 0
    cur_ids = suffix_token_ids

    while True:
        graph_positions = (cur_ids == GRAPH_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if len(graph_positions) == 0:
            if len(cur_ids) > 0:
                parts.append(embed_tokens(cur_ids.unsqueeze(0)).squeeze(0).half())
            break

        pos = graph_positions[0].item()
        if pos > 0:
            parts.append(embed_tokens(cur_ids[:pos].unsqueeze(0)).squeeze(0).half())
        parts.append(graph_projected_list[graph_idx])
        graph_idx += 1
        cur_ids = cur_ids[pos + 1:]

    return torch.cat(parts, dim=0)


def parse_batch_output(text, expected_count):
    """Parse numbered output lines like '1. cs.LG(Machine Learning)\\n2. cs.CR(...)'."""
    predictions = []
    pattern = re.compile(r'(\d+)\.\s*(.+)')
    for match in pattern.finditer(text):
        predictions.append(match.group(2).strip())

    if len(predictions) >= expected_count:
        return predictions[:expected_count]

    # Fallback: split by newline
    predictions = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line:
            line = re.sub(r'^\d+\.\s*', '', line)
            predictions.append(line)

    while len(predictions) < expected_count:
        predictions.append("")

    return predictions[:expected_count]


@torch.inference_mode()
def generate_batched(model, prefix_kv, prefix_len, per_prompt_embeds,
                     tokenizer, temperature=0.2, top_p=None, max_new_tokens=128):
    """
    Generate text for a batch using prefix KV cache.

    Args:
        prefix_kv: cached KV from prefix forward (tuple of (K,V) per layer, batch=1)
        prefix_len: number of tokens in cached prefix
        per_prompt_embeds: [B, seq_len, hidden_dim] — may vary in seq_len if padded
    Returns:
        list of generated text strings
    """
    B = per_prompt_embeds.shape[0]
    device = per_prompt_embeds.device

    batch_kv = tuple(
        (k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1))
        for k, v in prefix_kv
    )

    total_len = prefix_len + per_prompt_embeds.shape[1]
    attn_mask = torch.ones(B, total_len, dtype=torch.long, device=device)

    out = model.model(
        inputs_embeds=per_prompt_embeds,
        past_key_values=batch_kv,
        attention_mask=attn_mask,
        use_cache=True,
        return_dict=True,
    )
    current_kv = out.past_key_values
    current_logits = model.lm_head(out.last_hidden_state[:, -1:, :]).squeeze(1)

    eos_id = tokenizer.eos_token_id
    generated = [[] for _ in range(B)]
    done = [False] * B

    for _ in range(max_new_tokens):
        if temperature > 0:
            scaled = current_logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = current_logits.argmax(dim=-1, keepdim=True)

        for i in range(B):
            if not done[i]:
                tok = next_tokens[i].item()
                generated[i].append(tok)
                if tok == eos_id:
                    done[i] = True

        if all(done):
            break

        kv_len = current_kv[0][0].shape[2]
        step_attn = torch.ones(B, kv_len + 1, dtype=torch.long, device=device)
        out = model.model(
            input_ids=next_tokens,
            past_key_values=current_kv,
            attention_mask=step_attn,
            use_cache=True,
            return_dict=True,
        )
        current_kv = out.past_key_values
        current_logits = model.lm_head(out.last_hidden_state).squeeze(1)

    texts = []
    for i in range(B):
        text = tokenizer.decode(generated[i], skip_special_tokens=True).strip()
        texts.append(text)
    return texts


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loading model from {model_path}, base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()
    model.eval()

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

    N = args.nodes_per_prompt  # nodes per prompt

    # Build prompt
    if args.restructure_prompt:
        qs = build_nc_prompt_restructured(args.dataset, N)
        mode_str = f"restructured-N{N}"
    else:
        assert N == 1, "Original prompt only supports nodes_per_prompt=1"
        qs = build_nc_prompt_original(args.dataset, questions[0])
        mode_str = "original"

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Split at FIRST <graph> to separate prefix from the rest
    first_graph_pos = full_prompt.index(DEFAULT_GRAPH_TOKEN)
    prefix_text = full_prompt[:first_graph_pos]
    suffix_text_with_graphs = full_prompt[first_graph_pos:]  # "<graph>, <graph>, ... suffix"

    # Tokenize prefix (with BOS)
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
    prefix_len = prefix_ids.shape[1]

    # Tokenize suffix (with graph tokens) using tokenizer_graph_token
    suffix_token_ids = tokenizer_graph_token(
        suffix_text_with_graphs, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').cuda()
    # Count graph tokens in suffix
    n_graphs_in_suffix = (suffix_token_ids == GRAPH_TOKEN_INDEX).sum().item()
    n_text_tokens_in_suffix = len(suffix_token_ids) - n_graphs_in_suffix
    graph_tokens_total = N * 111

    print(f"Mode: {mode_str}")
    print(f"Prefix tokens: {prefix_len} (cached)")
    print(f"Per-prompt: {N} graphs x 111 = {graph_tokens_total} graph tokens + "
          f"{n_text_tokens_in_suffix} text tokens = {graph_tokens_total + n_text_tokens_in_suffix} total")
    print(f"Savings: {prefix_len}/{prefix_len + graph_tokens_total + n_text_tokens_in_suffix} = "
          f"{prefix_len / (prefix_len + graph_tokens_total + n_text_tokens_in_suffix) * 100:.0f}% cached")

    # Compute prefix KV cache (ONCE)
    print("Computing prefix KV cache...")
    prefix_embeds = model.get_model().embed_tokens(prefix_ids).half()
    with torch.inference_mode():
        prefix_out = model.model(
            inputs_embeds=prefix_embeds, use_cache=True, return_dict=True)
    prefix_kv = prefix_out.past_key_values
    print(f"Prefix KV cached: {prefix_len} tokens, "
          f"{sum(k.numel() + v.numel() for k, v in prefix_kv) * 2 / 1e6:.0f} MB")

    mm_projector = model.get_model().mm_projector
    embed_tokens = model.get_model().embed_tokens
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    existing_count = 0
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        existing_count = len(open(answers_file, 'r').readlines())
        if existing_count >= len(questions):
            print("Already done!")
            return
        questions = questions[existing_count:]
        print(f"Resuming from {existing_count} existing results")

    ans_file = open(answers_file, "a" if existing_count > 0 else "w")

    B = args.batch_size  # number of prompts in parallel
    total_gen_time = 0
    t_start = time.time()

    # Group test nodes: N nodes per prompt, B prompts per batch
    nodes_per_batch = N * B

    for batch_start in tqdm(range(0, len(questions), nodes_per_batch), desc=f"KV-cache N{N}"):
        batch_nodes = questions[batch_start:batch_start + nodes_per_batch]

        # Split into groups of N
        groups = []
        for g_start in range(0, len(batch_nodes), N):
            groups.append(batch_nodes[g_start:g_start + N])
        actual_B = len(groups)

        # Build per-prompt embeddings
        prompt_embeds_list = []
        group_ids = []
        group_gts = []

        for group in groups:
            actual_N = len(group)
            # Project each node's graph
            projected_graphs = []
            ids = []
            gts = []
            for node in group:
                graph_emb = build_graph_embedding_single(node, pretrained_emb, structure_emb)
                projected = mm_projector(graph_emb.half().cuda()).squeeze(0)  # [111, 4096]
                projected_graphs.append(projected)
                ids.append(node['id'])
                gts.append(node['conversations'][1]['value'])
            group_ids.append(ids)
            group_gts.append(gts)

            if N == 1:
                # Single node: just [graph_emb, suffix_emb]
                suffix_emb = embed_tokens(suffix_token_ids[suffix_token_ids != GRAPH_TOKEN_INDEX].unsqueeze(0)).squeeze(0).half()
                prompt_emb = torch.cat([projected_graphs[0], suffix_emb], dim=0)
            else:
                # Multi-node: replace each GRAPH_TOKEN_INDEX with corresponding graph
                # For last group that may have fewer nodes, rebuild suffix with actual_N graphs
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

        # Pad to same length and stack
        max_len = max(e.shape[0] for e in prompt_embeds_list)
        padded = []
        for e in prompt_embeds_list:
            if e.shape[0] < max_len:
                pad = torch.zeros(max_len - e.shape[0], e.shape[1], dtype=e.dtype, device=e.device)
                e = torch.cat([e, pad], dim=0)
            padded.append(e)
        per_prompt_embeds = torch.stack(padded)  # [actual_B, max_len, 4096]

        # Generate
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

        # Write results
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
                    "prompt": f"[kvcache-{mode_str}]",
                    "text": text,
                    "gt": gts[0],
                    "answer_id": shortuuid.uuid()
                }) + "\n")
            else:
                # Parse N predictions from numbered output
                predictions = parse_batch_output(text, actual_N)
                for i in range(actual_N):
                    ans_file.write(json.dumps({
                        "question_id": ids[i],
                        "prompt": f"[kvcache-{mode_str}, node {i+1}/{actual_N}]",
                        "text": predictions[i],
                        "gt": gts[i],
                        "answer_id": shortuuid.uuid()
                    }) + "\n")
        ans_file.flush()

    ans_file.close()
    total_time = time.time() - t_start
    n = len(questions)
    print(f"\nDone! {n} samples in {total_time:.1f}s total ({total_gen_time:.1f}s generation)")
    print(f"Throughput: {n/total_time:.1f} samples/s, {total_time/n*1000:.0f} ms/sample")
    print(f"Results saved to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="results/kvcache_nc.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of prompts to process in parallel")
    parser.add_argument("--nodes_per_prompt", type=int, default=1,
                        help="Number of nodes (graph tokens) per prompt")
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--restructure_prompt", action="store_true",
                        help="Put class list before <graph> for max prefix caching")
    args = parser.parse_args()

    eval_model(args)
