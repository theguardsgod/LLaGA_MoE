"""
Evaluation script for MoE-LLaGA.

Based on eval_pretrain.py but loads MoE model and provides routing_features
during inference.
"""

import sys
sys.path.append("./")
sys.path.append("./utils")

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from utils.constants import (
    GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID
)
from utils.conversation import conv_templates, SeparatorStyle
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from model.language_model.moe_llaga_llama import MoELlagaLlamaForCausalLM, MoELlagaConfig

from transformers import AutoTokenizer, AutoConfig

SMALL_DATASETS = ["pubmed", "cora"]

DATASET_DIRS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}


def load_pretrain_embedding_graph(data_dir, emb_type):
    if emb_type == "simteg":
        sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        return torch.cat([sbert, roberta, e5], dim=-1)
    return torch.load(os.path.join(data_dir, f"{emb_type}_x.pt"))


def load_pretrain_embedding_hop(data_dir, emb_type, hop, mask):
    if emb_type == "simteg":
        sbert = [torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))[mask]] + \
                [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        roberta = [torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))[mask]] + \
                  [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        e5 = [torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))[mask]] + \
             [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        return [torch.cat([sbert[i], roberta[i], e5[i]], dim=-1) for i in range(hop + 1)]
    return [torch.load(os.path.join(data_dir, f"{emb_type}_x.pt"))[mask]] + \
           [torch.load(os.path.join(data_dir, f"{emb_type}_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]


def load_moe_model(model_path, model_base, cache_dir="../../checkpoint"):
    """Load MoE-LLaGA model from checkpoint."""
    kwargs = {"torch_dtype": torch.float16}

    if model_base is not None:
        # Projector-only: load base LLM + MoE projector weights
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg = AutoConfig.from_pretrained(model_path)
        model = MoELlagaLlamaForCausalLM.from_pretrained(
            model_base, low_cpu_mem_usage=True, config=cfg,
            cache_dir=cache_dir, **kwargs
        )
        # Load MoE projector
        moe_path = os.path.join(model_path, 'moe_projector.bin')
        if os.path.exists(moe_path):
            moe_weights = torch.load(moe_path, map_location='cpu')
            moe_weights = {k.replace("moe_projector.", ""): v.to(torch.float16)
                           for k, v in moe_weights.items()}
            model.moe_projector = model.moe_projector or MoELlagaLlamaForCausalLM._build_moe_from_config(cfg)
            # Re-init MoE from config if needed
            if model.moe_projector is None:
                from model.moe_llaga import MoEGraphProjector
                model.moe_projector = MoEGraphProjector(
                    mm_hidden_size=cfg.mm_hidden_size,
                    llm_hidden_size=cfg.hidden_size,
                    num_experts=getattr(cfg, 'num_experts', 4),
                    top_k=getattr(cfg, 'top_k', 2),
                    projector_type=getattr(cfg, 'mm_projector_type', 'linear'),
                    routing_dim=getattr(cfg, 'routing_dim', 2432),
                )
            model.moe_projector.load_state_dict(moe_weights)
            print(f"Loaded MoE projector from {moe_path}")
        # Also load base mm_projector if exists
        mm_path = os.path.join(model_path, 'mm_projector.bin')
        if os.path.exists(mm_path):
            mm_weights = torch.load(mm_path, map_location='cpu')
            mm_weights = {k: v.to(torch.float16) for k, v in mm_weights.items()}
            model.load_state_dict(mm_weights, strict=False)
    else:
        # Full model checkpoint
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = MoELlagaLlamaForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

    # Ensure MoE projector is initialized and loaded.
    # __init__ sets self.moe_projector = None, so from_pretrained silently
    # ignores the moe_projector.* weights. We must create the module from
    # config and then load its weights from the separate checkpoint file.
    if model.moe_projector is None:
        cfg = model.config
        from model.moe_llaga import MoEGraphProjector
        model.moe_projector = MoEGraphProjector(
            mm_hidden_size=getattr(cfg, 'mm_hidden_size', 2543),
            llm_hidden_size=getattr(cfg, 'hidden_size', 4096),
            num_experts=getattr(cfg, 'num_experts', 4),
            top_k=getattr(cfg, 'top_k', 2),
            projector_type=getattr(cfg, 'mm_projector_type', 'linear'),
            routing_dim=getattr(cfg, 'routing_dim', 2432),
            noise_std=getattr(cfg, 'noise_std', 1.0),
        )
        moe_path = os.path.join(model_path, 'moe_projector.bin')
        if os.path.exists(moe_path):
            moe_weights = torch.load(moe_path, map_location='cpu')
            # Strip prefix if present
            clean = {}
            for k, v in moe_weights.items():
                key = k.replace("moe_projector.", "") if k.startswith("moe_projector.") else k
                clean[key] = v.to(torch.float16)
            model.moe_projector.load_state_dict(clean)
            print(f"Loaded MoE projector from {moe_path}")
        else:
            print(f"WARNING: moe_projector.bin not found at {moe_path}")

    model.resize_token_embeddings(len(tokenizer))
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, context_len


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    print(f"Loading MoE model from {model_path}, base: {args.model_base}")
    tokenizer, model, context_len = load_moe_model(
        model_path, args.model_base, cache_dir=args.cache_dir
    )
    model = model.to(torch.float16).cuda()
    model.eval()

    data_dir = DATASET_DIRS.get(args.dataset)
    if data_dir is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load community data
    comm_path = os.path.join(data_dir, "node_to_community.pt")
    feat_path = os.path.join(data_dir, "community_features.pt")
    if os.path.exists(comm_path) and os.path.exists(feat_path):
        node_to_community = torch.load(comm_path)
        community_features = torch.load(feat_path)
        print(f"Loaded community data: {community_features.shape[0]} communities")
    else:
        print("WARNING: No community data found, using zero routing features")
        node_to_community = None
        community_features = None

    # Load prompts
    if args.task in ["nc", "nd"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, "sampled_2_10_test.jsonl")
        else:
            prompt_file = os.path.join(data_dir,
                                       f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
    elif args.task == "lp":
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, "edge_sampled_2_10_only_test.jsonl")
        else:
            prompt_file = os.path.join(data_dir,
                                       f"edge_sampled_{args.use_hop}_{args.sample_neighbor_size}_only_test.jsonl")
    else:
        raise ValueError(f"Unknown task: {args.task}")

    data = torch.load(os.path.join(data_dir, "processed_data.pt"))
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        line_number = len(open(answers_file, 'r').readlines())
        print(f"{args.answers_file} already exists with {line_number} lines")
        if line_number >= len(lines):
            return
        lines = lines[line_number:]
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]

    # Load embeddings
    index = None
    if args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(
            f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
        )
    elif args.template == "HO":
        n = data.num_nodes
        mask = torch.full([n], fill_value=False, dtype=torch.bool)
        for q in questions:
            idx = q["id"]
            if args.task == "lp":
                mask[idx[0]] = True
                mask[idx[1]] = True
            elif args.task in ["nc", "nd"]:
                mask[idx] = True
        pretrained_emb = load_pretrain_embedding_hop(
            data_dir, args.pretrained_embedding_type, args.use_hop, mask
        )
        index = torch.full([n], fill_value=n + 1, dtype=torch.long)
        test_index = torch.arange(mask.sum())
        index[mask] = test_index
        structure_emb = None
    else:
        raise ValueError

    # Evaluate
    for line in tqdm(questions):
        idx = line["id"]

        # Build question prompt (same as eval_pretrain.py)
        if args.task == "nc":
            if args.dataset == "products":
                qs = (f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent "
                      f"products sold in Amazon, and edges between products indicate they are "
                      f"purchased together. We need to classify the center node into 47 classes: "
                      f"Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, "
                      f"Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, "
                      f"Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, "
                      f"Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, "
                      f"Office Products, Industrial & Scientific, Musical Instruments, "
                      f"Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, "
                      f"Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, "
                      f"Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, "
                      f"MP3 Players & Accessories, Gift Cards, Office & School Supplies, "
                      f"Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, "
                      f"Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D\u00e9cor, "
                      f"#508510, please tell me which class the center node belongs to?")
            else:
                qs = line["conversations"][0]['value']
        elif args.task == "nd":
            qs = f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "lp":
            qs = (f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, "
                  f"we need to predict whether these two nodes connect with each other. "
                  f"Please tell me whether two center nodes in the subgraphs should connect to each other.")
        else:
            raise ValueError

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_graph_token(
            prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        # Build graph embeddings
        if not isinstance(line['graph'][0], list):
            line['graph'] = [line['graph']]

        if args.template == "ND":
            graph = torch.LongTensor(line['graph'])
            mask = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = pretrained_emb[graph[mask]]
            s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
            graph_emb = torch.zeros((s, n, d))
            graph_emb[mask] = masked_graph_emb
            if structure_emb is not None:
                graph_emb = torch.cat(
                    [graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1
                )
        elif args.template == "HO":
            for g in range(len(line['graph'])):
                center_id = line['graph'][g][0]
                line['graph'][g] = [center_id] * (args.use_hop + 1)
            graph = torch.LongTensor(line['graph'])
            center_id = graph[:, 0]
            graph_emb = torch.stack(
                [emb[index[center_id]] for emb in pretrained_emb], dim=1
            )

        # Build routing features from community data
        center_ids = graph[:, 0]
        if node_to_community is not None and community_features is not None:
            comm_ids = node_to_community[center_ids]
            routing_feat = community_features[comm_ids]
        else:
            routing_feat = torch.zeros(center_ids.shape[0], 2432)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    routing_features=routing_feat.half().cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            input_token_len = input_ids.shape[1]
            n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff > 0:
                print(f'[Warning] {n_diff} output_ids differ from input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            print(f"Error: {e}")
            outputs = ""

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": qs,
            "graph": line['graph'],
            "text": outputs,
            "gt": line["conversations"][1]['value'],
            "answer_id": ans_id,
        }) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    args = parser.parse_args()

    eval_model(args)
