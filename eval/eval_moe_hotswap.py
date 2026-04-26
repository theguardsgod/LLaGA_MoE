"""
Multi-checkpoint MoE-LLaGA evaluation with hot-swap.

Loads the base LLM once and hot-swaps only the MoE projector weights
for each checkpoint/dataset, avoiding repeated ~14GB model loads.
"""

import sys
sys.path.append("./")
sys.path.append("./utils")

import copy
import gc
import argparse
import subprocess
import torch
# Compat: torch>=2.6 defaults weights_only=True; LLaGA .pt files contain
# torch_geometric Data objects, so force legacy behaviour.
_orig_torch_load = torch.load
def _compat_torch_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_torch_load(*a, **kw)
torch.load = _compat_torch_load
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
from model.moe_llaga import MoEGraphProjector

from transformers import AutoTokenizer

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


def load_base_model(first_checkpoint_path, model_base, cache_dir):
    """Load base LLM once and initialize empty MoE projector from config."""
    kwargs = {"torch_dtype": torch.float16}

    if model_base is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg = MoELlagaConfig.from_pretrained(first_checkpoint_path)
        model = MoELlagaLlamaForCausalLM.from_pretrained(
            model_base, low_cpu_mem_usage=True, config=cfg,
            cache_dir=cache_dir, **kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(first_checkpoint_path, use_fast=False)
        model = MoELlagaLlamaForCausalLM.from_pretrained(
            first_checkpoint_path, low_cpu_mem_usage=True, **kwargs
        )
        cfg = model.config

    # Initialize empty MoE projector (weights loaded later by swap_moe_projector)
    model.moe_projector = MoEGraphProjector(
        mm_hidden_size=getattr(cfg, 'mm_hidden_size', 2543),
        llm_hidden_size=getattr(cfg, 'hidden_size', 4096),
        num_experts=getattr(cfg, 'num_experts', 4),
        top_k=getattr(cfg, 'top_k', 2),
        projector_type=getattr(cfg, 'mm_projector_type', 'linear'),
        routing_dim=getattr(cfg, 'routing_dim', 2432),
        noise_std=getattr(cfg, 'noise_std', 1.0),
    )

    model.resize_token_embeddings(len(tokenizer))
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, context_len


def swap_moe_projector(model, checkpoint_path):
    """Hot-swap MoE projector weights from a checkpoint."""
    moe_path = os.path.join(checkpoint_path, 'moe_projector.bin')
    if not os.path.exists(moe_path):
        raise FileNotFoundError(f"moe_projector.bin not found at {moe_path}")

    # Check if config differs and rebuild projector if needed
    cfg_path = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(cfg_path):
        ckpt_cfg = MoELlagaConfig.from_pretrained(checkpoint_path)
        base_cfg = model.config
        config_keys = ['mm_hidden_size', 'num_experts', 'top_k', 'routing_dim',
                        'mm_projector_type', 'hidden_size']
        needs_rebuild = False
        for k in config_keys:
            if getattr(ckpt_cfg, k, None) != getattr(base_cfg, k, None):
                print(f"  Config mismatch: {k} = {getattr(ckpt_cfg, k, None)} "
                      f"(was {getattr(base_cfg, k, None)}), rebuilding projector")
                needs_rebuild = True
                break
        if needs_rebuild:
            del model.moe_projector
            gc.collect()
            model.moe_projector = MoEGraphProjector(
                mm_hidden_size=getattr(ckpt_cfg, 'mm_hidden_size', 2543),
                llm_hidden_size=getattr(ckpt_cfg, 'hidden_size', 4096),
                num_experts=getattr(ckpt_cfg, 'num_experts', 4),
                top_k=getattr(ckpt_cfg, 'top_k', 2),
                projector_type=getattr(ckpt_cfg, 'mm_projector_type', 'linear'),
                routing_dim=getattr(ckpt_cfg, 'routing_dim', 2432),
                noise_std=getattr(ckpt_cfg, 'noise_std', 1.0),
            )

    moe_weights = torch.load(moe_path, map_location='cpu')
    clean = {}
    for k, v in moe_weights.items():
        key = k.replace("moe_projector.", "") if k.startswith("moe_projector.") else k
        clean[key] = v.to(torch.float16)

    model.moe_projector.load_state_dict(clean)
    model.moe_projector.to(device=model.device)

    del moe_weights, clean
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Hot-swapped MoE projector from {moe_path}")


def load_dataset_data(dataset, args):
    """Load all dataset-specific data: embeddings, community features, test prompts."""
    data_dir = DATASET_DIRS.get(dataset)
    if data_dir is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    data = torch.load(os.path.join(data_dir, "processed_data.pt"))

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

    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    questions = [json.loads(q) for q in lines]

    # Load community data
    comm_path = os.path.join(data_dir, "node_to_community.pt")
    feat_path = os.path.join(data_dir, "community_features.pt")
    if os.path.exists(comm_path) and os.path.exists(feat_path):
        node_to_community = torch.load(comm_path)
        community_features = torch.load(feat_path)
        print(f"  Loaded community data: {community_features.shape[0]} communities")
    else:
        print("  WARNING: No community data found, using zero routing features")
        node_to_community = None
        community_features = None

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
        raise ValueError(f"Unknown template: {args.template}")

    return {
        "data": data,
        "questions": questions,
        "all_lines": lines,
        "pretrained_emb": pretrained_emb,
        "structure_emb": structure_emb,
        "node_to_community": node_to_community,
        "community_features": community_features,
        "index": index,
    }


def build_question(task, dataset, line):
    """Build question prompt for a given task and dataset."""
    if task == "nc":
        if dataset == "products":
            return (f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent "
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
        return line["conversations"][0]['value']
    if task == "nd":
        return f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
    if task == "lp":
        return (f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, "
                f"we need to predict whether these two nodes connect with each other. "
                f"Please tell me whether two center nodes in the subgraphs should connect to each other.")
    raise ValueError(f"Unknown task: {task}")


def prepare_single_sample(line, dataset, args, pretrained_emb, structure_emb,
                          node_to_community, community_features, index, tokenizer):
    """Pre-process a single sample into model-ready tensors.

    Returns dict with input_ids [1,L], graph_emb, graph, routing_feat,
    question text, and graph_out (for writing results).
    """
    qs = build_question(args.task, dataset, line)

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_graph_token(
        prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0)  # [1, L]

    graph_data = copy.deepcopy(line['graph'])
    if not isinstance(graph_data[0], list):
        graph_data = [graph_data]

    if args.template == "ND":
        graph = torch.LongTensor(graph_data)
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
        for g in range(len(graph_data)):
            center_id = graph_data[g][0]
            graph_data[g] = [center_id] * (args.use_hop + 1)
        graph = torch.LongTensor(graph_data)
        center_id = graph[:, 0]
        graph_emb = torch.stack(
            [emb[index[center_id]] for emb in pretrained_emb], dim=1
        )
    else:
        raise ValueError(f"Unknown template: {args.template}")

    center_ids = graph[:, 0]
    if node_to_community is not None and community_features is not None:
        comm_ids = node_to_community[center_ids]
        routing_feat = community_features[comm_ids]
    else:
        routing_feat = torch.zeros(center_ids.shape[0], 2432)

    return {
        'input_ids': input_ids,
        'graph_emb': graph_emb,
        'graph': graph,
        'routing_feat': routing_feat,
        'question': qs,
        'graph_out': graph_data,
    }


def run_eval_single_dataset(model, tokenizer, dataset_data, dataset, output_file, args):
    """Run inference on a single dataset using the currently loaded MoE projector.

    Supports batch inference (--batch_size > 1) when all prompts have the same
    token length, which is the common case for LLaGA tasks.
    """
    questions = dataset_data["questions"]
    pretrained_emb = dataset_data["pretrained_emb"]
    structure_emb = dataset_data["structure_emb"]
    node_to_community = dataset_data["node_to_community"]
    community_features = dataset_data["community_features"]
    index = dataset_data["index"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Resume support
    if "tmp" not in output_file and os.path.exists(output_file):
        done = len(open(output_file, 'r').readlines())
        print(f"  {output_file} already has {done} lines")
        if done >= len(questions):
            print(f"  Skipping {dataset} (already complete)")
            return
        questions = questions[done:]
        ans_file = open(output_file, "a")
    else:
        ans_file = open(output_file, "w")

    conv_ref = conv_templates[args.conv_mode]
    stop_str = conv_ref.sep if conv_ref.sep_style != SeparatorStyle.TWO else conv_ref.sep2
    batch_size = getattr(args, 'batch_size', 1)

    # Pre-compute all samples
    all_prepared = []
    for line in questions:
        all_prepared.append(prepare_single_sample(
            line, dataset, args, pretrained_emb, structure_emb,
            node_to_community, community_features, index, tokenizer
        ))

    # Check if batching is possible (requires uniform prompt lengths)
    if batch_size > 1:
        lengths = set(p['input_ids'].shape[1] for p in all_prepared)
        if len(lengths) > 1:
            print(f"  {len(lengths)} different prompt lengths detected, falling back to batch_size=1")
            batch_size = 1
        else:
            print(f"  Prompt length={next(iter(lengths))}, using batch_size={batch_size}")

    num_batches = (len(all_prepared) + batch_size - 1) // batch_size
    desc = f"  {dataset}" + (f" (bs={batch_size})" if batch_size > 1 else "")

    for b in tqdm(range(num_batches), desc=desc):
        start = b * batch_size
        end = min(start + batch_size, len(all_prepared))
        batch = all_prepared[start:end]
        B = len(batch)

        # Stack tensors: graph dims are concatenated (not nested) since the model
        # indexes graph_features linearly across all <graph> tokens in the batch
        batch_input_ids = torch.cat([p['input_ids'] for p in batch], dim=0).cuda()
        batch_graph_emb = torch.cat([p['graph_emb'] for p in batch], dim=0).half().cuda()
        batch_graph = torch.cat([p['graph'] for p in batch], dim=0).cuda()
        batch_routing = torch.cat([p['routing_feat'] for p in batch], dim=0).half().cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                batch_input_ids,
                graph_emb=batch_graph_emb,
                graph=batch_graph,
                routing_features=batch_routing,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        in_len = batch_input_ids.shape[1]
        n_diff = (batch_input_ids != output_ids[:, :in_len]).sum().item()
        if n_diff > 0:
            print(f'  [Warning] {n_diff} output_ids differ from input_ids')

        texts = tokenizer.batch_decode(
            output_ids[:, in_len:], skip_special_tokens=True
        )

        for i, text in enumerate(texts):
            text = text.strip()
            if text.endswith(stop_str):
                text = text[:-len(stop_str)].strip()

            line = questions[start + i]
            ans_file.write(json.dumps({
                "question_id": line["id"],
                "prompt": batch[i]['question'],
                "graph": batch[i]['graph_out'],
                "text": text,
                "gt": line["conversations"][1]["value"],
                "answer_id": shortuuid.uuid(),
            }) + "\n")
            ans_file.flush()

    ans_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', default=["arxiv", "products", "pubmed", "cora"])
    parser.add_argument("--model_base", type=str, default=None,
                        help="Base LLM path (recommended: avoids loading full checkpoint)")
    parser.add_argument("--checkpoint_pattern", type=str,
                        default="./checkpoints/{dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc")
    parser.add_argument("--output_dir", type=str, default="/localnvme/llaga/eval_output")
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument("--output_prefix", type=str, default="moe",
                        help="Prefix for output files (e.g. 'moe_v2')")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (>1 enables batch generation)")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max new tokens to generate per sample. For NC/LP tasks "
                             "32 is enough (labels are short). Large values OOM at high batch.")
    args = parser.parse_args()

    disable_torch_init()

    # Build checkpoint-dataset pairs
    pairs = []
    for ds in args.datasets:
        ckpt = args.checkpoint_pattern.format(dataset=ds)
        if os.path.isdir(ckpt):
            pairs.append((ds, ckpt))
        else:
            print(f"WARNING: checkpoint not found for {ds}: {ckpt}, skipping")

    if not pairs:
        print("No valid checkpoints found. Exiting.")
        return

    # Load base LLM once
    first_ds, first_ckpt = pairs[0]
    print(f"Loading base LLM (config from {first_ckpt})...")
    tokenizer, model, context_len = load_base_model(
        first_ckpt, args.model_base, args.cache_dir
    )
    model = model.to(torch.float16).cuda()
    model.eval()

    # Ensure pad_token_id is set (needed for batch generation output padding)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Base LLM loaded. Starting evaluation of {len(pairs)} checkpoints.\n")

    # Evaluate each checkpoint
    for ds, ckpt_path in pairs:
        print(f"{'='*60}")
        print(f"Evaluating: {ds}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        # Hot-swap MoE projector
        swap_moe_projector(model, ckpt_path)

        # Load dataset data
        dataset_data = load_dataset_data(ds, args)

        # Run evaluation
        output_file = os.path.join(args.output_dir, f"{args.output_prefix}_{args.task}_{ds}.jsonl")
        run_eval_single_dataset(model, tokenizer, dataset_data, ds, output_file, args)

        # Score results
        print(f"  Scoring {ds}...")
        subprocess.run([
            "python", "eval/eval_res.py",
            "--dataset", ds,
            "--task", args.task,
            "--res_path", output_file,
        ])

        # Free dataset data
        del dataset_data
        gc.collect()
        torch.cuda.empty_cache()
        print()

    print("=== All evaluations complete ===")


if __name__ == "__main__":
    main()
