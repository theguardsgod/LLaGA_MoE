"""
Unified quantization evaluation script for LLaGA baseline and MoE models.

Runs post-training quantization (PTQ) inference with bitsandbytes 4-bit NF4
or 8-bit INT8 and profiles memory usage. Supports both baseline (mm_projector)
and MoE (moe_projector) checkpoints.

Usage:
    python eval/eval_quantized_unified.py \
        --model_type baseline \
        --quant_mode 4bit \
        --dataset cora \
        --output_dir /localnvme/llaga/eval_output/quant_experiments
"""

import sys
sys.path.append("./")
sys.path.append("./utils")

import argparse
import gc
import json
import os
import time

import shortuuid
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from model.language_model.llaga_llama import LlagaLlamaForCausalLM
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
from utils.conversation import SeparatorStyle, conv_templates
from utils.utils import disable_torch_init, tokenizer_graph_token

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
                [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt"))[mask]
                 for i in range(1, hop + 1)]
        roberta = [torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))[mask]] + \
                  [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt"))[mask]
                   for i in range(1, hop + 1)]
        e5 = [torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))[mask]] + \
             [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt"))[mask]
              for i in range(1, hop + 1)]
        return [torch.cat([s, r, e], dim=-1) for s, r, e in zip(sbert, roberta, e5)]
    else:
        return [torch.load(os.path.join(data_dir, f"{emb_type}_x.pt"))[mask]] + \
               [torch.load(os.path.join(data_dir, f"{emb_type}_{i}hop_x.pt"))[mask]
                for i in range(1, hop + 1)]


def build_question_nc(dataset, line):
    """Build node classification question string."""
    if dataset == "products":
        return (
            f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in "
            f"Amazon, and edges between products indicate they are purchased together. We need to classify "
            f"the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & "
            f"Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, "
            f"Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, "
            f"Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & "
            f"Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby "
            f"Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, "
            f"Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & "
            f"Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & "
            f"Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & "
            f"D\u00e9cor, #508510, please tell me which class the center node belongs to?"
        )
    return line["conversations"][0]["value"]


def get_model_device(model):
    """Get the device where the model expects inputs."""
    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        for key in ("model.embed_tokens", "model", "lm_head"):
            for name, dev in device_map.items():
                if name == key or name.startswith(f"{key}."):
                    if isinstance(dev, int):
                        return torch.device(f"cuda:{dev}")
                    if isinstance(dev, str) and dev.startswith("cuda"):
                        return torch.device(dev)
        for dev in device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0")


def get_projector_device(model):
    """Get the device of the projector module."""
    proj = None
    if hasattr(model, "moe_projector") and model.moe_projector is not None:
        proj = model.moe_projector
    elif hasattr(model, "get_model"):
        proj = getattr(model.get_model(), "mm_projector", None)

    if proj is not None:
        try:
            return next(proj.parameters()).device
        except StopIteration:
            pass

    return get_model_device(model)


def load_baseline_model(ckpt_path, model_base, quant_mode, cache_dir):
    """Load baseline LLaGA model with optional quantization.

    When quantized, bitsandbytes replaces nn.Linear layers (including the
    mm_projector) with 4-bit/8-bit variants.  We therefore:
      1. Load the base LLM with quantization applied to all Linear layers.
      2. Replace the quantized mm_projector with a fresh FP16 nn.Linear.
      3. Load the trained projector weights into that fresh module.
    This keeps the LLM backbone quantized while the tiny projector stays FP16.
    """
    from model.llaga_arch import build_graph_projector

    cfg = AutoConfig.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    kwargs = {"low_cpu_mem_usage": True, "config": cfg, "cache_dir": cache_dir}

    if quant_mode == "4bit":
        kwargs["device_map"] = "auto"
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant_mode == "8bit":
        kwargs["device_map"] = "auto"
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = torch.float16

    model = LlagaLlamaForCausalLM.from_pretrained(model_base, **kwargs)
    model.resize_token_embeddings(len(tokenizer))

    # For quantized models, the mm_projector was also quantized by bnb.
    # Replace it with a fresh FP16 Linear and load the trained weights.
    proj_path = os.path.join(ckpt_path, "mm_projector.bin")
    if os.path.exists(proj_path):
        proj_weights_raw = torch.load(proj_path, map_location="cpu")
        # Strip "model.mm_projector." prefix to get clean keys
        proj_weights = {}
        for k, v in proj_weights_raw.items():
            clean_key = k.replace("model.mm_projector.", "")
            proj_weights[clean_key] = v.to(torch.float16)

        if quant_mode in ("4bit", "8bit"):
            # Determine which device to put the projector on
            target_device = get_model_device(model)
            # Replace the quantized projector with a fresh FP16 one
            fresh_proj = build_graph_projector(cfg).to(dtype=torch.float16, device=target_device)
            fresh_proj.load_state_dict(proj_weights)
            model.get_model().mm_projector = fresh_proj
            print(f"  Loaded mm_projector (FP16) from {proj_path} -> {target_device}")
        else:
            # FP16 mode: load_state_dict works fine with the full key prefix
            fp16_weights = {k: v.to(torch.float16) for k, v in proj_weights_raw.items()}
            model.load_state_dict(fp16_weights, strict=False)
            print(f"  Loaded mm_projector from {proj_path}")

    if quant_mode == "fp16":
        model = model.to(torch.float16).cuda()

    model.eval()
    return tokenizer, model


def load_moe_model(ckpt_path, model_base, quant_mode, cache_dir):
    """Load MoE LLaGA model with optional quantization.

    Similar to the baseline loader: the LLM backbone is quantized while the
    MoE projector stays in FP16.  We must import MoELlagaConfig before
    AutoConfig so that model_type='moe_llaga' is recognised.
    """
    from model.language_model.moe_llaga_llama import MoELlagaConfig, MoELlagaLlamaForCausalLM
    from model.moe_llaga import MoEGraphProjector

    # Use MoELlagaConfig directly since model_type="moe_llaga" is not in
    # the default AutoConfig registry for transformers==4.31.0.
    cfg = MoELlagaConfig.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    kwargs = {"low_cpu_mem_usage": True, "config": cfg, "cache_dir": cache_dir}

    if quant_mode == "4bit":
        kwargs["device_map"] = "auto"
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant_mode == "8bit":
        kwargs["device_map"] = "auto"
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = torch.float16

    model = MoELlagaLlamaForCausalLM.from_pretrained(model_base, **kwargs)

    # Build MoE projector
    model.moe_projector = MoEGraphProjector(
        mm_hidden_size=getattr(cfg, "mm_hidden_size", 2543),
        llm_hidden_size=getattr(cfg, "hidden_size", 4096),
        num_experts=getattr(cfg, "num_experts", 4),
        top_k=getattr(cfg, "top_k", 2),
        projector_type=getattr(cfg, "mm_projector_type", "linear"),
        routing_dim=getattr(cfg, "routing_dim", 2432),
        noise_std=getattr(cfg, "noise_std", 1.0),
    )

    # Load MoE projector weights
    moe_path = os.path.join(ckpt_path, "moe_projector.bin")
    if os.path.exists(moe_path):
        weights = torch.load(moe_path, map_location="cpu")
        clean = {}
        for k, v in weights.items():
            if k.startswith("moe_projector."):
                k = k[len("moe_projector."):]
            clean[k] = v.to(torch.float16)
        model.moe_projector.load_state_dict(clean)
        print(f"  Loaded moe_projector from {moe_path}")

    model.resize_token_embeddings(len(tokenizer))

    if quant_mode == "fp16":
        model = model.to(torch.float16).cuda()
    else:
        # Move MoE projector to the same device as the LLM
        target_device = get_model_device(model)
        model.moe_projector.to(dtype=torch.float16, device=target_device)

    model.eval()
    return tokenizer, model


def run_eval(model, tokenizer, dataset, args):
    """Run evaluation and return list of result dicts plus timing info."""
    data_dir = DATASET_DIRS[dataset]
    data = torch.load(os.path.join(data_dir, "processed_data.pt"))

    prompt_file = os.path.join(
        data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl"
    )
    lines = open(prompt_file, "r").readlines()
    print(f"  Total test samples: {len(lines)}")

    if args.start >= 0:
        end = args.end if args.end > 0 else len(lines)
        lines = lines[args.start:end]
    elif args.end > 0:
        lines = lines[:args.end]

    questions = [json.loads(q) for q in lines]
    print(f"  Evaluating {len(questions)} samples")

    # Load embeddings
    if args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(
            f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
        )
        index = None
    elif args.template == "HO":
        num_nodes = data.num_nodes
        mask = torch.full([num_nodes], fill_value=False, dtype=torch.bool)
        for question in questions:
            idx = question["id"]
            mask[idx] = True
        pretrained_emb = load_pretrain_embedding_hop(
            data_dir, args.pretrained_embedding_type, args.use_hop, mask
        )
        index = torch.full([num_nodes], fill_value=num_nodes + 1, dtype=torch.long)
        test_index = torch.arange(mask.sum())
        index[mask] = test_index
        structure_emb = None
    else:
        raise ValueError(f"Unknown template: {args.template}")

    # Load community data for MoE
    node_to_community = None
    community_features = None
    if args.model_type == "moe":
        comm_path = os.path.join(data_dir, "node_to_community.pt")
        feat_path = os.path.join(data_dir, "community_features.pt")
        if os.path.exists(comm_path) and os.path.exists(feat_path):
            node_to_community = torch.load(comm_path)
            community_features = torch.load(feat_path)
            print(f"  Loaded community data: {community_features.shape[0]} communities")
        else:
            print("  WARNING: No community data found, using zero routing features")

    input_device = get_model_device(model)
    graph_device = get_projector_device(model)

    results = []
    total_tokens = 0
    total_time = 0.0

    # Reset peak memory tracking
    torch.cuda.reset_peak_memory_stats()

    for line in tqdm(questions, desc=f"  {dataset}"):
        idx = line["id"]
        qs = build_question_nc(dataset, line)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(
            prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(input_device)

        if not isinstance(line["graph"][0], list):
            line["graph"] = [line["graph"]]

        if args.template == "ND":
            graph = torch.LongTensor(line["graph"])
            mask_g = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = pretrained_emb[graph[mask_g]]
            sc, gl, hd = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
            graph_emb = torch.zeros((sc, gl, hd))
            graph_emb[mask_g] = masked_graph_emb
            if structure_emb is not None:
                graph_emb = torch.cat(
                    [graph_emb, structure_emb.unsqueeze(0).expand(sc, -1, -1)], dim=-1
                )
        elif args.template == "HO":
            for gi in range(len(line["graph"])):
                cid = line["graph"][gi][0]
                line["graph"][gi] = [cid] * (args.use_hop + 1)
            graph = torch.LongTensor(line["graph"])
            center_id = graph[:, 0]
            graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        gen_kwargs = dict(
            graph_emb=graph_emb.to(device=graph_device, dtype=torch.float16),
            graph=graph.to(graph_device),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True,
        )

        # Add routing features for MoE
        if args.model_type == "moe":
            center_ids = graph[:, 0]
            if node_to_community is not None and community_features is not None:
                comm_ids = node_to_community[center_ids]
                routing_feat = community_features[comm_ids]
            else:
                routing_feat = torch.zeros(center_ids.shape[0], 2432)
            gen_kwargs["routing_features"] = routing_feat.to(
                device=graph_device, dtype=torch.float16
            )

        try:
            torch.cuda.synchronize()
            t0 = time.time()

            with torch.inference_mode():
                output_ids = model.generate(input_ids, **gen_kwargs)

            torch.cuda.synchronize()
            t1 = time.time()

            gen_tokens = output_ids.shape[1] - input_ids.shape[1]
            total_tokens += gen_tokens
            total_time += (t1 - t0)

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)].strip()

        except Exception as exc:
            print(f"  Error on sample {idx}: {exc}")
            outputs = ""

        results.append({
            "question_id": idx,
            "prompt": qs,
            "graph": line["graph"],
            "text": outputs,
            "gt": line["conversations"][1]["value"],
            "answer_id": shortuuid.uuid(),
        })

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    avg_time = total_time / len(questions) if questions else 0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

    stats = {
        "num_samples": len(questions),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "total_time_s": round(total_time, 2),
        "avg_time_per_sample_s": round(avg_time, 3),
        "total_tokens_generated": total_tokens,
        "tokens_per_second": round(tokens_per_sec, 1),
    }

    return results, stats


def score_results(dataset, output_file):
    """Run eval_res.py scoring and capture the output."""
    import subprocess
    result = subprocess.run(
        ["python", "eval/eval_res.py", "--dataset", dataset, "--task", "nc", "--res_path", output_file],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Unified quantization evaluation for LLaGA")
    parser.add_argument("--model_type", type=str, required=True, choices=["baseline", "moe"])
    parser.add_argument("--quant_mode", type=str, required=True, choices=["fp16", "4bit", "8bit"])
    parser.add_argument("--dataset", type=str, required=True, choices=["cora", "pubmed", "arxiv", "products"])
    parser.add_argument("--output_dir", type=str, default="/localnvme/llaga/eval_output/quant_experiments")
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    args = parser.parse_args()

    disable_torch_init()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"{args.model_type}_{args.quant_mode}_nc_{args.dataset}.jsonl"
    )

    # Check if already complete
    if os.path.exists(output_file) and "tmp" not in output_file:
        existing = len(open(output_file).readlines())
        data_dir = DATASET_DIRS[args.dataset]
        pf = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
        total = len(open(pf).readlines())
        if existing >= total:
            print(f"Output file already complete ({existing}/{total} samples): {output_file}")
            print("Scoring existing results...")
            score_results(args.dataset, output_file)
            return

    print("=" * 70)
    print(f"MODEL: {args.model_type} | QUANT: {args.quant_mode} | DATASET: {args.dataset}")
    print("=" * 70)

    # Determine checkpoint path
    if args.model_type == "baseline":
        ckpt_path = f"./checkpoints/{args.dataset}/llaga-vicuna-7b-simteg-2-10-linear-projector_nc_v2"
    else:
        ckpt_path = f"/localnvme/llaga/checkpoints/{args.dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc_v2"

    if not os.path.isdir(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    print(f"Checkpoint: {ckpt_path}")
    print(f"Output: {output_file}")

    # Load model
    torch.cuda.reset_peak_memory_stats()
    load_t0 = time.time()

    if args.model_type == "baseline":
        tokenizer, model = load_baseline_model(ckpt_path, args.model_base, args.quant_mode, args.cache_dir)
    else:
        tokenizer, model = load_moe_model(ckpt_path, args.model_base, args.quant_mode, args.cache_dir)

    load_time = time.time() - load_t0
    load_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"  Model loaded in {load_time:.1f}s. Memory after load: {load_memory_mb:.1f} MB")

    # Run evaluation
    results, stats = run_eval(model, tokenizer, args.dataset, args)

    # Write results
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n  Wrote {len(results)} results to {output_file}")
    print(f"  Peak GPU memory: {stats['peak_memory_mb']:.1f} MB")
    print(f"  Avg time/sample: {stats['avg_time_per_sample_s']:.3f}s")
    print(f"  Tokens/second: {stats['tokens_per_second']:.1f}")

    # Score
    print(f"\n{'=' * 70}")
    print(f"SCORING: {args.model_type} | {args.quant_mode} | {args.dataset}")
    print(f"{'=' * 70}")
    score_output = score_results(args.dataset, output_file)

    # Save stats
    stats_file = os.path.join(
        args.output_dir, f"{args.model_type}_{args.quant_mode}_nc_{args.dataset}_stats.json"
    )
    stats["model_type"] = args.model_type
    stats["quant_mode"] = args.quant_mode
    stats["dataset"] = args.dataset
    stats["checkpoint"] = ckpt_path
    stats["load_time_s"] = round(load_time, 1)
    stats["load_memory_mb"] = round(load_memory_mb, 1)
    stats["score_output"] = score_output.strip() if score_output else ""

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_file}")


if __name__ == "__main__":
    main()
