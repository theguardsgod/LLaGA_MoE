"""
Quantization experiment runner for LLaGA.

Evaluates MoE and baseline models under FP16 / 4-bit NF4 / 8-bit INT8 quantization.
Measures accuracy, peak GPU memory, and inference speed.

Both MoE and baseline models were trained with ND template (node-centered subgraph
with laplacian positional encoding appended), so eval uses ND template too.

Usage:
    # MoE model on cora, 4-bit quantization, GPU 0
    python eval/run_quant_experiments.py \
        --model_type moe \
        --datasets cora \
        --quant_mode 4bit \
        --gpu 0

    # Baseline model on cora+pubmed, all quant modes, GPU 2
    python eval/run_quant_experiments.py \
        --model_type baseline \
        --datasets cora pubmed \
        --quant_mode fp16 4bit 8bit \
        --gpu 2
"""

import sys
sys.path.append("./")
sys.path.append("./utils")

import argparse
import gc
import json
import os
import subprocess
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from eval.quantization_utils import (
    build_moe_projector_from_config,
    get_graph_device,
    get_input_device,
    load_clean_projector_weights,
    move_moe_projector_to_runtime_device,
    move_tensor_to_device,
    restore_fp16_mm_projector,
)
from model.language_model.moe_llaga_llama import MoELlagaLlamaForCausalLM, MoELlagaConfig
from model.language_model.llaga_llama import LlagaLlamaForCausalLM, LlagaConfig
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
from utils.conversation import SeparatorStyle, conv_templates
from utils.utils import disable_torch_init, tokenizer_graph_token

DATASET_DIRS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}

USE_HOP = 2
SAMPLE_NEIGHBOR_SIZE = 10

MOE_CHECKPOINT_PATTERN = "/localnvme/llaga/checkpoints/{dataset}/moe-llaga-vicuna-7b-simteg-2-10-linear-E4-K2_nc_v2"
BASELINE_CHECKPOINT_PATTERN = "./checkpoints/{dataset}/llaga-vicuna-7b-simteg-2-10-linear-projector_nc_v2"
MODEL_BASE = "lmsys/vicuna-7b-v1.5-16k"
CACHE_DIR = "../../checkpoint"


def load_pretrain_embedding_graph(data_dir, emb_type="simteg"):
    """Load graph-level (non-hop) embeddings for ND template."""
    if emb_type == "simteg":
        sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        return torch.cat([sbert, roberta, e5], dim=-1)
    return torch.load(os.path.join(data_dir, f"{emb_type}_x.pt"))


def build_quant_kwargs(quant_mode):
    """Build kwargs for from_pretrained based on quantization mode."""
    if quant_mode == "4bit":
        return {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        }
    elif quant_mode == "8bit":
        return {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "load_in_8bit": True,
        }
    else:  # fp16
        return {
            "torch_dtype": torch.float16,
        }


def load_moe_model(checkpoint_path, quant_mode, gpu_id):
    """Load MoE model with specified quantization."""
    kwargs = build_quant_kwargs(quant_mode)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=False)
    cfg = MoELlagaConfig.from_pretrained(checkpoint_path)

    model = MoELlagaLlamaForCausalLM.from_pretrained(
        MODEL_BASE,
        low_cpu_mem_usage=True,
        config=cfg,
        cache_dir=CACHE_DIR,
        **kwargs,
    )

    # Build and load MoE projector
    model.moe_projector = build_moe_projector_from_config(cfg)
    model.resize_token_embeddings(len(tokenizer))

    # Load MoE projector weights
    moe_path = os.path.join(checkpoint_path, "moe_projector.bin")
    clean = load_clean_projector_weights(moe_path, prefix_to_strip="moe_projector.")
    model.moe_projector.load_state_dict(clean)

    # Move model to correct device
    if quant_mode in ("4bit", "8bit"):
        move_moe_projector_to_runtime_device(model)
    else:
        model = model.to(torch.float16).cuda(gpu_id)

    model.eval()
    return tokenizer, model


def load_baseline_model(checkpoint_path, quant_mode, gpu_id):
    """Load baseline (single projector) model with specified quantization."""
    kwargs = build_quant_kwargs(quant_mode)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=False)
    cfg = LlagaConfig.from_pretrained(checkpoint_path)

    model = LlagaLlamaForCausalLM.from_pretrained(
        MODEL_BASE,
        low_cpu_mem_usage=True,
        config=cfg,
        cache_dir=CACHE_DIR,
        **kwargs,
    )

    # Load mm_projector weights
    proj_path = os.path.join(checkpoint_path, "mm_projector.bin")
    if os.path.exists(proj_path):
        if quant_mode in ("4bit", "8bit"):
            target_device = restore_fp16_mm_projector(model, cfg, proj_path)
            print(f"  Restored mm_projector (FP16) from {proj_path} -> {target_device}")
        else:
            proj_weights = torch.load(proj_path, map_location="cpu")
            proj_weights = {k: v.to(torch.float16) for k, v in proj_weights.items()}
            model.load_state_dict(proj_weights, strict=False)
            print(f"  Loaded mm_projector from {proj_path}")

    model.resize_token_embeddings(len(tokenizer))

    if quant_mode not in ("4bit", "8bit"):
        model = model.to(torch.float16).cuda(gpu_id)

    model.eval()
    return tokenizer, model


def build_nc_question(dataset, line):
    """Build node classification question."""
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


def build_graph_emb_nd(line, pretrained_emb, structure_emb):
    """Build graph embedding for ND template.

    For ND template:
    - graph is a list of node indices, padded with -500
    - graph_emb = pretrained_emb[valid_nodes] + laplacian structure_emb appended
    """
    if not isinstance(line["graph"][0], list):
        line["graph"] = [line["graph"]]

    graph = torch.LongTensor(line["graph"])
    mask = graph != DEFAULT_GRAPH_PAD_ID
    masked_graph_emb = pretrained_emb[graph[mask]]
    sample_count = graph.shape[0]
    graph_len = graph.shape[1]
    hidden_dim = masked_graph_emb.shape[1]
    graph_emb = torch.zeros((sample_count, graph_len, hidden_dim))
    graph_emb[mask] = masked_graph_emb

    if structure_emb is not None:
        graph_emb = torch.cat(
            [graph_emb, structure_emb.unsqueeze(0).expand(sample_count, -1, -1)],
            dim=-1,
        )

    return graph, graph_emb


def run_evaluation(model, tokenizer, questions, pretrained_emb, structure_emb,
                   dataset, node_to_community=None, community_features=None,
                   is_moe=False):
    """Run evaluation and return list of result dicts plus timing info."""
    input_device = get_input_device(model)
    graph_device = get_graph_device(model)

    results = []
    total_gen_time = 0.0
    total_tokens = 0

    for line in tqdm(questions, desc=f"  Evaluating {dataset}"):
        idx = line["id"]
        qs = build_nc_question(dataset, line)

        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(
            prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)
        input_ids = move_tensor_to_device(input_ids, input_device)

        graph, graph_emb = build_graph_emb_nd(line, pretrained_emb, structure_emb)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        gen_kwargs = dict(
            input_ids=input_ids,
            graph_emb=move_tensor_to_device(graph_emb, graph_device, dtype=torch.float16),
            graph=move_tensor_to_device(graph, graph_device),
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
        )

        if is_moe:
            center_ids = graph[:, 0]
            if node_to_community is not None and community_features is not None:
                comm_ids = node_to_community[center_ids]
                routing_feat = community_features[comm_ids]
            else:
                routing_feat = torch.zeros(center_ids.shape[0], 2432)
            gen_kwargs["routing_features"] = move_tensor_to_device(
                routing_feat, graph_device, dtype=torch.float16
            )

        try:
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.inference_mode():
                output_ids = model.generate(**gen_kwargs)
            torch.cuda.synchronize()
            t1 = time.time()

            total_gen_time += (t1 - t0)
            input_token_len = input_ids.shape[1]
            new_tokens = output_ids.shape[1] - input_token_len
            total_tokens += new_tokens

            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()
        except Exception as exc:
            print(f"  Error on sample {idx}: {exc}")
            outputs = ""

        results.append({
            "question_id": idx,
            "text": outputs,
            "gt": line["conversations"][1]["value"],
        })

    speed_info = {
        "total_gen_time": total_gen_time,
        "total_tokens": total_tokens,
        "avg_time_per_sample": total_gen_time / len(questions) if questions else 0,
        "tokens_per_second": total_tokens / total_gen_time if total_gen_time > 0 else 0,
    }
    return results, speed_info


def load_dataset(dataset, questions_limit=-1):
    """Load dataset with ND template embeddings."""
    data_dir = DATASET_DIRS[dataset]
    prompt_file = os.path.join(data_dir, f"sampled_{USE_HOP}_{SAMPLE_NEIGHBOR_SIZE}_test.jsonl")

    lines = open(prompt_file, "r").readlines()
    if questions_limit > 0:
        lines = lines[:questions_limit]
    questions = [json.loads(q) for q in lines]

    # ND template: load graph-level embeddings (not hop-level)
    pretrained_emb = load_pretrain_embedding_graph(data_dir, "simteg")

    # Load laplacian structure embeddings
    structure_emb_path = f"/localnvme/llaga/dataset/laplacian_{USE_HOP}_{SAMPLE_NEIGHBOR_SIZE}.pt"
    if os.path.exists(structure_emb_path):
        structure_emb = torch.load(structure_emb_path)
        print(f"  Loaded structure embeddings from {structure_emb_path}: {structure_emb.shape}")
    else:
        print(f"  WARNING: {structure_emb_path} not found, no structure embeddings")
        structure_emb = None

    # Load community data for MoE
    comm_path = os.path.join(data_dir, "node_to_community.pt")
    feat_path = os.path.join(data_dir, "community_features.pt")
    node_to_community = None
    community_features = None
    if os.path.exists(comm_path) and os.path.exists(feat_path):
        node_to_community = torch.load(comm_path)
        community_features = torch.load(feat_path)

    return questions, pretrained_emb, structure_emb, node_to_community, community_features


def get_peak_memory_mb(gpu_id=0):
    """Get peak GPU memory usage in MB."""
    return torch.cuda.max_memory_allocated(gpu_id) / (1024 * 1024)


def write_results(results, output_file):
    """Write eval results to JSONL."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="LLaGA quantization experiments")
    parser.add_argument("--model_type", type=str, choices=["moe", "baseline"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed"])
    parser.add_argument("--quant_mode", nargs="+", default=["fp16", "4bit", "8bit"],
                        choices=["fp16", "4bit", "8bit"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of samples per dataset (-1 for all)")
    parser.add_argument("--output_dir", type=str,
                        default="/localnvme/llaga/eval_output/quant_experiments")
    args = parser.parse_args()

    disable_torch_init()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Initialize CUDA context early so memory tracking calls work
    torch.cuda.init()

    # Results summary
    summary = []

    for quant_mode in args.quant_mode:
        print(f"\n{'=' * 70}")
        print(f"QUANTIZATION MODE: {quant_mode.upper()}")
        print(f"MODEL TYPE: {args.model_type.upper()}")
        print(f"{'=' * 70}")

        # Reset memory tracking
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.empty_cache()
        gc.collect()

        # Load model once per quantization mode
        first_dataset = args.datasets[0]
        if args.model_type == "moe":
            checkpoint_path = MOE_CHECKPOINT_PATTERN.format(dataset=first_dataset)
        else:
            checkpoint_path = BASELINE_CHECKPOINT_PATTERN.format(dataset=first_dataset)

        print(f"\nLoading model from {checkpoint_path} with {quant_mode} quantization...")

        if args.model_type == "moe":
            tokenizer, model = load_moe_model(checkpoint_path, quant_mode, 0)
        else:
            tokenizer, model = load_baseline_model(checkpoint_path, quant_mode, 0)

        mem_after_load = get_peak_memory_mb(0)
        print(f"  Model loaded. Peak memory after load: {mem_after_load:.1f} MB")

        for dataset in args.datasets:
            print(f"\n--- Dataset: {dataset} ---")

            # Hot-swap projector if needed
            if args.model_type == "moe" and dataset != first_dataset:
                ckpt = MOE_CHECKPOINT_PATTERN.format(dataset=dataset)
                if os.path.isdir(ckpt):
                    moe_path = os.path.join(ckpt, "moe_projector.bin")
                    clean = load_clean_projector_weights(moe_path, prefix_to_strip="moe_projector.")
                    model.moe_projector.load_state_dict(clean)
                    move_moe_projector_to_runtime_device(model)
                    print(f"  Hot-swapped MoE projector from {ckpt}")
                else:
                    print(f"  WARNING: checkpoint not found for {dataset}: {ckpt}, skipping")
                    continue
            elif args.model_type == "baseline" and dataset != first_dataset:
                ckpt = BASELINE_CHECKPOINT_PATTERN.format(dataset=dataset)
                if os.path.isdir(ckpt):
                    proj_path = os.path.join(ckpt, "mm_projector.bin")
                    if os.path.exists(proj_path):
                        proj_weights = torch.load(proj_path, map_location="cpu")
                        proj_weights = {k: v.to(torch.float16) for k, v in proj_weights.items()}
                        model.load_state_dict(proj_weights, strict=False)
                        print(f"  Swapped projector from {ckpt}")
                else:
                    print(f"  WARNING: checkpoint not found for {dataset}: {ckpt}, skipping")
                    continue

            # Load dataset
            questions, pretrained_emb, structure_emb, node_to_community, community_features = \
                load_dataset(dataset, args.limit)
            print(f"  Loaded {len(questions)} questions")

            # Reset peak memory before inference
            torch.cuda.reset_peak_memory_stats(0)

            # Run evaluation
            results, speed_info = run_evaluation(
                model, tokenizer, questions, pretrained_emb, structure_emb,
                dataset,
                node_to_community=node_to_community,
                community_features=community_features,
                is_moe=(args.model_type == "moe"),
            )

            peak_mem = get_peak_memory_mb(0)

            # Save results
            tag = f"{args.model_type}_{quant_mode}_nc_{dataset}"
            output_file = os.path.join(args.output_dir, f"{tag}.jsonl")
            write_results(results, output_file)
            print(f"  Results saved to {output_file}")

            # Score results
            print(f"  Scoring...")
            score_result = subprocess.run(
                ["python", "eval/eval_res.py",
                 "--dataset", dataset, "--task", "nc", "--res_path", output_file],
                capture_output=True, text=True,
            )
            score_output = score_result.stdout.strip()
            print(f"  {score_output}")

            # Parse accuracy from score output
            accuracy = "N/A"
            for score_line in score_output.split("\n"):
                if "overall_acc" in score_line:
                    accuracy = score_line.split(":")[-1].strip()
                    break
                elif "acc:" in score_line:
                    accuracy = score_line.split(":")[-1].strip()
                    break

            summary.append({
                "model_type": args.model_type,
                "quant_mode": quant_mode,
                "dataset": dataset,
                "num_samples": len(questions),
                "accuracy": accuracy,
                "peak_memory_mb": f"{peak_mem:.1f}",
                "avg_time_per_sample_s": f"{speed_info['avg_time_per_sample']:.3f}",
                "tokens_per_second": f"{speed_info['tokens_per_second']:.1f}",
            })

            # Clean up dataset tensors
            del questions, pretrained_emb, structure_emb, node_to_community, community_features
            gc.collect()
            torch.cuda.empty_cache()

        # Clean up model
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Print final summary
    print(f"\n{'=' * 90}")
    print("QUANTIZATION EXPERIMENT RESULTS SUMMARY")
    print(f"{'=' * 90}")
    header = (
        f"{'Model':<10} {'Quant':<8} {'Dataset':<10} {'Samples':<8} "
        f"{'Accuracy':<10} {'PeakMem(MB)':<13} {'Time/Sample(s)':<16} {'Tok/s':<8}"
    )
    print(header)
    print("-" * 90)
    for row in summary:
        line = (
            f"{row['model_type']:<10} "
            f"{row['quant_mode']:<8} "
            f"{row['dataset']:<10} "
            f"{row['num_samples']:<8} "
            f"{row['accuracy']:<10} "
            f"{row['peak_memory_mb']:<13} "
            f"{row['avg_time_per_sample_s']:<16} "
            f"{row['tokens_per_second']:<8}"
        )
        print(line)

    # Save summary JSON
    summary_file = os.path.join(args.output_dir, f"summary_{args.model_type}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
