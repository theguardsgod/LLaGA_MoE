import sys
sys.path.append("./")
sys.path.append("./utils")

import argparse
import json
import os
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from eval.awq_utils import (
    build_multimodal_inputs_embeds,
    decode_awq_outputs,
    get_embedding_device,
    load_awq_language_model,
    load_fp16_mm_projector,
)
from eval.eval_pretrain_awq import (
    build_question,
    load_llaga_config,
    load_pretrain_embedding_graph,
)
from eval.gptq_utils import load_gptq_language_model
from eval.quantization_utils import (
    get_graph_device,
    get_input_device,
    move_tensor_to_device,
    restore_fp16_mm_projector,
)
from model.language_model.llaga_llama import LlagaConfig, LlagaLlamaForCausalLM
from utils.constants import DEFAULT_GRAPH_PAD_ID, GRAPH_TOKEN_INDEX
from utils.conversation import SeparatorStyle, conv_templates
from utils.utils import disable_torch_init, tokenizer_graph_token


DATASET_DIRS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}


def build_bnb_quantization_config(mode):
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def build_checkpoint_path(dataset):
    return f"./checkpoints/{dataset}/llaga-vicuna-7b-simteg-2-10-linear-projector_nc_v2"


def load_baseline_or_bnb_model(args, mode):
    model_path = os.path.expanduser(args.model_path or build_checkpoint_path(args.dataset))
    cfg = LlagaConfig.from_pretrained(model_path)
    load_cfg = LlagaConfig.from_pretrained(model_path)
    if hasattr(load_cfg, "mm_hidden_size"):
        delattr(load_cfg, "mm_hidden_size")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False, cache_dir=args.cache_dir)

    kwargs = {
        "low_cpu_mem_usage": True,
        "config": load_cfg,
        "cache_dir": args.cache_dir,
        "torch_dtype": torch.float16,
    }
    if mode in {"4bit", "8bit"}:
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = build_bnb_quantization_config(mode)

    model = LlagaLlamaForCausalLM.from_pretrained(args.model_base, **kwargs)
    model.resize_token_embeddings(len(tokenizer))

    projector_path = os.path.join(model_path, "mm_projector.bin")
    target_device = restore_fp16_mm_projector(model, cfg, projector_path)

    if mode == "fp16":
        model = model.to(torch.float16).cuda()
        target_device = get_input_device(model)

    model.eval()
    return tokenizer, model, cfg, model_path, target_device


def load_awq_model(args):
    model_path = os.path.expanduser(args.model_path or build_checkpoint_path(args.dataset))
    cfg = load_llaga_config(model_path)
    tokenizer, model, backend = load_awq_language_model(
        os.path.expanduser(args.awq_model_path),
        awq_tokenizer_path=args.awq_tokenizer_path,
        awq_backend=args.awq_backend,
        cache_dir=args.cache_dir,
        fuse_awq_layers=args.fuse_awq_layers,
        awq_max_seq_len=args.awq_max_seq_len,
        awq_use_exllama=args.awq_use_exllama,
        awq_use_exllamav2=args.awq_use_exllamav2,
    )
    projector_device = get_embedding_device(model)
    mm_projector = load_fp16_mm_projector(model_path, cfg, projector_device)
    model.eval()
    return tokenizer, model, cfg, model_path, backend, mm_projector


def load_gptq_model(args):
    model_path = os.path.expanduser(args.model_path or build_checkpoint_path(args.dataset))
    cfg = load_llaga_config(model_path)
    tokenizer, model = load_gptq_language_model(
        os.path.expanduser(args.gptq_model_path),
        gptq_tokenizer_path=args.gptq_tokenizer_path,
        gptq_model_basename=args.gptq_model_basename,
        cache_dir=args.cache_dir,
        gptq_use_triton=args.gptq_use_triton,
        gptq_disable_exllama=args.gptq_disable_exllama,
        gptq_disable_exllamav2=args.gptq_disable_exllamav2,
        gptq_use_marlin=args.gptq_use_marlin,
    )
    projector_device = get_embedding_device(model)
    mm_projector = load_fp16_mm_projector(model_path, cfg, projector_device)
    model.eval()
    return tokenizer, model, cfg, model_path, mm_projector


def load_questions_and_embeddings(args):
    data_dir = DATASET_DIRS[args.dataset]
    prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
    data_path = os.path.join(data_dir, "processed_data.pt")
    data = torch.load(data_path, weights_only=False)
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        end = args.end if args.end > 0 else len(lines)
        lines = lines[args.start:end]
    elif args.end > 0:
        lines = lines[:args.end]

    questions = [json.loads(line) for line in lines]
    pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
    structure_emb = torch.load(
        f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt",
        weights_only=False,
    )
    return data, questions, pretrained_emb, structure_emb


def build_graph_tensors(line, pretrained_emb, structure_emb):
    if not isinstance(line["graph"][0], list):
        line["graph"] = [line["graph"]]

    graph = torch.LongTensor(line["graph"])
    mask_g = graph != DEFAULT_GRAPH_PAD_ID
    masked_graph_emb = pretrained_emb[graph[mask_g]]
    sample_count, graph_len, hidden_dim = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
    graph_emb = torch.zeros((sample_count, graph_len, hidden_dim))
    graph_emb[mask_g] = masked_graph_emb
    graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(sample_count, -1, -1)], dim=-1)
    return graph, graph_emb


def decode_standard_outputs(tokenizer, output_ids, prompt_length, stop_str):
    outputs = tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)].strip()
    return outputs


def benchmark_mode(args):
    disable_torch_init()
    data, questions, pretrained_emb, structure_emb = load_questions_and_embeddings(args)

    torch.cuda.reset_peak_memory_stats()
    load_t0 = time.time()

    if args.mode == "awq":
        tokenizer, model, cfg, model_path, backend, mm_projector = load_awq_model(args)
        input_device = None
        graph_device = None
        load_extra = {"awq_backend": backend}
    elif args.mode == "gptq":
        tokenizer, model, cfg, model_path, mm_projector = load_gptq_model(args)
        input_device = None
        graph_device = None
        load_extra = {
            "gptq_use_triton": args.gptq_use_triton,
            "gptq_disable_exllama": args.gptq_disable_exllama,
            "gptq_disable_exllamav2": args.gptq_disable_exllamav2,
            "gptq_use_marlin": args.gptq_use_marlin,
        }
    else:
        tokenizer, model, cfg, model_path, projector_device = load_baseline_or_bnb_model(args, args.mode)
        input_device = get_input_device(model)
        graph_device = get_graph_device(model)
        mm_projector = None
        load_extra = {"projector_device": str(projector_device)}

    load_time = time.time() - load_t0
    load_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()

    total_tokens = 0
    total_time = 0.0
    preprocess_time = 0.0
    generate_time = 0.0

    for line in tqdm(questions, desc=args.mode):
        try:
            torch.cuda.synchronize()
            preprocess_t0 = time.time()

            cur_prompt = build_question(args, data, line)
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], cur_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_graph_token(
                prompt,
                tokenizer,
                GRAPH_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0)
            graph, graph_emb = build_graph_tensors(line, pretrained_emb, structure_emb)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                if args.mode in {"awq", "gptq"}:
                    inputs_embeds, attention_mask = build_multimodal_inputs_embeds(
                        model,
                        mm_projector,
                        cfg,
                        input_ids,
                        graph,
                        graph_emb,
                    )
                    prompt_length = attention_mask.shape[1]
                else:
                    runtime_input_ids = move_tensor_to_device(input_ids, input_device)
                    runtime_graph_emb = move_tensor_to_device(graph_emb, graph_device, dtype=torch.float16)
                    runtime_graph = move_tensor_to_device(graph, graph_device)
                    prompt_length = input_ids.shape[1]

            torch.cuda.synchronize()
            preprocess_t1 = time.time()

            with torch.inference_mode():
                if args.mode in {"awq", "gptq"}:
                    output_ids = model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )
                else:
                    output_ids = model.generate(
                        runtime_input_ids,
                        graph_emb=runtime_graph_emb,
                        graph=runtime_graph,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

            torch.cuda.synchronize()
            generate_t1 = time.time()

            preprocess_time += preprocess_t1 - preprocess_t0
            generate_time += generate_t1 - preprocess_t1
            total_time += generate_t1 - preprocess_t0
            total_tokens += max(0, output_ids.shape[1] - prompt_length)

            if args.mode in {"awq", "gptq"}:
                decode_awq_outputs(tokenizer, output_ids, prompt_length=prompt_length)
            else:
                decode_standard_outputs(tokenizer, output_ids, prompt_length, stop_str)
        except Exception as exc:
            raise RuntimeError(f"Benchmark failed on question {line['id']}: {exc}") from exc

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    avg_time = total_time / len(questions) if questions else 0.0
    avg_preprocess_time = preprocess_time / len(questions) if questions else 0.0
    avg_generate_time = generate_time / len(questions) if questions else 0.0
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    stats = {
        "mode": args.mode,
        "dataset": args.dataset,
        "num_samples": len(questions),
        "load_time_s": round(load_time, 2),
        "load_memory_mb": round(load_memory_mb, 1),
        "torch_peak_mb": round(peak_memory_mb, 1),
        "preprocess_time_s": round(preprocess_time, 2),
        "generate_time_s": round(generate_time, 2),
        "total_time_s": round(total_time, 2),
        "avg_preprocess_time_s": round(avg_preprocess_time, 3),
        "avg_generate_time_s": round(avg_generate_time, 3),
        "avg_time_per_sample_s": round(avg_time, 3),
        "total_tokens_generated": int(total_tokens),
        "tokens_per_second": round(tokens_per_sec, 1),
        "model_path": model_path,
    }
    stats.update(load_extra)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline runtime across fp16, bnb, AWQ, and GPTQ.")
    parser.add_argument("--mode", type=str, required=True, choices=["fp16", "4bit", "8bit", "awq", "gptq"])
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "pubmed", "arxiv", "products"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default="lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument(
        "--awq_model_path",
        type=str,
        default="/localnvme/llaga/checkpoints/awq/vicuna-7b-v1.5-16k-cora-awq-w4g128",
    )
    parser.add_argument("--awq_tokenizer_path", type=str, default=None)
    parser.add_argument("--awq_backend", type=str, default="autoawq", choices=["auto", "transformers", "autoawq"])
    parser.add_argument("--fuse_awq_layers", action="store_true", default=False)
    parser.add_argument("--awq_max_seq_len", type=int, default=2048)
    parser.add_argument("--awq_use_exllama", action="store_true", default=False)
    parser.add_argument("--awq_use_exllamav2", action="store_true", default=False)
    parser.add_argument("--gptq_model_path", type=str, default=None)
    parser.add_argument("--gptq_tokenizer_path", type=str, default=None)
    parser.add_argument("--gptq_model_basename", type=str, default=None)
    parser.add_argument("--gptq_use_triton", action="store_true", default=False)
    parser.add_argument("--gptq_enable_exllama", dest="gptq_disable_exllama", action="store_false")
    parser.add_argument("--gptq_enable_exllamav2", dest="gptq_disable_exllamav2", action="store_false")
    parser.add_argument("--gptq_use_marlin", action="store_true", default=False)
    parser.set_defaults(gptq_disable_exllama=True, gptq_disable_exllamav2=True)
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=64)
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    if args.template != "ND":
        raise ValueError("This benchmark script currently supports only template=ND.")
    if args.task != "nc":
        raise ValueError("This benchmark script currently supports only task=nc.")

    stats = benchmark_mode(args)
    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w") as handle:
            json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
