import sys
sys.path.append("./")
sys.path.append("./utils")

import argparse
import json
import os

import shortuuid
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from eval.eval_moe import DATASET_DIRS, load_pretrain_embedding_graph, load_pretrain_embedding_hop
from eval.quantization_utils import (
    add_quantization_args,
    build_moe_projector_from_config,
    build_quantization_kwargs,
    get_graph_device,
    get_input_device,
    load_clean_projector_weights,
    move_tensor_to_device,
    prepare_model_for_inference,
    validate_quantization_args,
)
from model.language_model.moe_llaga_llama import MoELlagaLlamaForCausalLM
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
from utils.conversation import SeparatorStyle, conv_templates
from utils.utils import disable_torch_init, tokenizer_graph_token


def load_moe_model(
    model_path,
    model_base,
    cache_dir="../../checkpoint",
    load_4bit=False,
    load_8bit=False,
):
    kwargs = build_quantization_kwargs(load_4bit=load_4bit, load_8bit=load_8bit)

    if model_base is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg = AutoConfig.from_pretrained(model_path)
        model = MoELlagaLlamaForCausalLM.from_pretrained(
            model_base,
            low_cpu_mem_usage=True,
            config=cfg,
            cache_dir=cache_dir,
            **kwargs,
        )

        mm_path = os.path.join(model_path, "mm_projector.bin")
        if os.path.exists(mm_path):
            mm_weights = load_clean_projector_weights(mm_path)
            model.load_state_dict(mm_weights, strict=False)

        moe_path = os.path.join(model_path, "moe_projector.bin")
        if model.moe_projector is None:
            model.moe_projector = build_moe_projector_from_config(cfg)
        if os.path.exists(moe_path):
            moe_weights = load_clean_projector_weights(moe_path, prefix_to_strip="moe_projector.")
            model.moe_projector.load_state_dict(moe_weights)
            print(f"Loaded MoE projector from {moe_path}")
        else:
            print(f"WARNING: moe_projector.bin not found at {moe_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = MoELlagaLlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        if model.moe_projector is None:
            cfg = model.config
            model.moe_projector = build_moe_projector_from_config(cfg)
            moe_path = os.path.join(model_path, "moe_projector.bin")
            if os.path.exists(moe_path):
                moe_weights = load_clean_projector_weights(moe_path, prefix_to_strip="moe_projector.")
                model.moe_projector.load_state_dict(moe_weights)
                print(f"Loaded MoE projector from {moe_path}")
            else:
                print(f"WARNING: moe_projector.bin not found at {moe_path}")

    model.resize_token_embeddings(len(tokenizer))
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, context_len


def build_question(args, line):
    if args.task == "nc":
        if args.dataset == "products":
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
    if args.task == "nd":
        return f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
    if args.task == "lp":
        return (
            f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to "
            f"predict whether these two nodes connect with each other. Please tell me whether two center nodes "
            f"in the subgraphs should connect to each other."
        )
    raise ValueError(f"Unknown task: {args.task}")


def eval_model(args):
    disable_torch_init()
    validate_quantization_args(args)

    model_path = os.path.expanduser(args.model_path)
    print(f"Loading MoE model from {model_path}, base: {args.model_base}")
    tokenizer, model, context_len = load_moe_model(
        model_path,
        args.model_base,
        cache_dir=args.cache_dir,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )
    model = prepare_model_for_inference(
        model,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        move_moe_projector=True,
    )
    model.eval()
    input_device = get_input_device(model)
    graph_device = get_graph_device(model)

    data_dir = DATASET_DIRS.get(args.dataset)
    if data_dir is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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

    if args.task in ["nc", "nd"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, "sampled_2_10_test.jsonl")
        else:
            prompt_file = os.path.join(
                data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl"
            )
    elif args.task == "lp":
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, "edge_sampled_2_10_only_test.jsonl")
        else:
            prompt_file = os.path.join(
                data_dir, f"edge_sampled_{args.use_hop}_{args.sample_neighbor_size}_only_test.jsonl"
            )
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
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        line_number = len(open(answers_file, "r").readlines())
        print(f"{args.answers_file} already exists with {line_number} lines")
        if line_number >= len(lines):
            return
        lines = lines[line_number:]
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]

    index = None
    if args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(
            f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
        )
    elif args.template == "HO":
        num_nodes = data.num_nodes
        mask = torch.full([num_nodes], fill_value=False, dtype=torch.bool)
        for question in questions:
            idx = question["id"]
            if args.task == "lp":
                mask[idx[0]] = True
                mask[idx[1]] = True
            elif args.task in ["nc", "nd"]:
                mask[idx] = True
        pretrained_emb = load_pretrain_embedding_hop(
            data_dir,
            args.pretrained_embedding_type,
            args.use_hop,
            mask,
        )
        index = torch.full([num_nodes], fill_value=num_nodes + 1, dtype=torch.long)
        test_index = torch.arange(mask.sum())
        index[mask] = test_index
        structure_emb = None
    else:
        raise ValueError

    for line in tqdm(questions):
        idx = line["id"]
        qs = build_question(args, line)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_graph_token(
            prompt,
            tokenizer,
            GRAPH_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)
        input_ids = move_tensor_to_device(input_ids, input_device)

        if not isinstance(line["graph"][0], list):
            line["graph"] = [line["graph"]]

        if args.template == "ND":
            graph = torch.LongTensor(line["graph"])
            mask = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = pretrained_emb[graph[mask]]
            sample_count, graph_len, hidden_dim = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
            graph_emb = torch.zeros((sample_count, graph_len, hidden_dim))
            graph_emb[mask] = masked_graph_emb
            if structure_emb is not None:
                graph_emb = torch.cat(
                    [graph_emb, structure_emb.unsqueeze(0).expand(sample_count, -1, -1)],
                    dim=-1,
                )
        elif args.template == "HO":
            for graph_idx in range(len(line["graph"])):
                center_id = line["graph"][graph_idx][0]
                line["graph"][graph_idx] = [center_id] * (args.use_hop + 1)
            graph = torch.LongTensor(line["graph"])
            center_id = graph[:, 0]
            graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)
        else:
            raise ValueError

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
                    graph_emb=move_tensor_to_device(graph_emb, graph_device, dtype=torch.float16),
                    graph=move_tensor_to_device(graph, graph_device),
                    routing_features=move_tensor_to_device(
                        routing_feat, graph_device, dtype=torch.float16
                    ),
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
                print(f"[Warning] {n_diff} output_ids differ from input_ids")
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as exc:
            print(f"Error: {exc}")
            outputs = ""

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": qs,
                    "graph": line["graph"],
                    "text": outputs,
                    "gt": line["conversations"][1]["value"],
                    "answer_id": ans_id,
                }
            )
            + "\n"
        )
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
    add_quantization_args(parser)
    args = parser.parse_args()

    eval_model(args)
