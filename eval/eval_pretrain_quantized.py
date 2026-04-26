import sys
sys.path.append("./")
sys.path.append("./utils")

import argparse
import json
import os

import shortuuid
import torch
from torch_geometric.utils import add_self_loops, degree, k_hop_subgraph, remove_self_loops
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from eval.eval_pretrain import (
    MP,
    SMALL_DATASETS,
    load_pretrain_embedding_graph,
    load_pretrain_embedding_hop,
    load_pretrain_embedding_hop_lp,
)
from eval.quantization_utils import (
    add_quantization_args,
    build_quantization_kwargs,
    get_graph_device,
    get_input_device,
    move_tensor_to_device,
    prepare_model_for_inference,
    restore_fp16_mm_projector,
    validate_quantization_args,
)
from model.builder import load_pretrained_model
from model.language_model.llaga_llama import LlagaLlamaForCausalLM
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
from utils.conversation import SeparatorStyle, conv_templates
from utils.utils import disable_torch_init, get_model_name_from_path, tokenizer_graph_token


def build_question(args, data, line):
    if args.task in ["nd", "nda"]:
        return f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."

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
                f"D&#233;cor, #508510, please tell me which class the center node belongs to?"
            )
        return line["conversations"][0]["value"]

    if args.task == "nctext":
        text = data.raw_texts[line["id"]][:2000]
        if args.dataset == "arxiv":
            return (
                f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges "
                f"represent co-citations, the node feature of center node is {text}. We need to classify the "
                f"center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in "
                f"Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), "
                f"cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), "
                f"cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet "
                f"Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), "
                f"cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary "
                f"Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision "
                f"and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and "
                f"Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming "
                f"Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), "
                f"cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), "
                f"cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), "
                f"cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data "
                f"Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game "
                f"Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please "
                f"tell me which class the center node belongs to? Direct tell me the class name."
            )
        if args.dataset == "products":
            return (
                f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in "
                f"Amazon, and edges between products indicate they are purchased together, the node feature of "
                f"center node is {text}. We need to classify the center node into 47 classes: Home & Kitchen, "
                f"Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & "
                f"Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & "
                f"Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, "
                f"Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, "
                f"Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, "
                f"Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, "
                f"Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, "
                f"Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, "
                f"Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, "
                f"please tell me which class the center node belongs to? Direct tell me the class name."
            )
        if args.dataset == "pubmed":
            return (
                f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers about "
                f"Diabetes and edges represent co-citations, the node feature of center node is {text}. We "
                f"need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes "
                f"Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs "
                f"to? Direct tell me the class name."
            )
        if args.dataset == "cora":
            return (
                f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges "
                f"represent co-citations, the node feature of center node is {text}. We need to classify the "
                f"center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, "
                f"Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which "
                f"class the center node belongs to? Direct tell me the class name."
            )
        raise ValueError(f"Unsupported dataset for nctext: {args.dataset}")

    if args.task == "lp":
        return (
            f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to "
            f"predict whether these two nodes connect with each other. Please tell me whether two center nodes "
            f"in the subgraphs should connect to each other."
        )

    raise ValueError(f"Unsupported task: {args.task}")


def eval_model(args):
    disable_torch_init()
    validate_quantization_args(args)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    if (args.load_4bit or args.load_8bit) and args.model_base is not None:
        cfg = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
        model = LlagaLlamaForCausalLM.from_pretrained(
            args.model_base,
            low_cpu_mem_usage=True,
            config=cfg,
            cache_dir=args.cache_dir,
            **build_quantization_kwargs(load_4bit=args.load_4bit, load_8bit=args.load_8bit),
        )
        model.resize_token_embeddings(len(tokenizer))
        context_len = getattr(model.config, "max_sequence_length", 2048)
        projector_path = os.path.join(model_path, "mm_projector.bin")
        if os.path.exists(projector_path):
            target_device = restore_fp16_mm_projector(model, cfg, projector_path)
            print(f"Restored mm_projector to FP16 on {target_device}")
    else:
        tokenizer, model, context_len = load_pretrained_model(
            model_path,
            args.model_base,
            model_name,
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            cache_dir=args.cache_dir,
        )
        model = prepare_model_for_inference(model, args.load_4bit, args.load_8bit)
    model.eval()
    input_device = get_input_device(model)
    graph_device = get_graph_device(model)

    if args.dataset == "arxiv":
        data_dir = "/localnvme/llaga/dataset/ogbn-arxiv"
    elif args.dataset == "products":
        data_dir = "/localnvme/llaga/dataset/ogbn-products"
    elif args.dataset == "pubmed":
        data_dir = "/localnvme/llaga/dataset/pubmed"
    elif args.dataset == "cora":
        data_dir = "/localnvme/llaga/dataset/cora"
    else:
        raise ValueError(f"{args.dataset} not exists")

    if args.task in ["nc", "nd", "nda", "nctext"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, "sampled_2_10_test.jsonl")
        else:
            prompt_file = os.path.join(
                data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl"
            )
        data_path = os.path.join(data_dir, "processed_data.pt")
    elif args.task in ["lp"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, "edge_sampled_2_10_only_test.jsonl")
        else:
            prompt_file = os.path.join(
                data_dir, f"edge_sampled_{args.use_hop}_{args.sample_neighbor_size}_only_test.jsonl"
            )
        data_path = os.path.join(data_dir, "processed_data.pt")
    else:
        raise ValueError

    data = torch.load(data_path)
    print(f"Load from {prompt_file}\n")
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
        print(f"{args.answers_file} already exists! it has {line_number} lines!!")
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
        if args.dataset in SMALL_DATASETS and args.task == "lp":
            pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        elif args.task == "lp":
            pretrained_emb, mask = load_pretrain_embedding_hop_lp(
                data_dir,
                args.pretrained_embedding_type,
                args.use_hop,
            )
            index = torch.full([num_nodes], fill_value=num_nodes + 1, dtype=torch.long)
            test_index = torch.arange(mask.sum())
            index[mask] = test_index
        else:
            mask = torch.full([num_nodes], fill_value=False, dtype=torch.bool)
            for question in questions:
                idx = question["id"]
                if "lp" in args.task:
                    mask[idx[0]] = True
                    mask[idx[1]] = True
                elif args.task in ["nc", "nd", "nctext"]:
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
            if args.dataset in SMALL_DATASETS and args.task == "lp":
                mp = MP()
                center_nodes = []
                for graph_idx in range(len(line["graph"])):
                    center_id = line["graph"][graph_idx][0]
                    line["graph"][graph_idx] = [center_id] * (args.use_hop + 1)
                    center_nodes.append(center_id)

                graph = torch.LongTensor(line["graph"])
                center_id = graph[:, 0]
                graph_embs = [move_tensor_to_device(pretrained_emb[center_id], graph_device)]
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                    center_nodes,
                    args.use_hop,
                    data.edge_index,
                    relabel_nodes=True,
                )
                local_edge_mask = (
                    ((edge_index[0] == mapping[0]) & (edge_index[1] == mapping[1]))
                    | ((edge_index[0] == mapping[1]) & (edge_index[1] == mapping[0]))
                )
                edge_index = edge_index[:, ~local_edge_mask]
                local_x = move_tensor_to_device(pretrained_emb[subset], graph_device)
                local_num_nodes = subset.shape[0]
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index)
                edge_index = move_tensor_to_device(edge_index, graph_device)
                row, col = edge_index
                deg = degree(col, local_num_nodes, dtype=pretrained_emb.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                for _ in range(args.use_hop):
                    local_x = mp.propagate(edge_index, x=local_x, norm=norm)
                    graph_embs.append(local_x[mapping])
                graph_emb = torch.stack(graph_embs, dim=1)
            else:
                for graph_idx in range(len(line["graph"])):
                    center_id = line["graph"][graph_idx][0]
                    line["graph"][graph_idx] = [center_id] * (args.use_hop + 1)
                graph = torch.LongTensor(line["graph"])
                center_id = graph[:, 0]
                graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)
        else:
            raise ValueError

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=move_tensor_to_device(graph_emb, graph_device, dtype=torch.float16),
                    graph=move_tensor_to_device(graph, graph_device),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as exc:
            print(f"!!!!!!Error!!!!! {exc}")
            outputs = ""

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
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
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end", default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    add_quantization_args(parser)
    args = parser.parse_args()

    eval_model(args)
