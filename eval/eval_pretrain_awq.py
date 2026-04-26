import sys
sys.path.append("./")
sys.path.append("./utils")

import argparse
import json
import math
import os
from types import SimpleNamespace

import shortuuid
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, k_hop_subgraph, remove_self_loops
from tqdm import tqdm

from eval.awq_utils import (
    add_awq_args,
    build_multimodal_inputs_embeds,
    decode_awq_outputs,
    get_embedding_device,
    load_awq_language_model,
    load_fp16_mm_projector,
)
from eval.quantization_utils import move_tensor_to_device
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN, GRAPH_TOKEN_INDEX
from utils.conversation import SeparatorStyle, conv_templates
from utils.utils import disable_torch_init, tokenizer_graph_token


SMALL_DATASETS = ["pubmed", "cora"]


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        return torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    return torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))


def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop, mask):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = [torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))[mask]] + [
            torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)
        ]
        simteg_roberta = [torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))[mask]] + [
            torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)
        ]
        simteg_e5 = [torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))[mask]] + [
            torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)
        ]
        return [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]

    return [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]] + [
        torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt"))[mask]
        for i in range(1, hop + 1)
    ]


def load_pretrain_embedding_hop_lp(data_dir, pretrained_embedding_type, hop):
    mask = torch.load(os.path.join(data_dir, "no_test_link_mask.pt"))
    if pretrained_embedding_type == "simteg":
        simteg_sbert = [torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))[mask]] + [
            torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)
        ]
        simteg_roberta = [torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))[mask]] + [
            torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x_notestlink.pt"))
            for i in range(1, hop + 1)
        ]
        simteg_e5 = [torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))[mask]] + [
            torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)
        ]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]] + [
            torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x_notestlink.pt"))
            for i in range(1, hop + 1)
        ]
    return pretrained_embs, mask


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


def load_llaga_config(model_path):
    with open(os.path.join(model_path, "config.json"), "r") as handle:
        return SimpleNamespace(**json.load(handle))


def eval_model(args):
    disable_torch_init()

    llaga_model_path = os.path.expanduser(args.model_path)
    awq_model_path = os.path.expanduser(args.awq_model_path)

    cfg = load_llaga_config(llaga_model_path)
    tokenizer, model, backend = load_awq_language_model(
        awq_model_path,
        awq_tokenizer_path=args.awq_tokenizer_path,
        awq_backend=args.awq_backend,
        cache_dir=args.cache_dir,
        fuse_awq_layers=args.fuse_awq_layers,
        awq_max_seq_len=args.awq_max_seq_len,
        awq_use_exllama=args.awq_use_exllama,
        awq_use_exllamav2=args.awq_use_exllamav2,
    )
    print(f"Loaded AWQ language model from {awq_model_path} with backend={backend}")

    projector_device = get_embedding_device(model)
    mm_projector = load_fp16_mm_projector(llaga_model_path, cfg, projector_device)

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

    data = torch.load(data_path, weights_only=False)
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

    model.eval()

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
                graph_embs = [move_tensor_to_device(pretrained_emb[center_id], projector_device)]
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
                local_x = move_tensor_to_device(pretrained_emb[subset], projector_device)
                local_num_nodes = subset.shape[0]
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index)
                edge_index = move_tensor_to_device(edge_index, projector_device)
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
                inputs_embeds, attention_mask = build_multimodal_inputs_embeds(
                    model,
                    mm_projector,
                    cfg,
                    input_ids,
                    graph,
                    graph_emb,
                )
                output_ids = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            outputs = decode_awq_outputs(tokenizer, output_ids, prompt_length=attention_mask.shape[1])
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
    add_awq_args(parser)
    args = parser.parse_args()

    eval_model(args)
