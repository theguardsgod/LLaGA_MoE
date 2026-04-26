"""
Batch NC evaluation: groups N nodes per prompt with N <graph> tokens.
Parses numbered output lines to extract per-node predictions.

Supports:
- ND batched node-centered graphs
- HO node-only batching (`template=HO`, `use_hop=0`)

Usage:
    python eval/eval_pretrain_batch_nc.py \
        --model_path <path> --model_base <base> \
        --dataset arxiv --template ND \
        --pretrained_embedding_type simteg \
        --use_hop 2 --sample_neighbor_size 10 \
        --nc_batch_size 20 --answers_file results/batch_nc.jsonl
"""
import sys
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import os
import json
import re
from tqdm import tqdm
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path


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


def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    else:
        pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
    return pretrained_emb


def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = [torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))] + [
            torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt")) for i in range(1, hop + 1)
        ]
        simteg_roberta = [torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))] + [
            torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt")) for i in range(1, hop + 1)
        ]
        simteg_e5 = [torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))] + [
            torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt")) for i in range(1, hop + 1)
        ]
        return [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]

    return [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))] + [
        torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt")) for i in range(1, hop + 1)
    ]


def build_batch_prompt(dataset, batch_size, template="ND"):
    graph_tokens = ", ".join([DEFAULT_GRAPH_TOKEN] * batch_size)
    classes = NC_CLASSES[dataset]
    if template == "HO":
        return (
            f"Given {batch_size} center nodes: {graph_tokens}. "
            f"We need to classify each node into {classes}. "
            f"For each node (1 to {batch_size}), tell me which class the node belongs to. "
            f"Answer format: one class per line, e.g.:\n"
            f"1. <class>\n2. <class>\n..."
        )

    edge_desc = EDGE_DESC[dataset]
    return (
        f"Given {batch_size} node-centered graphs: {graph_tokens}{edge_desc}. "
        f"We need to classify each center node into {classes}. "
        f"For each node (1 to {batch_size}), tell me which class the center node belongs to. "
        f"Answer format: one class per line, e.g.:\n"
        f"1. <class>\n2. <class>\n..."
    )


def parse_batch_output(text, expected_count):
    """Parse numbered output lines like '1. cs.LG(Machine Learning)\n2. cs.CR(...)' """
    predictions = []
    # Try numbered format first: "1. xxx", "2. xxx"
    pattern = re.compile(r'(\d+)\.\s*(.+)')
    for match in pattern.finditer(text):
        predictions.append(match.group(2).strip())

    # If we got enough, return
    if len(predictions) >= expected_count:
        return predictions[:expected_count]

    # Fallback: split by newline, take non-empty lines
    if len(predictions) < expected_count:
        predictions = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if line:
                # Strip leading number + dot if present
                line = re.sub(r'^\d+\.\s*', '', line)
                predictions.append(line)

    # Pad with empty if not enough
    while len(predictions) < expected_count:
        predictions.append("")

    return predictions[:expected_count]


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()

    data_dir = DATASET_PATHS[args.dataset]
    data_path = os.path.join(data_dir, "processed_data.pt")
    data = torch.load(data_path)

    # Load test questions (individual samples)
    if args.template == "HO":
        prompt_file = os.path.join(data_dir, "sampled_2_10_test.jsonl")
    else:
        prompt_file = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_test.jsonl")
    print(f"Load from {prompt_file}")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    questions = [json.loads(q) for q in lines]
    print(f"Total test samples: {len(questions)}, batch_size: {args.nc_batch_size}")

    if args.template == "ND":
        pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        structure_emb = torch.load(
            f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt")
    elif args.template == "HO":
        if args.use_hop != 0:
            raise ValueError("Node-only batch NC expects `template=HO` with `use_hop=0`.")
        pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop)
        structure_emb = None
    else:
        raise ValueError(f"Unsupported template: {args.template}")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Check for resume
    existing_count = 0
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        existing_count = len(open(answers_file, 'r').readlines())
        print(f"{answers_file} already exists with {existing_count} lines")

    # Count how many individual samples have been processed
    # We write one line per individual node, so existing_count = samples done
    if existing_count >= len(questions):
        print("Already done!")
        return
    questions = questions[existing_count:]
    ans_file = open(answers_file, "a" if existing_count > 0 else "w")

    # Group into batches
    batch_size = args.nc_batch_size
    num_batches = (len(questions) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Batch NC"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(questions))
        batch = questions[start:end]
        actual_size = len(batch)

        # Build prompt
        qs = build_batch_prompt(args.dataset, actual_size, template=args.template)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Build graph embeddings for all nodes in batch
        all_graphs = []
        all_ids = []
        all_gts = []
        for line in batch:
            g = line['graph']
            if isinstance(g[0], list):
                all_graphs.extend(g)
            else:
                all_graphs.append(g)
            all_ids.append(line['id'])
            all_gts.append(line['conversations'][1]['value'])

        if args.template == "ND":
            graph = torch.LongTensor(all_graphs)
            mask = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = pretrained_emb[graph[mask]]
            s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
            graph_emb = torch.zeros((s, n, d))
            graph_emb[mask] = masked_graph_emb
            if structure_emb is not None:
                graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
        else:
            node_only_graphs = []
            for g in all_graphs:
                center_id = g[0]
                node_only_graphs.append([center_id] * (args.use_hop + 1))
            graph = torch.LongTensor(node_only_graphs)
            center_id = graph[:, 0]
            graph_emb = torch.stack([emb[center_id] for emb in pretrained_emb], dim=1)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            print(f"!!!!!!Error!!!!! {e}")
            outputs = "\n".join([""] * actual_size)

        # Parse batch output into individual predictions
        predictions = parse_batch_output(outputs, actual_size)

        # Write one result per node (compatible with eval_res.py)
        for i in range(actual_size):
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": all_ids[i],
                "prompt": f"[batch {batch_idx}, node {i+1}/{actual_size}]",
                "text": predictions[i],
                "gt": all_gts[i],
                "answer_id": ans_id
            }) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"Done! Results saved to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--answers_file", type=str, default="results/batch_nc.jsonl")
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
    parser.add_argument("--nc_batch_size", type=int, default=20)
    parser.add_argument("--mm_use_graph_start_end", default=False, action="store_true")
    args = parser.parse_args()

    eval_model(args)
