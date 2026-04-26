"""
Prepare batch NC training/test data: group N nodes into one sample with N <graph> tokens.

Supports:
- ND: node-centered graphs with structure/laplacian features.
- HO node-only: center node only (use_hop=0), one <graph> token per node.

Usage:
    python scripts/prepare_batch_nc_data.py --dataset arxiv --batch_size 20 --use_hop 2 --sample_neighbor_size 10
    python scripts/prepare_batch_nc_data.py --dataset arxiv --batch_size 20 --template HO --use_hop 0
"""
import sys
sys.path.append(".")
import json
import os
import argparse
import random
import torch

from utils.constants import DEFAULT_GRAPH_TOKEN


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


def build_batch_answer(labels):
    lines = [f"{i+1}. {label}" for i, label in enumerate(labels)]
    return "\n".join(lines)


def process_split(data_dir, split, dataset, batch_size, use_hop, sample_neighbor_size, data, template):
    if template == "HO":
        jsonl_name = f"sampled_2_10_{split}.jsonl"
        output_name = f"batch{batch_size}_nodeonly_nc_{split}.jsonl"
    elif split == "train":
        jsonl_name = f"sampled_{use_hop}_{sample_neighbor_size}_train.jsonl"
        output_name = f"batch{batch_size}_nc_train.jsonl"
    else:
        jsonl_name = f"sampled_{use_hop}_{sample_neighbor_size}_test.jsonl"
        output_name = f"batch{batch_size}_nc_test.jsonl"

    input_path = os.path.join(data_dir, jsonl_name)
    output_path = os.path.join(data_dir, output_name)

    # Load all samples
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    if split == "train":
        random.shuffle(samples)

    # Group into batches
    batched = []
    for start in range(0, len(samples), batch_size):
        group = samples[start:start + batch_size]
        actual_size = len(group)

        # Merge graph lists
        merged_graphs = []
        ids = []
        labels = []
        for s in group:
            graph = s['graph']
            if isinstance(graph[0], list):
                graph_list = graph
            else:
                graph_list = [graph]

            if template == "HO":
                for g in graph_list:
                    center_id = g[0]
                    merged_graphs.append([center_id] * (use_hop + 1))
            else:
                merged_graphs.extend(graph_list)

            node_id = s['id']
            ids.append(node_id)
            label = data.label_texts[data.y[node_id]]
            labels.append(label)

        user_prompt = build_batch_prompt(dataset, actual_size, template=template)
        assistant_answer = build_batch_answer(labels)

        batched_sample = {
            "id": ids,
            "graph": merged_graphs,
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_answer}
            ]
        }
        batched.append(batched_sample)

    with open(output_path, 'w') as f:
        for s in batched:
            f.write(json.dumps(s) + "\n")

    print(
        f"[{dataset}/{split}/{template}] {len(samples)} samples -> {len(batched)} batches "
        f"(batch_size={batch_size}), saved to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="arxiv", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--template", type=str, default="ND", choices=["ND", "HO"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.template == "HO" and args.use_hop != 0:
        raise ValueError("Node-only batch NC expects `template=HO` with `use_hop=0`.")

    random.seed(args.seed)
    data_dir = DATASET_PATHS[args.dataset]
    data = torch.load(os.path.join(data_dir, "processed_data.pt"), weights_only=False)

    for split in ["train", "test"]:
        if args.template == "HO":
            jsonl_name = f"sampled_2_10_{split}.jsonl"
        else:
            jsonl_name = f"sampled_{args.use_hop}_{args.sample_neighbor_size}_{split}.jsonl"
        if os.path.exists(os.path.join(data_dir, jsonl_name)):
            process_split(
                data_dir,
                split,
                args.dataset,
                args.batch_size,
                args.use_hop,
                args.sample_neighbor_size,
                data,
                args.template,
            )
        else:
            print(f"[{args.dataset}/{split}] {jsonl_name} not found, skipping")


if __name__ == "__main__":
    main()
