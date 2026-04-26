"""
Restructure NC training/test data: move class list BEFORE <graph> for KV-cache inference.

Original: "Given a node-centered graph: <graph>{edge_desc}, we need to classify ... into {classes}, please tell me ..."
New:      "We need to classify ... into {classes}. Given a node-centered graph: <graph>{edge_desc}, please tell me ..."

Usage:
    python scripts/restructure_nc_data.py --dataset arxiv --use_hop 2 --sample_neighbor_size 10
"""
import json
import os
import argparse

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


def build_restructured_prompt(dataset):
    """Must match eval_pretrain_kvcache.py build_nc_prompt_restructured() exactly."""
    classes = NC_CLASSES[dataset]
    edge_desc = EDGE_DESC[dataset]
    return (
        f"We need to classify the center node into {classes}. "
        f"Given a node-centered graph: <graph>{edge_desc}, "
        f"please tell me which class the center node belongs to? "
        f"Answer with only the class name, e.g., cs.AI(Artificial Intelligence)."
    )


def process_file(input_path, output_path, new_prompt):
    count = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            sample = json.loads(line)
            sample['conversations'][0]['value'] = new_prompt
            fout.write(json.dumps(sample) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="arxiv",
                        choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    args = parser.parse_args()

    data_dir = DATASET_PATHS[args.dataset]
    new_prompt = build_restructured_prompt(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"New prompt: {new_prompt[:100]}...")

    for split in ["train", "test"]:
        input_path = os.path.join(
            data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_{split}.jsonl")
        output_path = os.path.join(
            data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_{split}_kvcache.jsonl")

        if not os.path.exists(input_path):
            print(f"  {split}: {input_path} not found, skipping")
            continue

        count = process_file(input_path, output_path, new_prompt)
        print(f"  {split}: {count} samples -> {output_path}")


if __name__ == "__main__":
    main()
