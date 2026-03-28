"""
Offline preprocessing: partition graphs using Louvain and save community assignments.

Usage:
    python scripts/partition_graph.py --dataset arxiv --resolution 1.0
    python scripts/partition_graph.py --dataset arxiv products pubmed cora

Saves to each dataset directory:
    - node_to_community.pt  [N] int64 tensor
    - community_features.pt [C, d] float tensor (SimTEG mean-pooled)
    - partition_info.pt     {num_communities, resolution, sizes}
"""

import sys
sys.path.append("./")

import argparse
import os
import torch
from utils.graph_partition import LouvainGraphPartitioner

DATASET_DIRS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}


def load_simteg_embeddings(data_dir):
    """Load and concatenate SimTEG embeddings (SBERT + RoBERTa + E5)."""
    sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
    roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
    e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
    return torch.cat([sbert, roberta, e5], dim=-1)


def partition_dataset(dataset, resolution, min_size, max_size):
    data_dir = DATASET_DIRS[dataset]
    print(f"\n{'='*60}")
    print(f"Partitioning {dataset} (dir: {data_dir})")
    print(f"  resolution={resolution}, min_size={min_size}, max_size={max_size}")

    # Load graph
    data = torch.load(os.path.join(data_dir, "processed_data.pt"))
    print(f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")

    # Run Louvain
    partitioner = LouvainGraphPartitioner(
        resolution=resolution, min_size=min_size, max_size=max_size
    )
    node_to_community, num_communities = partitioner.partition(
        data.edge_index, data.num_nodes
    )
    print(f"  Communities: {num_communities}")

    # Community size statistics
    comm_ids, counts = torch.unique(node_to_community, return_counts=True)
    print(f"  Size stats: min={counts.min().item()}, max={counts.max().item()}, "
          f"mean={counts.float().mean().item():.1f}, median={counts.float().median().item():.1f}")

    # Compute community features using SimTEG embeddings
    print(f"  Computing community features...")
    node_emb = load_simteg_embeddings(data_dir)
    community_features = LouvainGraphPartitioner.compute_community_features(
        node_emb, node_to_community, num_communities
    )
    print(f"  Community features shape: {community_features.shape}")

    # Save
    torch.save(node_to_community, os.path.join(data_dir, "node_to_community.pt"))
    torch.save(community_features, os.path.join(data_dir, "community_features.pt"))
    torch.save({
        "num_communities": num_communities,
        "resolution": resolution,
        "min_size": min_size,
        "max_size": max_size,
        "community_sizes": counts,
    }, os.path.join(data_dir, "partition_info.pt"))

    print(f"  Saved: node_to_community.pt, community_features.pt, partition_info.pt")
    return num_communities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs='+', default=["arxiv"],
                        help="Dataset(s) to partition: arxiv, products, pubmed, cora")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Louvain resolution (higher = more communities)")
    parser.add_argument("--min_size", type=int, default=10,
                        help="Min community size (smaller ones get merged)")
    parser.add_argument("--max_size", type=int, default=5000,
                        help="Max community size")
    args = parser.parse_args()

    for ds in args.dataset:
        if ds not in DATASET_DIRS:
            print(f"Unknown dataset: {ds}, skipping")
            continue
        partition_dataset(ds, args.resolution, args.min_size, args.max_size)

    print(f"\nDone.")
