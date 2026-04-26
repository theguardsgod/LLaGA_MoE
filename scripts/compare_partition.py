"""
A/B comparison between the original LouvainGraphPartitioner and the
igraph-backed fast version.

Runs both partitioners on one dataset, reports:

  - Wall time per partitioner
  - Number of communities
  - Community size stats (min / max / mean / median)
  - Membership agreement via Adjusted Rand Index (sklearn)
  - Community feature similarity: for every community in A, match the
    nearest community in B by cosine similarity and report the mean / min
    of these best-match scores

Usage:
    python scripts/compare_partition.py --dataset arxiv
    python scripts/compare_partition.py --dataset arxiv --resolution 1.0
    python scripts/compare_partition.py --dataset pubmed cora

Note: Louvain is non-deterministic and the two implementations do not share
a tie-breaking policy, so a perfect match (ARI = 1.0) is not expected. What
you want to see is ARI well above 0.8 and mean best-match cosine > 0.95 —
that means the clustering is essentially the same at the structural level,
even if specific community IDs differ.
"""

import sys
sys.path.append("./")

import argparse
import os
import time
import types

# Workaround: if torch_sparse CUDA .so is broken (ABI mismatch), install a
# stub so that torch.load can still unpickle Data objects that reference it.
try:
    import torch_sparse  # noqa: F401
except OSError:
    # torch_sparse CUDA .so is broken (ABI mismatch). Create a minimal stub
    # so torch.load can unpickle Data objects that reference SparseTensor.
    class _SparseTensorStub:
        """Placeholder that accepts any pickle state without crashing."""
        def __init__(self, *args, **kwargs):
            pass
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})
        def __reduce_ex__(self, protocol):
            return (self.__class__, ())

    _stub = types.ModuleType("torch_sparse")
    sys.modules["torch_sparse"] = _stub
    for sub in ("storage", "tensor", "matmul"):
        m = types.ModuleType(f"torch_sparse.{sub}")
        sys.modules[f"torch_sparse.{sub}"] = m
        setattr(_stub, sub, m)
    # Register placeholder classes where the unpickler expects them.
    _stub.SparseTensor = _SparseTensorStub
    _stub.tensor.SparseTensor = _SparseTensorStub
    _stub.storage.SparseStorage = _SparseTensorStub

import torch
import numpy as np

from utils.graph_partition import LouvainGraphPartitioner as SlowPartitioner
from utils.graph_partition_fast import LouvainGraphPartitioner as FastPartitioner

DATASET_DIRS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def load_simteg_embeddings(data_dir):
    sbert = _load(os.path.join(data_dir, "simteg_sbert_x.pt"))
    roberta = _load(os.path.join(data_dir, "simteg_roberta_x.pt"))
    e5 = _load(os.path.join(data_dir, "simteg_e5_x.pt"))
    return torch.cat([sbert, roberta, e5], dim=-1)


def size_stats(node_to_community):
    _, counts = torch.unique(node_to_community, return_counts=True)
    c = counts.float()
    return {
        "num_comms": counts.numel(),
        "min": int(counts.min().item()),
        "max": int(counts.max().item()),
        "mean": float(c.mean().item()),
        "median": float(c.median().item()),
    }


def mean_best_match_cosine(feat_a, feat_b):
    """For every row in feat_a, find its nearest row in feat_b by cosine
    similarity. Returns (mean, min) over these best matches.

    Both inputs are [C, d] tensors (possibly with different C)."""
    a = torch.nn.functional.normalize(feat_a, dim=-1)
    b = torch.nn.functional.normalize(feat_b, dim=-1)
    sim = a @ b.t()                     # [Ca, Cb]
    best = sim.max(dim=1).values        # [Ca]
    return float(best.mean().item()), float(best.min().item())


def ari(labels_a, labels_b):
    """Adjusted Rand Index via sklearn. Falls back to None if unavailable."""
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(labels_a, labels_b))
    except ImportError:
        print("  [warn] sklearn not installed, skipping ARI")
        return None


def nmi(labels_a, labels_b):
    """Normalized Mutual Information via sklearn. None if unavailable."""
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(labels_a, labels_b))
    except ImportError:
        return None


def run_one(partitioner_cls, name, edge_index, num_nodes, node_emb,
            resolution, min_size, max_size, seed):
    print(f"\n--- {name} ---")
    part = partitioner_cls(
        resolution=resolution, min_size=min_size, max_size=max_size, seed=seed
    )
    t0 = time.perf_counter()
    node_to_comm, num_comms = part.partition(edge_index, num_nodes)
    t_partition = time.perf_counter() - t0

    t0 = time.perf_counter()
    comm_feat = partitioner_cls.compute_community_features(
        node_emb, node_to_comm, num_comms
    )
    t_features = time.perf_counter() - t0

    stats = size_stats(node_to_comm)
    print(f"  partition():                {t_partition:8.2f}s")
    print(f"  compute_community_features: {t_features:8.2f}s")
    print(f"  num_communities: {stats['num_comms']}")
    print(f"  sizes: min={stats['min']}, max={stats['max']}, "
          f"mean={stats['mean']:.1f}, median={stats['median']:.1f}")
    return {
        "node_to_comm": node_to_comm,
        "num_comms": num_comms,
        "comm_feat": comm_feat,
        "t_partition": t_partition,
        "t_features": t_features,
        "stats": stats,
    }


def compare_dataset(dataset, resolution, min_size, max_size, seed, skip_slow):
    print(f"\n{'='*64}")
    print(f"Dataset: {dataset}")
    print(f"  resolution={resolution}, min_size={min_size}, "
          f"max_size={max_size}, seed={seed}")

    data_dir = DATASET_DIRS[dataset]
    data = _load(os.path.join(data_dir, "processed_data.pt"))
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    print(f"  nodes={num_nodes}, edges={edge_index.shape[1]}")

    print("  Loading SimTEG embeddings for feature comparison...")
    node_emb = load_simteg_embeddings(data_dir)
    print(f"  node_emb shape: {tuple(node_emb.shape)}")

    fast = run_one(
        FastPartitioner, "FAST (igraph C backend)",
        edge_index, num_nodes, node_emb,
        resolution, min_size, max_size, seed,
    )

    if skip_slow:
        print("\n  (slow partitioner skipped by --skip-slow)")
        return

    slow = run_one(
        SlowPartitioner, "SLOW (python-louvain + networkx)",
        edge_index, num_nodes, node_emb,
        resolution, min_size, max_size, seed,
    )

    print(f"\n--- Comparison ---")
    speedup_part = slow["t_partition"] / max(fast["t_partition"], 1e-6)
    speedup_feat = slow["t_features"] / max(fast["t_features"], 1e-6)
    print(f"  partition() speedup:                 {speedup_part:6.1f}x")
    print(f"  compute_community_features speedup:  {speedup_feat:6.1f}x")

    slow_labels = slow["node_to_comm"].numpy()
    fast_labels = fast["node_to_comm"].numpy()
    ari_score = ari(slow_labels, fast_labels)
    nmi_score = nmi(slow_labels, fast_labels)
    if ari_score is not None:
        print(f"  Adjusted Rand Index (membership): {ari_score:.4f}")
    if nmi_score is not None:
        print(f"  Normalized Mutual Info:            {nmi_score:.4f}")

    mean_cos, min_cos = mean_best_match_cosine(
        fast["comm_feat"].float(), slow["comm_feat"].float()
    )
    print(f"  Mean best-match cosine (fast → slow): {mean_cos:.4f}")
    print(f"  Min  best-match cosine (fast → slow): {min_cos:.4f}")

    print(f"\n  Interpretation:")
    print(f"    ARI > 0.8 and mean cos > 0.95 → clusterings are essentially")
    print(f"    equivalent, only community IDs differ. Safe to switch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs='+', default=["cora"],
                        help="Dataset(s) to compare: arxiv, products, pubmed, cora")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min_size", type=int, default=10)
    parser.add_argument("--max_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-slow", action="store_true",
                        help="Only run the fast partitioner (useful for products "
                             "where the slow version takes hours).")
    args = parser.parse_args()

    for ds in args.dataset:
        if ds not in DATASET_DIRS:
            print(f"Unknown dataset: {ds}, skipping")
            continue
        compare_dataset(
            ds, args.resolution, args.min_size, args.max_size,
            args.seed, args.skip_slow,
        )
    print("\nDone.")
