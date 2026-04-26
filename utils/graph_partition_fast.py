"""
Fast subgraph partitioning via python-igraph's C backend.

Drop-in replacement for `utils/graph_partition.py::LouvainGraphPartitioner`.
Same public interface (`partition`, `compute_community_features`,
`get_community_info`), but:

  1. Louvain runs in C via `igraph.community_multilevel` instead of the pure
     Python `python-louvain` package (50-200x faster on large graphs).
  2. Graph construction goes directly from edge_index to igraph, skipping
     NetworkX's dict-of-dict overhead entirely.
  3. `_postprocess` is vectorized: a single scatter over all cross-community
     edges, replacing the O(small_comms * E) Python loop.
  4. `compute_community_features` uses `index_add_`: one pass over nodes,
     replacing the O(C * N) per-community mask + mean.

Usage — swap the import in `scripts/partition_graph.py`:

    # from utils.graph_partition import LouvainGraphPartitioner
    from utils.graph_partition_fast import LouvainGraphPartitioner

The original `graph_partition.py` is left untouched so you can A/B compare.

Requires: `pip install python-igraph`
"""

import torch
import numpy as np
from torch_geometric.utils import to_undirected

try:
    import igraph as ig
except ImportError as e:
    raise ImportError(
        "python-igraph is required for the fast partitioner. "
        "Install with: pip install python-igraph"
    ) from e


class LouvainGraphPartitioner:
    """Fast Louvain partitioner backed by igraph's C implementation."""

    def __init__(self, resolution=1.0, min_size=10, max_size=5000, seed=42):
        """
        Args:
            resolution: Louvain resolution. Higher = more smaller communities.
                        (Only honored on python-igraph >= 0.9; older versions
                        fall back to the default 1.0 with a warning.)
            min_size: Communities smaller than this are merged into their
                      most-connected neighbor community.
            max_size: Reserved for future use (recursive re-partitioning of
                      oversized communities). Currently unused to match the
                      original implementation.
            seed: Random seed for reproducibility.
        """
        self.resolution = resolution
        self.min_size = min_size
        self.max_size = max_size
        self.seed = seed

    def partition(self, edge_index, num_nodes):
        """
        Run Louvain community detection on the graph.

        Args:
            edge_index: [2, E] tensor of edge indices.
            num_nodes: Total number of nodes in the graph.

        Returns:
            node_to_community: LongTensor [num_nodes], mapping each node to
                               community ID.
            num_communities: int, total number of communities after
                             post-processing.
        """
        # Undirected edge index (PyG helper adds reverse edges where missing).
        edge_index_undirected = to_undirected(edge_index)

        # Build igraph directly from the edge list. `tolist()` on a [E, 2]
        # tensor yields List[List[int]], which igraph's Graph() accepts.
        # This bypasses NetworkX entirely — on products (~61M edges) this
        # alone saves several minutes.
        edges_list = edge_index_undirected.t().tolist()
        g = ig.Graph(n=num_nodes, edges=edges_list, directed=False)
        # Remove duplicate reverse edges and self-loops so modularity is
        # computed on the true underlying simple graph.
        g.simplify(multiple=True, loops=True)

        # Seed igraph's RNG so results are reproducible across runs.
        try:
            ig.set_random_number_generator(np.random.default_rng(self.seed))
        except Exception:
            # Older igraph versions expose a different RNG hook; ignore if
            # unavailable — reproducibility is best-effort.
            pass

        # Run Louvain (C implementation). `resolution` is supported in
        # python-igraph >= 0.9; older versions raise TypeError.
        try:
            clusters = g.community_multilevel(resolution=self.resolution)
        except TypeError:
            if self.resolution != 1.0:
                print(f"  [warn] igraph version ignores resolution="
                      f"{self.resolution}, using default 1.0")
            clusters = g.community_multilevel()

        membership = np.asarray(clusters.membership, dtype=np.int64)
        node_to_community = torch.from_numpy(membership)

        node_to_community = self._postprocess(
            node_to_community, edge_index_undirected, num_nodes
        )

        num_communities = int(node_to_community.max().item()) + 1
        return node_to_community, num_communities

    def _postprocess(self, node_to_community, edge_index, num_nodes):
        """
        Merge communities smaller than `min_size` into their most-connected
        neighbor community.

        Vectorized path:
          - Pull src/dst community ids for every edge at once.
          - Keep only cross-community edges whose src lives in a small
            community.
          - Encode (src_comm, dst_comm) as a single int key, count unique
            keys via `torch.unique`, then argmax per src community by sorting
            counts descending and keeping the first occurrence.

        Net effect: one O(E) pass instead of O(small_comms * E).
        """
        comm_ids, counts = torch.unique(node_to_community, return_counts=True)
        is_small = counts < self.min_size
        small_comm_tensor = comm_ids[is_small]

        if small_comm_tensor.numel() > 0:
            row, col = edge_index
            src_comm = node_to_community[row]                    # [E]
            dst_comm = node_to_community[col]                    # [E]

            src_is_small = torch.isin(src_comm, small_comm_tensor)
            cross = src_is_small & (src_comm != dst_comm)
            sc = src_comm[cross]
            dc = dst_comm[cross]

            C = int(node_to_community.max().item()) + 1

            # Encode each (small_comm, neighbor_comm) pair into a single key.
            keys = sc.to(torch.long) * C + dc.to(torch.long)
            unique_keys, key_counts = torch.unique(keys, return_counts=True)
            sc_of_key = unique_keys // C
            dc_of_key = unique_keys % C

            # For each small community pick the neighbor with the highest
            # cross-edge count: sort by count desc and take first occurrence.
            order = torch.argsort(key_counts, descending=True)
            sc_sorted = sc_of_key[order]
            dc_sorted = dc_of_key[order]

            remap = torch.arange(C, dtype=torch.long)
            seen = torch.zeros(C, dtype=torch.bool)
            for s, d in zip(sc_sorted.tolist(), dc_sorted.tolist()):
                if not seen[s]:
                    remap[s] = d
                    seen[s] = True

            # Isolated small communities (no cross-community edges) — match
            # the original behavior of assigning them to community 0.
            for s in small_comm_tensor.tolist():
                if not seen[s]:
                    remap[s] = 0

            node_to_community = remap[node_to_community]

        # Re-index to contiguous [0, C').
        _, inverse = torch.unique(node_to_community, return_inverse=True)
        return inverse

    @staticmethod
    def compute_community_features(node_embeddings, node_to_community, num_communities):
        """
        Mean-pool node embeddings into per-community features.

        Vectorized with `index_add_`: one pass over N nodes instead of the
        original O(C * N) per-community mask + mean.

        Args:
            node_embeddings: [N, d] node feature tensor.
            node_to_community: [N] community assignment tensor.
            num_communities: int.

        Returns:
            community_features: [num_communities, d] mean-pooled features.
        """
        d = node_embeddings.shape[1]
        dtype = node_embeddings.dtype
        device = node_embeddings.device
        n = node_embeddings.shape[0]

        idx = node_to_community.to(device=device, dtype=torch.long)

        sums = torch.zeros(num_communities, d, dtype=dtype, device=device)
        counts = torch.zeros(num_communities, dtype=dtype, device=device)

        sums.index_add_(0, idx, node_embeddings)
        counts.index_add_(0, idx, torch.ones(n, dtype=dtype, device=device))

        return sums / counts.clamp(min=1).unsqueeze(-1)

    @staticmethod
    def get_community_info(node_to_community, num_communities):
        """
        Get node indices per community.

        Args:
            node_to_community: [N] community assignment tensor.
            num_communities: int.

        Returns:
            community_nodes: Dict[int, LongTensor], comm_id -> node indices.
        """
        community_nodes = {}
        for comm_id in range(num_communities):
            mask = node_to_community == comm_id
            community_nodes[comm_id] = torch.where(mask)[0]
        return community_nodes
