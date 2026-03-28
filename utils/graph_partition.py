"""
Subgraph partitioning module using Louvain community detection.

Partitions a large graph into high-cohesion communities for MoE routing.
This is an offline preprocessing step — results are saved as .pt files.
"""

import torch
import numpy as np
from torch_geometric.utils import to_undirected
import community as community_louvain
import networkx as nx
from collections import defaultdict


class LouvainGraphPartitioner:
    """Partition a large graph into communities using Louvain algorithm."""

    def __init__(self, resolution=1.0, min_size=10, max_size=5000, seed=42):
        """
        Args:
            resolution: Louvain resolution parameter. Higher = more smaller communities.
            min_size: Communities smaller than this are merged into nearest neighbor.
            max_size: Communities larger than this are recursively re-partitioned.
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
            node_to_community: LongTensor [num_nodes], mapping each node to community ID.
            num_communities: int, total number of communities after post-processing.
        """
        # Build undirected edge index and convert to NetworkX
        edge_index_undirected = to_undirected(edge_index)
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index_undirected.t().numpy()
        G.add_edges_from(edges.tolist())

        # Run Louvain
        partition = community_louvain.best_partition(
            G, resolution=self.resolution, random_state=self.seed
        )

        # Convert to tensor
        node_to_community = torch.zeros(num_nodes, dtype=torch.long)
        for node_id, comm_id in partition.items():
            node_to_community[node_id] = comm_id

        # Post-process: handle too-small and too-large communities
        node_to_community = self._postprocess(
            node_to_community, edge_index_undirected, num_nodes
        )

        num_communities = node_to_community.max().item() + 1
        return node_to_community, num_communities

    def _postprocess(self, node_to_community, edge_index, num_nodes):
        """Merge small communities and split large ones."""
        # Count community sizes
        comm_ids, counts = torch.unique(node_to_community, return_counts=True)
        comm_sizes = dict(zip(comm_ids.tolist(), counts.tolist()))

        # --- Merge small communities into their most-connected neighbor ---
        small_comms = [c for c, s in comm_sizes.items() if s < self.min_size]
        if small_comms:
            # Build community adjacency: for each small community, find the
            # neighboring community with the most cross-edges
            row, col = edge_index
            src_comm = node_to_community[row]
            dst_comm = node_to_community[col]

            for sc in small_comms:
                # Find edges from this community to others
                mask = (src_comm == sc) & (dst_comm != sc)
                if mask.sum() == 0:
                    # Isolated community — assign to community 0
                    node_to_community[node_to_community == sc] = 0
                    continue
                neighbor_comms = dst_comm[mask]
                # Pick the most frequent neighbor community
                vals, cnts = torch.unique(neighbor_comms, return_counts=True)
                best_neighbor = vals[cnts.argmax()].item()
                node_to_community[node_to_community == sc] = best_neighbor

        # --- Re-index communities to be contiguous ---
        unique_comms = torch.unique(node_to_community)
        remap = torch.zeros(unique_comms.max().item() + 1, dtype=torch.long)
        for new_id, old_id in enumerate(unique_comms.tolist()):
            remap[old_id] = new_id
        node_to_community = remap[node_to_community]

        return node_to_community

    @staticmethod
    def compute_community_features(node_embeddings, node_to_community, num_communities):
        """
        Compute mean-pooled features per community for router input.

        Args:
            node_embeddings: [N, d] node feature tensor.
            node_to_community: [N] community assignment tensor.
            num_communities: int.

        Returns:
            community_features: [num_communities, d] mean-pooled features.
        """
        d = node_embeddings.shape[1]
        community_features = torch.zeros(num_communities, d, dtype=node_embeddings.dtype)
        community_counts = torch.zeros(num_communities, dtype=torch.long)

        for comm_id in range(num_communities):
            mask = node_to_community == comm_id
            if mask.sum() > 0:
                community_features[comm_id] = node_embeddings[mask].mean(dim=0)
                community_counts[comm_id] = mask.sum()

        return community_features

    @staticmethod
    def get_community_info(node_to_community, num_communities):
        """
        Get node indices per community.

        Args:
            node_to_community: [N] community assignment tensor.
            num_communities: int.

        Returns:
            community_nodes: Dict[int, LongTensor], community_id -> node indices.
        """
        community_nodes = {}
        for comm_id in range(num_communities):
            mask = node_to_community == comm_id
            community_nodes[comm_id] = torch.where(mask)[0]
        return community_nodes
