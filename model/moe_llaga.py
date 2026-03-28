"""
MoE (Mixture of Experts) components for LLaGA.

Provides:
- TopKRouter: gating network that routes subgraph features to top-K experts
- GraphProjectorExpert: single expert wrapping a graph-to-LLM projector
- MoEGraphProjector: full MoE module combining router + experts + dispatch/combine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re


def _build_single_projector(mm_hidden_size, llm_hidden_size, projector_type='linear'):
    """Build a single projector (same logic as build_graph_projector in llaga_arch.py)."""
    if projector_type == 'linear':
        return nn.Linear(mm_hidden_size, llm_hidden_size)
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, llm_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
        return nn.Sequential(*modules)
    raise ValueError(f'Unknown projector type: {projector_type}')


class TopKRouter(nn.Module):
    """
    Top-K gating network for MoE routing.

    Takes subgraph/community-level features and produces expert assignment
    weights via a learned linear gate + softmax + top-K selection.
    """

    def __init__(self, input_dim, num_experts, top_k=2, noise_std=1.0):
        """
        Args:
            input_dim: Dimension of routing input features (community embedding dim).
            num_experts: Total number of experts.
            top_k: Number of experts to activate per input.
            noise_std: Standard deviation of Gaussian noise added during training
                       (for load balancing via noisy gating).
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, routing_features):
        """
        Args:
            routing_features: [batch_size, input_dim]

        Returns:
            dispatch_weights: [batch_size, top_k] — normalized weights for selected experts.
            dispatch_indices: [batch_size, top_k] — expert IDs selected.
            aux_loss: Scalar load-balancing loss.
        """
        # Gate logits
        logits = self.gate(routing_features)  # [B, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Full softmax over all experts (for aux loss computation)
        full_probs = F.softmax(logits, dim=-1)  # [B, num_experts]

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # [B, K]
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, K] re-normalize

        # Load balancing auxiliary loss (Switch Transformer style)
        aux_loss = self._load_balance_loss(full_probs, top_k_indices)

        return top_k_weights, top_k_indices, aux_loss

    def _load_balance_loss(self, probs, indices):
        """
        Compute load-balancing loss to encourage uniform expert usage.

        L_balance = num_experts * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = mean routing probability for expert i
        """
        batch_size = probs.shape[0]
        if batch_size == 0:
            return torch.tensor(0.0, device=probs.device)

        # f_i: fraction of tokens dispatched to each expert
        one_hot = F.one_hot(indices, self.num_experts).float()  # [B, K, E]
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)  # [E]
        f = tokens_per_expert / (batch_size * self.top_k)  # [E]

        # P_i: mean routing probability per expert
        P = probs.mean(dim=0)  # [E]

        aux_loss = self.num_experts * (f * P).sum()
        return aux_loss


class GraphProjectorExpert(nn.Module):
    """
    A single MoE expert that projects graph embeddings into LLM hidden space.

    Has the same architecture as the original mm_projector in LLaGA.
    """

    def __init__(self, mm_hidden_size, llm_hidden_size, projector_type='linear'):
        super().__init__()
        self.projector = _build_single_projector(
            mm_hidden_size, llm_hidden_size, projector_type
        )

    def forward(self, graph_emb):
        """
        Args:
            graph_emb: [..., mm_hidden_size]

        Returns:
            projected: [..., llm_hidden_size]
        """
        return self.projector(graph_emb)


class MoEGraphProjector(nn.Module):
    """
    Mixture-of-Experts graph projector.

    Replaces the single mm_projector with multiple expert projectors
    and a learned router that dispatches based on community features.
    """

    def __init__(
        self,
        mm_hidden_size,
        llm_hidden_size,
        num_experts=4,
        top_k=2,
        projector_type='linear',
        routing_dim=2432,
        noise_std=1.0,
    ):
        """
        Args:
            mm_hidden_size: Input dim of graph embeddings (e.g., 2543 for ND, 2432 for HO).
            llm_hidden_size: LLM hidden dim (4096 for Vicuna-7B).
            num_experts: Number of expert projectors.
            top_k: Number of experts activated per input.
            projector_type: 'linear' or '2-layer-mlp'.
            routing_dim: Dimension of community features for routing.
            noise_std: Noise for router during training.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.llm_hidden_size = llm_hidden_size

        self.router = TopKRouter(routing_dim, num_experts, top_k, noise_std)
        self.experts = nn.ModuleList([
            GraphProjectorExpert(mm_hidden_size, llm_hidden_size, projector_type)
            for _ in range(num_experts)
        ])

    def forward(self, graph_emb, routing_features, graph_mask=None):
        """
        MoE forward: route → dispatch → expert compute → combine.

        Args:
            graph_emb: [num_graphs, seq_len, mm_hidden_size]
                Per-graph token embeddings.
            routing_features: [num_graphs, routing_dim]
                Community-level features for routing decisions.
            graph_mask: Optional [num_graphs, seq_len] bool mask.
                True for valid tokens, False for padding (DEFAULT_GRAPH_PAD_ID).

        Returns:
            combined: [num_graphs, seq_len, llm_hidden_size]
                Weighted combination of expert outputs.
            aux_loss: Scalar load-balancing loss.
        """
        B, S, D = graph_emb.shape

        # Step 1: Router — get top-K experts and weights per graph
        weights, indices, aux_loss = self.router(routing_features)
        # weights: [B, K], indices: [B, K]

        # Step 2: Compute ALL experts on full input for DDP/DeepSpeed compatibility.
        # In distributed training, all ranks must execute the same set of parameter
        # operations so that gradient reduction stays in sync across ranks.
        # We compute all experts, then select/weight via the routing decisions.
        all_expert_outputs = []
        for expert_id in range(self.num_experts):
            all_expert_outputs.append(self.experts[expert_id](graph_emb))  # [B, S, H]
        all_expert_outputs = torch.stack(all_expert_outputs, dim=1)  # [B, E, S, H]

        # Step 3: Gather top-K expert outputs and weight them
        # indices: [B, K] -> gather from [B, E, S, H]
        K = self.top_k
        idx_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(
            B, K, S, self.llm_hidden_size
        )  # [B, K, S, H]
        selected = torch.gather(all_expert_outputs, dim=1, index=idx_expanded)  # [B, K, S, H]

        # Weighted combination
        w = weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        combined = (w * selected).sum(dim=1)  # [B, S, H]

        # Apply graph padding mask
        if graph_mask is not None:
            combined[~graph_mask] = 0.0

        return combined, aux_loss
