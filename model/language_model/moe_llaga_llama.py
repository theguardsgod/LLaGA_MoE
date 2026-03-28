"""
MoE-LLaGA model built on top of LlagaLlamaForCausalLM.

Inherits the full LLaGA pipeline and replaces encode_graphs()
with MoE dispatch/combine logic. The LLM backbone is shared.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.llaga_arch import LlagaMetaModel, LlagaMetaForCausalLM
from model.moe_llaga import MoEGraphProjector
from utils.constants import IGNORE_INDEX, GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_PAD_ID


class MoELlagaConfig(LlamaConfig):
    model_type = "moe_llaga"

    def __init__(self, num_experts=4, top_k=2, routing_dim=2432,
                 aux_loss_weight=0.01, noise_std=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_dim = routing_dim
        self.aux_loss_weight = aux_loss_weight
        self.noise_std = noise_std


class MoELlagaLlamaModel(LlagaMetaModel, LlamaModel):
    """Base model: LlamaModel + LlagaMetaModel (projector init)."""
    config_class = MoELlagaConfig

    def __init__(self, config: MoELlagaConfig):
        super(MoELlagaLlamaModel, self).__init__(config)
        # Note: LlagaMetaModel.__init__ creates self.mm_projector
        # We will replace it with MoE projector in the CausalLM class


class MoELlagaLlamaForCausalLM(LlamaForCausalLM, LlagaMetaForCausalLM):
    """
    MoE-enhanced LLaGA model.

    The only difference from LlagaLlamaForCausalLM is that encode_graphs()
    uses the MoE projector (multiple experts + router) instead of a single
    mm_projector. The entire LLM backbone and multimodal preparation logic
    are inherited unchanged.
    """
    config_class = MoELlagaConfig

    def __init__(self, config: MoELlagaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MoELlagaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MoE projector — will be initialized in initialize_moe_modules()
        self.moe_projector = None
        self.aux_loss_weight = getattr(config, 'aux_loss_weight', 0.01)

        self.post_init()

    def get_model(self):
        return self.model

    def initialize_moe_modules(self, model_args):
        """
        Initialize the MoE graph projector.

        Must be called after the base model is loaded and mm_hidden_size is set.
        """
        mm_hidden_size = getattr(model_args, 'mm_hidden_size', None)
        if mm_hidden_size is None:
            mm_hidden_size = getattr(self.config, 'mm_hidden_size', None)
        assert mm_hidden_size is not None, "mm_hidden_size must be set before init MoE"

        llm_hidden_size = getattr(
            self.config, 'word_embed_proj_dim',
            getattr(self.config, 'hidden_size', 4096)
        )
        projector_type = getattr(model_args, 'mm_projector_type',
                                 getattr(self.config, 'mm_projector_type', 'linear'))
        num_experts = getattr(model_args, 'num_experts',
                              getattr(self.config, 'num_experts', 4))
        top_k = getattr(model_args, 'top_k',
                        getattr(self.config, 'top_k', 2))
        routing_dim = getattr(model_args, 'routing_dim',
                              getattr(self.config, 'routing_dim', 2432))
        noise_std = getattr(model_args, 'noise_std',
                            getattr(self.config, 'noise_std', 1.0))

        self.moe_projector = MoEGraphProjector(
            mm_hidden_size=mm_hidden_size,
            llm_hidden_size=llm_hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            projector_type=projector_type,
            routing_dim=routing_dim,
            noise_std=noise_std,
        )

        # Store config for serialization
        self.config.mm_hidden_size = mm_hidden_size
        self.config.mm_projector_type = projector_type
        self.config.num_experts = num_experts
        self.config.top_k = top_k
        self.config.routing_dim = routing_dim
        self.config.noise_std = noise_std

    def encode_graphs(self, graph, graph_emb, routing_features=None):
        """
        MoE graph encoding: route graph embeddings through expert projectors.

        If routing_features is None, falls back to single mm_projector
        (for backward compatibility).

        Args:
            graph: [num_graphs, seq_len] node index tensor.
            graph_emb: [num_graphs, seq_len, mm_hidden_size] embeddings.
            routing_features: [num_graphs, routing_dim] community features.

        Returns:
            graph_features: [num_graphs, seq_len, llm_hidden_size]
            aux_loss: scalar (0 if not using MoE)
        """
        if self.moe_projector is not None and routing_features is not None:
            graph_mask = graph != DEFAULT_GRAPH_PAD_ID
            graph_features, aux_loss = self.moe_projector(
                graph_emb, routing_features, graph_mask
            )
        else:
            # Fallback to single projector (inherited from LlagaMetaModel)
            graph_features = self.get_model().mm_projector(graph_emb)
            graph_features[graph == DEFAULT_GRAPH_PAD_ID] = 0.0
            aux_loss = torch.tensor(0.0, device=graph_emb.device)

        return graph_features, aux_loss

    def prepare_inputs_labels_for_multimodal_moe(
        self, input_ids, attention_mask, past_key_values, labels,
        graphs, graph_emb, routing_features=None
    ):
        """
        MoE version of prepare_inputs_labels_for_multimodal.

        Same logic as the parent class, but uses MoE encode_graphs
        and returns aux_loss.
        """
        if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones(
                (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                dtype=attention_mask.dtype, device=attention_mask.device
            )
            return input_ids, attention_mask, past_key_values, None, labels, torch.tensor(0.0)

        # MoE encoding
        graph_features, aux_loss = self.encode_graphs(graphs, graph_emb, routing_features)

        # --- Below is identical to LlagaMetaForCausalLM.prepare_inputs_labels_for_multimodal ---
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_graph_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == GRAPH_TOKEN_INDEX).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_graph_features = graph_features[cur_graph_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_graph_features[0:0], cur_input_embeds_2], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue

            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while graph_token_indices.numel() > 0:
                cur_graph_features = graph_features[cur_graph_idx]
                graph_token_start = graph_token_indices[0]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and \
                   getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:graph_token_start - 1]).detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[graph_token_start - 1:graph_token_start])
                    )
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[graph_token_start + 1:graph_token_start + 2])
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full(
                            (cur_graph_features.shape[0],), IGNORE_INDEX,
                            device=labels.device, dtype=labels.dtype
                        ))
                        cur_new_labels.append(cur_labels[graph_token_start:graph_token_start + 1])
                        cur_labels = cur_labels[graph_token_start + 2:]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:graph_token_start])
                    )
                    cur_new_input_embeds.append(cur_graph_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full(
                            (cur_graph_features.shape[0],), IGNORE_INDEX,
                            device=labels.device, dtype=labels.dtype
                        ))
                        cur_labels = cur_labels[graph_token_start + 1:]

                cur_graph_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and \
                   getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_input_ids = cur_input_ids[graph_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[graph_token_start + 1:]
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and \
                   getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # Padding & alignment (same as parent)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((
                    cur_new_embed,
                    torch.zeros(
                        (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                        dtype=cur_new_embed.dtype, device=cur_new_embed.device
                    )
                ), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((
                        cur_new_label,
                        torch.full(
                            (max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                            dtype=cur_new_label.dtype, device=cur_new_label.device
                        )
                    ), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],), True,
                        dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False,
                        dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                    True, dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, aux_loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        routing_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # MoE multimodal preparation
        (
            input_ids, attention_mask, past_key_values,
            inputs_embeds, labels, aux_loss
        ) = self.prepare_inputs_labels_for_multimodal_moe(
            input_ids, attention_mask, past_key_values, labels,
            graph, graph_emb, routing_features
        )

        # Standard LLM forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)

            # Total loss = LM loss + weighted auxiliary MoE loss
            loss = lm_loss + self.aux_loss_weight * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "graph": kwargs.get("graph", None),
            "graph_emb": kwargs.get("graph_emb", None),
            "routing_features": kwargs.get("routing_features", None),
        })
        return model_inputs
