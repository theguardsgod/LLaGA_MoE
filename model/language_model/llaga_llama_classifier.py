from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    LlamaConfig,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from ..llaga_arch import LlagaMetaModel, LlagaMetaForCausalLM


@dataclass
class LlagaSequenceClassifierOutputWithPast(SequenceClassifierOutputWithPast):
    pooled_embeddings: Optional[torch.FloatTensor] = None


class LlagaClassificationConfig(LlamaConfig):
    model_type = "llaga_cls"


class LlagaLlamaClassificationModel(LlagaMetaModel, LlamaModel):
    config_class = LlagaClassificationConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class LlagaLlamaForSequenceClassification(LlamaPreTrainedModel, LlagaMetaForCausalLM):
    config_class = LlagaClassificationConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlagaLlamaClassificationModel(config)
        dropout_prob = getattr(config, "classifier_dropout", 0.0)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def _pool_last_token(self, hidden_states, attention_mask):
        if attention_mask is None:
            return hidden_states[:, -1, :]
        sequence_lengths = attention_mask.long().sum(dim=-1) - 1
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, sequence_lengths]

    def _pool_graph_tokens(self, hidden_states, graph_position_mask, attention_mask):
        if graph_position_mask is None or not graph_position_mask.any():
            return self._pool_last_token(hidden_states, attention_mask)
        graph_position_mask = graph_position_mask.unsqueeze(-1).to(hidden_states.device)
        masked_hidden = hidden_states * graph_position_mask
        token_count = graph_position_mask.sum(dim=1).clamp_min(1)
        return masked_hidden.sum(dim=1) / token_count

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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlagaSequenceClassifierOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooling_type = getattr(self.config, "classifier_pooling", "graph_mean")
        graph_position_mask = None
        if pooling_type == "last":
            input_ids, attention_mask, past_key_values, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                None,
                graph,
                graph_emb,
            )
        else:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _,
                graph_position_mask,
            ) = self.prepare_inputs_labels_for_multimodal_with_graph_mask(
                input_ids,
                attention_mask,
                past_key_values,
                None,
                graph,
                graph_emb,
            )

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
        if pooling_type == "last":
            pooled_embeddings = self._pool_last_token(hidden_states, attention_mask)
        elif pooling_type == "graph_mean":
            pooled_embeddings = self._pool_graph_tokens(hidden_states, graph_position_mask, attention_mask)
        else:
            raise ValueError(f"Unsupported classifier_pooling: {pooling_type}")
        logits = self.classifier(self.dropout(pooled_embeddings))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return LlagaSequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            pooled_embeddings=pooled_embeddings,
        )


AutoConfig.register("llaga_cls", LlagaClassificationConfig)
AutoModelForSequenceClassification.register(LlagaClassificationConfig, LlagaLlamaForSequenceClassification)
