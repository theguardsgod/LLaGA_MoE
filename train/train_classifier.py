import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

from model.language_model.llaga_llama_classifier import (
    LlagaClassificationConfig,
    LlagaLlamaForSequenceClassification,
)
from utils import conversation as conversation_lib
from utils.constants import DEFAULT_GRAPH_PAD_ID, GRAPH_TOKEN_INDEX
from utils.utils import tokenizer_graph_token


DATASET_PATHS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5-16k")
    version: Optional[str] = field(default="v1")
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_graph_start_end: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    classifier_dropout: float = field(default=0.0)
    classifier_pooling: str = field(default="graph_mean")


@dataclass
class DataArguments:
    dataset: str = field(default="arxiv")
    pretrained_embedding_type: Optional[str] = field(default="simteg")
    use_hop: int = field(default=2)
    sample_neighbor_size: int = field(default=10)
    template: str = field(default="ND")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=4096)
    freeze_backbone: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)


def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):
    if pretrained_embedding_type == "simteg":
        simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        return torch.cat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
    return torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))


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


class LazySupervisedGraphClassificationDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super().__init__()
        if data_args.dataset not in DATASET_PATHS:
            raise ValueError(f"Unsupported dataset: {data_args.dataset}")

        self.tokenizer = tokenizer
        self.template = data_args.template
        self.use_hop = data_args.use_hop
        self.data_dir = DATASET_PATHS[data_args.dataset]
        self.data = torch.load(os.path.join(self.data_dir, "processed_data.pt"), weights_only=False)
        if self.template == "ND":
            self.pretrained_emb = load_pretrain_embedding_graph(self.data_dir, data_args.pretrained_embedding_type)
            self.structure_emb = torch.load(
                f"/localnvme/llaga/dataset/laplacian_{data_args.use_hop}_{data_args.sample_neighbor_size}.pt"
            )
            prompt_path = os.path.join(
                self.data_dir, f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl"
            )
        elif self.template == "HO":
            self.pretrained_emb = load_pretrain_embedding_hop(
                self.data_dir, data_args.pretrained_embedding_type, data_args.use_hop
            )
            self.structure_emb = None
            prompt_path = os.path.join(self.data_dir, "sampled_2_10_train.jsonl")
        else:
            raise ValueError(f"Unsupported template: {self.template}")

        self.samples = []
        with open(prompt_path, "r") as file:
            for line in file:
                sample = json.loads(line)
                node_id = sample["id"]
                self.samples.append(
                    {
                        "id": node_id,
                        "dataset": data_args.dataset,
                        "graph": sample["graph"],
                        "user_prompt": sample["conversations"][0]["value"],
                        "label": int(self.data.y[node_id]),
                    }
                )

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _build_prompt(self, user_prompt):
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def __getitem__(self, index):
        sample = copy.deepcopy(self.samples[index])
        prompt = self._build_prompt(sample["user_prompt"])
        input_ids = tokenizer_graph_token(prompt, self.tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt")

        graph = sample["graph"]
        if not isinstance(graph[0], list):
            graph = [graph]

        if self.template == "ND":
            graph = torch.LongTensor(graph)
            mask = graph != DEFAULT_GRAPH_PAD_ID
            masked_graph_emb = self.pretrained_emb[graph[mask]]
            sample_count, graph_len, hidden_dim = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
            graph_emb = torch.zeros((sample_count, graph_len, hidden_dim))
            graph_emb[mask] = masked_graph_emb
            graph_emb = torch.cat([graph_emb, self.structure_emb.unsqueeze(0).expand(sample_count, -1, -1)], dim=-1)
        else:
            for idx in range(len(graph)):
                center_id = graph[idx][0]
                graph[idx] = [center_id] * (self.use_hop + 1)
            graph = torch.LongTensor(graph)
            center_id = graph[:, 0]
            graph_emb = torch.stack([emb[center_id] for emb in self.pretrained_emb], dim=1)

        return {
            "input_ids": input_ids,
            "graph": graph,
            "graph_emb": graph_emb,
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }


@dataclass
class DataCollatorForGraphClassification:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = torch.stack([instance["labels"] for instance in instances], dim=0)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        batch = {
            "input_ids": input_ids[:, : self.tokenizer.model_max_length],
            "labels": labels,
            "attention_mask": input_ids[:, : self.tokenizer.model_max_length].ne(self.tokenizer.pad_token_id),
            "graph": torch.cat([instance["graph"] for instance in instances], dim=0),
            "graph_emb": torch.cat([instance["graph_emb"] for instance in instances], dim=0),
        }
        return batch


def _resolve_mm_projector_path(path):
    if path is None:
        return None
    if os.path.isdir(path):
        candidate = os.path.join(path, "mm_projector.bin")
        if os.path.exists(candidate):
            return candidate
    return path


def _infer_mm_hidden_size(data_args):
    if data_args.pretrained_embedding_type in ["sbert", "simteg_sbert"]:
        mm_hidden_size = 384
    elif data_args.pretrained_embedding_type in ["simteg_e5", "simteg_roberta", "roberta"]:
        mm_hidden_size = 1024
    elif data_args.pretrained_embedding_type == "simteg":
        mm_hidden_size = 2432
    else:
        raise ValueError(f"Unsupported embedding type: {data_args.pretrained_embedding_type}")
    if data_args.template == "ND":
        structure_dim = int(
            (data_args.sample_neighbor_size ** (data_args.use_hop + 1) - 1) / (data_args.sample_neighbor_size - 1)
        )
        mm_hidden_size += structure_dim
    return mm_hidden_size


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data = torch.load(os.path.join(DATASET_PATHS[data_args.dataset], "processed_data.pt"), weights_only=False)
    config = LlagaClassificationConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=len(data.label_texts),
    )
    config.mm_hidden_size = _infer_mm_hidden_size(data_args)
    config.mm_projector_type = model_args.mm_projector_type
    config.use_hop = data_args.use_hop
    config.sample_neighbor_size = data_args.sample_neighbor_size
    config.classifier_dropout = model_args.classifier_dropout
    config.classifier_pooling = model_args.classifier_pooling

    model = LlagaLlamaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize({"pad_token": "[PAD]"}, tokenizer, model)

    model_args.mm_hidden_size = config.mm_hidden_size
    model_args.pretrain_mm_mlp_adapter = _resolve_mm_projector_path(model_args.pretrain_mm_mlp_adapter)
    model.get_model().initialize_graph_modules(model_args=model_args, fsdp=training_args.fsdp)
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    model.initialize_graph_tokenizer(model_args, tokenizer)

    if training_args.freeze_backbone:
        model.model.requires_grad_(False)
        model.classifier.requires_grad_(True)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = not training_args.freeze_mm_mlp_adapter
    elif training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    train_dataset = LazySupervisedGraphClassificationDataset(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForGraphClassification(tokenizer=tokenizer),
    )

    if list(os.scandir(training_args.output_dir)) if os.path.isdir(training_args.output_dir) else False:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    random.seed(0)
    train()
