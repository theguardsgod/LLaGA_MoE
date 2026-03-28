"""
Training script for MoE-LLaGA.

Based on train.py but uses MoELlagaLlamaForCausalLM and adds routing_features
to the data pipeline. All original LLaGA training logic is preserved.
"""

import os
import sys
sys.path.append(".")
sys.path.append("./utils")
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import pandas as pd

import torch
import transformers

from utils.constants import (
    IGNORE_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_START_TOKEN,
    DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID
)
from torch.utils.data import Dataset
from llaga_trainer import LLaGATrainer

from model.language_model.moe_llaga_llama import MoELlagaLlamaForCausalLM, MoELlagaConfig

import random
from tqdm import trange
from utils import conversation as conversation_lib
from utils.utils import tokenizer_graph_token
import scipy.sparse as sp
import numpy as np


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_graph_start_end: bool = field(default=False)
    mm_use_graph_patch_token: bool = field(default=True)
    # MoE-specific
    num_experts: int = field(default=4)
    top_k: int = field(default=2)
    routing_dim: int = field(default=2432)
    aux_loss_weight: float = field(default=0.01)
    noise_std: float = field(default=1.0)


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    pretrained_embedding_type: Optional[str] = field(default='simteg')
    use_hop: Optional[int] = field(default=2)
    sample_neighbor_size: Optional[int] = field(default=-1)
    use_task: Optional[str] = field(default="nc")
    use_dataset: Optional[str] = field(default="arxiv")
    template: Optional[str] = field(default="ND")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length."},
    )
    group_by_modality_length: bool = field(default=False)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)


# ============================================================
# Conversation preprocessing (same as original train.py)
# ============================================================

def preprocess_multimodal(sources, data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_GRAPH_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
                sentence['value'] = DEFAULT_GRAPH_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
    return sources


def preprocess_v1(sources, tokenizer, has_graph=False):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_graph:
        input_ids = torch.stack(
            [tokenizer_graph_token(prompt, tokenizer, -200, return_tensors='pt') for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations, return_tensors="pt", padding="longest",
            max_length=tokenizer.model_max_length, truncation=True,
        ).input_ids

    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer, -200))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer, -200)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources, tokenizer, has_graph=False):
    return preprocess_v1(sources, tokenizer, has_graph=has_graph)


# ============================================================
# Dataset with community routing features
# ============================================================

DATASET_DIRS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}


class MoELazySupervisedGraphDataset(Dataset):
    """
    Dataset for MoE-LLaGA training.

    Extends original LazySupervisedGraphDataset by loading community partition
    info and returning routing_features per sample.
    """

    def __init__(self, tokenizer, data_args):
        super().__init__()
        self.use_dataset = data_args.use_dataset.split('-')
        self.use_hop = data_args.use_hop
        self.template = data_args.template
        self.datas = {}
        self.pretrained_embs = {}
        self.community_features = {}
        self.node_to_community = {}
        list_data_dict = []

        for d, dataset in enumerate(self.use_dataset):
            repeat = 1
            if "." in dataset:
                ds = dataset.split('.')
                repeat = int(ds[1])
                dataset = ds[0]

            if dataset not in DATASET_DIRS:
                raise ValueError(f"{dataset} not supported")
            data_dir = DATASET_DIRS[dataset]
            data_path = os.path.join(data_dir, "processed_data.pt")

            data = torch.load(data_path)
            self.datas[dataset] = data

            # Load embeddings (same as original)
            if data_args.template == "ND":
                pretrained_emb = self._load_pretrain_embedding_graph(
                    data_dir, data_args.pretrained_embedding_type
                )
                self.structure_emb = torch.load(
                    f"/localnvme/llaga/dataset/laplacian_{data_args.use_hop}_{data_args.sample_neighbor_size}.pt"
                )
            elif data_args.template == "HO":
                pretrained_emb = self._load_pretrain_embedding_hop(
                    data_dir, data_args.pretrained_embedding_type, data_args.use_hop
                )
                self.structure_emb = None
            else:
                raise ValueError(f"Unknown template: {data_args.template}")

            self.pretrained_embs[dataset] = pretrained_emb

            # Load community partition data
            comm_path = os.path.join(data_dir, "node_to_community.pt")
            feat_path = os.path.join(data_dir, "community_features.pt")
            if os.path.exists(comm_path) and os.path.exists(feat_path):
                self.node_to_community[dataset] = torch.load(comm_path)
                self.community_features[dataset] = torch.load(feat_path)
                rank0_print(f"  Loaded community data for {dataset}: "
                            f"{self.community_features[dataset].shape[0]} communities")
            else:
                rank0_print(f"  WARNING: No community data for {dataset}, "
                            f"routing_features will be zeros")
                self.node_to_community[dataset] = None
                self.community_features[dataset] = None

            # Load task data (same logic as original train.py)
            self.use_task = data_args.use_task.split('-')
            for task in self.use_task:
                task_list_data_dict = []
                if task == "nc":
                    if data_args.template == "HO":
                        dp = os.path.join(data_dir, f"sampled_2_10_train.jsonl")
                    else:
                        dp = os.path.join(data_dir,
                                          f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                    if os.path.exists(dp):
                        with open(dp, 'r') as f:
                            for line in f:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                if dataset == "products":
                                    l["conversations"][0]['value'] = (
                                        f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, "
                                        f"where nodes represent products sold in Amazon, and edges between "
                                        f"products indicate they are purchased together. We need to classify "
                                        f"the center node into 47 classes: Home & Kitchen, Health & Personal Care, "
                                        f"Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, "
                                        f"CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, "
                                        f"Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, "
                                        f"Movies & TV, Software, Video Games, Automotive, Pet Supplies, "
                                        f"Office Products, Industrial & Scientific, Musical Instruments, "
                                        f"Tools & Home Improvement, Magazine Subscriptions, Baby Products, "
                                        f"label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, "
                                        f"All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, "
                                        f"Purchase Circles, MP3 Players & Accessories, Gift Cards, "
                                        f"Office & School Supplies, Home Improvement, Camera & Photo, "
                                        f"GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, "
                                        f"Buy a Kindle, Furniture & D\u00e9cor, #508510, "
                                        f"please tell me which class the center node belongs to?"
                                    )
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError(f"Data file not found: {dp}")
                elif task == "lp":
                    if data_args.template == "HO":
                        dp = os.path.join(data_dir, f"edge_sampled_2_10_only_train.jsonl")
                    else:
                        dp = os.path.join(data_dir,
                                          f"edge_sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_only_train.jsonl")
                    if os.path.exists(dp):
                        with open(dp, 'r') as f:
                            for line in f:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                l["conversations"][0]['value'] = (
                                    f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and "
                                    f"{DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes "
                                    f"connect with each other. Please tell me whether two center nodes "
                                    f"in the subgraphs should connect to each other."
                                )
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError(f"Data file not found: {dp}")
                else:
                    raise ValueError(f"Task {task} not supported in MoE training")

                if repeat > 1:
                    base = copy.copy(task_list_data_dict)
                    for _ in range(repeat - 1):
                        task_list_data_dict += base

                rank0_print(f"Dataset {dataset} Task {task}, size {len(task_list_data_dict)}")
                list_data_dict.extend(task_list_data_dict)

        random.shuffle(list_data_dict)
        rank0_print(f"Total training samples: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def _load_pretrain_embedding_graph(self, data_dir, emb_type):
        if emb_type == "simteg":
            sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
            roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
            e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
            return torch.cat([sbert, roberta, e5], dim=-1)
        return torch.load(os.path.join(data_dir, f"{emb_type}_x.pt"))

    def _load_pretrain_embedding_hop(self, data_dir, emb_type, hop):
        if emb_type == "simteg":
            sbert = [torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))] + \
                    [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt")) for i in range(1, hop + 1)]
            roberta = [torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))] + \
                      [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt")) for i in range(1, hop + 1)]
            e5 = [torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))] + \
                 [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt")) for i in range(1, hop + 1)]
            return [torch.cat([sbert[i], roberta[i], e5[i]], dim=-1) for i in range(hop + 1)]
        return [torch.load(os.path.join(data_dir, f"{emb_type}_x.pt"))] + \
               [torch.load(os.path.join(data_dir, f"{emb_type}_{i}hop_x.pt")) for i in range(1, hop + 1)]

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            graph_token_size = len(sample.get('graphs', []))
            length_list.append(
                sum(len(conv['value'].split()) for conv in sample['conversations']) + graph_token_size
            )
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1

        sources_copy = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources_copy, self.tokenizer,
            has_graph=('graph' in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0]
            )

        if 'graph' in self.list_data_dict[i]:
            dataset = self.list_data_dict[i]["dataset"]
            if not isinstance(self.list_data_dict[i]['graph'][0], list):
                self.list_data_dict[i]['graph'] = [self.list_data_dict[i]['graph']]

            if self.template == "ND":
                graph = torch.LongTensor(self.list_data_dict[i]['graph'])
                mask = graph != DEFAULT_GRAPH_PAD_ID
                masked_graph_emb = self.pretrained_embs[dataset][graph[mask]]
                s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
                graph_emb = torch.zeros((s, n, d))
                graph_emb[mask] = masked_graph_emb
                if self.structure_emb is not None:
                    graph_emb = torch.cat(
                        [graph_emb, self.structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1
                    )
            elif self.template == "HO":
                for g in range(len(self.list_data_dict[i]['graph'])):
                    center_id = self.list_data_dict[i]['graph'][g][0]
                    self.list_data_dict[i]['graph'][g] = [center_id] * (self.use_hop + 1)
                graph = torch.LongTensor(self.list_data_dict[i]['graph'])
                center_id = graph[:, 0]
                graph_emb = torch.stack(
                    [emb[center_id] for emb in self.pretrained_embs[dataset]], dim=1
                )
            else:
                raise ValueError

            data_dict['graph'] = graph
            data_dict['graph_emb'] = graph_emb

            # --- MoE addition: routing features ---
            # Get center node ID(s) for this sample
            center_ids = graph[:, 0]  # [num_graphs]
            if self.node_to_community.get(dataset) is not None:
                comm_ids = self.node_to_community[dataset][center_ids]  # [num_graphs]
                routing_feat = self.community_features[dataset][comm_ids]  # [num_graphs, d]
            else:
                # Fallback: use zeros
                routing_feat = torch.zeros(
                    center_ids.shape[0], 2432, dtype=torch.float
                )
            data_dict['routing_features'] = routing_feat

        return data_dict


@dataclass
class MoEDataCollatorForSupervisedDataset:
    """Collate examples for MoE supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph' in instances[0]:
            graph = [instance['graph'] for instance in instances]
            graph_emb = [instance['graph_emb'] for instance in instances]
            routing_features = [instance['routing_features'] for instance in instances]
            batch['graph'] = torch.cat(graph, dim=0)
            batch['graph_emb'] = torch.cat(graph_emb, dim=0)
            batch['routing_features'] = torch.cat(routing_features, dim=0)

        return batch


def _train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if "tmp" not in training_args.output_dir and os.path.exists(training_args.output_dir):
        if bool(os.listdir(training_args.output_dir)):
            print(f"{training_args.output_dir} already exists and not empty!!!!")
            return
        print(f"{training_args.output_dir} already exists!!!!")

    # Determine mm_hidden_size
    if data_args.pretrained_embedding_type in ['sbert', 'simteg_sbert']:
        model_args.mm_hidden_size = 384
    elif data_args.pretrained_embedding_type in ["simteg_e5", "simteg_roberta", "roberta"]:
        model_args.mm_hidden_size = 1024
    elif data_args.pretrained_embedding_type in ["simteg"]:
        model_args.mm_hidden_size = 1024 * 2 + 384
    else:
        raise ValueError(f"Unknown embedding type: {data_args.pretrained_embedding_type}")

    if data_args.template == "ND":
        structure_dim = int(
            (data_args.sample_neighbor_size ** (data_args.use_hop + 1) - 1)
            / (data_args.sample_neighbor_size - 1)
        )
        model_args.mm_hidden_size += structure_dim

    # Set routing_dim to the raw SimTEG dimension (community features are always 2432)
    model_args.routing_dim = 2432

    print(f"mm_hidden_size: {model_args.mm_hidden_size}")
    print(f"MoE config: num_experts={model_args.num_experts}, top_k={model_args.top_k}")

    # Load base LLM as MoE-LLaGA
    model = MoELlagaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.float16 if training_args.fp16 else
                     (torch.bfloat16 if training_args.bf16 else torch.float32)),
    )
    model.config.use_cache = False

    # Initialize MoE modules
    model.initialize_moe_modules(model_args)

    # Also initialize the base mm_projector (for potential fallback)
    model.get_model().initialize_graph_modules(model_args)

    # Freeze LLM backbone, only train MoE projector
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.moe_projector.parameters():
            p.requires_grad = True
        # Also keep base projector trainable if it exists
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    if model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["default"]

    # Build dataset
    train_dataset = MoELazySupervisedGraphDataset(
        tokenizer=tokenizer, data_args=data_args
    )
    data_collator = MoEDataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Train
    trainer = LLaGATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # Save MoE projector weights separately
    moe_state = model.moe_projector.state_dict()
    moe_state = {f"moe_projector.{k}": v.cpu() for k, v in moe_state.items()}
    torch.save(moe_state, os.path.join(training_args.output_dir, "moe_projector.bin"))
    print(f"Saved moe_projector.bin to {training_args.output_dir}")

    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    _train()
