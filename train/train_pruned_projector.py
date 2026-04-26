"""
Train mm_projector on a pruned LLM loaded via torch.load.
Single-GPU, projector-only training (LLM frozen).

Usage:
    CUDA_VISIBLE_DEVICES=0 python train/train_pruned_projector.py \
        --pruned_model_path <path_to_pytorch_model.bin> \
        --dataset arxiv --task nc_kv --epochs 3 --batch_size 16
"""
import sys
sys.path.insert(0, "/home/23131884r/code/LLM-Pruner")
sys.path.append(".")
sys.path.append("./utils")

import os
import json
import argparse
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.constants import (
    IGNORE_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, GRAPH_TOKEN_INDEX,
)
from utils.conversation import conv_templates
from utils.utils import tokenizer_graph_token


DATASET_PATHS = {
    "arxiv": "/localnvme/llaga/dataset/ogbn-arxiv",
    "products": "/localnvme/llaga/dataset/ogbn-products",
    "pubmed": "/localnvme/llaga/dataset/pubmed",
    "cora": "/localnvme/llaga/dataset/cora",
}


def load_pretrain_embedding(data_dir, emb_type):
    if emb_type == "simteg":
        sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
        roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
        e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
        return torch.cat([sbert, roberta, e5], dim=-1)
    return torch.load(os.path.join(data_dir, f"{emb_type}_x.pt"))


class PrunedProjectorDataset(Dataset):
    """Dataset that returns tokenized conversations + graph embeddings."""

    def __init__(self, data_path, tokenizer, pretrained_emb, structure_emb, conv_mode="v1"):
        self.tokenizer = tokenizer
        self.pretrained_emb = pretrained_emb
        self.structure_emb = structure_emb
        self.conv_mode = conv_mode
        self.samples = []
        with open(data_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build graph embedding
        g = sample["graph"]
        if not isinstance(g[0], list):
            g = [g]
        graph = torch.LongTensor(g)
        mask = graph != DEFAULT_GRAPH_PAD_ID
        masked_emb = self.pretrained_emb[graph[mask]]
        s, n, d = graph.shape[0], graph.shape[1], masked_emb.shape[1]
        graph_emb = torch.zeros((s, n, d))
        graph_emb[mask] = masked_emb
        if self.structure_emb is not None:
            graph_emb = torch.cat(
                [graph_emb, self.structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1
            )

        # Tokenize conversation
        human_msg = sample["conversations"][0]["value"]
        gpt_msg = sample["conversations"][1]["value"]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], human_msg)
        conv.append_message(conv.roles[1], gpt_msg)
        full_prompt = conv.get_prompt()

        # Split into parts before/after graph tokens
        input_ids = tokenizer_graph_token(
            full_prompt, self.tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
        )

        # Build labels: mask everything before assistant response
        # Find the assistant turn separator
        sep = conv.sep + conv.roles[1] + ": "
        parts = full_prompt.split(sep)
        if len(parts) >= 2:
            prefix_text = parts[0] + sep
            prefix_ids = tokenizer_graph_token(
                prefix_text, self.tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
            )
            prefix_len = len(prefix_ids)
        else:
            prefix_len = 0

        labels = input_ids.clone()
        labels[:prefix_len] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "graph_emb": graph_emb,  # [num_graphs, 111, 2543]
        }


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    graph_embs = []

    for i, b in enumerate(batch):
        l = b["input_ids"].shape[0]
        input_ids[i, :l] = b["input_ids"]
        labels[i, :l] = b["labels"]
        attention_mask[i, :l] = 1
        graph_embs.append(b["graph_emb"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "graph_embs": graph_embs,
    }


def prepare_inputs_labels(model, mm_projector, embed_tokens, batch, device):
    """Replace GRAPH_TOKEN_INDEX in input_ids with projected graph embeddings."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    graph_embs = batch["graph_embs"]

    B, L = input_ids.shape
    new_inputs_embeds = []
    new_labels = []

    for i in range(B):
        cur_ids = input_ids[i]
        cur_labels = labels[i]
        cur_graph_emb = graph_embs[i].float().to(device)  # [num_graphs, 111, 2543] FP32

        # Project graphs in FP32, clamp, then cast to FP16 for LLM
        projected = mm_projector(cur_graph_emb)
        projected = projected.clamp(-65504, 65504).half()  # [num_graphs, 111, 4096]

        # Find GRAPH_TOKEN_INDEX positions and replace
        graph_positions = (cur_ids == GRAPH_TOKEN_INDEX).nonzero(as_tuple=True)[0]

        if len(graph_positions) == 0:
            # No graph tokens
            emb = embed_tokens(cur_ids.unsqueeze(0)).squeeze(0)
            new_inputs_embeds.append(emb)
            new_labels.append(cur_labels)
            continue

        parts_emb = []
        parts_labels = []
        prev = 0
        graph_idx = 0

        for pos in graph_positions:
            pos = pos.item()
            if pos > prev:
                text_ids = cur_ids[prev:pos]
                parts_emb.append(embed_tokens(text_ids.unsqueeze(0)).squeeze(0))
                parts_labels.append(cur_labels[prev:pos])

            if graph_idx < projected.shape[0]:
                parts_emb.append(projected[graph_idx])
                parts_labels.append(
                    torch.full((projected.shape[1],), IGNORE_INDEX, device=device, dtype=torch.long)
                )
                graph_idx += 1
            prev = pos + 1

        if prev < L:
            text_ids = cur_ids[prev:]
            parts_emb.append(embed_tokens(text_ids.unsqueeze(0)).squeeze(0))
            parts_labels.append(cur_labels[prev:])

        new_inputs_embeds.append(torch.cat(parts_emb, dim=0))
        new_labels.append(torch.cat(parts_labels, dim=0))

    # Pad to same length
    max_new_len = max(e.shape[0] for e in new_inputs_embeds)
    padded_embeds = torch.zeros(B, max_new_len, new_inputs_embeds[0].shape[1],
                                dtype=new_inputs_embeds[0].dtype, device=device)
    padded_labels = torch.full((B, max_new_len), IGNORE_INDEX, dtype=torch.long, device=device)
    padded_mask = torch.zeros(B, max_new_len, dtype=torch.long, device=device)

    for i in range(B):
        l = new_inputs_embeds[i].shape[0]
        padded_embeds[i, :l] = new_inputs_embeds[i]
        padded_labels[i, :l] = new_labels[i]
        padded_mask[i, :l] = 1

    return padded_embeds, padded_labels, padded_mask


def main(args):
    random.seed(42)
    torch.manual_seed(42)

    # Load pruned model (optionally with recovered weights)
    print(f"Loading pruned model from {args.pruned_model_path}")
    pruned = torch.load(args.pruned_model_path, map_location="cpu")
    model = pruned["model"]
    tokenizer = pruned["tokenizer"]
    if args.recovered_model_path:
        print(f"Loading recovered weights from {args.recovered_model_path}")
        recovered = torch.load(args.recovered_model_path, map_location="cpu")
        model.load_state_dict(recovered["state_dict"])
    model.half().cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable()

    hidden_size = model.config.hidden_size
    print(f"Pruned model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, hidden={hidden_size}")

    # Compute mm_hidden_size
    data_dir = DATASET_PATHS[args.dataset]
    pretrained_emb = load_pretrain_embedding(data_dir, args.emb_type)
    structure_emb = torch.load(
        f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
    )

    emb_dim = pretrained_emb.shape[1]
    struct_dim = structure_emb.shape[1]
    mm_hidden_size = emb_dim + struct_dim
    print(f"mm_hidden_size: {mm_hidden_size} (emb={emb_dim} + struct={struct_dim})")

    # Create projector — keep in FP32 for numerical stability
    if args.proj_type == "mlp":
        mm_projector = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        ).cuda()
    else:
        mm_projector = nn.Linear(mm_hidden_size, hidden_size).cuda()
    mm_projector.train()
    print(f"Projector ({args.proj_type}): {sum(p.numel() for p in mm_projector.parameters())/1e6:.1f}M params")

    # Load data
    if args.task == "nc_kv":
        data_path = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_train_kvcache.jsonl")
    else:
        data_path = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_train.jsonl")
    print(f"Loading data from {data_path}")

    dataset = PrunedProjectorDataset(data_path, tokenizer, pretrained_emb, structure_emb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(mm_projector.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * 0.03)

    # Cosine scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    embed_tokens = model.model.embed_tokens

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    grad_accum = args.grad_accum
    print(f"\nTraining: {args.epochs} epochs, {len(loader)} steps/epoch, total {total_steps} steps")
    print(f"Batch size: {args.batch_size}, grad_accum: {grad_accum}, effective: {args.batch_size * grad_accum}")
    print(f"Output: {output_dir}\n")

    global_step = 0
    for epoch in range(args.epochs):
        mm_projector.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for step_i, batch in enumerate(pbar):
            inputs_embeds, labels, attn_mask = prepare_inputs_labels(
                model, mm_projector, embed_tokens, batch, "cuda"
            )

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss.float() / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at step {step_i}, skipping")
                optimizer.zero_grad()
                continue

            loss.backward()

            if (step_i + 1) % grad_accum == 0 or (step_i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(mm_projector.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        save_path = os.path.join(output_dir, "mm_projector.bin")
        torch.save(mm_projector.state_dict(), save_path)
        print(f"Saved projector to {save_path}")

    # Save config
    config = model.config.to_dict()
    config["mm_hidden_size"] = mm_hidden_size
    config["mm_projector_type"] = args.proj_type
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone! Projector saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned_model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--task", type=str, default="nc_kv")
    parser.add_argument("--emb_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--proj_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--recovered_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="/localnvme/llaga/checkpoints/arxiv/pruned-25pct-projector_nc_kv")
    args = parser.parse_args()
    main(args)
