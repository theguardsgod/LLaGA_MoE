"""
Joint LoRA + projector training on pruned LLM with graph data.
LoRA recovers the pruned model's ability while projector learns graph→LLM mapping.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train/train_pruned_lora_projector.py \
        --pruned_model_path <path> --dataset arxiv --task nc_kv --epochs 1

    torchrun --nproc_per_node=4 train/train_pruned_lora_projector.py \
        --pruned_model_path <path> --dataset arxiv --task nc_kv --epochs 1
"""
import sys
sys.path.insert(0, "/home/23131884r/code/LLM-Pruner")
sys.path.append(".")
sys.path.append("./utils")

import os
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from peft import LoraConfig, get_peft_model, TaskType

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
    def __init__(self, data_path, tokenizer, pretrained_emb, structure_emb,
                 conv_mode="v1", node_only=False):
        self.tokenizer = tokenizer
        self.pretrained_emb = pretrained_emb
        self.structure_emb = structure_emb
        self.conv_mode = conv_mode
        self.node_only = node_only
        self.samples = []
        with open(data_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.node_only:
            # Node-only: just center node embedding, no structure
            g = sample["graph"]
            center_idx = g[0] if not isinstance(g[0], list) else g[0][0]
            if center_idx == DEFAULT_GRAPH_PAD_ID:
                d = self.pretrained_emb.shape[1]
                graph_emb = torch.zeros((1, 1, d))
            else:
                graph_emb = self.pretrained_emb[center_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, 2432]
        else:
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

        human_msg = sample["conversations"][0]["value"]
        gpt_msg = sample["conversations"][1]["value"]
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], human_msg)
        conv.append_message(conv.roles[1], gpt_msg)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(
            full_prompt, self.tokenizer, GRAPH_TOKEN_INDEX, return_tensors="pt"
        )

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

        return {"input_ids": input_ids, "labels": labels, "graph_emb": graph_emb}


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
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": attention_mask, "graph_embs": graph_embs}


def prepare_inputs_labels(mm_projector, embed_tokens, batch, device):
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
        cur_graph_emb = graph_embs[i].float().to(device)

        projected = mm_projector(cur_graph_emb)
        projected = projected.clamp(-65504, 65504).half()

        graph_positions = (cur_ids == GRAPH_TOKEN_INDEX).nonzero(as_tuple=True)[0]

        if len(graph_positions) == 0:
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
    # DDP setup
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = "cuda"

    is_main = rank == 0
    random.seed(42)
    torch.manual_seed(42)

    # Load pruned model
    if is_main:
        print(f"Loading pruned model from {args.pruned_model_path}")
    pruned = torch.load(args.pruned_model_path, map_location="cpu")
    model = pruned["model"]
    tokenizer = pruned["tokenizer"]

    if args.recovered_model_path:
        if is_main:
            print(f"Loading recovered weights from {args.recovered_model_path}")
        recovered = torch.load(args.recovered_model_path, map_location="cpu")
        model.load_state_dict(recovered["state_dict"])

    model = model.half().to(device)

    hidden_size = model.config.hidden_size
    if is_main:
        print(f"Pruned model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, hidden={hidden_size}")

    use_lora = not args.no_lora
    if use_lora:
        model.train()
        model.gradient_checkpointing_enable()
        # Apply LoRA to the LLM
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if is_main:
            model.print_trainable_parameters()
    else:
        # Freeze LLM entirely, only train projector
        # Keep train mode so gradient checkpointing works
        for p in model.parameters():
            p.requires_grad_(False)
        model.train()
        model.gradient_checkpointing_enable()
        if is_main:
            print("LLM frozen (--no_lora), projector-only training")

    # Projector in FP32
    data_dir = DATASET_PATHS[args.dataset]
    pretrained_emb = load_pretrain_embedding(data_dir, args.emb_type)
    if args.node_only:
        structure_emb = None
        mm_hidden_size = pretrained_emb.shape[1]  # 2432 for simteg
    else:
        structure_emb = torch.load(
            f"/localnvme/llaga/dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt"
        )
        mm_hidden_size = pretrained_emb.shape[1] + structure_emb.shape[1]

    if args.proj_type == "mlp":
        mm_projector = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        ).to(device)
    else:
        mm_projector = nn.Linear(mm_hidden_size, hidden_size).to(device)
    mm_projector.train()
    if is_main:
        print(f"Projector ({args.proj_type}): {sum(p.numel() for p in mm_projector.parameters())/1e6:.1f}M params")
        print(f"mm_hidden_size: {mm_hidden_size}")

    # Wrap model with DDP (only if LoRA — frozen models don't need DDP)
    if ddp and use_lora:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Data
    if args.task == "nc_kv":
        data_path = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_train_kvcache.jsonl")
    else:
        data_path = os.path.join(data_dir, f"sampled_{args.use_hop}_{args.sample_neighbor_size}_train.jsonl")
    if is_main:
        print(f"Loading data from {data_path}")

    dataset = PrunedProjectorDataset(data_path, tokenizer, pretrained_emb, structure_emb,
                                     node_only=args.node_only)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    grad_accum = args.grad_accum
    if ddp:
        grad_accum = max(grad_accum // world_size, 1)

    # Optimizer param groups
    if use_lora:
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
        all_params = [
            {"params": lora_params, "lr": args.lora_lr},
            {"params": mm_projector.parameters(), "lr": args.proj_lr},
        ]
    else:
        lora_params = []
        all_params = [{"params": mm_projector.parameters(), "lr": args.proj_lr}]
    optimizer = torch.optim.AdamW(all_params, weight_decay=0.0)

    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * 0.03)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    lr_lambdas = [lr_lambda, lr_lambda] if use_lora else [lr_lambda]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if is_main:
        eff_batch = args.batch_size * grad_accum * world_size
        print(f"\nTraining: {args.epochs} epochs, {len(loader)} steps/epoch")
        print(f"Batch: {args.batch_size} x {grad_accum} x {world_size}gpu = {eff_batch}")
        if use_lora:
            print(f"LoRA LR: {args.lora_lr}, Projector LR: {args.proj_lr}")
        else:
            print(f"Projector LR: {args.proj_lr} (LLM frozen)")
        print(f"Output: {output_dir}\n", flush=True)

    # Get embed_tokens from the raw model (unwrap DDP and PEFT)
    if use_lora:
        raw_model = model.module if ddp else model
        embed_tokens = raw_model.base_model.model.model.embed_tokens
    else:
        raw_model = model
        embed_tokens = model.model.embed_tokens

    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0
        valid_steps = 0
        optimizer.zero_grad()

        for step_i, batch in enumerate(loader):
            inputs_embeds, labels, attn_mask = prepare_inputs_labels(
                mm_projector, embed_tokens, batch, device
            )

            outputs = raw_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs.loss.float() / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main and step_i % 100 == 0:
                    print(f"WARNING: NaN/Inf loss at step {step_i}, skipping", flush=True)
                optimizer.zero_grad()
                continue

            loss.backward()

            if (step_i + 1) % grad_accum == 0 or (step_i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    list(lora_params) + list(mm_projector.parameters()), 0.5
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            valid_steps += 1
            global_step += 1

            if is_main and (step_i + 1) % 200 == 0:
                lrs = scheduler.get_last_lr()
                if use_lora:
                    lr_str = f"lora_lr={lrs[0]:.2e}, proj_lr={lrs[1]:.2e}"
                else:
                    lr_str = f"proj_lr={lrs[0]:.2e}"
                print(f"  step {step_i+1}/{len(loader)}, loss={loss.item():.4f}, {lr_str}",
                      flush=True)

        avg_loss = epoch_loss / max(valid_steps, 1)
        if is_main:
            print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}", flush=True)

            # Save projector
            proj_path = os.path.join(output_dir, "mm_projector.bin")
            torch.save(mm_projector.state_dict(), proj_path)
            print(f"Saved projector to {proj_path}", flush=True)

    # Save on main process
    if is_main:
        if use_lora:
            print("Merging LoRA weights...", flush=True)
            merged = raw_model.merge_and_unload()
            merged.half()

            model_path = os.path.join(output_dir, "pytorch_model.bin")
            print("Saving merged model...", flush=True)
            state_dict = {k: v.cpu() for k, v in merged.state_dict().items()}
            torch.save({"state_dict": state_dict, "config": merged.config}, model_path)
            del state_dict
            cfg_model = merged.config
        else:
            cfg_model = model.config

        config = cfg_model.to_dict()
        config["mm_hidden_size"] = mm_hidden_size
        config["mm_projector_type"] = args.proj_type
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        tokenizer.save_pretrained(output_dir)
        print(f"\nDone! Saved to {output_dir}", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned_model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--task", type=str, default="nc_kv")
    parser.add_argument("--emb_type", type=str, default="simteg")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_lr", type=float, default=2e-5)
    parser.add_argument("--proj_lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--proj_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--no_lora", action="store_true", help="Freeze LLM, only train projector")
    parser.add_argument("--node_only", action="store_true",
                        help="Node-only mode: only center node embedding, no structure")
    parser.add_argument("--recovered_model_path", type=str, default=None,
                        help="Path to LoRA-recovered pytorch_model.bin")
    parser.add_argument("--output_dir", type=str,
                        default="/localnvme/llaga/checkpoints/arxiv/pruned-25pct-lora-projector")
    args = parser.parse_args()
    main(args)
