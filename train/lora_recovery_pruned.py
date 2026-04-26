"""
LoRA recovery training for pruned LLM.
Finetunes the pruned model with LoRA on Alpaca data to restore instruction-following ability.

Single-GPU:
    CUDA_VISIBLE_DEVICES=0 python train/lora_recovery_pruned.py \
        --pruned_model_path <path> --output_dir <dir> --epochs 1

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train/lora_recovery_pruned.py \
        --pruned_model_path <path> --output_dir <dir> --epochs 1
"""
import sys
sys.path.insert(0, "/home/23131884r/code/LLM-Pruner")
sys.path.append(".")

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


class AlpacaDataset(Dataset):
    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
    PROMPT_TEMPLATE_NO_INPUT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )

    def __init__(self, tokenizer, max_len=256, split="train"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        data = load_dataset("yahma/alpaca-cleaned", split=split)
        self.samples = list(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if s.get("input", "").strip():
            prompt = self.PROMPT_TEMPLATE.format(instruction=s["instruction"], input=s["input"])
        else:
            prompt = self.PROMPT_TEMPLATE_NO_INPUT.format(instruction=s["instruction"])

        full_text = prompt + s["output"] + self.tokenizer.eos_token
        encoded = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)

        prompt_encoded = self.tokenizer(prompt, truncation=True, max_length=self.max_len)
        prompt_len = len(prompt_encoded.input_ids)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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
    torch.manual_seed(42)

    if is_main:
        print(f"Loading pruned model from {args.pruned_model_path}")
    pruned = torch.load(args.pruned_model_path, map_location="cpu")
    model = pruned["model"]
    tokenizer = pruned["tokenizer"]

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model.half().to(device)
    model.train()
    model.gradient_checkpointing_enable()

    if is_main:
        params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Model: {params:.2f}B params, {world_size} GPUs")

    # Apply LoRA
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

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Dataset
    if is_main:
        print("Loading Alpaca dataset...")
    dataset = AlpacaDataset(tokenizer, max_len=args.cutoff_len)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=4, pin_memory=True)

    # With DDP, effective batch = micro_batch * grad_accum * world_size
    grad_accum = args.batch_size // (args.micro_batch_size * world_size)
    grad_accum = max(grad_accum, 1)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * 0.03)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)
    if is_main:
        eff_batch = args.micro_batch_size * grad_accum * world_size
        print(f"\nTraining: {args.epochs} epochs, {len(loader)} steps/epoch")
        print(f"Batch: {args.micro_batch_size} x {grad_accum} x {world_size}gpu = {eff_batch}")
        print(f"LR: {args.lr}, Output: {args.output_dir}\n", flush=True)

    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0
        valid_steps = 0
        optimizer.zero_grad()

        for step_i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.float() / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            if (step_i + 1) % grad_accum == 0 or (step_i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            valid_steps += 1
            global_step += 1

            if is_main and (step_i + 1) % 200 == 0:
                print(f"  step {step_i+1}/{len(loader)}, loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = epoch_loss / max(valid_steps, 1)
        if is_main:
            print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}", flush=True)

    # Save on main process only
    if is_main:
        raw_model = model.module if ddp else model
        print("Merging LoRA weights on GPU...")
        raw_model = raw_model.merge_and_unload()
        raw_model.half()

        save_path = os.path.join(args.output_dir, "pytorch_model.bin")
        print("Moving to CPU and saving...")
        state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
        torch.save({"state_dict": state_dict, "config": raw_model.config}, save_path)
        del state_dict
        print(f"Saved merged state_dict to {save_path}")

        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved tokenizer to {args.output_dir}")
        print("Done!", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruned_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/localnvme/llaga/checkpoints/pruned-25pct-lora-recovery")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--cutoff_len", type=int, default=256)
    args = parser.parse_args()
    main(args)
