"""Convert LLM-Pruner checkpoint to HuggingFace format."""
import torch
import sys
import os
import json

sys.path.insert(0, "/home/23131884r/code/LLM-Pruner")

pruned = torch.load(
    "/home/23131884r/code/LLM-Pruner/prune_log/vicuna-7b-pruned-25pct-uniform/pytorch_model.bin",
    map_location="cpu",
)
model = pruned["model"]
tokenizer = pruned["tokenizer"]

# Update config to reflect pruned dimensions
model.config.num_attention_heads = 24  # was 32
model.config.num_key_value_heads = 24  # was 32
model.config.intermediate_size = 8256  # was 11008

output_dir = "/localnvme/llaga/checkpoints/vicuna-7b-pruned-25pct"
os.makedirs(output_dir, exist_ok=True)

model.half()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Saved to", output_dir)

with open(os.path.join(output_dir, "config.json")) as f:
    cfg = json.load(f)
print(
    f"Config: hidden={cfg['hidden_size']}, heads={cfg['num_attention_heads']}, "
    f"intermediate={cfg['intermediate_size']}, layers={cfg['num_hidden_layers']}"
)
