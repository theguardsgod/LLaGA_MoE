"""Quick latency test for pruned model vs original, no LLaGA projector needed."""
import sys
sys.path.insert(0, "/home/23131884r/code/LLM-Pruner")
import torch
import time
from transformers import AutoModelForCausalLM, LlamaTokenizer


def load_pruned(path):
    pruned = torch.load(path, map_location="cpu")
    return pruned["tokenizer"], pruned["model"]


def benchmark(model, tokenizer, device, num_runs=20, prompt="The meaning of life is"):
    model.half().to(device).eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            model.generate(input_ids, max_new_tokens=32, do_sample=False)

    torch.cuda.synchronize()
    times = []
    with torch.inference_mode():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.time()
            model.generate(input_ids, max_new_tokens=32, do_sample=False)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    avg = sum(times) / len(times)
    params = sum(p.numel() for p in model.parameters()) / 1e9
    mem = torch.cuda.max_memory_allocated() / 1e9
    return avg, params, mem


if __name__ == "__main__":
    device = "cuda"

    # Pruned model
    print("Loading pruned model...")
    tok_p, model_p = load_pruned(
        "/home/23131884r/code/LLM-Pruner/prune_log/vicuna-7b-pruned-25pct-uniform/pytorch_model.bin"
    )
    avg_p, params_p, mem_p = benchmark(model_p, tok_p, device)
    print(f"Pruned:   {params_p:.2f}B params, {avg_p*1000:.0f}ms/gen, {mem_p:.1f}GB peak")

    del model_p
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Original model
    from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM
    print("\nLoading original model...")
    tok_o = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5-16k", cache_dir="../../checkpoint")
    model_o = LlamaForCausalLM.from_pretrained(
        "lmsys/vicuna-7b-v1.5-16k", cache_dir="../../checkpoint", low_cpu_mem_usage=True
    )
    avg_o, params_o, mem_o = benchmark(model_o, tok_o, device)
    print(f"Original: {params_o:.2f}B params, {avg_o*1000:.0f}ms/gen, {mem_o:.1f}GB peak")

    print(f"\nSpeedup: {avg_o/avg_p:.2f}x")
    print(f"Memory reduction: {(1 - mem_p/mem_o)*100:.0f}%")
    print(f"Param reduction: {(1 - params_p/params_o)*100:.0f}%")
