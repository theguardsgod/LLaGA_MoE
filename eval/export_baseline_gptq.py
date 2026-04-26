import sys
sys.path.append("./")

import argparse
import os

from transformers import AutoTokenizer

from eval.gptq_utils import (
    build_gptq_calibration_examples,
    build_gptq_quantize_config,
)


def export_gptq(args):
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "AutoGPTQ is required for exporting a GPTQ checkpoint. Install `auto-gptq` in the target environment."
        ) from exc

    model_base = os.path.expanduser(args.model_base)
    output_dir = os.path.expanduser(args.output_dir)
    calib_prompt_file = os.path.expanduser(args.calib_prompt_file)

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, cache_dir=args.cache_dir)
    quantize_config = build_gptq_quantize_config(
        bits=args.bits,
        group_size=args.group_size,
        desc_act=not args.disable_desc_act,
        sym=not args.disable_sym,
        damp_percent=args.damp_percent,
        model_file_base_name=args.model_file_base_name,
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        model_base,
        quantize_config=quantize_config,
        trust_remote_code=False,
    )

    examples = build_gptq_calibration_examples(
        tokenizer,
        calib_prompt_file,
        limit=args.calib_limit,
        max_length=args.max_length,
    )

    print(f"Quantizing {model_base} into {output_dir}")
    print(
        "Calibration samples: "
        f"{len(examples)} | config={{'bits': {args.bits}, 'group_size': {args.group_size}, "
        f"'desc_act': {not args.disable_desc_act}, 'sym': {not args.disable_sym}}}"
    )

    model.quantize(
        examples,
        batch_size=args.batch_size,
        use_triton=args.use_triton,
        use_cuda_fp16=not args.disable_cuda_fp16,
        autotune_warmup_after_quantized=args.autotune_warmup_after_quantized,
        cache_examples_on_gpu=not args.disable_cache_examples_on_gpu,
    )
    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir, use_safetensors=not args.disable_safetensors)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved GPTQ checkpoint to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--calib_prompt_file", type=str, required=True)
    parser.add_argument("--calib_limit", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--damp_percent", type=float, default=0.01)
    parser.add_argument("--model_file_base_name", type=str, default=None)
    parser.add_argument("--disable_desc_act", action="store_true", default=False)
    parser.add_argument("--disable_sym", action="store_true", default=False)
    parser.add_argument("--use_triton", action="store_true", default=False)
    parser.add_argument("--disable_cuda_fp16", action="store_true", default=False)
    parser.add_argument("--autotune_warmup_after_quantized", action="store_true", default=False)
    parser.add_argument("--disable_cache_examples_on_gpu", action="store_true", default=False)
    parser.add_argument("--disable_safetensors", action="store_true", default=False)
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    export_gptq(parser.parse_args())
