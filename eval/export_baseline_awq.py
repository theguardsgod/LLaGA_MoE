import sys
sys.path.append("./")

import argparse
import os

from transformers import AutoTokenizer

from eval.awq_utils import load_calibration_texts
from utils.constants import DEFAULT_GRAPH_TOKEN


def export_awq(args):
    try:
        from awq import AutoAWQForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "AutoAWQ is required for exporting an AWQ checkpoint. Install `autoawq` in the target environment."
        ) from exc

    model_base = os.path.expanduser(args.model_base)
    output_dir = os.path.expanduser(args.output_dir)
    calib_prompt_file = os.path.expanduser(args.calib_prompt_file)

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, cache_dir=args.cache_dir)
    model = AutoAWQForCausalLM.from_pretrained(model_base, safetensors=False)

    calib_data = [text.replace(DEFAULT_GRAPH_TOKEN, "") for text in load_calibration_texts(calib_prompt_file, args.calib_limit)]
    quant_config = {
        "zero_point": not args.disable_zero_point,
        "q_group_size": args.group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }

    print(f"Quantizing {model_base} into {output_dir}")
    print(f"Calibration samples: {len(calib_data)} | config={quant_config}")

    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved AWQ checkpoint to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--calib_prompt_file", type=str, required=True)
    parser.add_argument("--calib_limit", type=int, default=128)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--version", type=str, default="GEMM")
    parser.add_argument("--disable_zero_point", action="store_true", default=False)
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    export_awq(parser.parse_args())
