import os

import torch
from transformers import AutoTokenizer

from eval.awq_utils import load_calibration_texts
from utils.constants import DEFAULT_GRAPH_TOKEN


def add_gptq_args(parser):
    parser.add_argument("--gptq_model_path", type=str, required=True)
    parser.add_argument("--gptq_tokenizer_path", type=str, default=None)
    parser.add_argument("--gptq_model_basename", type=str, default=None)
    parser.add_argument("--gptq_use_triton", action="store_true", default=False)
    parser.add_argument("--gptq_enable_exllama", dest="gptq_disable_exllama", action="store_false")
    parser.add_argument("--gptq_enable_exllamav2", dest="gptq_disable_exllamav2", action="store_false")
    parser.add_argument("--gptq_use_marlin", action="store_true", default=False)
    parser.set_defaults(gptq_disable_exllama=True, gptq_disable_exllamav2=True)
    return parser


def load_gptq_language_model(
    gptq_model_path,
    gptq_tokenizer_path=None,
    gptq_model_basename=None,
    cache_dir=None,
    gptq_use_triton=False,
    gptq_disable_exllama=True,
    gptq_disable_exllamav2=True,
    gptq_use_marlin=False,
):
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "AutoGPTQ is required for loading a GPTQ checkpoint. Install it in the target environment."
        ) from exc

    tokenizer_path = gptq_tokenizer_path or gptq_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoGPTQForCausalLM.from_quantized(
        os.path.expanduser(gptq_model_path),
        model_basename=gptq_model_basename,
        device=device,
        use_triton=gptq_use_triton,
        low_cpu_mem_usage=True,
        use_cuda_fp16=True,
        disable_exllama=gptq_disable_exllama,
        disable_exllamav2=gptq_disable_exllamav2,
        use_marlin=gptq_use_marlin,
        trust_remote_code=False,
    )
    return tokenizer, model


def build_gptq_quantize_config(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
    damp_percent=0.01,
    model_file_base_name=None,
):
    try:
        from auto_gptq import BaseQuantizeConfig
    except Exception as exc:
        raise RuntimeError(
            "AutoGPTQ is required for exporting a GPTQ checkpoint. Install it in the target environment."
        ) from exc

    return BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        damp_percent=damp_percent,
        model_file_base_name=model_file_base_name,
    )


def build_gptq_calibration_examples(tokenizer, calib_prompt_file, limit=128, max_length=2048):
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [
        text.replace(DEFAULT_GRAPH_TOKEN, "")
        for text in load_calibration_texts(calib_prompt_file, limit=limit)
    ]

    examples = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        examples.append(
            {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
            }
        )
    return examples
