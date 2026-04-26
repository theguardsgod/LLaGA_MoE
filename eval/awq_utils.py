import json
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.quantization_utils import get_input_device, move_tensor_to_device
from utils.constants import DEFAULT_GRAPH_PAD_ID, GRAPH_TOKEN_INDEX


def build_graph_projector(config):
    projector_type = getattr(config, "mm_projector_type", "linear")
    hidden_dim = getattr(config, "word_embed_proj_dim", getattr(config, "hidden_size", "linear"))

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, hidden_dim)

    prefix = "-layer-mlp"
    if projector_type.endswith(prefix):
        mlp_depth = int(projector_type[: -len(prefix)])
        modules = [nn.Linear(config.mm_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return nn.Sequential(*modules)

    raise ValueError(f"Unknown projector type: {projector_type}")


def add_awq_args(parser):
    parser.add_argument("--awq_model_path", type=str, required=True)
    parser.add_argument("--awq_tokenizer_path", type=str, default=None)
    parser.add_argument("--awq_backend", type=str, default="auto", choices=["auto", "transformers", "autoawq"])
    parser.add_argument("--fuse_awq_layers", action="store_true", default=False)
    parser.add_argument("--awq_max_seq_len", type=int, default=2048)
    parser.add_argument("--awq_use_exllama", action="store_true", default=False)
    parser.add_argument("--awq_use_exllamav2", action="store_true", default=False)
    return parser


def _normalize_projector_state_dict(weights):
    clean = {}
    for key, value in weights.items():
        if key.startswith("model.mm_projector."):
            key = key[len("model.mm_projector."):]
        elif key.startswith("mm_projector."):
            key = key[len("mm_projector."):]
        clean[key] = value.to(torch.float16)
    return clean


def load_fp16_mm_projector(model_path, cfg, device):
    projector_path = os.path.join(model_path, "mm_projector.bin")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Missing mm_projector.bin under {model_path}")

    weights = torch.load(projector_path, map_location="cpu")
    projector = build_graph_projector(cfg).to(dtype=torch.float16, device=device)
    projector.load_state_dict(_normalize_projector_state_dict(weights))
    projector.eval()
    return projector


def _load_awq_with_transformers(awq_model_path, cache_dir=None):
    return AutoModelForCausalLM.from_pretrained(
        awq_model_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )


def _load_awq_with_autoawq(
    awq_model_path,
    fuse_awq_layers=False,
    awq_max_seq_len=2048,
    awq_use_exllama=False,
    awq_use_exllamav2=False,
):
    from awq import AutoAWQForCausalLM

    if awq_use_exllama and awq_use_exllamav2:
        raise ValueError("`awq_use_exllama` and `awq_use_exllamav2` cannot be enabled together.")

    kwargs = {
        "fuse_layers": False if (awq_use_exllama or awq_use_exllamav2) else fuse_awq_layers,
        "use_exllama": awq_use_exllama,
        "use_exllama_v2": awq_use_exllamav2,
    }
    if awq_max_seq_len > 0:
        kwargs["max_seq_len"] = awq_max_seq_len

    try:
        return AutoAWQForCausalLM.from_quantized(awq_model_path, **kwargs)
    except TypeError:
        kwargs.pop("max_seq_len", None)
        return AutoAWQForCausalLM.from_quantized(awq_model_path, **kwargs)


def load_awq_language_model(
    awq_model_path,
    awq_tokenizer_path=None,
    awq_backend="auto",
    cache_dir=None,
    fuse_awq_layers=False,
    awq_max_seq_len=2048,
    awq_use_exllama=False,
    awq_use_exllamav2=False,
):
    tokenizer_path = awq_tokenizer_path or awq_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, cache_dir=cache_dir)

    errors = []

    if awq_backend in {"auto", "transformers"}:
        try:
            model = _load_awq_with_transformers(awq_model_path, cache_dir=cache_dir)
            return tokenizer, model, "transformers"
        except Exception as exc:
            errors.append(f"transformers: {type(exc).__name__}: {exc}")

    if awq_backend in {"auto", "autoawq"}:
        try:
            model = _load_awq_with_autoawq(
                awq_model_path,
                fuse_awq_layers=fuse_awq_layers,
                awq_max_seq_len=awq_max_seq_len,
                awq_use_exllama=awq_use_exllama,
                awq_use_exllamav2=awq_use_exllamav2,
            )
            return tokenizer, model, "autoawq"
        except Exception as exc:
            errors.append(f"autoawq: {type(exc).__name__}: {exc}")

    joined = "\n".join(errors) if errors else "no backend attempted"
    raise RuntimeError(
        "Unable to load the AWQ model. Install a compatible AWQ runtime and ensure the checkpoint is pre-quantized.\n"
        f"Attempted backends:\n{joined}"
    )


def get_embedding_model(model):
    if hasattr(model, "model") and hasattr(model.model, "get_input_embeddings"):
        return model.model
    return model


def get_embedding_device(model):
    return get_input_device(get_embedding_model(model))


def project_graph_embeddings(projector, graph, graph_emb):
    projector_device = next(projector.parameters()).device
    graph = move_tensor_to_device(graph, projector_device)
    graph_emb = move_tensor_to_device(graph_emb, projector_device, dtype=torch.float16)
    graph_features = projector(graph_emb)
    graph_features[graph == DEFAULT_GRAPH_PAD_ID] = 0
    return graph_features


def build_multimodal_inputs_embeds(model, projector, cfg, input_ids, graph, graph_emb):
    if getattr(cfg, "mm_use_graph_special_token", False):
        raise NotImplementedError("AWQ baseline path does not support `mm_use_graph_special_token` yet.")
    if getattr(cfg, "tune_mm_mlp_adapter", False) and getattr(cfg, "mm_use_graph_start_end", False):
        raise NotImplementedError(
            "AWQ baseline path does not support `tune_mm_mlp_adapter + mm_use_graph_start_end` yet."
        )

    embedding_model = get_embedding_model(model)
    embed_tokens = embedding_model.get_input_embeddings()
    embedding_device = embed_tokens.weight.device

    input_ids = move_tensor_to_device(input_ids, embedding_device)
    graph_features = project_graph_embeddings(projector, graph, graph_emb).to(device=embedding_device)

    new_input_embeds = []
    new_attention_masks = []
    cur_graph_idx = 0

    for cur_input_ids in input_ids:
        cur_graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
        cur_input_ids = cur_input_ids.to(device=embedding_device)
        cur_embeds = []

        while cur_graph_token_indices.numel() > 0:
            graph_token_start = cur_graph_token_indices[0]

            if graph_token_start > 0:
                cur_embeds.append(embed_tokens(cur_input_ids[:graph_token_start]))

            cur_embeds.append(graph_features[cur_graph_idx])
            cur_graph_idx += 1

            cur_input_ids = cur_input_ids[graph_token_start + 1 :]
            cur_graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]

        if cur_input_ids.numel() > 0:
            cur_embeds.append(embed_tokens(cur_input_ids))

        if not cur_embeds:
            raise ValueError("Failed to build multimodal embeddings: prompt produced an empty input sequence.")

        cur_input_embeds = torch.cat(cur_embeds, dim=0)
        new_input_embeds.append(cur_input_embeds)
        new_attention_masks.append(
            torch.ones(cur_input_embeds.shape[0], dtype=torch.long, device=embedding_device)
        )

    max_len = max(x.shape[0] for x in new_input_embeds)
    hidden_size = new_input_embeds[0].shape[-1]

    padded_embeds = []
    padded_attention_masks = []
    for cur_embeds, cur_mask in zip(new_input_embeds, new_attention_masks):
        pad_len = max_len - cur_embeds.shape[0]
        if pad_len > 0:
            cur_embeds = torch.cat(
                [
                    cur_embeds,
                    torch.zeros((pad_len, hidden_size), dtype=cur_embeds.dtype, device=cur_embeds.device),
                ],
                dim=0,
            )
            cur_mask = torch.cat(
                [cur_mask, torch.zeros(pad_len, dtype=cur_mask.dtype, device=cur_mask.device)],
                dim=0,
            )
        padded_embeds.append(cur_embeds)
        padded_attention_masks.append(cur_mask)

    return torch.stack(padded_embeds, dim=0), torch.stack(padded_attention_masks, dim=0)


def decode_awq_outputs(tokenizer, output_ids, prompt_length):
    if output_ids.shape[1] > prompt_length:
        decoded_ids = output_ids[:, prompt_length:]
    else:
        decoded_ids = output_ids
    return tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)[0].strip()


def load_calibration_texts(calib_prompt_file, limit=128):
    texts = []
    with open(calib_prompt_file, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                texts.append(line)
            else:
                if isinstance(data, dict) and "conversations" in data and data["conversations"]:
                    texts.append(str(data["conversations"][0]["value"]))
                else:
                    texts.append(line)

            if len(texts) >= limit:
                break

    if not texts:
        raise ValueError(f"No calibration texts found in {calib_prompt_file}")

    return texts
