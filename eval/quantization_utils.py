import torch
from transformers import BitsAndBytesConfig


def add_quantization_args(parser):
    parser.add_argument("--load_4bit", action="store_true", default=False)
    parser.add_argument("--load_8bit", action="store_true", default=False)
    return parser


def validate_quantization_args(args):
    if args.load_4bit and args.load_8bit:
        raise ValueError("`--load_4bit` and `--load_8bit` are mutually exclusive.")


def is_quantized(load_4bit=False, load_8bit=False):
    return load_4bit or load_8bit


def build_quantization_kwargs(load_4bit=False, load_8bit=False):
    if load_4bit and load_8bit:
        raise ValueError("`load_4bit` and `load_8bit` cannot both be enabled.")

    if load_4bit:
        return {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        }

    if load_8bit:
        return {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "load_in_8bit": True,
        }

    return {"torch_dtype": torch.float16}


def _normalize_device(device_ref):
    if isinstance(device_ref, torch.device):
        return device_ref
    if isinstance(device_ref, int):
        return torch.device(f"cuda:{device_ref}")
    if isinstance(device_ref, str) and device_ref in {"cpu", "mps"}:
        return torch.device(device_ref)
    if isinstance(device_ref, str) and device_ref.startswith("cuda"):
        return torch.device(device_ref)
    return None


def _module_device(module):
    if module is None:
        return None

    try:
        return next(module.parameters()).device
    except StopIteration:
        pass

    try:
        return next(module.buffers()).device
    except StopIteration:
        return None


def _device_from_hf_device_map(model):
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return None

    preferred_keys = ("model.embed_tokens", "model", "lm_head")
    for preferred in preferred_keys:
        for module_name, device_ref in device_map.items():
            if module_name == preferred or module_name.startswith(f"{preferred}."):
                device = _normalize_device(device_ref)
                if device is not None:
                    return device

    for device_ref in device_map.values():
        device = _normalize_device(device_ref)
        if device is not None and device.type != "cpu":
            return device

    for device_ref in device_map.values():
        device = _normalize_device(device_ref)
        if device is not None:
            return device

    return None


def get_input_device(model):
    if hasattr(model, "get_input_embeddings"):
        device = _module_device(model.get_input_embeddings())
        if device is not None:
            return device

    device = _device_from_hf_device_map(model)
    if device is not None:
        return device

    device = _module_device(model)
    if device is not None:
        return device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_graph_device(model):
    device = _module_device(getattr(model, "moe_projector", None))
    if device is not None:
        return device

    if hasattr(model, "get_model"):
        device = _module_device(getattr(model.get_model(), "mm_projector", None))
        if device is not None:
            return device

    return get_input_device(model)


def move_moe_projector_to_runtime_device(model):
    if getattr(model, "moe_projector", None) is None:
        return

    target_device = get_input_device(model)
    model.moe_projector.to(dtype=torch.float16, device=target_device)


def prepare_model_for_inference(model, load_4bit=False, load_8bit=False, move_moe_projector=False):
    if not is_quantized(load_4bit=load_4bit, load_8bit=load_8bit):
        model = model.to(torch.float16).cuda()

    if move_moe_projector:
        move_moe_projector_to_runtime_device(model)

    return model


def build_moe_projector_from_config(cfg):
    """Build MoEGraphProjector from a model config."""
    from model.moe_llaga import MoEGraphProjector
    return MoEGraphProjector(
        mm_hidden_size=getattr(cfg, "mm_hidden_size", 2543),
        llm_hidden_size=getattr(cfg, "hidden_size", 4096),
        num_experts=getattr(cfg, "num_experts", 4),
        top_k=getattr(cfg, "top_k", 2),
        projector_type=getattr(cfg, "mm_projector_type", "linear"),
        routing_dim=getattr(cfg, "routing_dim", 2432),
        noise_std=getattr(cfg, "noise_std", 1.0),
    )


def restore_fp16_mm_projector(model, cfg, projector_path):
    """Replace a quantized mm_projector with a fresh FP16 projector and load weights."""
    from model.llaga_arch import build_graph_projector

    weights = torch.load(projector_path, map_location="cpu")
    clean = {}
    for key, value in weights.items():
        if key.startswith("model.mm_projector."):
            key = key[len("model.mm_projector."):]
        elif key.startswith("mm_projector."):
            key = key[len("mm_projector."):]
        clean[key] = value.to(torch.float16)

    target_device = get_input_device(model)
    fresh_projector = build_graph_projector(cfg).to(dtype=torch.float16, device=target_device)
    fresh_projector.load_state_dict(clean)
    model.get_model().mm_projector = fresh_projector
    return target_device


def load_clean_projector_weights(projector_path, prefix_to_strip=None):
    """Load projector weights from .bin file, strip prefix and cast to FP16."""
    weights = torch.load(projector_path, map_location="cpu")
    clean = {}
    for key, value in weights.items():
        if prefix_to_strip and key.startswith(prefix_to_strip):
            key = key[len(prefix_to_strip):]
        clean[key] = value.to(torch.float16)
    return clean


def move_tensor_to_device(tensor, device, dtype=None):
    kwargs = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return tensor.to(**kwargs)
