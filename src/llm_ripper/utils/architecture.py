"""Architecture-aware helpers for resolving transformer submodules.

Provides utilities to find attention/FFN modules across common model families
using declarative pattern lists.
"""

from typing import Optional, List
import torch.nn as nn


# Known layer patterns per family
FAMILY_PATTERNS = {
    "gpt2": {
        "attn": ["transformer.h.{i}.attn", "h.{i}.attn"],
        "ffn": ["transformer.h.{i}.mlp", "h.{i}.mlp"],
    },
    "llama": {
        "attn": [
            "model.layers.{i}.self_attn",
            "layers.{i}.self_attn",
            "model.layers.{i}.attention",
            "layers.{i}.attention",
        ],
        "ffn": [
            "model.layers.{i}.mlp",
            "layers.{i}.mlp",
            "model.layers.{i}.feed_forward",
            "layers.{i}.feed_forward",
        ],
    },
    "mistral": {
        "attn": [
            "model.layers.{i}.self_attn",
            "layers.{i}.self_attn",
            "model.layers.{i}.attention",
            "layers.{i}.attention",
        ],
        "ffn": [
            "model.layers.{i}.mlp",
            "layers.{i}.mlp",
            # Mixtral-style MoE containers
            "model.layers.{i}.block_sparse_moe",
            "layers.{i}.block_sparse_moe",
        ],
    },
    # Alias for Mixtral (explicit name)
    "mixtral": {
        "attn": ["model.layers.{i}.self_attn", "layers.{i}.self_attn"],
        "ffn": [
            "model.layers.{i}.block_sparse_moe",
            "layers.{i}.block_sparse_moe",
            "model.layers.{i}.mlp",
            "layers.{i}.mlp",
        ],
    },
    "bloom": {
        "attn": ["transformer.h.{i}.self_attention", "h.{i}.self_attention"],
        "ffn": ["transformer.h.{i}.mlp", "h.{i}.mlp"],
    },
    "opt": {
        "attn": ["model.decoder.layers.{i}.self_attn", "decoder.layers.{i}.self_attn"],
        "ffn": ["model.decoder.layers.{i}.fc1", "decoder.layers.{i}.fc1"],
    },
    "gptj": {
        "attn": ["transformer.h.{i}.attn", "h.{i}.attn"],
        "ffn": ["transformer.h.{i}.mlp", "h.{i}.mlp"],
    },
    "t5": {
        # Encoder-decoder: use both encoder/decoder blocks
        "attn": [
            "encoder.block.{i}.layer.0.SelfAttention",
            "decoder.block.{i}.layer.0.SelfAttention",
        ],
        "ffn": [
            "encoder.block.{i}.layer.1.DenseReluDense",
            "decoder.block.{i}.layer.1.DenseReluDense",
            # Some T5 variants use gated FFN naming
            "encoder.block.{i}.layer.1.DenseGatedActDense",
            "decoder.block.{i}.layer.1.DenseGatedActDense",
        ],
    },
    # Google Gemma style (similar to LLaMA)
    "gemma": {
        "attn": ["model.layers.{i}.self_attn", "layers.{i}.self_attn"],
        "ffn": ["model.layers.{i}.mlp", "layers.{i}.mlp"],
    },
    # Microsoft Phi family
    "phi": {
        "attn": [
            "model.layers.{i}.self_attn",
            "layers.{i}.self_attn",
            "transformer.layers.{i}.self_attn",
        ],
        "ffn": ["model.layers.{i}.mlp", "layers.{i}.mlp", "transformer.layers.{i}.mlp"],
    },
    "generic": {
        "attn": [
            "transformer.layers.{i}.attention",
            "layers.{i}.attention",
            "decoder.layers.{i}.self_attn",
            "encoder.layers.{i}.self_attn",
            "transformer.blocks.{i}.attn",
            "blocks.{i}.attn",
            # be liberal with alternative names
            "model.decoder.layers.{i}.self_attn",
            "model.layers.{i}.self_attn",
        ],
        "ffn": [
            "transformer.layers.{i}.ffn",
            "layers.{i}.ffn",
            "decoder.layers.{i}.mlp",
            "encoder.layers.{i}.mlp",
            "transformer.blocks.{i}.ffn",
            "blocks.{i}.ffn",
            "model.decoder.layers.{i}.mlp",
            "model.layers.{i}.mlp",
        ],
    },
    # Additional families approximated to common patterns
    "falcon": {
        "attn": ["transformer.h.{i}.self_attention", "h.{i}.self_attention"],
        "ffn": ["transformer.h.{i}.mlp", "h.{i}.mlp"],
    },
    "mpt": {
        "attn": ["transformer.blocks.{i}.attn", "blocks.{i}.attn"],
        "ffn": ["transformer.blocks.{i}.ffn", "blocks.{i}.ffn"],
    },
    "qwen": {
        "attn": ["transformer.h.{i}.attn", "h.{i}.attn"],
        "ffn": ["transformer.h.{i}.mlp", "h.{i}.mlp"],
    },
}


def resolve_family(model_type: Optional[str]) -> List[str]:
    mt = (model_type or "").lower()
    if "t5" in mt:
        return ["t5", "generic"]
    if any(k in mt for k in ("llama", "meta-llama")):
        return ["llama", "generic"]
    if "mistral" in mt:
        return ["mistral", "generic"]
    if "mixtral" in mt:
        return ["mixtral", "mistral", "generic"]
    if "falcon" in mt:
        return ["falcon", "generic"]
    if "mpt" in mt:
        return ["mpt", "generic"]
    if "qwen" in mt:
        return ["qwen", "generic"]
    if "gemma" in mt:
        return ["gemma", "llama", "generic"]
    if "phi" in mt:
        return ["phi", "generic"]
    if "gpt2" in mt or "gpt" in mt:
        return ["gpt2", "generic"]
    if "bloom" in mt:
        return ["bloom", "generic"]
    if "opt" in mt:
        return ["opt", "generic"]
    if "gptj" in mt:
        return ["gptj", "generic"]
    return [
        "gpt2",
        "llama",
        "mistral",
        "bloom",
        "opt",
        "gptj",
        "falcon",
        "mpt",
        "qwen",
        "generic",
    ]


def get_layer_module(
    model: nn.Module, layer_idx: int, kind: str, model_type: Optional[str] = None
) -> Optional[nn.Module]:
    """Find a submodule for given layer and kind ('attn' or 'ffn')."""
    families = resolve_family(model_type)
    for fam in families:
        patterns = FAMILY_PATTERNS.get(fam, {}).get(kind, [])
        for p in patterns:
            path = p.format(i=layer_idx)
            try:
                mod = model
                for attr in path.split("."):
                    mod = getattr(mod, attr)
                return mod
            except AttributeError:
                continue
    return None


def _families_for(model_type: Optional[str]) -> List[str]:
    return resolve_family(model_type)


def replace_layer_submodule(
    model: nn.Module,
    layer_idx: int,
    kind: str,
    new_module: nn.Module,
    model_type: Optional[str] = None,
) -> bool:
    """Replace a layer submodule (attn|ffn) in-place using family patterns.
    Returns True if replacement succeeded, else False.
    """
    families = _families_for(model_type)
    for fam in families:
        patterns = FAMILY_PATTERNS.get(fam, {}).get(kind, [])
        for p in patterns:
            path = p.format(i=layer_idx)
            parts = path.split(".")
            # Walk to parent
            try:
                parent = model
                for attr in parts[:-1]:
                    parent = getattr(parent, attr)
                name = parts[-1]
                # Validate current exists
                if not hasattr(parent, name):
                    continue
                setattr(parent, name, new_module)
                return True
            except AttributeError:
                continue
    return False
