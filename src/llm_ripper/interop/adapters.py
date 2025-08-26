from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..core.transplant import KnowledgeTransplanter, TransplantedModule, BridgeNetwork
from ..utils.config import ConfigManager


def _load_lora(path: str) -> Optional[Dict[str, torch.Tensor]]:
    p = Path(path)
    if p.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file

            return load_file(str(p))
        except Exception:
            return None
    if p.suffix == ".pt" or p.suffix == ".bin":
        try:
            sd = torch.load(str(p), map_location="cpu")
            return sd if isinstance(sd, dict) else None
        except Exception:
            return None
    return None


def import_lora_and_inject(
    model_path: str, lora_path: str, layer: int, alpha: float = 1.0
) -> Dict[str, Any]:
    """Import a simple LoRA (A,B) set and inject as an additional adapter on a target layer.
    This supports a minimal format where tensors contain keys with suffixes 'lora_A' and 'lora_B'.
    """
    sd = _load_lora(lora_path)
    if sd is None:
        raise RuntimeError(
            "Unsupported or unreadable LoRA file; expected safetensors or pt with dict"
        )
    # Construct a linear module from A,B: W_delta = B @ A
    A = None
    B = None
    for k, t in sd.items():
        name = k.lower()
        if name.endswith("lora_a") or name.endswith("lora_a.weight"):
            A = t.detach().cpu()
        if name.endswith("lora_b") or name.endswith("lora_b.weight"):
            B = t.detach().cpu()
    if A is None or B is None:
        raise RuntimeError("LoRA tensors with keys '*lora_A' and '*lora_B' not found")
    W = B @ A  # [out, in]
    donor = nn.Linear(W.shape[1], W.shape[0], bias=False)
    donor.weight.data[:] = W
    # Load target model and inject
    cfg = ConfigManager()
    kt = KnowledgeTransplanter(cfg)
    model, _, mcfg = kt.model_loader.load_model_and_tokenizer(model_path)
    # Build bridges if dims mismatch
    kt._get_target_layer(model, layer)
    hidden = getattr(getattr(model, "config", object()), "hidden_size", W.shape[1])
    ib = None
    ob = None
    if W.shape[1] != hidden:
        ib = BridgeNetwork(W.shape[1], hidden, hidden_dim=min(W.shape[1], hidden))
    if W.shape[0] != hidden:
        ob = BridgeNetwork(hidden, W.shape[0], hidden_dim=min(W.shape[0], hidden))
    tm = TransplantedModule(donor, ib, ob, freeze_donor=True)
    kt._inject_module(model, tm, layer)
    # Save updated model in-place
    out_dir = Path(model_path)
    (out_dir / "transplant_artifacts").mkdir(parents=True, exist_ok=True)
    torch.save(W, out_dir / f"transplant_artifacts/lora_delta_layer_{layer}.pt")
    return {"layer": layer, "in": int(W.shape[1]), "out": int(W.shape[0])}


def fuse_layer_adapters(model_path: str, layer: int) -> Dict[str, Any]:
    """Attach a simple fusion gate over all adapters present on a layer.
    Works if `transplanted_modules['layer_{layer}']` exists on the model.
    """
    cfg = ConfigManager()
    kt = KnowledgeTransplanter(cfg)
    model, _, mcfg = kt.model_loader.load_model_and_tokenizer(model_path)
    layer_key = f"layer_{layer}"
    if (
        not hasattr(model, "transplanted_modules")
        or layer_key not in model.transplanted_modules
    ):
        raise RuntimeError("No adapters found on requested layer")
    mods = model.transplanted_modules[layer_key]
    if isinstance(mods, nn.ModuleList):
        adapters = list(mods)
    else:
        adapters = [mods]
    hidden_size = getattr(getattr(model, "config", object()), "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("hidden_size not found on model.config")
    target_layer_module = kt._get_target_layer(model, layer)

    class Gate(nn.Module):
        def __init__(self, hs, k):
            super().__init__()
            self.alpha = nn.Parameter(torch.zeros(k))

        def forward(self, base, outs):
            w = torch.softmax(self.alpha, dim=0)
            return sum(w[i] * outs[i] for i in range(len(outs)))

    gate = Gate(hidden_size, len(adapters))

    def _hook(module, inputs, output):
        base_out = output[0] if isinstance(output, (tuple, list)) else output
        x = (
            inputs[0]
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0
            else base_out
        )
        outs = [a(x) for a in adapters]
        return gate(base_out, outs)

    if not hasattr(model, "adapter_fusion_hooks"):
        model.adapter_fusion_hooks = {}
    if layer_key in model.adapter_fusion_hooks:
        try:
            model.adapter_fusion_hooks[layer_key].remove()
        except Exception:
            pass
    model.adapter_fusion_hooks[layer_key] = target_layer_module.register_forward_hook(
        _hook
    )
    if not hasattr(model, "transplant_fusion_gates"):
        model.transplant_fusion_gates = nn.ModuleDict()
    model.transplant_fusion_gates[layer_key] = gate
    # Save gate state into artifacts
    out_dir = Path(model_path) / "transplant_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gate.state_dict(), out_dir / f"{layer_key}_fusion_gate.pt")
    return {"layer": layer, "adapters": len(adapters)}
