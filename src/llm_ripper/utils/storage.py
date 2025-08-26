"""Storage helpers for loading saved tensors (pt/safetensors), including sharded .pt files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import torch
from safetensors.torch import load_file as safeload


def load_pt_or_safetensors(file_path: Path) -> Dict[str, torch.Tensor]:
    """Load a tensor file into a dict with 'weight' key when applicable.
    Supports .pt (tensor) and .safetensors ({'weight': tensor}).
    """
    if file_path.suffix == ".safetensors":
        data = safeload(str(file_path))
        return dict(data)
    else:
        t = torch.load(str(file_path), map_location="cpu")
        if isinstance(t, torch.Tensor):
            return {"weight": t}
        return t  # already a dict


def load_sharded_pt(index_file: Path) -> torch.Tensor:
    """Load a sharded .pt tensor using an index JSON file listing parts.
    Assumes sharding along dimension 0 and concatenates parts accordingly.
    """
    meta = json.loads(index_file.read_text())
    parts = meta.get("parts", [])
    tensors: List[torch.Tensor] = []
    for name in parts:
        p = index_file.parent / name
        tensors.append(torch.load(str(p), map_location="cpu"))
    if not tensors:
        raise ValueError(f"No parts found in {index_file}")
    return torch.cat(tensors, dim=0)


def load_component_weights_dir(dir_path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load all projection weights from a component directory.
    Returns a mapping {name: {"weight": tensor, ...}} supporting:
    - sharded .pt via <name>.index.json
    - single-file .safetensors or .pt
    Bias files are loaded under key 'bias' if present.
    """
    weights: Dict[str, Dict[str, torch.Tensor]] = {}
    # First handle sharded indices
    for index_file in dir_path.glob("*.index.json"):
        base = index_file.stem.replace(".index", "")
        try:
            tensor = load_sharded_pt(index_file)
            weights.setdefault(base, {})["weight"] = tensor
        except Exception:
            pass
    # Then safetensors / pt
    for f in dir_path.glob("*.safetensors"):
        base = f.stem
        if base not in weights:
            data = safeload(str(f))
            # prefer 'weight' if exists, else first tensor
            if "weight" in data:
                weights.setdefault(base, {})["weight"] = data["weight"]
            else:
                for k, v in data.items():
                    weights.setdefault(base, {})[k] = v
                    break
    for f in dir_path.glob("*.pt"):
        base = f.stem
        # skip sharded part files (handled by index)
        # match suffixes like _part0, _part1, _part10, etc.
        try:
            import re as _re

            if _re.search(r"_part\d+$", base):
                continue
        except Exception:
            if base.endswith("_part0") or base.endswith("_part1"):
                continue
        if base not in weights:
            t = torch.load(str(f), map_location="cpu")
            if isinstance(t, torch.Tensor):
                weights.setdefault(base, {})["weight"] = t
            elif isinstance(t, dict):
                for k, v in t.items():
                    weights.setdefault(base, {})[k] = v
    # Bias files: <name>_bias.*
    for bias in list(dir_path.glob("*_bias.safetensors")) + list(
        dir_path.glob("*_bias.pt")
    ):
        name = bias.stem.replace("_bias", "")
        try:
            if bias.suffix == ".safetensors":
                b = safeload(str(bias)).get("bias")
            else:
                t = torch.load(str(bias), map_location="cpu")
                b = (
                    t["bias"]
                    if isinstance(t, dict) and "bias" in t
                    else (t if isinstance(t, torch.Tensor) else None)
                )
            if b is not None:
                weights.setdefault(name, {})["bias"] = b
        except Exception:
            pass
    return weights
