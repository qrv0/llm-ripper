from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _load_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    p = Path(model_path)
    if (p / "model.safetensors").exists():
        from safetensors.torch import load_file

        return load_file(str(p / "model.safetensors"))
    if (p / "pytorch_model.bin").exists():
        return torch.load(str(p / "pytorch_model.bin"), map_location="cpu")
    # Support directory with shards: just use main bin if present
    raise FileNotFoundError(f"No recognized model weights in {p}")


def _save_state_dict(sd: Dict[str, torch.Tensor], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out / "pytorch_model.bin")


def merge_models_average(spec_path: str, out_dir: str) -> Dict[str, Any]:
    """Average weights across models listed in spec (YAML or JSON)."""
    spec_p = Path(spec_path)
    if spec_p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            spec = yaml.safe_load(spec_p.read_text())
        except Exception as e:
            raise RuntimeError(f"Failed to read YAML spec: {e}")
    else:
        spec = json.loads(spec_p.read_text())
    models: List[str] = spec.get("models", [])
    if not models:
        raise ValueError("Spec must include 'models': [paths]")
    sds = [_load_state_dict(m) for m in models]
    # Intersect keys
    keys = set(sds[0].keys())
    for sd in sds[1:]:
        keys &= set(sd.keys())
    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [sd[k].float() for sd in sds]
        merged[k] = sum(tensors) / len(tensors)
    _save_state_dict(merged, out_dir)
    return {
        "merged_keys": len(keys),
        "models": models,
        "out": str(Path(out_dir) / "pytorch_model.bin"),
    }


def _copy_base_dir(base_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in base_dir.iterdir():
        # Skip large weights; we overwrite with merged
        if p.name in ("pytorch_model.bin", "model.safetensors"):
            continue
        dst = out_dir / p.name
        if p.is_dir():
            import shutil

            if dst.exists():
                continue
            shutil.copytree(p, dst)
        else:
            if not dst.exists():
                dst.write_bytes(p.read_bytes())


def merge_with_micro(spec_path: str, out_dir: str) -> Dict[str, Any]:
    """Merge models and optionally apply microtransplants in a single pipeline.
    Spec (YAML/JSON) keys:
      models: [paths]
      base_dir: path to HF model dir to copy configs/tokenizer from
      micro: [ { knowledge_bank, source_component, target_layer, strategy } ]
    """
    spec_p = Path(spec_path)
    if spec_p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            spec = yaml.safe_load(spec_p.read_text())
        except Exception as e:
            raise RuntimeError(f"Failed to read YAML spec: {e}")
    else:
        spec = json.loads(spec_p.read_text())
    models: List[str] = spec.get("models", [])
    base_dir = spec.get("base_dir")
    if not models or not base_dir:
        raise ValueError("Spec requires 'models' and 'base_dir'")
    out = Path(out_dir)
    _copy_base_dir(Path(base_dir), out)
    # Merge weights
    sds = [_load_state_dict(m) for m in models]
    keys = set(sds[0].keys())
    for sd in sds[1:]:
        keys &= set(sd.keys())
    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [sd[k].float() for sd in sds]
        merged[k] = sum(tensors) / len(tensors)
    _save_state_dict(merged, out_dir)
    # Apply microtransplants
    micro = spec.get("micro", [])
    applied = []
    if micro:
        from ..core.transplant import KnowledgeTransplanter, TransplantConfig
        from ..utils.config import ConfigManager

        cfg = ConfigManager()
        kt = KnowledgeTransplanter(cfg)
        model, tok, mcfg = kt.model_loader.load_model_and_tokenizer(out_dir)
        for entry in micro:
            kb = entry["knowledge_bank"]
            sc = entry["source_component"]
            tl = (
                int(entry["target_layer"])
                if isinstance(entry.get("target_layer"), int)
                else int(str(entry.get("target_layer", 0)))
            )
            st = entry.get("strategy", "module_injection")
            tc = TransplantConfig(
                sc, tl, cfg.get("adapter_hidden_size", 64), True, False, st
            )
            # Apply transplant on in-memory model
            # Use internal helpers
            knowledge_bank_path = Path(kb)
            target_config = mcfg
            # Leverage private API for precision
            try:
                (
                    kt._transplant_module(model, target_config, knowledge_bank_path, tc)
                    if st != "embedding_init"
                    else kt._transplant_embeddings(model, knowledge_bank_path, tc)
                )
                applied.append(
                    {"source_component": sc, "target_layer": tl, "strategy": st}
                )
            except Exception as e:
                applied.append(
                    {
                        "source_component": sc,
                        "target_layer": tl,
                        "strategy": st,
                        "error": str(e),
                    }
                )
        # Save back to out_dir
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    return {
        "merged_keys": len(keys),
        "models": models,
        "out": str(Path(out_dir)),
        "micro_applied": applied,
    }
