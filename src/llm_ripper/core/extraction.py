"""
Knowledge extraction module for LLM Ripper.

This module implements Part I of the framework: Architectural dissection and knowledge extraction.
"""

import torch
import torch.nn as nn

# from torch.fx import symbolic_trace  # unused
# Avoid importing heavy libs at module import time; import on demand
try:
    from safetensors.torch import save_file as _safetensors_save
except Exception:  # safetensors optional
    _safetensors_save = None  # type: ignore
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import hashlib
import uuid

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader
from ..utils.architecture import get_layer_module

logger = logging.getLogger(__name__)


@dataclass
class ComponentMetadata:
    """Metadata for extracted components."""

    component_type: str
    layer_idx: Optional[int]
    head_idx: Optional[int]
    dimensions: Tuple[int, ...]
    source_model: str
    attention_type: Optional[str] = None
    activation_function: Optional[str] = None


class KnowledgeExtractor:
    """
    Extracts static knowledge (weights) from transformer models.

    Implements the static knowledge extraction protocol described in Section 2
    of the framework specification.
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_loader = ModelLoader(
            cache_dir=config.get("model_cache_dir"), device=config.get("device")
        )

    def _sha256(self, tensor: torch.Tensor) -> str:
        try:
            arr = tensor.detach().cpu().numpy().tobytes()
            return hashlib.sha256(arr).hexdigest()
        except Exception:
            return ""

    def extract_model_components(
        self,
        model_name: str,
        output_dir: str,
        components: List[str] = None,
        force_model_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract all components from a model.

        Args:
            model_name: Name or path of the model to extract from
            output_dir: Directory to save extracted components
            components: List of components to extract (default: all)

        Returns:
            Dictionary containing extraction metadata
        """
        if components is None:
            components = ["embeddings", "attention_heads", "ffn_layers", "lm_head"]

        logger.info(f"Starting extraction from model: {model_name}")

        # Decide model type: if we will not extract lm_head, we can load the base model
        needs_lm_head = "lm_head" in components
        model_type = (
            force_model_type
            if force_model_type in ("base", "causal_lm")
            else ("causal_lm" if needs_lm_head else "base")
        )

        # Load model
        model, tokenizer, config = self.model_loader.load_model_and_tokenizer(
            model_name,
            model_type=model_type,
            load_in_8bit=self.config.get("load_in_8bit"),
            load_in_4bit=self.config.get("load_in_4bit"),
            trust_remote_code=self.config.get("trust_remote_code"),
        )
        arch_info = self.model_loader.get_model_architecture_info(model, config)

        # Create output directory structure
        output_path = Path(output_dir)
        self._create_knowledge_bank_structure(output_path)

        # Library versions for provenance
        try:
            import transformers as _tf

            tf_ver = getattr(_tf, "__version__", "unknown")
        except Exception:
            tf_ver = "unknown"
        torch_ver = getattr(torch, "__version__", "unknown")

        extraction_metadata = {
            "source_model": model_name,
            "architecture_info": arch_info,
            "extracted_components": {},
            "extraction_config": self.config.config.copy(),
            "run_id": str(uuid.uuid4()),
            "seed": self.config.get("seed"),
            "library_versions": {"torch": torch_ver, "transformers": tf_ver},
        }

        # Extract components
        if "embeddings" in components:
            embedding_metadata = self._extract_embeddings(model, output_path)
            extraction_metadata["extracted_components"][
                "embeddings"
            ] = embedding_metadata

        if "attention_heads" in components:
            attention_metadata = self._extract_attention_heads(
                model, config, output_path
            )
            extraction_metadata["extracted_components"][
                "attention_heads"
            ] = attention_metadata

        if "ffn_layers" in components:
            ffn_metadata = self._extract_ffn_layers(model, config, output_path)
            extraction_metadata["extracted_components"]["ffn_layers"] = ffn_metadata

        if "lm_head" in components:
            lm_head_metadata = self._extract_lm_head(model, output_path)
            extraction_metadata["extracted_components"]["lm_head"] = lm_head_metadata

        # Save extraction metadata
        with open(output_path / "extraction_metadata.json", "w") as f:
            json.dump(extraction_metadata, f, indent=2)

        logger.info(f"Extraction completed. Components saved to: {output_path}")

        return extraction_metadata

    def _create_knowledge_bank_structure(self, output_path: Path) -> None:
        """Create the knowledge bank directory structure."""
        directories = ["embeddings", "heads", "ffns", "lm_head", "concepts", "metadata"]

        for directory in directories:
            (output_path / directory).mkdir(parents=True, exist_ok=True)

    def _extract_embeddings(
        self, model: nn.Module, output_path: Path
    ) -> Dict[str, Any]:
        """Extract input embeddings."""
        logger.info("Extracting input embeddings...")

        # Get input embeddings
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is None:
            logger.warning("No input embeddings found")
            return {}

        weight_tensor = input_embeddings.weight.detach().cpu()

        # Check for weight tying with output embeddings
        output_embeddings = model.get_output_embeddings()
        weight_tied = False
        if output_embeddings is not None:
            weight_tied = torch.equal(
                input_embeddings.weight.detach().cpu(),
                output_embeddings.weight.detach().cpu(),
            )

        # Save embeddings (with sharding for .pt if very large)
        use_st = self.config.get("use_safetensors") and (_safetensors_save is not None)
        if use_st:
            _safetensors_save(
                {"weight": weight_tensor},
                output_path / "embeddings" / "embeddings.safetensors",
            )
            file_path = "embeddings/embeddings.safetensors"
            sharded = False
        else:
            fp = self._save_linear_weight(
                weight_tensor, output_path / "embeddings", "embeddings"
            )
            file_path = f"embeddings/{fp}"
            sharded = fp.endswith(".index.json")

        # Save metadata
        metadata = ComponentMetadata(
            component_type="embeddings",
            layer_idx=None,
            head_idx=None,
            dimensions=tuple(weight_tensor.shape),
            source_model=(
                model.config.name_or_path if hasattr(model, "config") else "unknown"
            ),
        )

        config_data = {
            "dimensions": list(weight_tensor.shape),
            "weight_tied": weight_tied,
            "vocab_size": weight_tensor.shape[0],
            "hidden_size": weight_tensor.shape[1],
        }

        with open(output_path / "embeddings" / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Embeddings extracted: {weight_tensor.shape}")

        return {
            "metadata": metadata.__dict__,
            "config": config_data,
            "file_path": file_path,
            "sharded": sharded,
            "sha256": self._sha256(weight_tensor),
        }

    def _extract_attention_heads(
        self, model: nn.Module, config: Dict[str, Any], output_path: Path
    ) -> Dict[str, Any]:
        """Extract attention heads with architecture-sensitive handling."""
        logger.info("Extracting attention heads...")

        attention_metadata = {}
        num_layers = config.get("num_hidden_layers", 0)
        attention_type = self._determine_attention_type(config)

        for layer_idx in tqdm(range(num_layers), desc="Extracting attention heads"):
            layer_metadata = self._extract_layer_attention(
                model, layer_idx, attention_type, output_path
            )
            attention_metadata[f"layer_{layer_idx}"] = layer_metadata

        return attention_metadata

    def _determine_attention_type(self, config: Dict[str, Any]) -> str:
        """Determine the attention mechanism type."""
        num_heads = config.get("num_attention_heads", 0)
        num_kv_heads = config.get("num_key_value_heads", num_heads)

        if num_kv_heads == num_heads:
            return "MHA"
        elif num_kv_heads == 1:
            return "MQA"
        else:
            return "GQA"

    def _extract_layer_attention(
        self, model: nn.Module, layer_idx: int, attention_type: str, output_path: Path
    ) -> Dict[str, Any]:
        """Extract attention components from a specific layer."""

        # Get the attention module for this layer
        attention_module = self._get_attention_module(model, layer_idx)
        if attention_module is None:
            logger.warning(f"No attention module found for layer {layer_idx}")
            return {}

        layer_dir = output_path / "heads" / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        if attention_type == "MHA":
            return self._extract_mha_weights(attention_module, layer_idx, layer_dir)
        elif attention_type in ["GQA", "MQA"]:
            return self._extract_gqa_weights(
                attention_module, layer_idx, layer_dir, attention_type
            )
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

    def _get_attention_module(
        self, model: nn.Module, layer_idx: int
    ) -> Optional[nn.Module]:
        """Get the attention module for a specific layer."""
        # Try architecture-aware resolution using config model_type
        mt = getattr(getattr(model, "config", object()), "model_type", None)
        mod = get_layer_module(model, layer_idx, kind="attn", model_type=mt)
        if mod is not None:
            return mod
        # Fallback to previous heuristics
        for pattern in (
            f"transformer.h.{layer_idx}.attn",
            f"model.layers.{layer_idx}.self_attn",
            f"transformer.layers.{layer_idx}.attention",
            f"h.{layer_idx}.attn",
            f"layers.{layer_idx}.self_attn",
        ):
            try:
                module = model
                for attr in pattern.split("."):
                    module = getattr(module, attr)
                return module
            except AttributeError:
                continue
        return None

    def _extract_mha_weights(
        self, attention_module: nn.Module, layer_idx: int, output_dir: Path
    ) -> Dict[str, Any]:
        """Extract weights for Multi-Head Attention."""
        weights_data = {}

        # Extract Q, K, V, O projection weights
        projection_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

        for proj_name in projection_names:
            if hasattr(attention_module, proj_name):
                proj_module = getattr(attention_module, proj_name)
                weight = proj_module.weight.detach().cpu()

                fp = self._save_tensor(weight, output_dir, proj_name)
                entry = {
                    "shape": list(weight.shape),
                    "file_path": fp,
                    "sha256": self._sha256(weight),
                }
                # Save bias if present
                if hasattr(proj_module, "bias") and proj_module.bias is not None:
                    bias = proj_module.bias.detach().cpu()
                    if self.config.get("use_safetensors") and (
                        _safetensors_save is not None
                    ):
                        _safetensors_save(
                            {"bias": bias}, output_dir / f"{proj_name}_bias.safetensors"
                        )
                        entry["bias_file"] = f"{proj_name}_bias.safetensors"
                    else:
                        torch.save(bias, output_dir / f"{proj_name}_bias.pt")
                        entry["bias_file"] = f"{proj_name}_bias.pt"
                    entry["bias_shape"] = list(bias.shape)
                    entry["bias_sha256"] = self._sha256(bias)
                weights_data[proj_name] = entry

        # Save configuration
        config_data = {
            "attention_type": "MHA",
            "layer_idx": layer_idx,
            "hidden_size": weights_data.get("q_proj", {}).get("shape", [0, 0])[1],
            "num_heads": getattr(attention_module, "num_heads", 0),
            "head_dim": getattr(attention_module, "head_dim", 0),
        }

        with open(output_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        return {"weights": weights_data, "config": config_data}

    def _extract_gqa_weights(
        self,
        attention_module: nn.Module,
        layer_idx: int,
        output_dir: Path,
        attention_type: str,
    ) -> Dict[str, Any]:
        """Extract weights for Grouped-Query Attention or Multi-Query Attention."""
        weights_data = {}

        # For GQA/MQA, we need to handle the shared K,V projections
        if hasattr(attention_module, "q_proj"):
            q_weight = attention_module.q_proj.weight.detach().cpu()
            fpq = self._save_tensor(q_weight, output_dir, "q_proj")
            weights_data["q_proj"] = {
                "shape": list(q_weight.shape),
                "file_path": fpq,
                "sha256": self._sha256(q_weight),
            }
            # Save bias if present
            if (
                hasattr(attention_module.q_proj, "bias")
                and attention_module.q_proj.bias is not None
            ):
                bias = attention_module.q_proj.bias.detach().cpu()
                if self.config.get("use_safetensors") and (
                    _safetensors_save is not None
                ):
                    _safetensors_save(
                        {"bias": bias}, output_dir / "q_proj_bias.safetensors"
                    )
                    weights_data["q_proj"]["bias_file"] = "q_proj_bias.safetensors"
                else:
                    torch.save(bias, output_dir / "q_proj_bias.pt")
                    weights_data["q_proj"]["bias_file"] = "q_proj_bias.pt"
                weights_data["q_proj"]["bias_shape"] = list(bias.shape)
                weights_data["q_proj"]["bias_sha256"] = self._sha256(bias)

        # Handle shared K,V projections
        if hasattr(attention_module, "k_proj"):
            k_weight = attention_module.k_proj.weight.detach().cpu()
            fpk = self._save_tensor(k_weight, output_dir, "kv_proj")
            weights_data["kv_proj"] = {
                "shape": list(k_weight.shape),
                "file_path": fpk,
                "sha256": self._sha256(k_weight),
            }
            # Save k bias if available (associated to kv)
            if (
                hasattr(attention_module.k_proj, "bias")
                and attention_module.k_proj.bias is not None
            ):
                bias = attention_module.k_proj.bias.detach().cpu()
                if self.config.get("use_safetensors") and (
                    _safetensors_save is not None
                ):
                    _safetensors_save(
                        {"bias": bias}, output_dir / "kv_proj_bias.safetensors"
                    )
                    weights_data.setdefault("kv_proj", {})[
                        "bias_file"
                    ] = "kv_proj_bias.safetensors"
                else:
                    torch.save(bias, output_dir / "kv_proj_bias.pt")
                    weights_data.setdefault("kv_proj", {})[
                        "bias_file"
                    ] = "kv_proj_bias.pt"
                weights_data["kv_proj"]["bias_shape"] = list(bias.shape)
                weights_data["kv_proj"]["bias_sha256"] = self._sha256(bias)

        if hasattr(attention_module, "v_proj"):
            v_weight = attention_module.v_proj.weight.detach().cpu()
            # Concatenate K and V if separate
            if "kv_proj" not in weights_data:
                kv_weight = torch.cat(
                    [
                        getattr(attention_module, "k_proj").weight.detach().cpu(),
                        v_weight,
                    ],
                    dim=0,
                )

                fpkv = self._save_tensor(kv_weight, output_dir, "kv_proj")
                weights_data["kv_proj"] = {
                    "shape": list(kv_weight.shape),
                    "file_path": fpkv,
                    "sha256": self._sha256(kv_weight),
                }
            # Save v bias into kv bias if present and kv bias missing
            if (
                hasattr(attention_module.v_proj, "bias")
                and attention_module.v_proj.bias is not None
                and "bias_file" not in weights_data.get("kv_proj", {})
            ):
                bias = attention_module.v_proj.bias.detach().cpu()
                if self.config.get("use_safetensors") and (
                    _safetensors_save is not None
                ):
                    _safetensors_save(
                        {"bias": bias}, output_dir / "kv_proj_bias.safetensors"
                    )
                    weights_data.setdefault("kv_proj", {})[
                        "bias_file"
                    ] = "kv_proj_bias.safetensors"
                else:
                    torch.save(bias, output_dir / "kv_proj_bias.pt")
                    weights_data.setdefault("kv_proj", {})[
                        "bias_file"
                    ] = "kv_proj_bias.pt"
                weights_data["kv_proj"]["bias_shape"] = list(bias.shape)
                weights_data["kv_proj"]["bias_sha256"] = self._sha256(bias)

        # Output projection
        if hasattr(attention_module, "o_proj"):
            o_weight = attention_module.o_proj.weight.detach().cpu()
            fpo = self._save_tensor(o_weight, output_dir, "o_proj")
            weights_data["o_proj"] = {
                "shape": list(o_weight.shape),
                "file_path": fpo,
                "sha256": self._sha256(o_weight),
            }
            if (
                hasattr(attention_module.o_proj, "bias")
                and attention_module.o_proj.bias is not None
            ):
                bias = attention_module.o_proj.bias.detach().cpu()
                if self.config.get("use_safetensors") and (
                    _safetensors_save is not None
                ):
                    _safetensors_save(
                        {"bias": bias}, output_dir / "o_proj_bias.safetensors"
                    )
                    weights_data["o_proj"]["bias_file"] = "o_proj_bias.safetensors"
                else:
                    torch.save(bias, output_dir / "o_proj_bias.pt")
                    weights_data["o_proj"]["bias_file"] = "o_proj_bias.pt"
                weights_data["o_proj"]["bias_shape"] = list(bias.shape)
                weights_data["o_proj"]["bias_sha256"] = self._sha256(bias)

        # Save configuration with GQA/MQA specific metadata
        config_data = {
            "attention_type": attention_type,
            "layer_idx": layer_idx,
            "hidden_size": weights_data.get("q_proj", {}).get("shape", [0, 0])[1],
            "num_query_heads": getattr(attention_module, "num_heads", 0),
            "num_key_value_heads": getattr(attention_module, "num_key_value_heads", 0),
            "head_dim": getattr(attention_module, "head_dim", 0),
        }

        if attention_type in ("GQA", "MQA"):
            kvh = int(config_data.get("num_key_value_heads") or 0)
            qh = int(config_data.get("num_query_heads") or 0)
            group_size = (qh // kvh) if (kvh and qh) else None
            config_data["group_size"] = group_size
            if group_size:
                # mapping from query head index -> kv head index
                mapping = [i // group_size for i in range(qh)]
                config_data["q_to_kv_mapping"] = mapping

        with open(output_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        return {"weights": weights_data, "config": config_data}

    def _extract_ffn_layers(
        self, model: nn.Module, config: Dict[str, Any], output_path: Path
    ) -> Dict[str, Any]:
        """Extract Feed-Forward Network layers."""
        logger.info("Extracting FFN layers...")

        ffn_metadata = {}
        num_layers = config.get("num_hidden_layers", 0)

        for layer_idx in tqdm(range(num_layers), desc="Extracting FFN layers"):
            layer_metadata = self._extract_layer_ffn(model, layer_idx, output_path)
            ffn_metadata[f"layer_{layer_idx}"] = layer_metadata

        return ffn_metadata

    def _extract_layer_ffn(
        self, model: nn.Module, layer_idx: int, output_path: Path
    ) -> Dict[str, Any]:
        """Extract FFN components from a specific layer."""
        ffn_module = self._get_ffn_module(model, layer_idx)
        if ffn_module is None:
            logger.warning(f"No FFN module found for layer {layer_idx}")
            return {}

        layer_dir = output_path / "ffns" / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Detect MoE-style FFN (experts + router)
        if self._is_moe_ffn(ffn_module):
            return self._extract_moe_ffn_layer(ffn_module, layer_idx, layer_dir)
        else:
            return self._extract_dense_ffn_layer(ffn_module, layer_idx, layer_dir)

    def _get_ffn_module(self, model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
        """Get the FFN module for a specific layer."""
        mt = getattr(getattr(model, "config", object()), "model_type", None)
        mod = get_layer_module(model, layer_idx, kind="ffn", model_type=mt)
        if mod is not None:
            return mod
        for pattern in (
            f"transformer.h.{layer_idx}.mlp",
            f"model.layers.{layer_idx}.mlp",
            f"transformer.layers.{layer_idx}.ffn",
            f"h.{layer_idx}.mlp",
            f"layers.{layer_idx}.mlp",
        ):
            try:
                module = model
                for attr in pattern.split("."):
                    module = getattr(module, attr)
                return module
            except AttributeError:
                continue
        return None

    def _save_linear_weight(
        self, weight: torch.Tensor, out_path: Path, filename: str
    ) -> str:
        """Helper to save a weight tensor respecting safetensors config with sharding for .pt."""
        if self.config.get("use_safetensors") and (_safetensors_save is not None):
            _safetensors_save(
                {"weight": weight.detach().cpu()}, out_path / f"{filename}.safetensors"
            )
            return f"{filename}.safetensors"
        tensor = weight.detach().cpu()
        shard_mb = max(1, int(self.config.get("tensor_shard_mb", 512)))
        bytes_total = tensor.element_size() * tensor.nelement()
        if bytes_total <= shard_mb * 1024 * 1024:
            torch.save(tensor, out_path / f"{filename}.pt")
            return f"{filename}.pt"
        # Shard along first dimension
        dim0 = tensor.shape[0] if tensor.ndim > 0 else 1
        if dim0 <= 1:
            torch.save(tensor, out_path / f"{filename}.pt")
            return f"{filename}.pt"
        bytes_per_row = max(1, bytes_total // dim0)
        rows_per_shard = max(1, (shard_mb * 1024 * 1024) // bytes_per_row)
        parts = []
        start = 0
        idx = 0
        while start < dim0:
            end = min(dim0, start + rows_per_shard)
            shard = tensor[start:end].contiguous()
            p = out_path / f"{filename}_part{idx}.pt"
            torch.save(shard, p)
            parts.append(p.name)
            start = end
            idx += 1
        index = out_path / f"{filename}.index.json"
        with open(index, "w") as f:
            json.dump({"parts": parts, "sharded": True}, f)
        return f"{filename}.index.json"

    def _save_tensor(self, tensor: torch.Tensor, out_path: Path, base_name: str) -> str:
        return self._save_linear_weight(tensor, out_path, base_name)

    def _extract_dense_ffn_layer(
        self, ffn_module: nn.Module, layer_idx: int, layer_dir: Path
    ) -> Dict[str, Any]:
        """Extract standard dense FFN (non-MoE)."""
        weights_data: Dict[str, Any] = {}
        projection_patterns = {
            "gate_proj": ["gate_proj", "w1", "fc1"],
            "up_proj": ["up_proj", "w3", "fc2"],
            "down_proj": ["down_proj", "w2", "fc3", "dense"],
        }
        for proj_type, possible_names in projection_patterns.items():
            for name in possible_names:
                if hasattr(ffn_module, name):
                    proj_module = getattr(ffn_module, name)
                    if hasattr(proj_module, "weight"):
                        file_path = self._save_linear_weight(
                            proj_module.weight, layer_dir, proj_type
                        )
                        entry = {
                            "shape": list(proj_module.weight.shape),
                            "original_name": name,
                            "file_path": file_path,
                            "sha256": self._sha256(proj_module.weight),
                        }
                        if (
                            hasattr(proj_module, "bias")
                            and proj_module.bias is not None
                        ):
                            bias = proj_module.bias.detach().cpu()
                            if self.config.get("use_safetensors") and (
                                _safetensors_save is not None
                            ):
                                _safetensors_save(
                                    {"bias": bias},
                                    layer_dir / f"{proj_type}_bias.safetensors",
                                )
                                entry["bias_file"] = f"{proj_type}_bias.safetensors"
                            else:
                                torch.save(bias, layer_dir / f"{proj_type}_bias.pt")
                                entry["bias_file"] = f"{proj_type}_bias.pt"
                            entry["bias_shape"] = list(bias.shape)
                            entry["bias_sha256"] = self._sha256(bias)
                        weights_data[proj_type] = entry
                        break
        activation_function = self._detect_activation_function(ffn_module)
        config_data = {
            "layer_idx": layer_idx,
            "activation_function": activation_function,
            "projections": weights_data,
            "hidden_size": (
                weights_data.get("gate_proj", {}).get("shape", [0, 0])[1]
                if "gate_proj" in weights_data
                else 0
            ),
            "intermediate_size": (
                weights_data.get("gate_proj", {}).get("shape", [0, 0])[0]
                if "gate_proj" in weights_data
                else 0
            ),
            "ffn_type": "dense",
        }
        with open(layer_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        return {"weights": weights_data, "config": config_data}

    def _is_moe_ffn(self, ffn_module: nn.Module) -> bool:
        """Heuristic check if FFN module is MoE with experts and router (broader coverage)."""
        # Common attribute names
        for cname in (
            "experts",
            "moe_experts",
            "experts_list",
            "expert_modules",
            "experts_module",
            "mixture_of_experts",
        ):
            if hasattr(ffn_module, cname) and isinstance(
                getattr(ffn_module, cname), (nn.ModuleList, list, tuple)
            ):
                return True
        # Some impls wrap experts under nested modules or with names containing 'experts'
        for name, sub in ffn_module.named_children():
            if ("expert" in name.lower()) and isinstance(
                sub, (nn.ModuleList, list, tuple)
            ):
                return True
            for cname in ("experts", "moe_experts", "experts_list"):
                if hasattr(sub, cname) and isinstance(
                    getattr(sub, cname), (nn.ModuleList, list, tuple)
                ):
                    return True
        return False

    def _extract_moe_ffn_layer(
        self, ffn_module: nn.Module, layer_idx: int, layer_dir: Path
    ) -> Dict[str, Any]:
        """Extract MoE FFN: per-expert projections and router/gate if present."""
        experts_container = None
        experts_name = None
        # Locate experts container (broader search)
        for cname in (
            "experts",
            "moe_experts",
            "experts_list",
            "expert_modules",
            "experts_module",
            "mixture_of_experts",
        ):
            if hasattr(ffn_module, cname):
                cont = getattr(ffn_module, cname)
                if isinstance(cont, (nn.ModuleList, list, tuple)) and len(cont) > 0:
                    experts_container = cont
                    experts_name = cname
                    break
        if experts_container is None:
            for name, sub in ffn_module.named_children():
                if ("expert" in name.lower()) and isinstance(
                    sub, (nn.ModuleList, list, tuple)
                ):
                    experts_container = sub
                    experts_name = name
                    break
                for cname in ("experts", "moe_experts", "experts_list"):
                    if hasattr(sub, cname):
                        cont = getattr(sub, cname)
                        if (
                            isinstance(cont, (nn.ModuleList, list, tuple))
                            and len(cont) > 0
                        ):
                            experts_container = cont
                            experts_name = f"{name}.{cname}"
                            break
                if experts_container is not None:
                    break
        if experts_container is None:
            # fallback to dense extraction if experts not found
            return self._extract_dense_ffn_layer(ffn_module, layer_idx, layer_dir)

        experts_metadata = {}
        num_experts = len(experts_container)
        # Extract each expert
        for idx, expert in enumerate(experts_container):
            exp_dir = layer_dir / f"expert_{idx}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            proj_info = {}
            projection_patterns = {
                "gate_proj": ["gate_proj", "w1", "fc1"],
                "up_proj": ["up_proj", "w3", "fc2"],
                "down_proj": ["down_proj", "w2", "fc3", "dense"],
            }
            for proj_type, names in projection_patterns.items():
                for name in names:
                    if hasattr(expert, name):
                        mod = getattr(expert, name)
                        if hasattr(mod, "weight"):
                            file_path = self._save_linear_weight(
                                mod.weight, exp_dir, proj_type
                            )
                            proj_info[proj_type] = {
                                "shape": list(mod.weight.shape),
                                "original_name": name,
                                "file_path": f"expert_{idx}/{file_path}",
                            }
                            break
            experts_metadata[f"expert_{idx}"] = {
                "projections": proj_info,
                "activation_function": self._detect_activation_function(expert),
            }

        # Extract router/gate if present
        router_info = {}
        router_module = None
        for candidate in (
            "router",
            "gate",
            "routing",
            "switch",
            "gating",
            "gating_network",
            "router_gate",
            "router_layer",
        ):  # common names
            if hasattr(ffn_module, candidate):
                router_module = getattr(ffn_module, candidate)
                break
        if router_module is None:
            # search nested
            for name, sub in ffn_module.named_children():
                if any(
                    k in name.lower()
                    for k in ("router", "gate", "routing", "switch", "gating")
                ):
                    router_module = sub
                    break
        if router_module is not None:
            router_dir = layer_dir / "router"
            router_dir.mkdir(parents=True, exist_ok=True)
            # Save linear weights if present
            if hasattr(router_module, "weight"):
                fp = self._save_linear_weight(
                    router_module.weight, router_dir, "router"
                )
                router_info["weight"] = {
                    "shape": list(router_module.weight.shape),
                    "file_path": f"router/{fp}",
                }
            # Save bias if present
            if (
                hasattr(router_module, "bias")
                and getattr(router_module, "bias") is not None
            ):
                if self.config.get("use_safetensors") and (
                    _safetensors_save is not None
                ):
                    _safetensors_save(
                        {"bias": router_module.bias.detach().cpu()},
                        router_dir / "router_bias.safetensors",
                    )
                    router_info["bias_file"] = "router/router_bias.safetensors"
                else:
                    torch.save(
                        router_module.bias.detach().cpu(), router_dir / "router_bias.pt"
                    )
                    router_info["bias_file"] = "router/router_bias.pt"

        # Determine dims from first expert if possible
        hidden_size = 0
        intermediate_size = 0
        first = experts_metadata.get("expert_0", {}).get("projections", {})
        if "gate_proj" in first:
            shp = first["gate_proj"].get("shape", [0, 0])
            intermediate_size, hidden_size = shp[0], shp[1]

        # Try to capture router hyperparams if present
        def _getattr_any(obj, names, default=None):
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return default

        top_k = _getattr_any(
            ffn_module, ["top_k", "k", "num_experts_per_tok", "num_active_experts"]
        )
        capacity_factor = _getattr_any(
            ffn_module, ["capacity_factor", "capacity", "router_capacity"]
        )
        router_z_loss = _getattr_any(ffn_module, ["router_z_loss_coef", "z_loss_coef"])
        aux_loss_coef = _getattr_any(
            ffn_module, ["aux_loss_coef", "load_balancing_loss_coef"]
        )
        routing_strategy = _getattr_any(
            ffn_module, ["routing_strategy", "router_type", "gate_type"]
        )
        score_scale = _getattr_any(ffn_module, ["score_scale", "router_score_scale"])
        jitter = _getattr_any(ffn_module, ["router_jitter_noise", "jitter_noise"])
        router_temp = _getattr_any(ffn_module, ["router_temperature", "temperature"])
        config_data = {
            "layer_idx": layer_idx,
            "ffn_type": "moe",
            "experts_container": experts_name,
            "num_experts": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "router": router_info,
            "top_k": (
                int(top_k)
                if isinstance(top_k, (int,))
                else (top_k if top_k is not None else None)
            ),
            "capacity_factor": (
                float(capacity_factor) if capacity_factor is not None else None
            ),
            "router_z_loss_coef": (
                float(router_z_loss) if router_z_loss is not None else None
            ),
            "aux_loss_coef": (
                float(aux_loss_coef) if aux_loss_coef is not None else None
            ),
            "routing_strategy": (
                str(routing_strategy) if routing_strategy is not None else None
            ),
            "score_scale": float(score_scale) if score_scale is not None else None,
            "router_jitter_noise": float(jitter) if jitter is not None else None,
            "router_temperature": (
                float(router_temp) if router_temp is not None else None
            ),
        }
        with open(layer_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        return {"experts": experts_metadata, "config": config_data}

    def _detect_activation_function(self, ffn_module: nn.Module) -> str:
        """Detect the activation function used in the FFN."""
        # Look for activation function in the module
        for name, module in ffn_module.named_modules():
            if isinstance(module, torch.nn.SiLU):
                return "silu"
            elif isinstance(module, torch.nn.GELU):
                return "gelu"
            elif isinstance(module, torch.nn.ReLU):
                return "relu"
            elif hasattr(module, "activation_fn"):
                return str(module.activation_fn)

        # Check module attributes for activation function names
        if hasattr(ffn_module, "activation_fn"):
            return str(ffn_module.activation_fn)

        return "unknown"

    def _extract_lm_head(self, model: nn.Module, output_path: Path) -> Dict[str, Any]:
        """Extract language modeling head."""
        logger.info("Extracting LM head...")

        lm_head = model.get_output_embeddings()
        if lm_head is None:
            logger.warning("No LM head found")
            return {}

        weight_tensor = lm_head.weight.detach().cpu()

        # Check for weight tying with input embeddings
        input_embeddings = model.get_input_embeddings()
        weight_tied = False
        if input_embeddings is not None:
            weight_tied = torch.equal(
                weight_tensor, input_embeddings.weight.detach().cpu()
            )

        lm_head_dir = output_path / "lm_head"
        lm_head_dir.mkdir(parents=True, exist_ok=True)

        # Save LM head weights only if not tied to input embeddings
        if not weight_tied:
            if self.config.get("use_safetensors") and (_safetensors_save is not None):
                _safetensors_save(
                    {"weight": weight_tensor}, lm_head_dir / "lm_head.safetensors"
                )
            else:
                torch.save(weight_tensor, lm_head_dir / "lm_head.pt")

        # Save metadata
        config_data = {
            "dimensions": list(weight_tensor.shape),
            "weight_tied": weight_tied,
            "vocab_size": weight_tensor.shape[0],
            "hidden_size": weight_tensor.shape[1],
        }

        with open(lm_head_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        file_path = None
        if not weight_tied:
            file_path = (
                "lm_head/lm_head.safetensors"
                if (
                    self.config.get("use_safetensors")
                    and (_safetensors_save is not None)
                )
                else "lm_head/lm_head.pt"
            )

        return {
            "config": config_data,
            "file_path": file_path,
            "weight_tied": weight_tied,
        }
