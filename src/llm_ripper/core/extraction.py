"""
Knowledge extraction module for LLM Ripper.

This module implements Part I of the framework: Architectural dissection and knowledge extraction.
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import save_file, load_file
import h5py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader

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
            cache_dir=config.get("model_cache_dir"),
            device=config.get("device")
        )
        
    def extract_model_components(
        self, 
        model_name: str, 
        output_dir: str,
        components: List[str] = None
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
        
        # Load model
        model, tokenizer, config = self.model_loader.load_model_and_tokenizer(
            model_name,
            load_in_8bit=self.config.get("load_in_8bit"),
            load_in_4bit=self.config.get("load_in_4bit"),
            trust_remote_code=self.config.get("trust_remote_code"),
        )
        arch_info = self.model_loader.get_model_architecture_info(model, config)
        
        # Create output directory structure
        output_path = Path(output_dir)
        self._create_knowledge_bank_structure(output_path)
        
        extraction_metadata = {
            "source_model": model_name,
            "architecture_info": arch_info,
            "extracted_components": {},
            "extraction_config": self.config.config.copy()
        }
        
        # Extract components
        if "embeddings" in components:
            embedding_metadata = self._extract_embeddings(model, output_path)
            extraction_metadata["extracted_components"]["embeddings"] = embedding_metadata
            
        if "attention_heads" in components:
            attention_metadata = self._extract_attention_heads(model, config, output_path)
            extraction_metadata["extracted_components"]["attention_heads"] = attention_metadata
            
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
        directories = [
            "embeddings",
            "heads",
            "ffns", 
            "lm_head",
            "concepts",
            "metadata"
        ]
        
        for directory in directories:
            (output_path / directory).mkdir(parents=True, exist_ok=True)
    
    def _extract_embeddings(self, model: nn.Module, output_path: Path) -> Dict[str, Any]:
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
                output_embeddings.weight.detach().cpu()
            )
        
        # Save embeddings
        if self.config.get("use_safetensors"):
            save_file(
                {"weight": weight_tensor},
                output_path / "embeddings" / "embeddings.safetensors"
            )
        else:
            torch.save(weight_tensor, output_path / "embeddings" / "embeddings.pt")
        
        # Save metadata
        metadata = ComponentMetadata(
            component_type="embeddings",
            layer_idx=None,
            head_idx=None,
            dimensions=tuple(weight_tensor.shape),
            source_model=model.config.name_or_path if hasattr(model, 'config') else "unknown"
        )
        
        config_data = {
            "dimensions": list(weight_tensor.shape),
            "weight_tied": weight_tied,
            "vocab_size": weight_tensor.shape[0],
            "hidden_size": weight_tensor.shape[1]
        }
        
        with open(output_path / "embeddings" / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Embeddings extracted: {weight_tensor.shape}")
        
        return {
            "metadata": metadata.__dict__,
            "config": config_data,
            "file_path": "embeddings/embeddings.safetensors" if self.config.get("use_safetensors") else "embeddings/embeddings.pt"
        }
    
    def _extract_attention_heads(self, model: nn.Module, config: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
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
        self, 
        model: nn.Module, 
        layer_idx: int, 
        attention_type: str,
        output_path: Path
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
            return self._extract_gqa_weights(attention_module, layer_idx, layer_dir, attention_type)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
    
    def _get_attention_module(self, model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
        """Get the attention module for a specific layer."""
        # Common patterns for accessing transformer layers
        layer_patterns = [
            f"transformer.h.{layer_idx}.attn",
            f"model.layers.{layer_idx}.self_attn",
            f"transformer.layers.{layer_idx}.attention",
            f"h.{layer_idx}.attn",
            f"layers.{layer_idx}.self_attn",
        ]
        
        for pattern in layer_patterns:
            try:
                module = model
                for attr in pattern.split('.'):
                    module = getattr(module, attr)
                return module
            except AttributeError:
                continue
        
        return None
    
    def _extract_mha_weights(
        self, 
        attention_module: nn.Module, 
        layer_idx: int, 
        output_dir: Path
    ) -> Dict[str, Any]:
        """Extract weights for Multi-Head Attention."""
        weights_data = {}
        
        # Extract Q, K, V, O projection weights
        projection_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
        for proj_name in projection_names:
            if hasattr(attention_module, proj_name):
                proj_module = getattr(attention_module, proj_name)
                weight = proj_module.weight.detach().cpu()
                
                if self.config.get("use_safetensors"):
                    save_file(
                        {"weight": weight},
                        output_dir / f"{proj_name}.safetensors"
                    )
                else:
                    torch.save(weight, output_dir / f"{proj_name}.pt")
                
                weights_data[proj_name] = {
                    "shape": list(weight.shape),
                    "file_path": f"{proj_name}.safetensors" if self.config.get("use_safetensors") else f"{proj_name}.pt"
                }
        
        # Save configuration
        config_data = {
            "attention_type": "MHA",
            "layer_idx": layer_idx,
            "hidden_size": weights_data.get("q_proj", {}).get("shape", [0, 0])[1],
            "num_heads": getattr(attention_module, 'num_heads', 0),
            "head_dim": getattr(attention_module, 'head_dim', 0),
        }
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        return {
            "weights": weights_data,
            "config": config_data
        }
    
    def _extract_gqa_weights(
        self, 
        attention_module: nn.Module, 
        layer_idx: int, 
        output_dir: Path,
        attention_type: str
    ) -> Dict[str, Any]:
        """Extract weights for Grouped-Query Attention or Multi-Query Attention."""
        weights_data = {}
        
        # For GQA/MQA, we need to handle the shared K,V projections
        if hasattr(attention_module, 'q_proj'):
            q_weight = attention_module.q_proj.weight.detach().cpu()
            if self.config.get("use_safetensors"):
                save_file({"weight": q_weight}, output_dir / "q_proj.safetensors")
            else:
                torch.save(q_weight, output_dir / "q_proj.pt")
            
            weights_data["q_proj"] = {
                "shape": list(q_weight.shape),
                "file_path": "q_proj.safetensors" if self.config.get("use_safetensors") else "q_proj.pt"
            }
        
        # Handle shared K,V projections
        if hasattr(attention_module, 'k_proj'):
            k_weight = attention_module.k_proj.weight.detach().cpu()
            if self.config.get("use_safetensors"):
                save_file({"weight": k_weight}, output_dir / "kv_proj.safetensors")
            else:
                torch.save(k_weight, output_dir / "kv_proj.pt")
            
            weights_data["kv_proj"] = {
                "shape": list(k_weight.shape),
                "file_path": "kv_proj.safetensors" if self.config.get("use_safetensors") else "kv_proj.pt"
            }
        
        if hasattr(attention_module, 'v_proj'):
            v_weight = attention_module.v_proj.weight.detach().cpu()
            # Concatenate K and V if separate
            if "kv_proj" not in weights_data:
                kv_weight = torch.cat([
                    getattr(attention_module, 'k_proj').weight.detach().cpu(),
                    v_weight
                ], dim=0)
                
                if self.config.get("use_safetensors"):
                    save_file({"weight": kv_weight}, output_dir / "kv_proj.safetensors")
                else:
                    torch.save(kv_weight, output_dir / "kv_proj.pt")
                
                weights_data["kv_proj"] = {
                    "shape": list(kv_weight.shape),
                    "file_path": "kv_proj.safetensors" if self.config.get("use_safetensors") else "kv_proj.pt"
                }
        
        # Output projection
        if hasattr(attention_module, 'o_proj'):
            o_weight = attention_module.o_proj.weight.detach().cpu()
            if self.config.get("use_safetensors"):
                save_file({"weight": o_weight}, output_dir / "o_proj.safetensors")
            else:
                torch.save(o_weight, output_dir / "o_proj.pt")
            
            weights_data["o_proj"] = {
                "shape": list(o_weight.shape),
                "file_path": "o_proj.safetensors" if self.config.get("use_safetensors") else "o_proj.pt"
            }
        
        # Save configuration with GQA/MQA specific metadata
        config_data = {
            "attention_type": attention_type,
            "layer_idx": layer_idx,
            "hidden_size": weights_data.get("q_proj", {}).get("shape", [0, 0])[1],
            "num_query_heads": getattr(attention_module, 'num_heads', 0),
            "num_key_value_heads": getattr(attention_module, 'num_key_value_heads', 0),
            "head_dim": getattr(attention_module, 'head_dim', 0),
        }
        
        if attention_type == "GQA":
            kvh = int(config_data.get("num_key_value_heads") or 0)
            config_data["group_size"] = (config_data["num_query_heads"] // kvh) if kvh > 0 else None
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        return {
            "weights": weights_data,
            "config": config_data
        }

    def _extract_ffn_layers(self, model: nn.Module, config: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
        """Extract Feed-Forward Network layers."""
        logger.info("Extracting FFN layers...")
        
        ffn_metadata = {}
        num_layers = config.get("num_hidden_layers", 0)
        
        for layer_idx in tqdm(range(num_layers), desc="Extracting FFN layers"):
            layer_metadata = self._extract_layer_ffn(model, layer_idx, output_path)
            ffn_metadata[f"layer_{layer_idx}"] = layer_metadata
        
        return ffn_metadata
    
    def _extract_layer_ffn(self, model: nn.Module, layer_idx: int, output_path: Path) -> Dict[str, Any]:
        """Extract FFN components from a specific layer."""
        ffn_module = self._get_ffn_module(model, layer_idx)
        if ffn_module is None:
            logger.warning(f"No FFN module found for layer {layer_idx}")
            return {}
        
        layer_dir = output_path / "ffns" / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        weights_data = {}
        
        # Common FFN projection names for different architectures
        projection_patterns = {
            "gate_proj": ["gate_proj", "w1", "fc1"],
            "up_proj": ["up_proj", "w3", "fc2"],
            "down_proj": ["down_proj", "w2", "fc3", "dense"]
        }
        
        for proj_type, possible_names in projection_patterns.items():
            for name in possible_names:
                if hasattr(ffn_module, name):
                    proj_module = getattr(ffn_module, name)
                    weight = proj_module.weight.detach().cpu()
                    
                    if self.config.get("use_safetensors"):
                        save_file(
                            {"weight": weight},
                            layer_dir / f"{proj_type}.safetensors"
                        )
                    else:
                        torch.save(weight, layer_dir / f"{proj_type}.pt")
                    
                    weights_data[proj_type] = {
                        "shape": list(weight.shape),
                        "original_name": name,
                        "file_path": f"{proj_type}.safetensors" if self.config.get("use_safetensors") else f"{proj_type}.pt"
                    }
                    break
        
        # Determine activation function
        activation_function = self._detect_activation_function(ffn_module)
        
        # Save configuration
        config_data = {
            "layer_idx": layer_idx,
            "activation_function": activation_function,
            "projections": weights_data,
            "hidden_size": weights_data.get("gate_proj", {}).get("shape", [0, 0])[1] if "gate_proj" in weights_data else 0,
            "intermediate_size": weights_data.get("gate_proj", {}).get("shape", [0, 0])[0] if "gate_proj" in weights_data else 0,
        }
        
        with open(layer_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        return {
            "weights": weights_data,
            "config": config_data
        }
    
    def _get_ffn_module(self, model: nn.Module, layer_idx: int) -> Optional[nn.Module]:
        """Get the FFN module for a specific layer."""
        ffn_patterns = [
            f"transformer.h.{layer_idx}.mlp",
            f"model.layers.{layer_idx}.mlp", 
            f"transformer.layers.{layer_idx}.ffn",
            f"h.{layer_idx}.mlp",
            f"layers.{layer_idx}.mlp",
        ]
        
        for pattern in ffn_patterns:
            try:
                module = model
                for attr in pattern.split('.'):
                    module = getattr(module, attr)
                return module
            except AttributeError:
                continue
        
        return None
    
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
        if hasattr(ffn_module, 'activation_fn'):
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
                weight_tensor,
                input_embeddings.weight.detach().cpu()
            )
        
        lm_head_dir = output_path / "lm_head"
        lm_head_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LM head weights only if not tied to input embeddings
        if not weight_tied:
            if self.config.get("use_safetensors"):
                save_file(
                    {"weight": weight_tensor},
                    lm_head_dir / "lm_head.safetensors"
                )
            else:
                torch.save(weight_tensor, lm_head_dir / "lm_head.pt")
        
        # Save metadata
        config_data = {
            "dimensions": list(weight_tensor.shape),
            "weight_tied": weight_tied,
            "vocab_size": weight_tensor.shape[0],
            "hidden_size": weight_tensor.shape[1]
        }
        
        with open(lm_head_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        file_path = None
        if not weight_tied:
            file_path = "lm_head/lm_head.safetensors" if self.config.get("use_safetensors") else "lm_head/lm_head.pt"
        
        return {
            "config": config_data,
            "file_path": file_path,
            "weight_tied": weight_tied
        }