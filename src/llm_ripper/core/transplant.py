"""
Knowledge transplantation module for LLM Ripper.

This module implements Part III of the framework: Recomposition and validation
in new architectures using Bridge Networks and adapters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from safetensors.torch import load_file
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class TransplantConfig:
    """Configuration for knowledge transplantation."""
    source_component: str
    target_layer: int
    bridge_hidden_size: int
    freeze_donor: bool
    freeze_target: bool
    strategy: str  # "embedding_init", "module_injection", "adapter_fusion"


class BridgeNetwork(nn.Module):
    """
    Bridge Network for adapting between different model dimensions.
    
    Implements the adapter architecture described in Section 6.1.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Bottleneck architecture: down -> activation -> up
        self.down_projection = nn.Linear(input_dim, hidden_dim)
        self.up_projection = nn.Linear(hidden_dim, output_dim)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if dimensions match
        self.use_residual = (input_dim == output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.normal_(self.down_projection.weight, std=0.02)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.normal_(self.up_projection.weight, std=0.02)
        nn.init.zeros_(self.up_projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bridge network."""
        residual = x if self.use_residual else None
        
        # Bottleneck transformation
        h = self.down_projection(x)
        h = self.activation(h)
        h = self.dropout(h)
        output = self.up_projection(h)
        
        # Add residual connection if applicable
        if residual is not None:
            output = output + residual
        
        return output


class TransplantedModule(nn.Module):
    """
    Wrapper for transplanted components with bridge networks.
    
    Implements the module injection strategy from Section 6.2.
    """
    
    def __init__(
        self,
        donor_module: nn.Module,
        input_bridge: Optional[BridgeNetwork] = None,
        output_bridge: Optional[BridgeNetwork] = None,
        freeze_donor: bool = True
    ):
        super().__init__()
        
        self.donor_module = donor_module
        self.input_bridge = input_bridge
        self.output_bridge = output_bridge
        
        # Freeze donor weights if specified
        if freeze_donor:
            for param in self.donor_module.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass through transplanted module."""
        
        # Apply input bridge if present
        if self.input_bridge is not None:
            x = self.input_bridge(x)
        
        # Pass through donor module
        if args or kwargs:
            output = self.donor_module(x, *args, **kwargs)
        else:
            output = self.donor_module(x)
        
        # Apply output bridge if present
        if self.output_bridge is not None:
            output = self.output_bridge(output)
        
        return output


class KnowledgeTransplanter:
    """
    Handles transplantation of knowledge components between models.
    
    Implements Section 6: Knowledge reinjection strategies.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_loader = ModelLoader(
            cache_dir=config.get("model_cache_dir"),
            device=config.get("device")
        )
    
    def transplant_knowledge(
        self,
        source_knowledge_bank: str,
        target_model_name: str,
        transplant_configs: List[TransplantConfig],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Transplant knowledge from source bank to target model.
        
        Args:
            source_knowledge_bank: Path to knowledge bank directory
            target_model_name: Name of target model
            transplant_configs: List of transplant configurations
            output_dir: Directory to save transplanted model
            
        Returns:
            Dictionary containing transplant metadata
        """
        logger.info(f"Starting knowledge transplantation to: {target_model_name}")
        
        # Load target model
        target_model, target_tokenizer, target_config = self.model_loader.load_model_and_tokenizer(
            target_model_name,
            load_in_8bit=self.config.get("load_in_8bit"),
            load_in_4bit=self.config.get("load_in_4bit"),
            trust_remote_code=self.config.get("trust_remote_code"),
        )
        
        # Load source knowledge bank metadata
        knowledge_bank_path = Path(source_knowledge_bank)
        with open(knowledge_bank_path / "extraction_metadata.json", "r") as f:
            source_metadata = json.load(f)
        
        transplant_metadata = {
            "source_model": source_metadata["source_model"],
            "target_model": target_model_name,
            "transplant_configs": [config.__dict__ for config in transplant_configs],
            "transplanted_components": {}
        }
        
        # Apply each transplant configuration
        for transplant_config in transplant_configs:
            component_result = self._apply_transplant_config(
                target_model,
                target_config,
                knowledge_bank_path,
                transplant_config
            )
            
            transplant_metadata["transplanted_components"][
                f"{transplant_config.strategy}_{transplant_config.source_component}"
            ] = component_result
        
        # Save transplanted model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        target_model.save_pretrained(output_path / "model")
        target_tokenizer.save_pretrained(output_path / "tokenizer")
        
        # Save transplant metadata
        with open(output_path / "transplant_metadata.json", "w") as f:
            json.dump(transplant_metadata, f, indent=2)
        
        logger.info(f"Transplantation completed. Model saved to: {output_path}")
        
        return transplant_metadata
    
    def _apply_transplant_config(
        self,
        target_model: nn.Module,
        target_config: Dict[str, Any],
        knowledge_bank_path: Path,
        transplant_config: TransplantConfig
    ) -> Dict[str, Any]:
        """Apply a specific transplant configuration."""
        
        strategy = transplant_config.strategy
        
        if strategy == "embedding_init":
            return self._transplant_embeddings(
                target_model, target_config, knowledge_bank_path, transplant_config
            )
        elif strategy == "module_injection":
            return self._transplant_module(
                target_model, target_config, knowledge_bank_path, transplant_config
            )
        elif strategy == "adapter_fusion":
            return self._transplant_with_adapter_fusion(
                target_model, target_config, knowledge_bank_path, transplant_config
            )
        else:
            raise ValueError(f"Unknown transplant strategy: {strategy}")
    
    def _transplant_embeddings(
        self,
        target_model: nn.Module,
        target_config: Dict[str, Any],
        knowledge_bank_path: Path,
        transplant_config: TransplantConfig
    ) -> Dict[str, Any]:
        """
        Transplant embeddings using initialization strategy.
        
        Implements Section 6.2: Embedding initialization.
        """
        logger.info("Transplanting embeddings...")
        
        # Load source embeddings
        embeddings_dir = knowledge_bank_path / "embeddings"
        
        with open(embeddings_dir / "config.json", "r") as f:
            source_embedding_config = json.load(f)
        
        if (embeddings_dir / "embeddings.safetensors").exists():
            source_embeddings = load_file(embeddings_dir / "embeddings.safetensors")["weight"]
        else:
            source_embeddings = torch.load(embeddings_dir / "embeddings.pt")
        
        # Get target embedding layer
        target_embeddings = target_model.get_input_embeddings()
        
        source_dim = source_embeddings.shape[1]
        target_dim = target_embeddings.weight.shape[1]
        
        # Create bridge network if dimensions don't match
        if source_dim != target_dim:
            bridge = BridgeNetwork(
                input_dim=source_dim,
                output_dim=target_dim,
                hidden_dim=transplant_config.bridge_hidden_size
            )
            
            # Transform source embeddings
            with torch.no_grad():
                transformed_embeddings = bridge(source_embeddings)
            
            # Initialize target embeddings
            target_embeddings.weight.data[:transformed_embeddings.shape[0]] = transformed_embeddings
            
            # Add bridge to model for training
            if not hasattr(target_model, 'embedding_bridge'):
                target_model.embedding_bridge = bridge
        else:
            # Direct copy if dimensions match
            with torch.no_grad():
                min_vocab = min(source_embeddings.shape[0], target_embeddings.weight.shape[0])
                target_embeddings.weight.data[:min_vocab] = source_embeddings[:min_vocab]
        
        # Freeze embeddings if specified
        if transplant_config.freeze_donor:
            target_embeddings.weight.requires_grad = False
        
        return {
            "strategy": "embedding_init",
            "source_dim": source_dim,
            "target_dim": target_dim,
            "bridge_used": source_dim != target_dim,
            "vocab_overlap": min(source_embeddings.shape[0], target_embeddings.weight.shape[0])
        }
    
    def _transplant_module(
        self,
        target_model: nn.Module,
        target_config: Dict[str, Any],
        knowledge_bank_path: Path,
        transplant_config: TransplantConfig
    ) -> Dict[str, Any]:
        """
        Transplant a complete module (attention head or FFN).
        
        Implements Section 6.2: Module injection strategy.
        """
        logger.info(f"Transplanting module: {transplant_config.source_component}")
        
        # Load source component
        component_data = self._load_component(knowledge_bank_path, transplant_config.source_component)
        
        if component_data is None:
            raise ValueError(f"Could not load component: {transplant_config.source_component}")
        
        # Create donor module from loaded weights
        donor_module = self._create_donor_module(component_data)
        
        # Get target layer for injection
        target_layer = self._get_target_layer(target_model, transplant_config.target_layer)
        
        # Create bridge networks if needed
        input_bridge, output_bridge = self._create_bridges_for_module(
            donor_module, target_layer, transplant_config
        )
        
        # Create transplanted module
        transplanted_module = TransplantedModule(
            donor_module=donor_module,
            input_bridge=input_bridge,
            output_bridge=output_bridge,
            freeze_donor=transplant_config.freeze_donor
        )
        
        # Inject into target model
        self._inject_module(target_model, transplanted_module, transplant_config.target_layer)
        
        return {
            "strategy": "module_injection",
            "source_component": transplant_config.source_component,
            "target_layer": transplant_config.target_layer,
            "bridges_created": {
                "input_bridge": input_bridge is not None,
                "output_bridge": output_bridge is not None
            }
        }
    
    def _transplant_with_adapter_fusion(
        self,
        target_model: nn.Module,
        target_config: Dict[str, Any],
        knowledge_bank_path: Path,
        transplant_config: TransplantConfig
    ) -> Dict[str, Any]:
        """
        Transplant using AdapterFusion strategy.
        
        Implements Section 6.3: Advanced composition with AdapterFusion.
        """
        logger.info("Transplanting with AdapterFusion...")
        
        # This is a simplified implementation
        # Full AdapterFusion would require multiple donor modules and fusion layers
        
        # For now, implement as module injection with additional fusion layer
        module_result = self._transplant_module(
            target_model, target_config, knowledge_bank_path, transplant_config
        )
        
        # Add fusion layer (placeholder)
        fusion_layer = nn.Linear(target_config.get("hidden_size", 768), target_config.get("hidden_size", 768))
        
        if not hasattr(target_model, 'adapter_fusion_layers'):
            target_model.adapter_fusion_layers = nn.ModuleList()
        
        target_model.adapter_fusion_layers.append(fusion_layer)
        
        module_result["strategy"] = "adapter_fusion"
        module_result["fusion_layer_added"] = True
        
        return module_result
    
    def _load_component(self, knowledge_bank_path: Path, component_name: str) -> Optional[Dict[str, Any]]:
        """Load a component from the knowledge bank."""
        
        # Parse component name (e.g., "layer_5_attention", "layer_3_ffn")
        parts = component_name.split("_")
        
        if "attention" in component_name or "head" in component_name:
            component_dir = knowledge_bank_path / "heads" / f"layer_{parts[1]}"
        elif "ffn" in component_name or "mlp" in component_name:
            component_dir = knowledge_bank_path / "ffns" / f"layer_{parts[1]}"
        elif "embeddings" in component_name:
            component_dir = knowledge_bank_path / "embeddings"
        elif "lm_head" in component_name:
            component_dir = knowledge_bank_path / "lm_head"
        else:
            return None
        
        if not component_dir.exists():
            return None
        
        # Load configuration
        with open(component_dir / "config.json", "r") as f:
            config = json.load(f)
        
        # Load weights
        weights = {}
        for weight_file in component_dir.glob("*.safetensors"):
            weight_name = weight_file.stem
            weights[weight_name] = load_file(str(weight_file))
        
        for weight_file in component_dir.glob("*.pt"):
            weight_name = weight_file.stem
            if weight_name not in weights:  # Prefer safetensors
                weights[weight_name] = torch.load(str(weight_file))
        
        return {
            "config": config,
            "weights": weights
        }
    
    def _create_donor_module(self, component_data: Dict[str, Any]) -> nn.Module:
        """Create a donor module from loaded component data."""
        
        config = component_data["config"]
        weights = component_data["weights"]
        
        # Create appropriate module based on component type
        if "attention_type" in config:
            return self._create_attention_module(config, weights)
        elif "activation_function" in config:
            return self._create_ffn_module(config, weights)
        else:
            # Generic linear module
            return self._create_generic_module(config, weights)
    
    def _create_attention_module(self, config: Dict[str, Any], weights: Dict[str, Any]) -> nn.Module:
        """Create an attention module from config and weights."""
        
        hidden_size = config.get("hidden_size", 768)
        num_heads = config.get("num_heads", 12)
        
        # Simplified attention module
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads
                
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)
            
            def forward(self, x):
                # Simplified attention computation
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                
                # Simple attention (not full multi-head)
                attn_output = torch.matmul(q, k.transpose(-2, -1))
                attn_output = torch.softmax(attn_output, dim=-1)
                attn_output = torch.matmul(attn_output, v)
                
                return self.o_proj(attn_output)
        
        module = SimpleAttention()
        
        # Load weights
        if "q_proj" in weights and "weight" in weights["q_proj"]:
            module.q_proj.weight.data = weights["q_proj"]["weight"]
        if "k_proj" in weights and "weight" in weights["k_proj"]:
            module.k_proj.weight.data = weights["k_proj"]["weight"]
        if "v_proj" in weights and "weight" in weights["v_proj"]:
            module.v_proj.weight.data = weights["v_proj"]["weight"]
        if "o_proj" in weights and "weight" in weights["o_proj"]:
            module.o_proj.weight.data = weights["o_proj"]["weight"]
        
        return module
    
    def _create_ffn_module(self, config: Dict[str, Any], weights: Dict[str, Any]) -> nn.Module:
        """Create an FFN module from config and weights."""
        
        hidden_size = config.get("hidden_size", 768)
        intermediate_size = config.get("intermediate_size", 3072)
        activation = config.get("activation_function", "relu")
        
        # Simplified FFN module
        class SimpleFfn(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, intermediate_size)
                self.up_proj = nn.Linear(hidden_size, intermediate_size)
                self.down_proj = nn.Linear(intermediate_size, hidden_size)
                
                if activation == "silu":
                    self.activation = nn.SiLU()
                elif activation == "gelu":
                    self.activation = nn.GELU()
                else:
                    self.activation = nn.ReLU()
            
            def forward(self, x):
                gate = self.activation(self.gate_proj(x))
                up = self.up_proj(x)
                return self.down_proj(gate * up)
        
        module = SimpleFfn()
        
        # Load weights
        if "gate_proj" in weights and "weight" in weights["gate_proj"]:
            module.gate_proj.weight.data = weights["gate_proj"]["weight"]
        if "up_proj" in weights and "weight" in weights["up_proj"]:
            module.up_proj.weight.data = weights["up_proj"]["weight"]
        if "down_proj" in weights and "weight" in weights["down_proj"]:
            module.down_proj.weight.data = weights["down_proj"]["weight"]
        
        return module
    
    def _create_generic_module(self, config: Dict[str, Any], weights: Dict[str, Any]) -> nn.Module:
        """Create a generic linear module."""
        
        # Find the main weight tensor
        main_weight = None
        for weight_dict in weights.values():
            if "weight" in weight_dict:
                main_weight = weight_dict["weight"]
                break
        
        if main_weight is None:
            raise ValueError("No weight tensor found in component")
        
        input_dim, output_dim = main_weight.shape[1], main_weight.shape[0]
        
        module = nn.Linear(input_dim, output_dim)
        module.weight.data = main_weight
        
        return module
    
    def _get_target_layer(self, target_model: nn.Module, layer_idx: int) -> nn.Module:
        """Get a specific layer from the target model."""
        
        # Common patterns for accessing transformer layers
        layer_patterns = [
            f"transformer.h.{layer_idx}",
            f"model.layers.{layer_idx}",
            f"transformer.layers.{layer_idx}",
            f"h.{layer_idx}",
            f"layers.{layer_idx}",
        ]
        
        for pattern in layer_patterns:
            try:
                module = target_model
                for attr in pattern.split('.'):
                    module = getattr(module, attr)
                return module
            except AttributeError:
                continue
        
        raise ValueError(f"Could not find layer {layer_idx} in target model")
    
    def _create_bridges_for_module(
        self,
        donor_module: nn.Module,
        target_layer: nn.Module,
        transplant_config: TransplantConfig
    ) -> Tuple[Optional[BridgeNetwork], Optional[BridgeNetwork]]:
        """Create bridge networks for dimensional compatibility."""
        
        # This is a simplified implementation
        # In practice, you'd need to analyze the actual input/output dimensions
        
        # For demonstration, assume we need bridges
        bridge_hidden_size = transplant_config.bridge_hidden_size
        
        input_bridge = BridgeNetwork(
            input_dim=768,  # Target model hidden size
            output_dim=768,  # Donor module expected input
            hidden_dim=bridge_hidden_size
        )
        
        output_bridge = BridgeNetwork(
            input_dim=768,  # Donor module output
            output_dim=768,  # Target model expected size
            hidden_dim=bridge_hidden_size
        )
        
        return input_bridge, output_bridge
    
    def _inject_module(
        self,
        target_model: nn.Module,
        transplanted_module: TransplantedModule,
        target_layer: int
    ) -> None:
        """Inject transplanted module into target model."""
        
        # Add as a new attribute to the model
        if not hasattr(target_model, 'transplanted_modules'):
            target_model.transplanted_modules = nn.ModuleDict()
        
        target_model.transplanted_modules[f"layer_{target_layer}"] = transplanted_module