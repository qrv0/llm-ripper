"""Model loading utilities for LLM Ripper."""

import torch
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    AutoModelForCausalLM
)
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and managing transformer models."""
    
    def __init__(self, cache_dir: str = "./models", device: str = "auto"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._determine_device(device)
        
    def _determine_device(self, device: str) -> torch.device:
        """Determine the appropriate device for model loading."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def load_model_and_tokenizer(
        self, 
        model_name: str,
        model_type: str = "causal_lm",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False
    ) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
        """
        Load a model, tokenizer, and config.
        
        Args:
            model_name: HuggingFace model identifier or local path
            model_type: Type of model to load ('causal_lm', 'base')
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Tuple of (model, tokenizer, config)
        """
        logger.info(f"Loading model: {model_name}")
        
        # Load configuration
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Model loading arguments
        # Choose dtype for GPU automatically (bf16 if supported, else fp16)
        torch_dtype = torch.float32
        if self.device.type == "cuda":
            try:
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                torch_dtype = torch.float16
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        
        # Add quantization options
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["device_map"] = self.device
        
        # Load model based on type
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        elif model_type == "base":
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move to device if not using quantization
        if not (load_in_8bit or load_in_4bit):
            model = model.to(self.device)
        
        model.eval()
        
        logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer, config.to_dict()
    
    def get_model_architecture_info(self, model: torch.nn.Module, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract architectural information from a model.
        
        Args:
            model: The loaded model
            config: Model configuration dictionary
            
        Returns:
            Dictionary containing architectural information
        """
        arch_info = {
            "model_type": config.get("model_type", "unknown"),
            "hidden_size": config.get("hidden_size", 0),
            "num_hidden_layers": config.get("num_hidden_layers", 0),
            "num_attention_heads": config.get("num_attention_heads", 0),
            "intermediate_size": config.get("intermediate_size", 0),
            "vocab_size": config.get("vocab_size", 0),
            "max_position_embeddings": config.get("max_position_embeddings", 0),
        }
        
        # Add attention-specific information
        if "num_key_value_heads" in config:
            arch_info["num_key_value_heads"] = config["num_key_value_heads"]
            arch_info["attention_type"] = "GQA" if config["num_key_value_heads"] < config["num_attention_heads"] else "MHA"
        else:
            arch_info["attention_type"] = "MHA"
        
        # Add activation function
        arch_info["hidden_act"] = config.get("hidden_act", "unknown")
        
        # Add model size information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        arch_info["total_parameters"] = total_params
        arch_info["trainable_parameters"] = trainable_params
        arch_info["parameter_size_mb"] = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return arch_info
    
    def analyze_attention_pattern(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the attention pattern of a model.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Dictionary containing attention pattern analysis
        """
        attention_info = {}
        
        num_heads = config.get("num_attention_heads", 0)
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        
        if num_kv_heads == num_heads:
            attention_info["pattern"] = "MHA"
            attention_info["description"] = "Multi-Head Attention - each head has its own K,V projections"
        elif num_kv_heads == 1:
            attention_info["pattern"] = "MQA"
            attention_info["description"] = "Multi-Query Attention - all heads share single K,V projections"
        elif num_kv_heads < num_heads:
            attention_info["pattern"] = "GQA"
            attention_info["description"] = f"Grouped-Query Attention - {num_heads // num_kv_heads} query heads per K,V group"
            attention_info["group_size"] = num_heads // num_kv_heads
        
        attention_info["num_query_heads"] = num_heads
        attention_info["num_key_value_heads"] = num_kv_heads
        
        return attention_info
    
    def get_layer_modules(self, model: torch.nn.Module) -> Dict[str, Dict[str, torch.nn.Module]]:
        """
        Extract layer modules from a transformer model.
        
        Args:
            model: The loaded model
            
        Returns:
            Dictionary mapping layer indices to their modules
        """
        layers = {}
        
        # Common patterns for transformer layers
        layer_patterns = [
            "layers",           # Common pattern
            "h",               # GPT-style
            "transformer.h",   # Alternative GPT-style
            "decoder.layers",  # Decoder-only models
            "encoder.layers"   # Encoder models
        ]
        
        for pattern in layer_patterns:
            try:
                layer_container = model
                for attr in pattern.split('.'):
                    layer_container = getattr(layer_container, attr)
                
                for i, layer in enumerate(layer_container):
                    layer_modules = {}
                    
                    # Extract attention modules
                    for name, module in layer.named_modules():
                        if any(attn_name in name.lower() for attn_name in ['attention', 'attn']):
                            layer_modules[f"attention.{name}"] = module
                        elif any(ffn_name in name.lower() for ffn_name in ['ffn', 'mlp', 'feed_forward']):
                            layer_modules[f"ffn.{name}"] = module
                        elif any(norm_name in name.lower() for norm_name in ['norm', 'layernorm']):
                            layer_modules[f"norm.{name}"] = module
                    
                    layers[i] = layer_modules
                
                break  # Found layers, no need to try other patterns
                
            except AttributeError:
                continue
        
        return layers