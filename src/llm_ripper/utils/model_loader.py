"""Model loading utilities for LLM Ripper."""

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from typing import Tuple, Dict, Any
import logging
import warnings as _warnings
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
            # Guard CUDA availability check to avoid noisy warnings in restricted envs
            has_cuda = False
            try:
                with _warnings.catch_warnings():
                    _warnings.filterwarnings(
                        "ignore",
                        message=r"CUDA initialization: Unexpected error.*",
                        category=UserWarning,
                        module=r"torch\.cuda.*",
                    )
                    has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False
            if has_cuda:
                return torch.device("cuda")
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
            except Exception:
                pass
            return torch.device("cpu")
        else:
            return torch.device(device)

    def load_model_and_tokenizer(
        self,
        model_name: str,
        model_type: str = "causal_lm",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
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
            model_name, cache_dir=self.cache_dir, trust_remote_code=trust_remote_code
        )
        # Apply optional attention/rope configuration hints from env
        # Some configs may ignore these fields; we set them best-effort.
        import os as _os

        attn_impl = _os.getenv("ATTN_IMPLEMENTATION")
        if attn_impl:
            try:
                setattr(config, "attn_implementation", attn_impl)
            except Exception:
                pass
        rope_scaling_env = _os.getenv("ROPE_SCALING")
        if rope_scaling_env:
            try:
                import json as _json

                rs = _json.loads(rope_scaling_env)
            except Exception:
                rs = rope_scaling_env  # allow simple strings
            try:
                setattr(config, "rope_scaling", rs)
            except Exception:
                pass

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir, trust_remote_code=trust_remote_code
        )

        # Add padding token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Model loading arguments
        # Choose dtype for GPU automatically (bf16 if supported, else fp16)
        torch_dtype = torch.float32
        if self.device.type == "cuda":
            try:
                torch_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            except Exception:
                torch_dtype = torch.float16
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }

        # Add quantization options
        def _has_bitsandbytes():
            try:
                import bitsandbytes  # noqa: F401

                return True
            except Exception:
                return False

        if load_in_8bit:
            if _has_bitsandbytes():
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            else:
                logger.warning(
                    "bitsandbytes not found; ignoring --load-in-8bit and loading in default precision."
                )
        elif load_in_4bit:
            if _has_bitsandbytes():
                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"
            else:
                logger.warning(
                    "bitsandbytes not found; ignoring --load-in-4bit and loading in default precision."
                )
        # do not set device_map for full-precision load; we'll move model with .to(self.device)

        # Load model based on type
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        elif model_type == "base":
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Move to device if not using quantization
        if not (load_in_8bit or load_in_4bit):
            try:
                model = model.to(self.device)
            except Exception:
                # Some HF models return device_map even without quantization; as fallback, keep as is
                pass

        # Optional compile for PyTorch 2.x
        import os as _os

        if _os.getenv("COMPILE_MODEL", "false").lower() == "true":
            try:
                model = torch.compile(model)  # type: ignore
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile unavailable or failed: {e}")

        model.eval()

        logger.info(
            f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        return model, tokenizer, config.to_dict()

    def get_model_architecture_info(
        self, model: torch.nn.Module, config: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            arch_info["attention_type"] = (
                "GQA"
                if config["num_key_value_heads"] < config["num_attention_heads"]
                else "MHA"
            )
        else:
            arch_info["attention_type"] = "MHA"

        # Add activation function
        arch_info["hidden_act"] = config.get("hidden_act", "unknown")

        # Add model size information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        arch_info["total_parameters"] = total_params
        arch_info["trainable_parameters"] = trainable_params
        arch_info["parameter_size_mb"] = (
            total_params * 4 / (1024 * 1024)
        )  # Assuming float32

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
            attention_info["description"] = (
                "Multi-Head Attention - each head has its own K,V projections"
            )
        elif num_kv_heads == 1:
            attention_info["pattern"] = "MQA"
            attention_info["description"] = (
                "Multi-Query Attention - all heads share single K,V projections"
            )
        elif num_kv_heads < num_heads:
            attention_info["pattern"] = "GQA"
            attention_info["description"] = (
                f"Grouped-Query Attention - {num_heads // num_kv_heads} query heads per K,V group"
            )
            attention_info["group_size"] = num_heads // num_kv_heads

        attention_info["num_query_heads"] = num_heads
        attention_info["num_key_value_heads"] = num_kv_heads

        return attention_info

    def get_layer_modules(
        self, model: torch.nn.Module
    ) -> Dict[str, Dict[str, torch.nn.Module]]:
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
            "layers",  # Common pattern
            "h",  # GPT-style
            "transformer.h",  # Alternative GPT-style
            "decoder.layers",  # Decoder-only models
            "encoder.layers",  # Encoder models
        ]

        for pattern in layer_patterns:
            try:
                layer_container = model
                for attr in pattern.split("."):
                    layer_container = getattr(layer_container, attr)

                for i, layer in enumerate(layer_container):
                    layer_modules = {}

                    # Extract attention modules
                    for name, module in layer.named_modules():
                        if any(
                            attn_name in name.lower()
                            for attn_name in ["attention", "attn"]
                        ):
                            layer_modules[f"attention.{name}"] = module
                        elif any(
                            ffn_name in name.lower()
                            for ffn_name in ["ffn", "mlp", "feed_forward"]
                        ):
                            layer_modules[f"ffn.{name}"] = module
                        elif any(
                            norm_name in name.lower()
                            for norm_name in ["norm", "layernorm"]
                        ):
                            layer_modules[f"norm.{name}"] = module

                    layers[i] = layer_modules

                break  # Found layers, no need to try other patterns

            except AttributeError:
                continue

        return layers
