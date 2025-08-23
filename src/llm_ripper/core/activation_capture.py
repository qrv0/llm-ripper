"""
Activation capture module for LLM Ripper.

This module implements dynamic knowledge capture using torch.fx for efficient
activation extraction during model inference.
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace, Node
try:
    from torchvision.models.feature_extraction import create_feature_extractor  # type: ignore
except Exception:  # torchvision may be absent; we'll fallback to hooks
    create_feature_extractor = None  # type: ignore
import h5py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Iterator
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from datasets import Dataset

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class ActivationData:
    """Container for activation data."""
    layer_name: str
    layer_idx: int
    activations: torch.Tensor
    input_text: str
    token_ids: List[int]


class ActivationCapture:
    """
    Captures dynamic activations from transformer models during inference.
    
    Implements Section 3 of the framework: Dynamic knowledge capture
    using torch.fx for efficient extraction.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_loader = ModelLoader(
            cache_dir=config.get("model_cache_dir"),
            device=config.get("device")
        )
        
    def capture_model_activations(
        self,
        model_name: str,
        corpus_dataset: Dataset,
        output_file: str,
        layers_to_capture: Optional[List[str]] = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Capture activations from a model on a given corpus.
        
        Args:
            model_name: Name or path of the model
            corpus_dataset: Dataset containing text samples
            output_file: HDF5 file to save activations
            layers_to_capture: Specific layers to capture (default: all)
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary containing capture metadata
        """
        logger.info(f"Starting activation capture from model: {model_name}")
        
        # Load model and tokenizer
        model, tokenizer, config = self.model_loader.load_model_and_tokenizer(
            model_name,
            load_in_8bit=self.config.get("load_in_8bit"),
            load_in_4bit=self.config.get("load_in_4bit"),
            trust_remote_code=self.config.get("trust_remote_code"),
        )
        
        # Determine layers to capture
        if layers_to_capture is None:
            layers_to_capture = self._get_all_capturable_layers(model, config)
        
        # Create feature extractor using torch.fx
        feature_extractor = self._create_feature_extractor(model, layers_to_capture)
        
        # Setup HDF5 file for efficient storage
        capture_metadata = self._setup_hdf5_storage(
            output_file, 
            model_name, 
            config, 
            layers_to_capture,
            len(corpus_dataset) if max_samples is None else min(max_samples, len(corpus_dataset))
        )
        
        # Process corpus and capture activations
        with h5py.File(output_file, 'a') as hdf5_file:
            self._process_corpus(
                feature_extractor,
                tokenizer,
                corpus_dataset,
                hdf5_file,
                layers_to_capture,
                max_samples
            )
        
        logger.info(f"Activation capture completed. Data saved to: {output_file}")
        
        return capture_metadata
    
    def _get_all_capturable_layers(self, model: nn.Module, config: Dict[str, Any]) -> List[str]:
        """Get all layers that can be captured from the model."""
        capturable_layers = []
        
        # Get number of layers
        num_layers = config.get("num_hidden_layers", 0)
        
        # Add layer patterns for different transformer architectures
        layer_patterns = [
            "transformer.h.{}.attn",
            "transformer.h.{}.mlp", 
            "model.layers.{}.self_attn",
            "model.layers.{}.mlp",
            "layers.{}.attention",
            "layers.{}.ffn"
        ]
        
        for layer_idx in range(num_layers):
            for pattern in layer_patterns:
                layer_name = pattern.format(layer_idx)
                try:
                    # Check if this layer exists in the model
                    module = model
                    for attr in layer_name.split('.'):
                        module = getattr(module, attr)
                    capturable_layers.append(layer_name)
                except AttributeError:
                    continue
        
        # Add embeddings and final layers
        embedding_patterns = [
            "transformer.wte",
            "model.embed_tokens",
            "embeddings.word_embeddings"
        ]
        
        for pattern in embedding_patterns:
            try:
                module = model
                for attr in pattern.split('.'):
                    module = getattr(module, attr)
                capturable_layers.append(pattern)
                break
            except AttributeError:
                continue
        
        return capturable_layers
    
    def _create_feature_extractor(self, model: nn.Module, return_nodes: List[str]) -> nn.Module:
        """Create a feature extractor using torch.fx."""
        if create_feature_extractor is not None:
            try:
                # Use torchvision's feature extractor which handles torch.fx internally
                return create_feature_extractor(model, return_nodes=return_nodes)
            except Exception as e:
                logger.warning(f"Failed to create torch.fx feature extractor: {e}")
        # Fallback to hook-based extraction
        return self._create_hook_based_extractor(model, return_nodes)
    
    def _create_hook_based_extractor(self, model: nn.Module, return_nodes: List[str]) -> nn.Module:
        """Fallback hook-based feature extractor."""
        
        class HookBasedExtractor(nn.Module):
            def __init__(self, base_model, target_layers):
                super().__init__()
                self.base_model = base_model
                self.target_layers = target_layers
                self.activations = {}
                self.hooks = []
                self._register_hooks()
            
            def _register_hooks(self):
                for layer_name in self.target_layers:
                    try:
                        module = self.base_model
                        for attr in layer_name.split('.'):
                            module = getattr(module, attr)
                        
                        def make_hook(name):
                            def _hook(module, input, output):
                                self.activations[name] = output
                            return _hook
                        hook = module.register_forward_hook(make_hook(layer_name))
                        
                        self.hooks.append(hook)
                    except AttributeError:
                        logger.warning(f"Could not register hook for layer: {layer_name}")
            
            def forward(self, *args, **kwargs):
                self.activations.clear()
                output = self.base_model(*args, **kwargs)
                return self.activations
            
            def __del__(self):
                for hook in self.hooks:
                    hook.remove()
        
        return HookBasedExtractor(model, return_nodes)
    
    def _setup_hdf5_storage(
        self,
        output_file: str,
        model_name: str,
        config: Dict[str, Any],
        layers_to_capture: List[str],
        num_samples: int
    ) -> Dict[str, Any]:
        """Setup HDF5 file structure for efficient storage."""
        
        # Create HDF5 file with hierarchical structure
        with h5py.File(output_file, 'w') as hdf5_file:
            # Create metadata group
            metadata_group = hdf5_file.create_group("metadata")
            metadata_group.attrs["model_name"] = model_name
            metadata_group.attrs["num_samples"] = num_samples
            metadata_group.attrs["layers_captured"] = json.dumps(layers_to_capture)
            
            # Store model config
            config_group = metadata_group.create_group("model_config")
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    config_group.attrs[key] = value
            
            # Create data group with subgroups for each layer
            data_group = hdf5_file.create_group("activations")
            
            for layer_name in layers_to_capture:
                layer_group = data_group.create_group(layer_name.replace(".", "_"))
                layer_group.attrs["original_name"] = layer_name
        
        return {
            "output_file": output_file,
            "model_name": model_name,
            "layers_captured": layers_to_capture,
            "num_samples": num_samples,
            "config": config
        }
    
    def _process_corpus(
        self,
        feature_extractor: nn.Module,
        tokenizer,
        corpus_dataset: Dataset,
        hdf5_file: h5py.File,
        layers_to_capture: List[str],
        max_samples: Optional[int] = None
    ) -> None:
        """Process corpus and capture activations."""
        
        batch_size = self.config.get("batch_size", 8)
        max_length = self.config.get("max_sequence_length", 512)
        
        # Limit samples if specified
        if max_samples:
            corpus_dataset = corpus_dataset.select(range(min(max_samples, len(corpus_dataset))))
        
        # Process in batches
        for i in tqdm(range(0, len(corpus_dataset), batch_size), desc="Processing corpus"):
            batch_end = min(i + batch_size, len(corpus_dataset))
            batch_texts = [corpus_dataset[j]["text"] for j in range(i, batch_end)]
            
            # Tokenize batch
            tokenized = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = tokenized["input_ids"].to(feature_extractor.base_model.device)
            attention_mask = tokenized["attention_mask"].to(feature_extractor.base_model.device)
            
            # Capture activations
            with torch.no_grad():
                activations = feature_extractor(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Store activations in HDF5
            self._store_batch_activations(
                hdf5_file,
                activations,
                batch_texts,
                tokenized["input_ids"],
                i,
                layers_to_capture
            )
    
    def _store_batch_activations(
        self,
        hdf5_file: h5py.File,
        activations: Dict[str, torch.Tensor],
        batch_texts: List[str],
        input_ids: torch.Tensor,
        batch_start_idx: int,
        layers_to_capture: List[str]
    ) -> None:
        """Store batch activations in HDF5 file."""
        
        data_group = hdf5_file["activations"]
        
        for layer_name in layers_to_capture:
            layer_group_name = layer_name.replace(".", "_")
            
            if layer_group_name not in data_group:
                continue
                
            layer_group = data_group[layer_group_name]
            
            # Get activation tensor for this layer
            if layer_name in activations:
                value = activations[layer_name]
                # Normalize to a torch.Tensor
                import torch as _torch
                def _first_tensor(obj):
                    if _torch.is_tensor(obj):
                        return obj
                    if hasattr(obj, 'last_hidden_state') and _torch.is_tensor(obj.last_hidden_state):
                        return obj.last_hidden_state
                    if isinstance(obj, (list, tuple)):
                        for el in obj:
                            t = _first_tensor(el)
                            if t is not None:
                                return t
                    if isinstance(obj, dict):
                        # common keys
                        for k in ('last_hidden_state','hidden_states','logits','attentions','output'):
                            if k in obj and _torch.is_tensor(obj[k]):
                                return obj[k]
                        for v in obj.values():
                            t = _first_tensor(v)
                            if t is not None:
                                return t
                    return None
                tensor = _first_tensor(value)
                if tensor is None:
                    # skip if no tensor found
                    continue
                activation_tensor = tensor.detach().cpu().numpy()
                
                # Store each sample in the batch
                for batch_idx, (text, input_id_seq) in enumerate(zip(batch_texts, input_ids)):
                    sample_idx = batch_start_idx + batch_idx
                    sample_group_name = f"sample_{sample_idx}"
                    
                    if sample_group_name not in layer_group:
                        sample_group = layer_group.create_group(sample_group_name)
                    else:
                        sample_group = layer_group[sample_group_name]
                    
                    # Store activation data
                    if "activations" not in sample_group:
                        sample_group.create_dataset(
                            "activations",
                            data=activation_tensor[batch_idx],
                            compression=self.config.get("hdf5_compression", "gzip")
                        )
                    
                    # Store metadata
                    sample_group.attrs["text"] = text
                    sample_group.attrs["input_ids"] = input_id_seq.cpu().numpy()
                    sample_group.attrs["sequence_length"] = len(input_id_seq)
    
    def load_activations(
        self,
        hdf5_file: str,
        layer_name: Optional[str] = None,
        sample_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Load activations from HDF5 file.
        
        Args:
            hdf5_file: Path to HDF5 file containing activations
            layer_name: Specific layer to load (default: all)
            sample_indices: Specific samples to load (default: all)
            
        Returns:
            Dictionary containing loaded activations and metadata
        """
        
        activations_data = {}
        
        with h5py.File(hdf5_file, 'r') as f:
            # Load metadata
            metadata = dict(f["metadata"].attrs)
            activations_data["metadata"] = metadata
            
            # Get available layers
            available_layers = list(f["activations"].keys())
            
            if layer_name:
                layers_to_load = [layer_name.replace(".", "_")] if layer_name.replace(".", "_") in available_layers else []
            else:
                layers_to_load = available_layers
            
            # Load activations
            for layer_key in layers_to_load:
                layer_group = f["activations"][layer_key]
                original_name = layer_group.attrs.get("original_name", layer_key)
                
                layer_data = {}
                sample_groups = list(layer_group.keys())
                
                if sample_indices:
                    sample_groups = [f"sample_{idx}" for idx in sample_indices if f"sample_{idx}" in sample_groups]
                
                for sample_key in sample_groups:
                    sample_group = layer_group[sample_key]
                    
                    layer_data[sample_key] = {
                        "activations": np.array(sample_group["activations"]),
                        "text": sample_group.attrs.get("text", ""),
                        "input_ids": sample_group.attrs.get("input_ids", []),
                        "sequence_length": sample_group.attrs.get("sequence_length", 0)
                    }
                
                activations_data[original_name] = layer_data
        
        return activations_data
