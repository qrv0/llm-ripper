"""
Activation capture module for LLM Ripper.

This module implements dynamic knowledge capture using torch.fx for efficient
activation extraction during model inference.
"""

import torch
import torch.nn as nn

try:
    from torchvision.models.feature_extraction import create_feature_extractor  # type: ignore
except Exception:  # torchvision may be absent; we'll fallback to hooks
    create_feature_extractor = None  # type: ignore
try:
    import h5py  # type: ignore
except Exception:  # optional, used only when capturing/reading HDF5
    h5py = None  # type: ignore
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

try:
    from datasets import Dataset  # type: ignore
except Exception:  # typing-only fallback to avoid import-time failure

    class Dataset:  # type: ignore
        ...


import uuid

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
            cache_dir=config.get("model_cache_dir"), device=config.get("device")
        )

    def capture_model_activations(
        self,
        model_name: str,
        corpus_dataset: Dataset,
        output_file: str,
        layers_to_capture: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
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

        use_outputs = self.config.get("use_model_outputs", False)
        if use_outputs:
            # Derive layer identifiers for outputs path
            num_layers = config.get("num_hidden_layers", 0)
            layers_to_capture = ["hidden.last"] + [
                f"attn.layer_{i}" for i in range(num_layers)
            ]
            feature_extractor = None  # not used
        else:
            # Determine layers to capture
            if layers_to_capture is None:
                layers_to_capture = self._get_all_capturable_layers(model, config)
            # Create feature extractor using torch.fx or hooks
            feature_extractor = self._create_feature_extractor(model, layers_to_capture)

        # Setup HDF5 file for efficient storage
        capture_metadata = self._setup_hdf5_storage(
            output_file,
            model_name,
            config,
            layers_to_capture,
            (
                len(corpus_dataset)
                if max_samples is None
                else min(max_samples, len(corpus_dataset))
            ),
        )

        # Process corpus and capture activations
        with h5py.File(output_file, "a") as hdf5_file:
            if use_outputs:
                self._process_corpus_via_outputs(
                    model,
                    tokenizer,
                    corpus_dataset,
                    hdf5_file,
                    layers_to_capture,
                    max_samples,
                )
            else:
                self._process_corpus(
                    feature_extractor,
                    tokenizer,
                    corpus_dataset,
                    hdf5_file,
                    layers_to_capture,
                    max_samples,
                )

        logger.info(f"Activation capture completed. Data saved to: {output_file}")

        return capture_metadata

    def _get_all_capturable_layers(
        self, model: nn.Module, config: Dict[str, Any]
    ) -> List[str]:
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
            "layers.{}.ffn",
        ]

        for layer_idx in range(num_layers):
            for pattern in layer_patterns:
                layer_name = pattern.format(layer_idx)
                try:
                    # Check if this layer exists in the model
                    module = model
                    for attr in layer_name.split("."):
                        module = getattr(module, attr)
                    capturable_layers.append(layer_name)
                except AttributeError:
                    continue

        # Add embeddings and final layers
        embedding_patterns = [
            "transformer.wte",
            "model.embed_tokens",
            "embeddings.word_embeddings",
        ]

        for pattern in embedding_patterns:
            try:
                module = model
                for attr in pattern.split("."):
                    module = getattr(module, attr)
                capturable_layers.append(pattern)
                break
            except AttributeError:
                continue

        return capturable_layers

    def _create_feature_extractor(
        self, model: nn.Module, return_nodes: List[str]
    ) -> nn.Module:
        """Create a feature extractor using torch.fx."""
        if create_feature_extractor is not None:
            try:
                # Use torchvision's feature extractor which handles torch.fx internally
                return create_feature_extractor(model, return_nodes=return_nodes)
            except Exception as e:
                logger.warning(f"Failed to create torch.fx feature extractor: {e}")
        # Fallback to hook-based extraction
        return self._create_hook_based_extractor(model, return_nodes)

    def _create_hook_based_extractor(
        self, model: nn.Module, return_nodes: List[str]
    ) -> nn.Module:
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
                        for attr in layer_name.split("."):
                            module = getattr(module, attr)

                        def make_hook(name):
                            def _hook(module, input, output):
                                self.activations[name] = output

                            return _hook

                        hook = module.register_forward_hook(make_hook(layer_name))

                        self.hooks.append(hook)
                    except AttributeError:
                        logger.warning(
                            f"Could not register hook for layer: {layer_name}"
                        )

            def forward(self, *args, **kwargs):
                self.activations.clear()
                self.base_model(*args, **kwargs)
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
        num_samples: int,
    ) -> Dict[str, Any]:
        """Setup HDF5 file structure for efficient storage."""

        # Create HDF5 file with hierarchical structure
        if h5py is None:
            raise RuntimeError(
                "h5py is required for activation capture storage but is not installed."
            )
        with h5py.File(output_file, "w") as hdf5_file:
            # Create metadata group
            metadata_group = hdf5_file.create_group("metadata")
            metadata_group.attrs["model_name"] = model_name
            metadata_group.attrs["num_samples"] = num_samples
            metadata_group.attrs["layers_captured"] = json.dumps(layers_to_capture)
            metadata_group.attrs["run_id"] = str(uuid.uuid4())
            metadata_group.attrs["seed"] = int(self.config.get("seed", 0))
            # Library versions
            try:
                import transformers as _tf

                metadata_group.attrs["transformers_version"] = getattr(
                    _tf, "__version__", "unknown"
                )
            except Exception:
                metadata_group.attrs["transformers_version"] = "unknown"
            try:
                import torch as _torch

                metadata_group.attrs["torch_version"] = getattr(
                    _torch, "__version__", "unknown"
                )
            except Exception:
                metadata_group.attrs["torch_version"] = "unknown"

            # Store model config
            config_group = metadata_group.create_group("model_config")
            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    config_group.attrs[key] = value

            # Create data group; layout depends on store mode
            data_group = hdf5_file.create_group("activations")
            store_mode = self.config.get("hdf5_store_mode", "groups")
            data_group.attrs["store_mode"] = store_mode

            if store_mode == "groups":
                for layer_name in layers_to_capture:
                    layer_group = data_group.create_group(layer_name.replace(".", "_"))
                    layer_group.attrs["original_name"] = layer_name
            elif store_mode == "dataset":
                # For dataset mode, datasets are created lazily on first write
                # Pre-create samples datasets for texts and input_ids
                samples = hdf5_file.create_group("samples")
                dt = h5py.string_dtype(encoding="utf-8")
                samples.create_dataset("texts", shape=(num_samples,), dtype=dt)
                max_len = self.config.get("max_sequence_length", 512)
                samples.create_dataset(
                    "input_ids", shape=(num_samples, max_len), dtype="int32"
                )
                samples["input_ids"].attrs["pad_token"] = -1
            else:
                data_group.attrs["store_mode"] = "groups"
                for layer_name in layers_to_capture:
                    layer_group = data_group.create_group(layer_name.replace(".", "_"))
                    layer_group.attrs["original_name"] = layer_name

        return {
            "output_file": output_file,
            "model_name": model_name,
            "layers_captured": layers_to_capture,
            "num_samples": num_samples,
            "config": config,
        }

    def _process_corpus(
        self,
        feature_extractor: nn.Module,
        tokenizer,
        corpus_dataset: Dataset,
        hdf5_file: h5py.File,
        layers_to_capture: List[str],
        max_samples: Optional[int] = None,
    ) -> None:
        """Process corpus and capture activations."""

        batch_size = self.config.get("batch_size", 8)
        max_length = self.config.get("max_sequence_length", 512)

        # Limit samples if specified
        if max_samples:
            corpus_dataset = corpus_dataset.select(
                range(min(max_samples, len(corpus_dataset)))
            )

        # Process in batches
        for i in tqdm(
            range(0, len(corpus_dataset), batch_size), desc="Processing corpus"
        ):
            batch_end = min(i + batch_size, len(corpus_dataset))
            batch_texts = [corpus_dataset[j]["text"] for j in range(i, batch_end)]

            # Tokenize batch
            tokenized = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move to device
            input_ids = tokenized["input_ids"].to(feature_extractor.base_model.device)
            attention_mask = tokenized["attention_mask"].to(
                feature_extractor.base_model.device
            )

            # Capture activations
            with torch.no_grad():
                activations = feature_extractor(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            # Store activations in HDF5
            self._store_batch_activations(
                hdf5_file,
                activations,
                batch_texts,
                tokenized["input_ids"],
                i,
                layers_to_capture,
            )

    def _store_batch_activations(
        self,
        hdf5_file: h5py.File,
        activations: Dict[str, torch.Tensor],
        batch_texts: List[str],
        input_ids: torch.Tensor,
        batch_start_idx: int,
        layers_to_capture: List[str],
    ) -> None:
        """Store batch activations in HDF5 file."""

        if h5py is None:
            raise RuntimeError(
                "h5py is required for activation capture storage but is not installed."
            )
        data_group = hdf5_file["activations"]
        store_mode = data_group.attrs.get("store_mode", "groups")

        for layer_name in layers_to_capture:
            layer_group_name = layer_name.replace(".", "_")

            # Get activation tensor for this layer
            if layer_name not in activations:
                continue

            value = activations[layer_name]
            # Normalize to a torch.Tensor
            import torch as _torch

            def _first_tensor(obj):
                if _torch.is_tensor(obj):
                    return obj
                if hasattr(obj, "last_hidden_state") and _torch.is_tensor(
                    obj.last_hidden_state
                ):
                    return obj.last_hidden_state
                if isinstance(obj, (list, tuple)):
                    for el in obj:
                        t = _first_tensor(el)
                        if t is not None:
                            return t
                if isinstance(obj, dict):
                    for k in (
                        "last_hidden_state",
                        "hidden_states",
                        "logits",
                        "attentions",
                        "output",
                    ):
                        if k in obj and _torch.is_tensor(obj[k]):
                            return obj[k]
                    for v in obj.values():
                        t = _first_tensor(v)
                        if t is not None:
                            return t
                return None

            tensor = _first_tensor(value)
            if tensor is None:
                continue

            if store_mode == "dataset":
                max_len = self.config.get("max_sequence_length", 512)
                arr = tensor.detach().cpu().numpy()
                # Hidden states [B,T,H]
                if arr.ndim == 3:
                    B, T, H = arr.shape
                    if T > max_len:
                        arr = arr[:, :max_len, :]
                        T = max_len
                    if layer_group_name not in data_group:
                        layer_group = data_group.create_group(layer_group_name)
                        layer_group.attrs["original_name"] = layer_name
                        # Allow overriding chunks via config key 'hdf5_hidden_chunks' as 'b,t,h'
                        chunks_hidden = (max(1, min(8, B)), max_len, H)
                        ch_conf = self.config.get("hdf5_hidden_chunks")
                        if ch_conf:
                            try:
                                cb, ct, ch = [int(x) for x in str(ch_conf).split(",")]
                                chunks_hidden = (cb, ct, ch)
                            except Exception:
                                pass
                        dset = layer_group.create_dataset(
                            "data",
                            shape=(0, max_len, H),
                            maxshape=(None, max_len, H),
                            chunks=chunks_hidden,
                            dtype="float32",
                            compression=self.config.get("hdf5_compression", "gzip"),
                        )
                    else:
                        layer_group = data_group[layer_group_name]
                        dset = layer_group["data"]
                    cur = dset.shape[0]
                    dset.resize(cur + B, axis=0)
                    if T < dset.shape[1]:
                        pad = np.zeros((B, dset.shape[1] - T, H), dtype=arr.dtype)
                        arr = np.concatenate([arr, pad], axis=1)
                    dset[cur : cur + B, :, :] = arr.astype("float32")
                    # Optional reduced projections for smaller files
                    if self.config.get("store_hidden_mean", False):
                        # mean over time -> [B, H]
                        red = arr.mean(axis=1)
                        red_name = "hidden_mean"
                        if red_name not in layer_group:
                            rd = layer_group.create_dataset(
                                red_name,
                                shape=(0, red.shape[-1]),
                                maxshape=(None, red.shape[-1]),
                                chunks=(max(1, min(32, B)), red.shape[-1]),
                                dtype="float32",
                                compression=self.config.get("hdf5_compression", "gzip"),
                            )
                        else:
                            rd = layer_group[red_name]
                        cur2 = rd.shape[0]
                        rd.resize(cur2 + B, axis=0)
                        rd[cur2 : cur2 + B, :] = red.astype("float32")
                # Attention [B, Hh?, T, T] or [B,T,T] -> store mean or per-head
                elif arr.ndim >= 3 and arr.shape[-1] == arr.shape[-2]:
                    store_heads = self.config.get("store_attn_per_head", False)
                    if store_heads and arr.ndim == 4:
                        # [B, Hh, T, T]
                        B, Hh, T, _ = arr.shape
                        if T > max_len:
                            arr = arr[:, :, :max_len, :max_len]
                            T = max_len
                        if layer_group_name not in data_group:
                            layer_group = data_group.create_group(layer_group_name)
                            layer_group.attrs["original_name"] = layer_name
                            chunks_heads = (max(1, min(2, B)), Hh, max_len, max_len)
                            ch_conf = self.config.get("hdf5_attn_heads_chunks")
                            if ch_conf:
                                try:
                                    cb, chh, ct1, ct2 = [
                                        int(x) for x in str(ch_conf).split(",")
                                    ]
                                    chunks_heads = (cb, chh, ct1, ct2)
                                except Exception:
                                    pass
                            dset = layer_group.create_dataset(
                                "attn_heads",
                                shape=(0, Hh, max_len, max_len),
                                maxshape=(None, Hh, max_len, max_len),
                                chunks=chunks_heads,
                                dtype="float32",
                                compression=self.config.get("hdf5_compression", "gzip"),
                            )
                        else:
                            layer_group = data_group[layer_group_name]
                            dset = (
                                layer_group["attn_heads"]
                                if "attn_heads" in layer_group
                                else layer_group.create_dataset(
                                    "attn_heads",
                                    shape=(0, arr.shape[1], max_len, max_len),
                                    maxshape=(None, arr.shape[1], max_len, max_len),
                                    chunks=(
                                        max(1, min(2, B)),
                                        arr.shape[1],
                                        max_len,
                                        max_len,
                                    ),
                                    dtype="float32",
                                    compression=self.config.get(
                                        "hdf5_compression", "gzip"
                                    ),
                                )
                            )
                        cur = dset.shape[0]
                        dset.resize(cur + B, axis=0)
                        # pad if needed
                        if T < dset.shape[2]:
                            padT = dset.shape[2] - T
                            pad = np.zeros((B, arr.shape[1], padT, T), dtype=arr.dtype)
                            arr = np.concatenate([arr, pad], axis=2)
                            pad2 = np.zeros(
                                (B, arr.shape[1], dset.shape[2], dset.shape[3] - T),
                                dtype=arr.dtype,
                            )
                            arr = np.concatenate([arr, pad2], axis=3)
                        dset[cur : cur + B, :, :, :] = arr.astype("float32")
                        # Reduced projections for per-head attention
                        if self.config.get("store_attn_mean", False):
                            mean_name = "attn_mean"
                            mean_arr = arr.mean(axis=(2, 3))  # [B, Hh]
                            if mean_name not in layer_group:
                                md = layer_group.create_dataset(
                                    mean_name,
                                    shape=(0, mean_arr.shape[1]),
                                    maxshape=(None, mean_arr.shape[1]),
                                    chunks=(max(1, min(16, B)), mean_arr.shape[1]),
                                    dtype="float32",
                                    compression=self.config.get(
                                        "hdf5_compression", "gzip"
                                    ),
                                )
                            else:
                                md = layer_group[mean_name]
                            curm = md.shape[0]
                            md.resize(curm + B, axis=0)
                            md[curm : curm + B, :] = mean_arr.astype("float32")
                    else:
                        # reduce to [B, T, T]
                        while arr.ndim > 3:
                            arr = arr.mean(axis=1)
                        B, T, _ = arr.shape
                        if T > max_len:
                            arr = arr[:, :max_len, :max_len]
                            T = max_len
                        if layer_group_name not in data_group:
                            layer_group = data_group.create_group(layer_group_name)
                            layer_group.attrs["original_name"] = layer_name
                            chunks_attn = (max(1, min(4, B)), max_len, max_len)
                            ch_conf = self.config.get("hdf5_attn_chunks")
                            if ch_conf:
                                try:
                                    cb, ct1, ct2 = [
                                        int(x) for x in str(ch_conf).split(",")
                                    ]
                                    chunks_attn = (cb, ct1, ct2)
                                except Exception:
                                    pass
                            dset = layer_group.create_dataset(
                                "attn",
                                shape=(0, max_len, max_len),
                                maxshape=(None, max_len, max_len),
                                chunks=chunks_attn,
                                dtype="float32",
                                compression=self.config.get("hdf5_compression", "gzip"),
                            )
                        else:
                            layer_group = data_group[layer_group_name]
                            dset = (
                                layer_group["attn"]
                                if "attn" in layer_group
                                else layer_group.create_dataset(
                                    "attn",
                                    shape=(0, max_len, max_len),
                                    maxshape=(None, max_len, max_len),
                                    chunks=(max(1, min(4, B)), max_len, max_len),
                                    dtype="float32",
                                    compression=self.config.get(
                                        "hdf5_compression", "gzip"
                                    ),
                                )
                            )
                        cur = dset.shape[0]
                        dset.resize(cur + B, axis=0)
                        if T < dset.shape[1]:
                            pad = np.zeros((B, dset.shape[1] - T, T), dtype=arr.dtype)
                            arr = np.concatenate([arr, pad], axis=1)
                            pad2 = np.zeros(
                                (B, dset.shape[1], dset.shape[2] - T), dtype=arr.dtype
                            )
                            arr = np.concatenate([arr, pad2], axis=2)
                        dset[cur : cur + B, :, :] = arr.astype("float32")
                        # Reduced projections for attention
                        if self.config.get("store_attn_mean", False):
                            mean_name = "attn_mean"
                            mean_arr = arr.mean(axis=(1, 2))  # [B]
                            if mean_name not in layer_group:
                                md = layer_group.create_dataset(
                                    mean_name,
                                    shape=(0,),
                                    maxshape=(None,),
                                    chunks=(max(1, min(64, B)),),
                                    dtype="float32",
                                    compression=self.config.get(
                                        "hdf5_compression", "gzip"
                                    ),
                                )
                            else:
                                md = layer_group[mean_name]
                            curm = md.shape[0]
                            md.resize(curm + B, axis=0)
                            md[curm : curm + B] = mean_arr.astype("float32")
                        if self.config.get("store_attn_diag", False):
                            diag_name = "attn_diag"
                            # take diagonal and pad to max_len
                            diag = np.zeros((B, dset.shape[1]), dtype="float32")
                            m = min(T, dset.shape[1])
                            for bi in range(B):
                                diag[bi, :m] = np.diag(arr[bi, :m, :m]).astype(
                                    "float32"
                                )
                            if diag_name not in layer_group:
                                dd = layer_group.create_dataset(
                                    diag_name,
                                    shape=(0, dset.shape[1]),
                                    maxshape=(None, dset.shape[1]),
                                    chunks=(max(1, min(16, B)), dset.shape[1]),
                                    dtype="float32",
                                    compression=self.config.get(
                                        "hdf5_compression", "gzip"
                                    ),
                                )
                            else:
                                dd = layer_group[diag_name]
                            curd = dd.shape[0]
                            dd.resize(curd + B, axis=0)
                            dd[curd : curd + B, :] = diag
                else:
                    # unsupported; skip this layer for dataset mode
                    continue
                # Store sample texts and input_ids
                samples = hdf5_file["samples"]
                texts_ds = samples["texts"]
                ids_ds = samples["input_ids"]
                for b, (text, ids) in enumerate(zip(batch_texts, input_ids)):
                    idx = batch_start_idx + b
                    texts_ds[idx] = text
                    ids = ids.cpu().numpy()
                    if ids.shape[0] > ids_ds.shape[1]:
                        ids = ids[: ids_ds.shape[1]]
                    padded = np.full(
                        (ids_ds.shape[1],),
                        ids_ds.attrs.get("pad_token", -1),
                        dtype=ids_ds.dtype,
                    )
                    padded[: ids.shape[0]] = ids
                    ids_ds[idx, :] = padded
            else:
                # groups mode (original)
                if layer_group_name not in data_group:
                    continue
                layer_group = data_group[layer_group_name]
                activation_tensor = tensor.detach().cpu().numpy()
                for batch_idx, (text, input_id_seq) in enumerate(
                    zip(batch_texts, input_ids)
                ):
                    sample_idx = batch_start_idx + batch_idx
                    sample_group_name = f"sample_{sample_idx}"
                    if sample_group_name not in layer_group:
                        sample_group = layer_group.create_group(sample_group_name)
                    else:
                        sample_group = layer_group[sample_group_name]
                    if "activations" not in sample_group:
                        sample_group.create_dataset(
                            "activations",
                            data=activation_tensor[batch_idx],
                            compression=self.config.get("hdf5_compression", "gzip"),
                        )
                    sample_group.attrs["text"] = text
                    sample_group.attrs["input_ids"] = input_id_seq.cpu().numpy()
                    sample_group.attrs["sequence_length"] = len(input_id_seq)

    def _process_corpus_via_outputs(
        self,
        model: nn.Module,
        tokenizer,
        corpus_dataset: Dataset,
        hdf5_file: h5py.File,
        layers_to_capture: List[str],
        max_samples: Optional[int] = None,
    ) -> None:
        batch_size = self.config.get("batch_size", 8)
        max_length = self.config.get("max_sequence_length", 512)
        if max_samples:
            corpus_dataset = corpus_dataset.select(
                range(min(max_samples, len(corpus_dataset)))
            )
        for i in tqdm(
            range(0, len(corpus_dataset), batch_size), desc="Processing corpus(outputs)"
        ):
            batch_end = min(i + batch_size, len(corpus_dataset))
            batch_texts = [corpus_dataset[j]["text"] for j in range(i, batch_end)]
            tokenized = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {
                k: v.to(next(model.parameters()).device) for k, v in tokenized.items()
            }
            with torch.no_grad():
                outputs = model(
                    **tokenized, output_hidden_states=True, output_attentions=True
                )
            acts: Dict[str, torch.Tensor] = {}
            # Last hidden state
            if hasattr(outputs, "last_hidden_state") and torch.is_tensor(
                outputs.last_hidden_state
            ):
                acts["hidden.last"] = outputs.last_hidden_state
            # Attentions list per layer -> mean over heads
            try:
                atts = outputs.attentions
                if atts:
                    for li, att in enumerate(atts):
                        if att.dim() == 4:  # [B, heads, T, T]
                            acts[f"attn.layer_{li}"] = att.mean(dim=1)
                        elif att.dim() == 3:  # [B, T, T]
                            acts[f"attn.layer_{li}"] = att
            except Exception:
                pass
            self._store_batch_activations(
                hdf5_file,
                acts,
                batch_texts,
                tokenized["input_ids"].cpu(),
                i,
                layers_to_capture,
            )

    def load_activations(
        self,
        hdf5_file: str,
        layer_name: Optional[str] = None,
        sample_indices: Optional[List[int]] = None,
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

        if h5py is None:
            raise RuntimeError(
                "h5py is required to read activation datasets but is not installed."
            )
        with h5py.File(hdf5_file, "r") as f:
            # Load metadata
            metadata = dict(f["metadata"].attrs)
            activations_data["metadata"] = metadata

            # Get available layers
            available_layers = list(f["activations"].keys())

            if layer_name:
                layers_to_load = (
                    [layer_name.replace(".", "_")]
                    if layer_name.replace(".", "_") in available_layers
                    else []
                )
            else:
                layers_to_load = available_layers

            # Load activations
            for layer_key in layers_to_load:
                layer_group = f["activations"][layer_key]
                original_name = layer_group.attrs.get("original_name", layer_key)

                layer_data = {}
                sample_groups = list(layer_group.keys())

                if sample_indices:
                    sample_groups = [
                        f"sample_{idx}"
                        for idx in sample_indices
                        if f"sample_{idx}" in sample_groups
                    ]

                for sample_key in sample_groups:
                    sample_group = layer_group[sample_key]

                    layer_data[sample_key] = {
                        "activations": np.array(sample_group["activations"]),
                        "text": sample_group.attrs.get("text", ""),
                        "input_ids": sample_group.attrs.get("input_ids", []),
                        "sequence_length": sample_group.attrs.get("sequence_length", 0),
                    }

                activations_data[original_name] = layer_data

        return activations_data

    def load_dataset_slices(
        self, hdf5_file: str, layer_name: str, indices: List[int], tensor: str = "data"
    ) -> np.ndarray:
        """Load slices from dataset-mode HDF5 for a given layer.
        Args:
            hdf5_file: path to HDF5
            layer_name: original layer name (e.g., 'transformer.h.0.attn') or resolved (dots ok)
            indices: list of sample indices to load
            tensor: 'data' (hidden) | 'attn' | 'attn_heads'
        Returns:
            numpy array with stacked slices
        """
        key = layer_name.replace(".", "_")
        with h5py.File(hdf5_file, "r") as f:
            if f["activations"].attrs.get("store_mode", "groups") != "dataset":
                raise ValueError("HDF5 not in dataset mode")
            if key not in f["activations"]:
                raise KeyError(f"Layer not found: {layer_name}")
            layer_group = f["activations"][key]
            if tensor not in layer_group:
                raise KeyError(f"Tensor '{tensor}' not found in layer group")
            dset = layer_group[tensor]
            return np.stack([dset[i] for i in indices], axis=0)
