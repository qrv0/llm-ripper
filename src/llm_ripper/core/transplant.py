"""
Knowledge transplantation module for LLM Ripper.

This module implements Part III of the framework: Recomposition and validation
in new architectures using Bridge Networks and adapters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader
from ..utils.architecture import replace_layer_submodule

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
        dropout: float = 0.1,
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
        self.use_residual = input_dim == output_dim

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
        freeze_donor: bool = True,
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
            cache_dir=config.get("model_cache_dir"), device=config.get("device")
        )

    def transplant_knowledge(
        self,
        source_knowledge_bank: str,
        target_model_name: str,
        transplant_configs: List[TransplantConfig],
        output_dir: str,
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
        target_model, target_tokenizer, target_config = (
            self.model_loader.load_model_and_tokenizer(
                target_model_name,
                load_in_8bit=self.config.get("load_in_8bit"),
                load_in_4bit=self.config.get("load_in_4bit"),
                trust_remote_code=self.config.get("trust_remote_code"),
            )
        )

        # Load source knowledge bank metadata
        knowledge_bank_path = Path(source_knowledge_bank)
        with open(knowledge_bank_path / "extraction_metadata.json", "r") as f:
            source_metadata = json.load(f)

        transplant_metadata = {
            "source_model": source_metadata["source_model"],
            "target_model": target_model_name,
            "transplant_configs": [config.__dict__ for config in transplant_configs],
            "transplanted_components": {},
            "layers_adapters": {},
        }
        # add knowledge bank path for future reattachment
        transplant_metadata["source_knowledge_bank"] = str(knowledge_bank_path)

        # Apply each transplant configuration
        for transplant_config in transplant_configs:
            component_result = self._apply_transplant_config(
                target_model, target_config, knowledge_bank_path, transplant_config
            )

            transplant_metadata["transplanted_components"][
                f"{transplant_config.strategy}_{transplant_config.source_component}"
            ] = component_result
            lkey = f"layer_{transplant_config.target_layer}"
            lst = transplant_metadata["layers_adapters"].setdefault(lkey, [])
            lst.append(
                {
                    "source_component": transplant_config.source_component,
                    "strategy": transplant_config.strategy,
                }
            )

        # Save transplanted model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        target_model.save_pretrained(output_path / "model")
        target_tokenizer.save_pretrained(output_path / "tokenizer")

        # Save transplant artifacts (bridges and gates)
        artifacts_dir = output_path / "model" / "transplant_artifacts"
        artifacts = {"layers": {}}
        try:
            if hasattr(target_model, "transplanted_modules"):
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                for layer_key, mod in target_model.transplanted_modules.items():

                    def _save_mod(m: nn.Module, idx: int = None):
                        layer_art = artifacts["layers"].setdefault(layer_key, {})
                        suffix = f"_idx{idx}" if idx is not None else ""
                        if hasattr(m, "input_bridge") and m.input_bridge is not None:
                            ib_file = (
                                artifacts_dir / f"{layer_key}{suffix}_input_bridge.pt"
                            )
                            torch.save(m.input_bridge.state_dict(), ib_file)
                            layer_art[f"input_bridge{suffix}"] = str(ib_file)
                        if hasattr(m, "output_bridge") and m.output_bridge is not None:
                            ob_file = (
                                artifacts_dir / f"{layer_key}{suffix}_output_bridge.pt"
                            )
                            torch.save(m.output_bridge.state_dict(), ob_file)
                            layer_art[f"output_bridge{suffix}"] = str(ob_file)

                    if isinstance(mod, nn.ModuleList):
                        for i, m in enumerate(mod):
                            _save_mod(m, idx=i)
                    else:
                        _save_mod(mod)
            if hasattr(target_model, "transplant_fusion_gates"):
                for layer_key, gate in target_model.transplant_fusion_gates.items():
                    gf = artifacts_dir / f"{layer_key}_fusion_gate.pt"
                    torch.save(gate.state_dict(), gf)
                    artifacts.setdefault("layers", {}).setdefault(layer_key, {})[
                        "fusion_gate"
                    ] = str(gf)
        except Exception as e:
            logger.warning(f"Failed to save transplant artifacts: {e}")
        with open((artifacts_dir.parent / "transplant_artifacts.json"), "w") as f:
            json.dump(artifacts, f, indent=2)

        # Save transplant metadata
        with open(output_path / "transplant_metadata.json", "w") as f:
            json.dump(transplant_metadata, f, indent=2)

        logger.info(f"Transplantation completed. Model saved to: {output_path}")

        return transplant_metadata

    def fine_tune_fusion_gates(
        self,
        model: nn.Module,
        steps: int = 100,
        lr: float = 1e-3,
        target_gate: float = 0.1,
    ) -> Dict[str, Any]:
        """Lightweight synthetic fine-tuning for fusion gates only.
        Optimizes gate alphas towards a target sigmoid value so they become trainable and non-zero.
        This does not require tokenizers or datasets and is safe for offline/testing environments.
        Returns a report with initial/final gate values per layer.
        """
        if not hasattr(model, "transplant_fusion_gates") or not isinstance(
            model.transplant_fusion_gates, nn.ModuleDict
        ):
            return {"updated": 0, "layers": {}, "message": "no fusion gates found"}
        gates = model.transplant_fusion_gates
        # Collect parameters
        params = []
        init_vals: Dict[str, float] = {}
        for name, gate in gates.items():
            if hasattr(gate, "alpha"):
                init_vals[name] = float(torch.sigmoid(gate.alpha.detach()).cpu())
                params.append(gate.alpha)
        if not params:
            return {"updated": 0, "layers": {}, "message": "no alpha parameters found"}
        opt = optim.Adam([{"params": params, "lr": lr}])
        target = torch.tensor(
            [target_gate], dtype=torch.float32, device=params[0].device
        )
        for _ in range(max(1, steps)):
            opt.zero_grad()
            loss = None
            for gate in gates.values():
                if hasattr(gate, "alpha"):
                    v = torch.sigmoid(gate.alpha)
                    loss_term = (v - target).pow(2).mean()
                    loss = loss_term if loss is None else (loss + loss_term)
            if loss is None:
                break
            loss.backward()
            opt.step()
        out: Dict[str, Any] = {"updated": len(params), "layers": {}}
        for name, gate in gates.items():
            if hasattr(gate, "alpha"):
                out["layers"][name] = {
                    "initial": init_vals.get(name, 0.0),
                    "final": float(torch.sigmoid(gate.alpha.detach()).cpu()),
                }
        return out

    def reattach_full_from_artifacts(self, transplanted_dir: str) -> Dict[str, Any]:
        """Rebuild transplanted wrappers + bridges + gates from saved artifacts and donor KB.
        - Loads model from `<transplanted_dir>/model`
        - Reads `transplant_metadata.json` for configs and the source_knowledge_bank
        - Recreates donor modules, restores bridge/gate state, and reinjects
        Returns a report mapping layers to reattachment status.
        """
        root = Path(transplanted_dir)
        model_dir = root / "model"
        meta_file = root / "transplant_metadata.json"
        if not (model_dir.exists() and meta_file.exists()):
            return {"error": "Missing model/ or transplant_metadata.json"}
        # Load target model
        target_model, _, _ = self.model_loader.load_model_and_tokenizer(
            str(model_dir),
            load_in_8bit=self.config.get("load_in_8bit"),
            load_in_4bit=self.config.get("load_in_4bit"),
            trust_remote_code=self.config.get("trust_remote_code"),
        )
        meta = json.loads(meta_file.read_text())
        kb_dir = meta.get("source_knowledge_bank")
        if not kb_dir or not Path(kb_dir).exists():
            return {"error": "source_knowledge_bank not found or missing"}
        artifacts_dir = model_dir / "transplant_artifacts"
        report: Dict[str, Any] = {"layers": {}, "errors": []}
        # Prefer explicit ordered adapters if present
        if isinstance(meta.get("layers_adapters"), dict) and meta["layers_adapters"]:
            for layer_key, entries in meta["layers_adapters"].items():
                try:
                    target_layer_idx = int(str(layer_key).split("_")[1])
                except Exception:
                    continue
                for idx, entry in enumerate(entries):
                    try:
                        src = entry.get("source_component")
                        donor_data = self._load_component(Path(kb_dir), src)
                        if donor_data is None:
                            report["errors"].append(
                                {
                                    "layer": target_layer_idx,
                                    "error": f"component not found: {src}",
                                }
                            )
                            continue
                        donor_module = self._create_donor_module(donor_data)
                        target_layer_module = self._get_target_layer(
                            target_model, target_layer_idx
                        )
                        # Build bridges if needed
                        dummy_tc = TransplantConfig(
                            src,
                            target_layer_idx,
                            getattr(
                                getattr(target_model, "config", object()),
                                "hidden_size",
                                768,
                            ),
                            True,
                            False,
                            entry.get("strategy", "module_injection"),
                        )
                        input_bridge, output_bridge = self._create_bridges_for_module(
                            donor_module, target_layer_module, dummy_tc
                        )
                        # Load per-index bridge states
                        ib = artifacts_dir / f"{layer_key}_idx{idx}_input_bridge.pt"
                        ob = artifacts_dir / f"{layer_key}_idx{idx}_output_bridge.pt"
                        if input_bridge is not None and ib.exists():
                            try:
                                input_bridge.load_state_dict(
                                    torch.load(str(ib), map_location="cpu"),
                                    strict=False,
                                )
                            except Exception:
                                pass
                        if output_bridge is not None and ob.exists():
                            try:
                                output_bridge.load_state_dict(
                                    torch.load(str(ob), map_location="cpu"),
                                    strict=False,
                                )
                            except Exception:
                                pass
                        # Inject
                        transplanted_module = TransplantedModule(
                            donor_module, input_bridge, output_bridge, freeze_donor=True
                        )
                        self._inject_module(
                            target_model, transplanted_module, target_layer_idx
                        )
                    except Exception as e:
                        report["errors"].append(
                            {"layer": target_layer_idx, "error": str(e)}
                        )
                # Restore layer-level fusion gate if present
                gf = artifacts_dir / f"{layer_key}_fusion_gate.pt"
                if (
                    gf.exists()
                    and hasattr(target_model, "transplant_fusion_gates")
                    and layer_key in target_model.transplant_fusion_gates
                ):
                    try:
                        target_model.transplant_fusion_gates[layer_key].load_state_dict(
                            torch.load(str(gf), map_location="cpu"), strict=False
                        )
                    except Exception:
                        pass
                report["layers"][layer_key] = {
                    "adapters": [e.get("source_component") for e in entries],
                    "gate": (
                        layer_key
                        in getattr(target_model, "transplant_fusion_gates", {})
                    ),
                }
            return report
        # Fallback path: iterate transplant_configs order and infer indices
        layer_counts: Dict[str, int] = {}
        for cfg in meta.get("transplant_configs", []):
            try:
                tc = TransplantConfig(**cfg)
                donor_data = self._load_component(Path(kb_dir), tc.source_component)
                if donor_data is None:
                    report["errors"].append(
                        {
                            "layer": tc.target_layer,
                            "error": f"component not found: {tc.source_component}",
                        }
                    )
                    continue
                donor_module = self._create_donor_module(donor_data)
                target_layer_module = self._get_target_layer(
                    target_model, tc.target_layer
                )
                input_bridge, output_bridge = self._create_bridges_for_module(
                    donor_module, target_layer_module, tc
                )
                layer_key = f"layer_{tc.target_layer}"
                # Determine adapter index for this layer
                idx = layer_counts.get(layer_key, 0)
                # Load bridge states if available
                ib_idx = artifacts_dir / f"{layer_key}_idx{idx}_input_bridge.pt"
                ob_idx = artifacts_dir / f"{layer_key}_idx{idx}_output_bridge.pt"
                ib = (
                    ib_idx
                    if ib_idx.exists()
                    else (artifacts_dir / f"{layer_key}_input_bridge.pt")
                )
                ob = (
                    ob_idx
                    if ob_idx.exists()
                    else (artifacts_dir / f"{layer_key}_output_bridge.pt")
                )
                if input_bridge is not None and ib.exists():
                    try:
                        input_bridge.load_state_dict(
                            torch.load(str(ib), map_location="cpu"), strict=False
                        )
                    except Exception:
                        pass
                if output_bridge is not None and ob.exists():
                    try:
                        output_bridge.load_state_dict(
                            torch.load(str(ob), map_location="cpu"), strict=False
                        )
                    except Exception:
                        pass
                # Inject wrapper
                transplanted_module = TransplantedModule(
                    donor_module,
                    input_bridge,
                    output_bridge,
                    freeze_donor=tc.freeze_donor,
                )
                self._inject_module(target_model, transplanted_module, tc.target_layer)
                layer_counts[layer_key] = idx + 1
                # Restore fusion gate if present
                gf = artifacts_dir / f"{layer_key}_fusion_gate.pt"
                gate_restored = False
                if (
                    gf.exists()
                    and hasattr(target_model, "transplant_fusion_gates")
                    and layer_key in target_model.transplant_fusion_gates
                ):
                    try:
                        target_model.transplant_fusion_gates[layer_key].load_state_dict(
                            torch.load(str(gf), map_location="cpu"), strict=False
                        )
                        gate_restored = True
                    except Exception:
                        pass
                report["layers"][layer_key] = {
                    "donor": tc.source_component,
                    "bridges": {
                        "input": input_bridge is not None,
                        "output": output_bridge is not None,
                    },
                    "gate": gate_restored,
                }
            except Exception as e:
                report["errors"].append(
                    {"layer": cfg.get("target_layer"), "error": str(e)}
                )
        return report

    # ---------------- Mixture-of-Bridges training (offline self-supervised) ----------------
    def train_mixture_on_layer(
        self,
        model: nn.Module,
        layer: int,
        k: int = 2,
        steps: int = 50,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """Train k adapter bridges and a fusion gate to approximate the target layer output.
        Uses a small synthetic corpus from DataManager; optimizes MSE between fused adapters and base output.
        Only adapter and gate params are updated.
        """
        from ..utils.data_manager import DataManager

        dm = DataManager(self.config)
        texts = list(dm._create_diverse_corpus()["text"])[:16]  # small batch
        # Tokenizer for encoding
        _, tok, _ = self.model_loader.load_model_and_tokenizer(
            model.config.name_or_path if hasattr(model, "config") else "distilgpt2"
        )
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        target_layer_module = self._get_target_layer(model, layer)
        hidden_size = getattr(getattr(model, "config", object()), "hidden_size", None)
        if hidden_size is None:
            return {"error": "hidden_size not found"}
        # Capture base outputs
        with torch.no_grad():

            def _cap_hook(mod, inp, out):
                return out

            target_layer_module.register_forward_hook(lambda m, i, o: o)
            model(**enc)
            # Retrieve last captured via unsafe closure; fallback to recompute
        # Build k adapters as identity-ish bridges
        adapters = nn.ModuleList(
            [
                BridgeNetwork(
                    hidden_size, hidden_size, hidden_dim=max(8, hidden_size // (i + 2))
                )
                for i in range(k)
            ]
        )

        # Simple fusion gate over adapters (reuse from _transplant_with_adapter_fusion but simplified)
        class Gate(nn.Module):
            def __init__(self, hs, k):
                super().__init__()
                self.alpha = nn.Parameter(torch.zeros(k))

            def forward(self, outs: List[torch.Tensor]):
                w = torch.softmax(self.alpha, dim=0)
                return sum(w[i] * outs[i] for i in range(len(outs)))

        gate = Gate(hidden_size, k)
        # Optimization
        params = list(adapters.parameters()) + list(gate.parameters())
        opt = optim.Adam(params, lr=lr)
        losses: List[float] = []
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            # Forward to get layer input by short-circuiting
            # We approximate using embeddings as input proxy
            h = model.get_input_embeddings()(enc["input_ids"]).to(device)
            outs = [ad(h) for ad in adapters]
            fused = gate(outs)
            # Recompute base output for the same h using target layer
            base_out = target_layer_module(h)
            loss = (fused - base_out).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        # Attach as layer adapters under transplanted_modules for downstream fusion
        layer_key = f"layer_{layer}"
        if not hasattr(model, "transplanted_modules"):
            model.transplanted_modules = {}
        model.transplanted_modules[layer_key] = adapters
        if not hasattr(model, "transplant_fusion_gates"):
            model.transplant_fusion_gates = nn.ModuleDict()
        model.transplant_fusion_gates[layer_key] = gate
        return {
            "layer": layer,
            "adapters": k,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        }

    def _apply_transplant_config(
        self,
        target_model: nn.Module,
        target_config: Dict[str, Any],
        knowledge_bank_path: Path,
        transplant_config: TransplantConfig,
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
        transplant_config: TransplantConfig,
    ) -> Dict[str, Any]:
        """
        Transplant embeddings using initialization strategy.

        Implements Section 6.2: Embedding initialization.
        """
        logger.info("Transplanting embeddings...")

        # Load source embeddings
        embeddings_dir = knowledge_bank_path / "embeddings"

        with open(embeddings_dir / "config.json", "r"):
            pass
        # Load embeddings with storage helper (supports sharded pt)
        from ..utils.storage import load_sharded_pt

        source_embeddings = None
        idx = embeddings_dir / "embeddings.index.json"
        if idx.exists():
            try:
                source_embeddings = load_sharded_pt(idx)
            except Exception:
                source_embeddings = None
        if source_embeddings is None:
            from ..utils.storage import load_pt_or_safetensors

            st = embeddings_dir / "embeddings.safetensors"
            pt = embeddings_dir / "embeddings.pt"
            if st.exists():
                source_embeddings = load_pt_or_safetensors(st)["weight"]
            elif pt.exists():
                t = torch.load(str(pt))
                source_embeddings = (
                    t if isinstance(t, torch.Tensor) else t.get("weight")
                )
            else:
                raise FileNotFoundError("Embeddings not found in knowledge bank")

        # Get target embedding layer
        target_embeddings = target_model.get_input_embeddings()

        source_dim = source_embeddings.shape[1]
        target_dim = target_embeddings.weight.shape[1]

        # Create bridge network if dimensions don't match
        if source_dim != target_dim:
            bridge = BridgeNetwork(
                input_dim=source_dim,
                output_dim=target_dim,
                hidden_dim=transplant_config.bridge_hidden_size,
            )

            # Transform source embeddings
            with torch.no_grad():
                transformed_embeddings = bridge(source_embeddings)

            # Initialize target embeddings
            target_embeddings.weight.data[: transformed_embeddings.shape[0]] = (
                transformed_embeddings
            )

            # Add bridge to model for training
            if not hasattr(target_model, "embedding_bridge"):
                target_model.embedding_bridge = bridge
        else:
            # Direct copy if dimensions match
            with torch.no_grad():
                min_vocab = min(
                    source_embeddings.shape[0], target_embeddings.weight.shape[0]
                )
                target_embeddings.weight.data[:min_vocab] = source_embeddings[
                    :min_vocab
                ]

        # Freeze embeddings if specified
        if transplant_config.freeze_donor:
            target_embeddings.weight.requires_grad = False

        return {
            "strategy": "embedding_init",
            "source_dim": source_dim,
            "target_dim": target_dim,
            "bridge_used": source_dim != target_dim,
            "vocab_overlap": min(
                source_embeddings.shape[0], target_embeddings.weight.shape[0]
            ),
        }

    def _transplant_module(
        self,
        target_model: nn.Module,
        target_config: Dict[str, Any],
        knowledge_bank_path: Path,
        transplant_config: TransplantConfig,
    ) -> Dict[str, Any]:
        """
        Transplant a complete module (attention head or FFN).

        Implements Section 6.2: Module injection strategy.
        """
        logger.info(f"Transplanting module: {transplant_config.source_component}")

        # Load source component
        component_data = self._load_component(
            knowledge_bank_path, transplant_config.source_component
        )

        if component_data is None:
            raise ValueError(
                f"Could not load component: {transplant_config.source_component}"
            )

        # Create donor module from loaded weights
        donor_module = self._create_donor_module(component_data)

        # Get target layer for injection
        target_layer = self._get_target_layer(
            target_model, transplant_config.target_layer
        )

        # Create bridge networks if needed
        input_bridge, output_bridge = self._create_bridges_for_module(
            donor_module, target_layer, transplant_config
        )

        # Create transplanted module
        transplanted_module = TransplantedModule(
            donor_module=donor_module,
            input_bridge=input_bridge,
            output_bridge=output_bridge,
            freeze_donor=transplant_config.freeze_donor,
        )

        # Inject into target model
        self._inject_module(
            target_model, transplanted_module, transplant_config.target_layer
        )

        dims = self._infer_module_dims(donor_module, target_layer, target_config)
        # Diagnostics for shape/signature mismatches
        try:
            if dims.get("donor_in") != dims.get("target_hidden") or dims.get(
                "donor_out"
            ) != dims.get("target_hidden"):
                logger.warning(
                    f"Dimension mismatch on layer {transplant_config.target_layer}: "
                    f"donor_in={dims.get('donor_in')} donor_out={dims.get('donor_out')} "
                    f"target_hidden={dims.get('target_hidden')} — bridges were installed to reconcile."
                )
        except Exception:
            pass
        return {
            "strategy": "module_injection",
            "source_component": transplant_config.source_component,
            "target_layer": transplant_config.target_layer,
            "bridges_created": {
                "input_bridge": input_bridge is not None,
                "output_bridge": output_bridge is not None,
            },
            "dimension_mapping": dims,
        }

    def _transplant_with_adapter_fusion(
        self,
        target_model: nn.Module,
        target_config: Dict[str, Any],
        knowledge_bank_path: Path,
        transplant_config: TransplantConfig,
    ) -> Dict[str, Any]:
        """
        Transplant using AdapterFusion strategy.

        Implements Section 6.3: Advanced composition with AdapterFusion.
        """
        logger.info("Transplanting with AdapterFusion...")

        # If adapters already exist for this layer, skip reinjection and only attach fusion
        layer_key = f"layer_{transplant_config.target_layer}"
        if (
            not hasattr(target_model, "transplanted_modules")
            or layer_key not in target_model.transplanted_modules
        ):
            module_result = self._transplant_module(
                target_model, target_config, knowledge_bank_path, transplant_config
            )
        else:
            module_result = {
                "strategy": "adapter_fusion",
                "source_component": transplant_config.source_component,
                "target_layer": transplant_config.target_layer,
                "bridges_created": {"input_bridge": False, "output_bridge": False},
                "dimension_mapping": {
                    "target_hidden": int(target_config.get("hidden_size", 768))
                },
            }

        # Implement a simple gated fusion across all transplanted modules at the target layer
        hidden_size = target_config.get("hidden_size", 768)

        class AdapterFusionGate(nn.Module):
            def __init__(self, hidden_size: int, num_adapters: int):
                super().__init__()
                self.query = nn.Linear(hidden_size, hidden_size)
                self.keys = nn.ModuleList(
                    [nn.Linear(hidden_size, hidden_size) for _ in range(num_adapters)]
                )
                self.values = nn.ModuleList(
                    [nn.Linear(hidden_size, hidden_size) for _ in range(num_adapters)]
                )
                self.out = nn.Linear(hidden_size, hidden_size)
                self.scale = hidden_size**-0.5

            def forward(
                self, h: torch.Tensor, adapters_out: List[torch.Tensor]
            ) -> torch.Tensor:
                q = self.query(h)
                # Compute attention weights over adapters
                scores = []
                for i, a in enumerate(adapters_out):
                    k = self.keys[i](a)
                    s = (q * k).sum(dim=-1, keepdim=True) * self.scale
                    scores.append(s)
                attn = torch.softmax(torch.cat(scores, dim=-1), dim=-1)  # [B, T, N]
                fused = 0.0
                for i, a in enumerate(adapters_out):
                    v = self.values[i](a)
                    w = attn[..., i : i + 1]
                    fused = fused + w * v
                return self.out(fused)

        # Collect transplanted modules for this layer (support multiple adapters per layer)
        if (
            not hasattr(target_model, "transplanted_modules")
            or layer_key not in target_model.transplanted_modules
        ):
            return module_result
        # Normalize to list of adapters
        adapters: List[nn.Module] = []
        mods = target_model.transplanted_modules[layer_key]
        if isinstance(mods, nn.ModuleList):
            adapters = list(mods)
        else:
            adapters = [mods]
        fusion_gate = AdapterFusionGate(hidden_size, len(adapters))

        # Register a hook to fuse outputs at this layer
        target_layer_module = self._get_target_layer(
            target_model, transplant_config.target_layer
        )

        def _fusion_hook(module, inputs, output):
            base_out = output[0] if isinstance(output, (tuple, list)) else output
            x = (
                inputs[0]
                if isinstance(inputs, (tuple, list)) and len(inputs) > 0
                else base_out
            )
            adapters_out = [adp(x) for adp in adapters]
            fused = fusion_gate(base_out, adapters_out)
            return fused

        if not hasattr(target_model, "adapter_fusion_hooks"):
            target_model.adapter_fusion_hooks = {}
        # Remove existing hook if present
        if layer_key in target_model.adapter_fusion_hooks:
            try:
                target_model.adapter_fusion_hooks[layer_key].remove()
            except Exception:
                pass
        target_model.adapter_fusion_hooks[layer_key] = (
            target_layer_module.register_forward_hook(_fusion_hook)
        )
        if not hasattr(target_model, "transplant_fusion_gates"):
            target_model.transplant_fusion_gates = nn.ModuleDict()
        target_model.transplant_fusion_gates[layer_key] = fusion_gate

        module_result["strategy"] = "adapter_fusion"
        module_result["fusion_gate"] = True

        return module_result

    def _load_component(
        self, knowledge_bank_path: Path, component_name: str
    ) -> Optional[Dict[str, Any]]:
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

        # Load weights via storage helper
        from ..utils.storage import load_component_weights_dir

        weights = load_component_weights_dir(component_dir)

        return {"config": config, "weights": weights}

    # Public helper for external callers
    def build_donor_module_from_kb(
        self, knowledge_bank_path: str, component_name: str
    ) -> nn.Module:
        """Build a donor nn.Module from a knowledge bank component.
        Raises if component not found or cannot be constructed.
        """
        data = self._load_component(Path(knowledge_bank_path), component_name)
        if data is None:
            raise ValueError(f"Component not found in knowledge bank: {component_name}")
        return self._create_donor_module(data)

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

    def _create_attention_module(
        self, config: Dict[str, Any], weights: Dict[str, Any]
    ) -> nn.Module:
        """Create a multi-head attention module using provided tensors.
        Supports both MHA (q/k/v/o) and GQA/MQA with combined kv_proj.
        """

        # Infer dimensions from weight tensors
        def wshape(name):
            return (
                list(weights[name]["weight"].shape)
                if name in weights and "weight" in weights[name]
                else None
            )

        q_shape = wshape("q_proj")
        k_shape = wshape("k_proj")
        v_shape = wshape("v_proj")
        kv_shape = wshape("kv_proj")
        o_shape = wshape("o_proj")
        if q_shape is None or o_shape is None:
            raise ValueError(
                "Attention component requires at least q_proj and o_proj weights"
            )

        in_dim = q_shape[1]
        q_out = q_shape[0]
        # Heads
        num_q_heads = int(config.get("num_heads") or config.get("num_query_heads") or 1)
        head_dim = q_out // max(1, num_q_heads)
        num_kv_heads = int(config.get("num_key_value_heads") or num_q_heads)

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                # Build linear projections with exact shapes from tensors
                self.q_proj = nn.Linear(in_dim, q_out, bias=False)
                if kv_shape is not None and (k_shape is None or v_shape is None):
                    # Single combined kv projection matrix
                    self.kv_proj = nn.Linear(in_dim, kv_shape[0], bias=False)
                    self.k_proj = None
                    self.v_proj = None
                    self._kv_combined = True
                else:
                    self.k_proj = (
                        nn.Linear(in_dim, k_shape[0], bias=False)
                        if k_shape is not None
                        else None
                    )
                    self.v_proj = (
                        nn.Linear(in_dim, v_shape[0], bias=False)
                        if v_shape is not None
                        else None
                    )
                    self.kv_proj = None
                    self._kv_combined = False
                self.o_proj = nn.Linear(o_shape[1], o_shape[0], bias=False)
                self.num_q_heads = max(1, num_q_heads)
                self.num_kv_heads = max(1, num_kv_heads)
                self.head_dim = (
                    head_dim if head_dim > 0 else max(1, q_out // self.num_q_heads)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B, T, H = x.shape
                q = self.q_proj(x)  # [B,T,q_out]
                if self._kv_combined:
                    kv = self.kv_proj(x)  # [B,T,kv_out]
                    # Split K and V along last dim (projected dim)
                    kv_dim = kv.shape[-1]
                    half = kv_dim // 2
                    k = kv[..., :half]
                    v = kv[..., half:]
                else:
                    k = self.k_proj(x) if self.k_proj is not None else q
                    v = self.v_proj(x) if self.v_proj is not None else q
                # Reshape to heads
                q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(
                    1, 2
                )  # [B, hq, T, d]
                # Infer kv head_dim from k size
                kv_head_dim = k.shape[-1] // self.num_kv_heads
                k = k.view(B, T, self.num_kv_heads, kv_head_dim).transpose(
                    1, 2
                )  # [B, hk, T, dkv]
                v = v.view(B, T, self.num_kv_heads, kv_head_dim).transpose(
                    1, 2
                )  # [B, hk, T, dkv]
                # Map query heads to kv heads (GQA/MQA)
                if self.num_kv_heads == self.num_q_heads:
                    k_exp = k
                    v_exp = v
                else:
                    # Repeat or expand kv across groups
                    group_size = max(1, self.num_q_heads // self.num_kv_heads)
                    k_exp = k.repeat_interleave(group_size, dim=1)
                    v_exp = v.repeat_interleave(group_size, dim=1)
                scale = self.head_dim**-0.5
                attn_scores = (
                    torch.matmul(q, k_exp.transpose(-2, -1)) * scale
                )  # [B,hq,T,T]
                attn = torch.softmax(attn_scores, dim=-1)
                ctx = torch.matmul(attn, v_exp)  # [B,hq,T,d]
                ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)
                return self.o_proj(ctx)

        module = Attention()
        # Load weights
        module.q_proj.weight.data.copy_(weights["q_proj"]["weight"])  # type: ignore
        if module._kv_combined:
            module.kv_proj.weight.data.copy_(weights["kv_proj"]["weight"])  # type: ignore
        else:
            if module.k_proj is not None and "k_proj" in weights:
                module.k_proj.weight.data.copy_(weights["k_proj"]["weight"])  # type: ignore
            if module.v_proj is not None and "v_proj" in weights:
                module.v_proj.weight.data.copy_(weights["v_proj"]["weight"])  # type: ignore
        module.o_proj.weight.data.copy_(weights["o_proj"]["weight"])  # type: ignore
        return module

    def _create_ffn_module(
        self, config: Dict[str, Any], weights: Dict[str, Any]
    ) -> nn.Module:
        """Create an FFN module from config and weights using exact tensor shapes.
        Supports gated (SwiGLU-style) FFN when gate_proj and up_proj exist.
        """

        def shp(name):
            return (
                list(weights[name]["weight"].shape)
                if name in weights and "weight" in weights[name]
                else None
            )

        g_shape, u_shape, d_shape = shp("gate_proj"), shp("up_proj"), shp("down_proj")
        if d_shape is None:
            raise ValueError("FFN requires down_proj weight")
        hidden_in = d_shape[1]
        inter = d_shape[0]
        act_name = str(config.get("activation_function", "relu")).lower()

        class FFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.has_gate = g_shape is not None and u_shape is not None
                if self.has_gate:
                    self.gate_proj = nn.Linear(hidden_in, g_shape[0], bias=False)
                    self.up_proj = nn.Linear(hidden_in, u_shape[0], bias=False)
                    inter_dim = u_shape[0]
                else:
                    # Fallback: single hidden projection
                    inter_dim = inter
                    self.ff = nn.Linear(hidden_in, inter_dim, bias=False)
                self.down_proj = nn.Linear(inter_dim, hidden_in, bias=False)
                if act_name == "silu":
                    self.activation = nn.SiLU()
                elif act_name == "gelu":
                    self.activation = nn.GELU()
                else:
                    self.activation = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.has_gate:
                    return self.down_proj(
                        self.activation(self.gate_proj(x)) * self.up_proj(x)
                    )
                h = self.ff(x)
                return self.down_proj(self.activation(h))

        module = FFN()
        # Load weights
        if hasattr(module, "gate_proj") and "gate_proj" in weights:
            module.gate_proj.weight.data.copy_(weights["gate_proj"]["weight"])  # type: ignore
        if hasattr(module, "up_proj") and "up_proj" in weights:
            module.up_proj.weight.data.copy_(weights["up_proj"]["weight"])  # type: ignore
        if (
            hasattr(module, "ff")
            and hasattr(module, "ff")
            and "gate_proj" not in weights
            and "up_proj" not in weights
        ):
            # Try to use down_proj^T as proxy input if available isn't ideal; keep random init for ff
            pass
        module.down_proj.weight.data.copy_(weights["down_proj"]["weight"])  # type: ignore
        return module

    def _create_generic_module(
        self, config: Dict[str, Any], weights: Dict[str, Any]
    ) -> nn.Module:
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
                for attr in pattern.split("."):
                    module = getattr(module, attr)
                return module
            except AttributeError:
                continue

        raise ValueError(f"Could not find layer {layer_idx} in target model")

    def _create_bridges_for_module(
        self,
        donor_module: nn.Module,
        target_layer: nn.Module,
        transplant_config: TransplantConfig,
    ) -> Tuple[Optional[BridgeNetwork], Optional[BridgeNetwork]]:
        """Create bridge networks for dimensional compatibility.
        Infers donor and target dimensions and builds down/up bridges if needed.
        """
        # Infer target hidden size
        target_hidden = getattr(
            getattr(target_layer, "config", object()), "hidden_size", None
        )
        if target_hidden is None:
            target_hidden = getattr(
                getattr(target_layer, "self_attn", object()), "hidden_size", None
            )
        if target_hidden is None:
            target_hidden = getattr(
                getattr(target_layer, "mlp", object()), "hidden_size", None
            )
        if target_hidden is None:
            target_hidden = getattr(
                getattr(target_layer, "dense", object()), "out_features", None
            )
        # Fallback to model-level config if available
        if target_hidden is None:
            target_hidden = getattr(
                getattr(target_layer, "parent", object()), "config", object()
            )
        # Try model.config.hidden_size via the module tree
        try:
            # walk up modules to find a .config
            m = target_layer
            while m is not None and not hasattr(m, "config"):
                m = getattr(m, "parent", None)
            if m is not None and hasattr(m.config, "hidden_size"):
                target_hidden = getattr(m.config, "hidden_size")
        except Exception:
            pass
        if target_hidden is None:
            target_hidden = 768  # conservative default

        # Infer donor input/output dims
        donor_in = None
        donor_out = None
        for attr in ("q_proj", "gate_proj", "in_proj", "fc1", "w1"):
            if hasattr(donor_module, attr) and hasattr(
                getattr(donor_module, attr), "in_features"
            ):
                donor_in = getattr(getattr(donor_module, attr), "in_features")
                break
        for attr in ("o_proj", "down_proj", "out_proj", "fc2", "w2"):
            if hasattr(donor_module, attr) and hasattr(
                getattr(donor_module, attr), "out_features"
            ):
                donor_out = getattr(getattr(donor_module, attr), "out_features")
                break
        # Fallbacks if linear layers not found
        if donor_in is None or donor_out is None:
            if hasattr(donor_module, "hidden_size"):
                donor_in = donor_in or donor_module.hidden_size
                donor_out = donor_out or donor_module.hidden_size
            else:
                donor_in = donor_in or target_hidden
                donor_out = donor_out or target_hidden

        # Create bridges only if needed
        input_bridge = None
        output_bridge = None
        if donor_in != target_hidden:
            input_bridge = BridgeNetwork(
                input_dim=target_hidden,
                output_dim=donor_in,
                hidden_dim=transplant_config.bridge_hidden_size,
            )
        if donor_out != target_hidden:
            output_bridge = BridgeNetwork(
                input_dim=donor_out,
                output_dim=target_hidden,
                hidden_dim=transplant_config.bridge_hidden_size,
            )

        return input_bridge, output_bridge

    def _inject_module(
        self,
        target_model: nn.Module,
        transplanted_module: TransplantedModule,
        target_layer: int,
    ) -> None:
        """Inject transplanted module into target model.
        Prefer structured submodule replacement; fallback to a forward hook composition if needed.
        """

        # Try structured submodule replacement
        target_layer_module = self._get_target_layer(target_model, target_layer)

        class _FusionGate(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.alpha = nn.Parameter(torch.zeros(1))

            def forward(
                self, base_out: torch.Tensor, transplanted_out: torch.Tensor
            ) -> torch.Tensor:
                gate = torch.sigmoid(self.alpha)
                return base_out + gate * transplanted_out

        hidden_size = getattr(
            getattr(target_model, "config", object()), "hidden_size", 768
        )
        fusion_gate = _FusionGate(hidden_size)

        class _ComposedModule(nn.Module):
            def __init__(
                self,
                base_module: nn.Module,
                transplant: TransplantedModule,
                gate: _FusionGate,
            ):
                super().__init__()
                self.base = base_module
                self.transplant = transplant
                self.gate = gate

            def forward(self, *args, **kwargs):
                base_out = self.base(*args, **kwargs)
                # normalize to tensor
                x = None
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    x = args[0]
                elif (
                    isinstance(base_out, (tuple, list))
                    and len(base_out) > 0
                    and isinstance(base_out[0], torch.Tensor)
                ):
                    x = base_out[0]
                elif isinstance(base_out, dict) and "last_hidden_state" in base_out:
                    x = base_out["last_hidden_state"]
                if x is None:
                    return base_out
                try:
                    transplanted_out = self.transplant(x)
                    bout = (
                        base_out[0] if isinstance(base_out, (tuple, list)) else base_out
                    )
                    if isinstance(bout, dict) and "last_hidden_state" in bout:
                        bout = bout["last_hidden_state"]
                    fused = self.gate(bout, transplanted_out)
                    return fused
                except Exception:
                    return base_out

        replaced = False
        # Try architecture-aware replacement first
        model_type = getattr(
            getattr(target_model, "config", object()), "model_type", None
        )
        # Prefer replacing FFN if donor is FFN-like (has down_proj), else ATTention
        kind = (
            "ffn"
            if (
                hasattr(transplanted_module.donor_module, "down_proj")
                or hasattr(transplanted_module.donor_module, "gate_proj")
            )
            else "attn"
        )
        try:
            # Attempt to replace in the full model tree to preserve structure
            base_module_found = replace_layer_submodule(
                target_model,
                target_layer,
                kind,
                _ComposedModule(
                    getattr(
                        target_layer_module,
                        kind if hasattr(target_layer_module, kind) else "mlp",
                        target_layer_module,
                    ),
                    transplanted_module,
                    fusion_gate,
                ),
                model_type=model_type,
            )
            replaced = bool(base_module_found)
        except Exception:
            replaced = False
        # Fallback: local common names on the resolved layer
        if not replaced:
            for name in (
                "mlp",
                "ffn",
                "feed_forward",
                "self_attn",
                "attn",
                "attention",
            ):
                if hasattr(target_layer_module, name):
                    try:
                        base_sub = getattr(target_layer_module, name)
                        setattr(
                            target_layer_module,
                            name,
                            _ComposedModule(base_sub, transplanted_module, fusion_gate),
                        )
                        replaced = True
                        break
                    except Exception:
                        continue

        if replaced:
            if not hasattr(target_model, "transplanted_modules"):
                target_model.transplanted_modules = nn.ModuleDict()
            lkey = f"layer_{target_layer}"
            if lkey in target_model.transplanted_modules:
                existing = target_model.transplanted_modules[lkey]
                if isinstance(existing, nn.ModuleList):
                    existing.append(transplanted_module)
                else:
                    target_model.transplanted_modules[lkey] = nn.ModuleList(
                        [existing, transplanted_module]
                    )
            else:
                target_model.transplanted_modules[lkey] = transplanted_module
            if not hasattr(target_model, "transplant_fusion_gates"):
                target_model.transplant_fusion_gates = nn.ModuleDict()
            target_model.transplant_fusion_gates[f"layer_{target_layer}"] = fusion_gate
            return

        # Fallback to forward hook composition
        if not hasattr(target_model, "transplanted_modules"):
            target_model.transplanted_modules = nn.ModuleDict()
        lkey = f"layer_{target_layer}"
        if lkey in target_model.transplanted_modules:
            existing = target_model.transplanted_modules[lkey]
            if isinstance(existing, nn.ModuleList):
                existing.append(transplanted_module)
            else:
                ml = nn.ModuleList([existing, transplanted_module])
                target_model.transplanted_modules[lkey] = ml
        else:
            target_model.transplanted_modules[lkey] = transplanted_module

        def _hook(module, inputs, output):
            try:
                base_out = output[0] if isinstance(output, (tuple, list)) else output
                if isinstance(base_out, dict) and "last_hidden_state" in base_out:
                    base_out = base_out["last_hidden_state"]
                x = (
                    inputs[0]
                    if isinstance(inputs, (tuple, list)) and len(inputs) > 0
                    else base_out
                )
                transplanted_out = transplanted_module(x)
                return fusion_gate(base_out, transplanted_out)
            except Exception:
                return output

        if not hasattr(target_model, "transplant_hooks"):
            target_model.transplant_hooks = {}
        if f"layer_{target_layer}" in target_model.transplant_hooks:
            try:
                target_model.transplant_hooks[f"layer_{target_layer}"].remove()
            except Exception:
                pass
        target_model.transplant_hooks[f"layer_{target_layer}"] = (
            target_layer_module.register_forward_hook(_hook)
        )
        if not hasattr(target_model, "transplant_fusion_gates"):
            target_model.transplant_fusion_gates = nn.ModuleDict()
        target_model.transplant_fusion_gates[f"layer_{target_layer}"] = fusion_gate

    def load_transplant_artifacts(
        self, model: nn.Module, artifacts_dir: str
    ) -> Dict[str, Any]:
        """Load saved transplant artifacts (bridges/gates) and reattach gates as hooks.
        This does not recreate donor modules, but reinstalls fusion gates so they can be fine-tuned.
        """
        artifacts_path = Path(artifacts_dir)
        report = {"attached_gates": [], "missing_layers": []}
        try:
            meta_file = artifacts_path.parent / "transplant_artifacts.json"
            if not meta_file.exists():
                return {"error": "transplant_artifacts.json not found"}
            meta = json.loads(meta_file.read_text())
            layers = meta.get("layers", {})
            for layer_key, info in layers.items():
                # Reattach fusion gate if present
                gate_file = info.get("fusion_gate")
                if not gate_file:
                    continue
                # Parse layer index
                try:
                    target_layer_idx = int(layer_key.split("_")[1])
                except Exception:
                    continue
                target_layer_module = self._get_target_layer(model, target_layer_idx)
                # Create fusion gate and load state
                hidden_size = getattr(
                    getattr(model, "config", object()), "hidden_size", 768
                )

                class _FusionGate(nn.Module):
                    def __init__(self, hidden_size: int):
                        super().__init__()
                        self.alpha = nn.Parameter(torch.zeros(1))

                    def forward(
                        self, base_out: torch.Tensor, transplanted_out: torch.Tensor
                    ) -> torch.Tensor:
                        gate = torch.sigmoid(self.alpha)
                        return base_out + gate * transplanted_out

                fusion_gate = _FusionGate(hidden_size)
                try:
                    sd = torch.load(gate_file, map_location="cpu")
                    fusion_gate.load_state_dict(sd, strict=False)
                except Exception:
                    pass

                def _hook(module, inputs, output):
                    try:
                        base_out = (
                            output[0] if isinstance(output, (tuple, list)) else output
                        )
                        if (
                            isinstance(base_out, dict)
                            and "last_hidden_state" in base_out
                        ):
                            base_out = base_out["last_hidden_state"]
                        # no donor module available; identity transplanted_out
                        transplanted_out = torch.zeros_like(base_out)
                        return fusion_gate(base_out, transplanted_out)
                    except Exception:
                        return output

                if not hasattr(model, "transplant_hooks"):
                    model.transplant_hooks = {}
                # Remove previous hook if any
                if layer_key in model.transplant_hooks:
                    try:
                        model.transplant_hooks[layer_key].remove()
                    except Exception:
                        pass
                model.transplant_hooks[layer_key] = (
                    target_layer_module.register_forward_hook(_hook)
                )
                report["attached_gates"].append(layer_key)
        except Exception as e:
            return {"error": str(e)}
        return report

    def _infer_module_dims(
        self,
        donor_module: nn.Module,
        target_layer: nn.Module,
        target_config: Dict[str, Any],
    ) -> Dict[str, int]:
        target_hidden = target_config.get("hidden_size", 768)
        donor_in, donor_out = None, None
        for attr in (
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj",
            "fc1",
            "w1",
            "w3",
        ):
            if hasattr(donor_module, attr) and hasattr(
                getattr(donor_module, attr), "in_features"
            ):
                donor_in = getattr(getattr(donor_module, attr), "in_features")
                break
        for attr in ("o_proj", "down_proj", "out_proj", "fc2", "w2"):
            if hasattr(donor_module, attr) and hasattr(
                getattr(donor_module, attr), "out_features"
            ):
                donor_out = getattr(getattr(donor_module, attr), "out_features")
                break
        if donor_in is None:
            donor_in = getattr(donor_module, "hidden_size", target_hidden)
        if donor_out is None:
            donor_out = getattr(donor_module, "hidden_size", target_hidden)
        return {
            "donor_in": int(donor_in),
            "donor_out": int(donor_out),
            "target_hidden": int(target_hidden),
        }
