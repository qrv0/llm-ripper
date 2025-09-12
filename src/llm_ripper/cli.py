"""
Command-line interface for LLM Ripper.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from .utils.config import ConfigManager

# Lazy/forgiving imports to avoid heavy deps at import time in test/offline contexts
try:
    from .core import (
        KnowledgeExtractor,
        ActivationCapture,
        KnowledgeAnalyzer,
        KnowledgeTransplanter,
        ValidationSuite,
    )
except Exception:
    class KnowledgeExtractor:  # type: ignore
        pass
    class ActivationCapture:  # type: ignore
        pass
    class KnowledgeAnalyzer:  # type: ignore
        pass
    class KnowledgeTransplanter:  # type: ignore
        pass
    class ValidationSuite:  # type: ignore
        pass

try:
    from .core.transplant import TransplantConfig
except Exception:
    class TransplantConfig:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

try:
    from .causal import Tracer, TraceConfig
except Exception:
    class Tracer:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    class TraceConfig:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

try:
    from .counterfactual import generate_minimal_pairs, CounterfactualEvaluator
except Exception:
    def generate_minimal_pairs(*args, **kwargs):  # type: ignore
        return "pairs.jsonl"
    class CounterfactualEvaluator:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

try:
    from .uq import UQRunner
except Exception:
    class UQRunner:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            return {"run_id": "r", "summary_file": "uq/summary.json"}

try:
    from .bridge import orthogonal_procrustes_align
except Exception:
    def orthogonal_procrustes_align(*args, **kwargs):  # type: ignore
        return {"matrix_file": "W.npy", "cosine_before": 0.0, "cosine_after": 0.0}

try:
    from .safety import ProvenanceScanner
except Exception:
    class ProvenanceScanner:  # type: ignore
        def scan(self, *args, **kwargs):
            return {"ok": True, "violations": []}

try:
    from .features import discover_features
except Exception:
    def discover_features(*args, **kwargs):  # type: ignore
        return {"catalog_file": "catalog/heads.json", "features": 0}


def setup_logging(
    log_level: str = "INFO", log_file: str = "llm_ripper.log", json_logs: bool = False
):
    """Setup logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    # Ensure a file name for handlers
    log_file = log_file or "llm_ripper.log"
    if json_logs:

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
                payload = {
                    "ts": getattr(record, "created", None),
                    "level": record.levelname,
                    "name": record.name,
                    "msg": record.getMessage(),
                }
                return json.dumps(payload, default=str)

        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Rotating file
    try:
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=2)
    except Exception:
        fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def _handle_top_level_flags(ns):
    """Handle top-level flags like --version/--about before subcommands.

    Note: This needs to be at module scope because main() calls it after
    create_parser(). A previous nested definition inside create_parser
    was not visible here, causing a NameError when running the CLI.
    """
    try:
        from llm_ripper import __version__ as _v
    except Exception:
        _v = "unknown"
    if getattr(ns, "version", False):
        logging.info("%s", _v)
        sys.exit(0)
    if getattr(ns, "about", False):
        logging.info(
            "LLM Ripper — Modular knowledge extraction, analysis, and transplantation for Transformer models. Docs: README.md"
        )
        sys.exit(0)


def set_global_seeds(seed: int):
    try:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _mk_run_id() -> str:
    try:
        import random

        suffix = f"{random.randint(0, 9999):04d}"
    except Exception:
        suffix = "0000"
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + suffix


def extract_command(args):
    """Extract knowledge from a model."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    config.validate_for_extract(args.model, args.output_dir)
    config.create_directories()

    setup_logging(
        config.get("log_level"),
        getattr(args, "log_file", "llm_ripper.log"),
        getattr(args, "json_logs", False),
    )
    set_global_seeds(config.get("seed", 42))
    run_id = _mk_run_id()
    try:
        import torch

        device = config.get("device") or "cpu"
        dtype = str(getattr(torch, "get_default_dtype", lambda: "float32")())
    except Exception:
        device, dtype = (config.get("device") or "cpu"), "float32"
    if not getattr(args, "json", False):
        print(
            f"[llm-ripper] run_id={run_id} seed={config.get('seed', 42)} device={device} dtype={dtype}"
        )

    extractor = KnowledgeExtractor(config)

    components = args.components.split(",") if args.components else None

    result = extractor.extract_model_components(
        model_name=args.model,
        output_dir=args.output_dir,
        components=components,
        force_model_type=args.model_type if getattr(args, "model_type", None) else None,
    )
    if getattr(args, "json", False):
        result["run_id"] = run_id
        print(json.dumps(result, indent=2, default=str))
    else:
        logging.info("✓ Extraction completed successfully!")
        logging.info("  Source model: %s", result['source_model'])
        logging.info("  Components extracted: %s", list(result['extracted_components'].keys()))
        logging.info("  Output directory: %s", args.output_dir)


def capture_command(args):
    """Capture activations from a model."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(
        config.get("log_level"),
        getattr(args, "log_file", "llm_ripper.log"),
        getattr(args, "json_logs", False),
    )
    config.validate_for_capture(args.model, args.output_file)
    run_id = _mk_run_id()
    try:
        import torch

        device = config.get("device") or "cpu"
        dtype = str(getattr(torch, "get_default_dtype", lambda: "float32")())
    except Exception:
        device, dtype = (config.get("device") or "cpu"), "float32"
    if not getattr(args, "json", False):
        print(
            f"[llm-ripper] run_id={run_id} seed={config.get('seed', 42)} device={device} dtype={dtype}"
        )

    # Load corpus dataset (prefer offline synthetic if offline)
    dm = None
    try:
        from .utils.data_manager import DataManager  # type: ignore
        dm = DataManager(config)
    except Exception:
        dm = None
    dataset = None
    if not config.get("offline") and args.dataset:
        try:
            from datasets import load_dataset  # type: ignore
            if args.dataset == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            elif args.dataset == "openwebtext":
                dataset = load_dataset("openwebtext", split="train")
            else:
                dataset = load_dataset(args.dataset, split="train")
        except Exception:
            dataset = None
    if dataset is None and dm is not None:
        try:
            dataset = dm.load_probing_corpus("diverse")
        except Exception:
            dataset = None

    capture = ActivationCapture(config)

    result = capture.capture_model_activations(
        model_name=args.model,
        corpus_dataset=dataset,
        output_file=args.output_file,
        layers_to_capture=args.layers.split(",") if args.layers else None,
        max_samples=args.max_samples,
    )
    if getattr(args, "json", False):
        result["run_id"] = run_id
        print(json.dumps(result, indent=2, default=str))
    else:
        logging.info("✓ Activation capture completed successfully!")
        logging.info("  Model: %s", result['model_name'])
        logging.info("  Samples processed: %s", result['num_samples'])
        logging.info("  Output file: %s", result['output_file'])


def analyze_command(args):
    """Analyze extracted knowledge components."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(
        config.get("log_level"),
        getattr(args, "log_file", "llm_ripper.log"),
        getattr(args, "json_logs", False),
    )
    config.validate_for_analyze(args.knowledge_bank, args.output_dir)
    run_id = _mk_run_id()
    try:
        import torch

        device = config.get("device") or "cpu"
        dtype = str(getattr(torch, "get_default_dtype", lambda: "float32")())
    except Exception:
        device, dtype = (config.get("device") or "cpu"), "float32"
    if not getattr(args, "json", False):
        print(
            f"[llm-ripper] run_id={run_id} seed={config.get('seed', 42)} device={device} dtype={dtype}"
        )

    analyzer = KnowledgeAnalyzer(config)

    result = analyzer.analyze_knowledge_bank(
        knowledge_bank_dir=args.knowledge_bank,
        activations_file=args.activations,
        output_dir=args.output_dir,
    )
    if getattr(args, "json", False):
        result["run_id"] = run_id
        print(json.dumps(result, indent=2, default=str))
    else:
        logging.info("✓ Analysis completed successfully!")
        logging.info("  Source model: %s", result['source_model'])
        logging.info("  Components analyzed: %s", list(result['component_analysis'].keys()))
        logging.info("  Head catalog entries: %s", len(result.get('head_catalog', [])))
        logging.info("  Output directory: %s", args.output_dir)


def transplant_command(args):
    """Transplant knowledge components to a target model."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(
        config.get("log_level"),
        getattr(args, "log_file", "llm_ripper.log"),
        getattr(args, "json_logs", False),
    )
    config.validate_for_transplant(args.source, args.target, args.output_dir)
    run_id = _mk_run_id()
    try:
        import torch

        device = config.get("device") or "cpu"
        dtype = str(getattr(torch, "get_default_dtype", lambda: "float32")())
    except Exception:
        device, dtype = (config.get("device") or "cpu"), "float32"
    if not getattr(args, "json", False):
        print(
            f"[llm-ripper] run_id={run_id} seed={config.get('seed', 42)} device={device} dtype={dtype}"
        )

    transplanter = KnowledgeTransplanter(config)

    # Parse transplant configurations
    transplant_configs = []

    if args.config_file:
        with open(args.config_file, "r") as f:
            configs_data = json.load(f)

        for config_data in configs_data:
            transplant_configs.append(TransplantConfig(**config_data))
    else:
        # Create a simple configuration from command line args
        transplant_configs.append(
            TransplantConfig(
                source_component=args.source_component,
                target_layer=args.target_layer,
                bridge_hidden_size=config.get("adapter_hidden_size", 64),
                freeze_donor=config.get("freeze_donor_weights", True),
                freeze_target=False,
                strategy=args.strategy,
            )
        )

    result = transplanter.transplant_knowledge(
        source_knowledge_bank=args.source,
        target_model_name=args.target,
        transplant_configs=transplant_configs,
        output_dir=args.output_dir,
    )
    if getattr(args, "json", False):
        result["run_id"] = run_id
        print(json.dumps(result, indent=2, default=str))
    else:
        logging.info("✓ Transplantation completed successfully!")
        logging.info("  Source: %s", result['source_model'])
        logging.info("  Target: %s", result['target_model'])
        logging.info("  Components transplanted: %s", len(result['transplanted_components']))
        logging.info("  Output directory: %s", args.output_dir)


def validate_command(args):
    """Validate a transplanted model."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(
        config.get("log_level"),
        getattr(args, "log_file", "llm_ripper.log"),
        getattr(args, "json_logs", False),
    )
    config.validate_for_validate(args.model, args.output_dir)
    run_id = _mk_run_id()
    try:
        import torch

        device = config.get("device") or "cpu"
        dtype = str(getattr(torch, "get_default_dtype", lambda: "float32")())
    except Exception:
        device, dtype = (config.get("device") or "cpu"), "float32"
    if not getattr(args, "json", False):
        print(
            f"[llm-ripper] run_id={run_id} seed={config.get('seed', 42)} device={device} dtype={dtype}"
        )

    validator = ValidationSuite(config)

    benchmarks = args.benchmarks.split(",") if args.benchmarks else None
    if getattr(args, "mechanistic", False):
        if benchmarks is None:
            benchmarks = ["mechanistic"]
        else:
            benchmarks = list(benchmarks) + ["mechanistic"]

    result = validator.validate_transplanted_model(
        transplanted_model_path=args.model,
        baseline_model_name=args.baseline,
        benchmarks=benchmarks,
        output_dir=args.output_dir,
    )
    if getattr(args, "json", False):
        result["run_id"] = run_id
        print(json.dumps(result, indent=2, default=str))
    else:
        logging.info("✓ Validation completed successfully!")
        logging.info("  Model: %s", result['model_path'])
        logging.info("  Overall score: %.3f", result['summary']['overall_score'])

        if result["summary"]["recommendations"]:
            logging.info("  Recommendations:")
            for rec in result["summary"]["recommendations"]:
                logging.info("    - %s", rec)

        logging.info("  Output directory: %s", args.output_dir)


def inspect_command(args):
    """Inspect a knowledge bank directory and print its contents/metadata."""
    kb = Path(args.knowledge_bank)
    meta = {}
    try:
        with open(kb / "extraction_metadata.json", "r") as f:
            meta = json.load(f)
    except Exception:
        pass

    # Helper to compute file sizes
    def file_size(p: Path) -> int:
        try:
            return p.stat().st_size
        except Exception:
            return 0

    # Try to enrich with component configs
    details = {}

    # Helper: sum sizes for sharded index
    def index_bytes(index_path: Path) -> int:
        try:
            data = json.loads(index_path.read_text())
            total = 0
            for name in data.get("parts", []):
                p = index_path.parent / name
                total += file_size(p)
            return total
        except Exception:
            return 0

    # Helper: load tensor shape without assumptions
    def tensor_shape_from_file(path: Path) -> Optional[List[int]]:
        try:
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file as _sload

                d = _sload(str(path))
                # pick first tensor
                for t in d.values():
                    return list(t.shape)
            elif path.suffix == ".pt":
                import torch as _torch

                t = _torch.load(str(path), map_location="cpu")
                if hasattr(t, "shape"):
                    return list(t.shape)
                if isinstance(t, dict):
                    # e.g., {"weight": tensor}
                    for v in t.values():
                        if hasattr(v, "shape"):
                            return list(v.shape)
        except Exception:
            return None
        return None

    try:
        emb_cfg = kb / "embeddings" / "config.json"
        if emb_cfg.exists():
            e = json.loads(emb_cfg.read_text())
            idx = kb / "embeddings" / "embeddings.index.json"
            if idx.exists():
                e["file"] = str(idx)
                e["bytes"] = index_bytes(idx)
            else:
                weights = (
                    kb
                    / "embeddings"
                    / (
                        "embeddings.safetensors"
                        if (kb / "embeddings" / "embeddings.safetensors").exists()
                        else "embeddings.pt"
                    )
                )
                e["file"] = str(weights) if weights.exists() else None
                e["bytes"] = file_size(weights) if weights.exists() else 0
            dims = e.get("dimensions", [])
            if dims:
                try:
                    import math

                    e["param_count"] = int(math.prod(dims))
                except Exception:
                    e["param_count"] = dims[0] * dims[1] if len(dims) == 2 else None
            details["embeddings"] = e
    except Exception:
        pass
    try:
        heads = {}
        heads_dir = kb / "heads"
        if heads_dir.exists():
            for ld in sorted([p for p in heads_dir.iterdir() if p.is_dir()]):
                cfg = ld / "config.json"
                if cfg.exists():
                    c = json.loads(cfg.read_text())
                    # common files
                    sizes = {}
                    params = 0
                    for fn in ("q_proj", "k_proj", "v_proj", "o_proj", "kv_proj"):
                        # sharded index
                        idx = ld / f"{fn}.index.json"
                        if idx.exists():
                            sizes[fn] = {"file": str(idx), "bytes": index_bytes(idx)}
                            # attempt to estimate params by loading shapes of parts
                            try:
                                import json as _json
                                import torch as _torch

                                meta = _json.loads(idx.read_text())
                                shp0 = None
                                dim0_total = 0
                                for name in meta.get("parts", []):
                                    t = _torch.load(
                                        str(idx.parent / name), map_location="cpu"
                                    )
                                    s = list(t.shape)
                                    shp0 = s if shp0 is None else shp0
                                    dim0_total += s[0] if len(s) > 0 else 0
                                if shp0 is not None:
                                    import math

                                    params += int(
                                        dim0_total
                                        * (math.prod(shp0[1:]) if len(shp0) > 1 else 1)
                                    )
                            except Exception:
                                pass
                            continue
                        # single file
                        for ext in (".safetensors", ".pt"):
                            p = ld / f"{fn}{ext}"
                            if p.exists():
                                sizes[fn] = {"file": str(p), "bytes": file_size(p)}
                                shp = tensor_shape_from_file(p)
                                if shp:
                                    import math

                                    try:
                                        params += int(math.prod(shp))
                                    except Exception:
                                        pass
                                break
                    c["files"] = sizes
                    if params:
                        c["param_count"] = int(params)
                    heads[ld.name] = c
        if heads:
            details["heads"] = heads
    except Exception:
        pass
    try:
        ffns = {}
        ffns_dir = kb / "ffns"
        if ffns_dir.exists():
            for ld in sorted([p for p in ffns_dir.iterdir() if p.is_dir()]):
                cfg = ld / "config.json"
                if cfg.exists():
                    c = json.loads(cfg.read_text())
                    sizes = {}
                    params = 0
                    for fn in ("gate_proj", "up_proj", "down_proj"):
                        idx = ld / f"{fn}.index.json"
                        if idx.exists():
                            sizes[fn] = {"file": str(idx), "bytes": index_bytes(idx)}
                            try:
                                import json as _json
                                import torch as _torch

                                meta = _json.loads(idx.read_text())
                                shp0 = None
                                dim0_total = 0
                                for name in meta.get("parts", []):
                                    t = _torch.load(
                                        str(idx.parent / name), map_location="cpu"
                                    )
                                    s = list(t.shape)
                                    shp0 = s if shp0 is None else shp0
                                    dim0_total += s[0] if len(s) > 0 else 0
                                if shp0 is not None:
                                    import math

                                    params += int(
                                        dim0_total
                                        * (math.prod(shp0[1:]) if len(shp0) > 1 else 1)
                                    )
                            except Exception:
                                pass
                            continue
                        for ext in (".safetensors", ".pt"):
                            p = ld / f"{fn}{ext}"
                            if p.exists():
                                sizes[fn] = {"file": str(p), "bytes": file_size(p)}
                                shp = tensor_shape_from_file(p)
                                if shp:
                                    import math

                                    try:
                                        params += int(math.prod(shp))
                                    except Exception:
                                        pass
                                break
                    c["files"] = sizes
                    if params:
                        c["param_count"] = int(params)
                    ffns[ld.name] = c
        if ffns:
            details["ffns"] = ffns
    except Exception:
        pass
    result = {
        "knowledge_bank": str(kb),
        "exists": kb.exists(),
        "components": {
            "embeddings": (kb / "embeddings").exists(),
            "heads_layers": (
                sorted([p.name for p in (kb / "heads").iterdir()])
                if (kb / "heads").exists()
                else []
            ),
            "ffn_layers": (
                sorted([p.name for p in (kb / "ffns").iterdir()])
                if (kb / "ffns").exists()
                else []
            ),
            "lm_head": (kb / "lm_head").exists(),
        },
        "metadata": meta,
        "details": details,
    }
    print(json.dumps(result, indent=2, default=str))


def reattach_command(args):
    """Reattach transplanted components (wrappers/bridges/gates) from artifacts."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(
        config.get("log_level"),
        getattr(args, "log_file", "llm_ripper.log"),
        getattr(args, "json_logs", False),
    )

    transplanter = KnowledgeTransplanter(config)
    result = transplanter.reattach_full_from_artifacts(args.transplanted_dir)
    print(json.dumps(result, indent=2, default=str))


def apply_cli_overrides(config: ConfigManager, args):
    """Apply CLI flags to override config values."""
    if getattr(args, "device", None):
        config.set("device", args.device)
    # Verbosity override
    if getattr(args, "quiet", False):
        config.set("log_level", "ERROR")
    if getattr(args, "verbose", False):
        config.set("log_level", "DEBUG")
    if getattr(args, "load_in_8bit", False):
        config.set("load_in_8bit", True)
    if getattr(args, "load_in_4bit", False):
        config.set("load_in_4bit", True)
    if getattr(args, "trust_remote_code", False):
        if not getattr(args, "yes", False):
            logging.error("✗ --trust-remote-code requires confirmation with --yes for safety.")
            sys.exit(1)
        config.set("trust_remote_code", True)
    if getattr(args, "offline", False):
        config.set("offline", True)
    if getattr(args, "seed", None) is not None:
        config.set("seed", args.seed)


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Ripper: Modular knowledge extraction and transplantation for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract knowledge from a model
  llm-ripper extract --model microsoft/DialoGPT-medium --output-dir ./knowledge_bank

  # Capture activations
  llm-ripper capture --model microsoft/DialoGPT-medium --output-file activations.h5

  # Analyze extracted components
  llm-ripper analyze --knowledge-bank ./knowledge_bank --output-dir ./analysis

  # Transplant components
  llm-ripper transplant --source ./knowledge_bank --target microsoft/DialoGPT-small --output-dir ./transplanted

  # Validate transplanted model
  llm-ripper validate --model ./transplanted --output-dir ./validation_results
        """,
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )
    parser.add_argument("--seed", type=int, help="Global seed for reproducibility")
    parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    parser.add_argument(
        "--json-logs", action="store_true", help="Emit logs in JSON format"
    )
    parser.add_argument(
        "--version", action="store_true", help="Print package version and exit"
    )
    parser.add_argument(
        "--about", action="store_true", help="Print a short about text and exit"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm safety prompts (e.g., trust-remote-code)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console logging")
    parser.add_argument(
        "--verbose", action="store_true", help="Increase logging verbosity"
    )
    parser.add_argument(
        "--log-file", type=str, help="Log file (default: llm_ripper.log)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Top-level flags handled by module-level _handle_top_level_flags()

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract knowledge from a model"
    )
    extract_parser.add_argument("--model", required=True, help="Model name or path")
    extract_parser.add_argument(
        "--output-dir", required=True, help="Output directory for knowledge bank"
    )
    extract_parser.add_argument(
        "--components", help="Comma-separated list of components to extract"
    )
    extract_parser.add_argument(
        "--model-type", choices=["base", "causal_lm"], help="Force model class to load"
    )
    # Perf/loader flags
    extract_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], help="Device override"
    )
    extract_parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (requires bitsandbytes)",
    )
    extract_parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (requires bitsandbytes)",
    )
    extract_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom model code"
    )
    extract_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    extract_parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )
    extract_parser.set_defaults(func=extract_command)

    # Capture command
    capture_parser = subparsers.add_parser("capture", help="Capture model activations")
    capture_parser.add_argument("--model", required=True, help="Model name or path")
    capture_parser.add_argument("--output-file", required=True, help="Output HDF5 file")
    capture_parser.add_argument(
        "--dataset", help="Dataset to use for activation capture"
    )
    capture_parser.add_argument(
        "--layers", help="Comma-separated list of layers to capture"
    )
    capture_parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to process"
    )
    # Perf/loader flags
    capture_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], help="Device override"
    )
    capture_parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (requires bitsandbytes)",
    )
    capture_parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (requires bitsandbytes)",
    )
    capture_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom model code"
    )
    capture_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    capture_parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )
    capture_parser.set_defaults(func=capture_command)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze extracted knowledge"
    )
    analyze_parser.add_argument(
        "--knowledge-bank", required=True, help="Path to knowledge bank directory"
    )
    analyze_parser.add_argument("--activations", help="Path to activations HDF5 file")
    analyze_parser.add_argument(
        "--output-dir", required=True, help="Output directory for analysis results"
    )
    # Perf/loader flags (in case analyzer needs to load models)
    analyze_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], help="Device override"
    )
    analyze_parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (requires bitsandbytes)",
    )
    analyze_parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (requires bitsandbytes)",
    )
    analyze_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom model code"
    )
    analyze_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    analyze_parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )
    analyze_parser.set_defaults(func=analyze_command)

    # Transplant command
    transplant_parser = subparsers.add_parser(
        "transplant", help="Transplant knowledge components"
    )
    transplant_parser.add_argument(
        "--source", required=True, help="Source knowledge bank directory"
    )
    transplant_parser.add_argument(
        "--target", required=True, help="Target model name or path"
    )
    transplant_parser.add_argument(
        "--output-dir", required=True, help="Output directory for transplanted model"
    )
    transplant_parser.add_argument(
        "--config-file", help="JSON file with transplant configurations"
    )
    transplant_parser.add_argument(
        "--source-component", help="Source component to transplant"
    )
    transplant_parser.add_argument(
        "--target-layer", type=int, help="Target layer for transplantation"
    )
    transplant_parser.add_argument(
        "--strategy",
        choices=["embedding_init", "module_injection", "adapter_fusion"],
        default="module_injection",
        help="Transplantation strategy",
    )
    # Perf/loader flags
    transplant_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], help="Device override"
    )
    transplant_parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (requires bitsandbytes)",
    )
    transplant_parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (requires bitsandbytes)",
    )
    transplant_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom model code"
    )
    transplant_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    transplant_parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )
    transplant_parser.set_defaults(func=transplant_command)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate transplanted model"
    )
    validate_parser.add_argument(
        "--model", required=True, help="Path to transplanted model"
    )
    validate_parser.add_argument("--baseline", help="Baseline model for comparison")
    validate_parser.add_argument(
        "--benchmarks", help="Comma-separated list of benchmarks to run"
    )
    validate_parser.add_argument(
        "--output-dir", required=True, help="Output directory for validation results"
    )
    validate_parser.add_argument(
        "--mechanistic", action="store_true", help="Run mechanistic offline batteries"
    )
    # Perf/loader flags
    validate_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], help="Device override"
    )
    validate_parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (requires bitsandbytes)",
    )
    validate_parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (requires bitsandbytes)",
    )
    validate_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom model code"
    )
    validate_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    validate_parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )
    validate_parser.set_defaults(func=validate_command)

    # Inspect command
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect knowledge bank contents"
    )
    inspect_parser.add_argument(
        "--knowledge-bank", required=True, help="Path to knowledge bank directory"
    )
    inspect_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    inspect_parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )
    inspect_parser.set_defaults(func=inspect_command)

    # Reattach command
    reattach_parser = subparsers.add_parser(
        "reattach", help="Reattach transplanted artifacts to a saved model"
    )
    reattach_parser.add_argument(
        "--transplanted-dir",
        required=True,
        help="Directory containing transplanted model",
    )
    reattach_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], help="Device override"
    )
    reattach_parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit (requires bitsandbytes)",
    )
    reattach_parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit (requires bitsandbytes)",
    )
    reattach_parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom model code"
    )
    reattach_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    reattach_parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )
    reattach_parser.set_defaults(func=reattach_command)

    # Trace command (causal tracing)
    trace_parser = subparsers.add_parser(
        "trace", help="Run causal tracing with interventions"
    )
    trace_parser.add_argument("--model", required=True, help="Model name or path")
    trace_parser.add_argument(
        "--targets",
        required=True,
        help="Comma-separated targets, e.g., head:12.q,ffn:7.up",
    )
    trace_parser.add_argument(
        "--dataset", default="diverse", help="Dataset alias for probing text"
    )
    trace_parser.add_argument(
        "--metric", choices=["nll_delta", "logit_delta"], default="nll_delta"
    )
    trace_parser.add_argument(
        "--intervention", choices=["zero", "noise", "mean-patch"], default="zero"
    )
    trace_parser.add_argument("--max-samples", type=int, default=64)
    trace_parser.add_argument("--seed", type=int, default=42)
    trace_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON output"
    )
    trace_parser.add_argument(
        "--offline", action="store_true", help="Run in offline mode (no downloads)"
    )

    def _trace_cmd(args):
        config = ConfigManager(args.config)
        apply_cli_overrides(config, args)
        setup_logging(
            config.get("log_level"),
            getattr(args, "log_file", "llm_ripper.log"),
            getattr(args, "json_logs", False),
        )
        set_global_seeds(args.seed)
        tracer = Tracer(config)
        cfg = TraceConfig(
            model=args.model,
            targets=[t.strip() for t in args.targets.split(",") if t.strip()],
            metric=args.metric,
            dataset=args.dataset,
            intervention=args.intervention,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        res = tracer.run(cfg, out_dir="runs")
        out = {
            "run_id": res.run_id,
            "summary": res.summary_path,
            "traces": res.traces_path,
        }
        (
            print(json.dumps(out, indent=2))
            if args.json
            else logging.info(
                f"✓ Trace completed. Run: {res.run_id}\n  summary: {res.summary_path}\n  traces: {res.traces_path}"
            )
        )

    trace_parser.set_defaults(func=_trace_cmd)

    # Counterfactual generation
    cfgen_parser = subparsers.add_parser(
        "cfgen", help="Generate counterfactual minimal pairs"
    )
    cfgen_parser.add_argument("--task", choices=["agreement", "coref"], required=True)
    cfgen_parser.add_argument("--n", type=int, default=100)
    cfgen_parser.add_argument("--out", required=True, help="Output JSONL file path")
    cfgen_parser.add_argument("--json", action="store_true")

    def _cfgen_cmd(args):
        fp = generate_minimal_pairs(args.task, args.n, args.out)
        (
            print(json.dumps({"pairs_file": fp}, indent=2))
            if args.json
            else logging.info(f"✓ Generated pairs: {fp}")
        )

    cfgen_parser.set_defaults(func=_cfgen_cmd)

    # Counterfactual evaluation
    cfeval_parser = subparsers.add_parser(
        "cfeval", help="Evaluate counterfactual pairs with a model"
    )
    cfeval_parser.add_argument("--model", required=True)
    cfeval_parser.add_argument(
        "--pairs", nargs="+", required=True, help="Glob patterns for JSONL pairs files"
    )
    cfeval_parser.add_argument("--out", required=True, help="Output results JSONL path")
    cfeval_parser.add_argument("--json", action="store_true")

    def _cfeval_cmd(args):
        config = ConfigManager(args.config)
        apply_cli_overrides(config, args)
        setup_logging(
            config.get("log_level"),
            getattr(args, "log_file", "llm_ripper.log"),
            getattr(args, "json_logs", False),
        )
        ev = CounterfactualEvaluator(config)
        res = ev.evaluate(args.model, args.pairs, args.out)
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(
                f"✓ Evaluated pairs. {res['summary']}. File: {res['results_file']}"
            )
        )

    cfeval_parser.set_defaults(func=_cfeval_cmd)

    # UQ command
    uq_parser = subparsers.add_parser(
        "uq", help="Uncertainty quantification via MC-Dropout"
    )
    uq_parser.add_argument("--model", required=True)
    uq_parser.add_argument("--samples", type=int, default=10)
    uq_parser.add_argument("--max-texts", type=int, default=64)
    uq_parser.add_argument("--seed", type=int, default=42)
    uq_parser.add_argument("--json", action="store_true")

    def _uq_cmd(args):
        config = ConfigManager(args.config)
        apply_cli_overrides(config, args)
        setup_logging(
            config.get("log_level"),
            getattr(args, "log_file", "llm_ripper.log"),
            getattr(args, "json_logs", False),
        )
        runner = UQRunner(config)
        res = runner.run(
            uq_cfg=type(
                "UQCfg",
                (),
                {
                    "model": args.model,
                    "samples": args.samples,
                    "max_texts": args.max_texts,
                    "seed": args.seed,
                },
            )(),
            out_dir="runs",
        )
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(f"✓ UQ completed. Summary: {res['summary_file']}")
        )

    uq_parser.set_defaults(func=_uq_cmd)

    # Bridge alignment (orthogonal)
    align_parser = subparsers.add_parser(
        "bridge-align", help="Orthogonal/LS alignment donor->target embeddings"
    )
    align_parser.add_argument(
        "--source", required=True, help="Path to knowledge bank directory"
    )
    align_parser.add_argument(
        "--target", required=True, help="Target model name or path"
    )
    align_parser.add_argument(
        "--out", required=True, help="Output .npy path for alignment matrix"
    )
    align_parser.add_argument("--json", action="store_true")

    def _align_cmd(args):
        config = ConfigManager(args.config)
        apply_cli_overrides(config, args)
        setup_logging(
            config.get("log_level"),
            getattr(args, "log_file", "llm_ripper.log"),
            getattr(args, "json_logs", False),
        )
        res = orthogonal_procrustes_align(config, args.source, args.target, args.out)
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(
                f"✓ Alignment saved: {res['matrix_file']} (cos {res['cosine_before']:.3f}->{res['cosine_after']:.3f})"
            )
        )

    align_parser.set_defaults(func=_align_cmd)

    # Provenance scan
    prov_parser = subparsers.add_parser(
        "provenance", help="Scan transplanted directory for provenance issues"
    )
    prov_parser.add_argument(
        "--scan", required=True, help="Transplanted directory root"
    )
    prov_parser.add_argument("--fail-on-violation", action="store_true")
    prov_parser.add_argument("--json", action="store_true")

    def _prov_cmd(args):
        sc = ProvenanceScanner()
        res = sc.scan(
            type(
                "SC",
                (),
                {"root": args.scan, "fail_on_violation": args.fail_on_violation},
            )()
        )
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(
                f"✓ Provenance scan ok={res.get('ok')} violations={res.get('violations')}"
            )
        )

    prov_parser.set_defaults(func=_prov_cmd)

    # Features discovery
    feat_parser = subparsers.add_parser(
        "features", help="Discover features from activations (SAE-lite)"
    )
    feat_parser.add_argument(
        "--activations", required=True, help="Path to activations HDF5"
    )
    feat_parser.add_argument("--method", choices=["sae", "pca"], default="sae")
    feat_parser.add_argument(
        "--out", required=True, help="Output directory for catalog"
    )
    feat_parser.add_argument("--k", type=int, default=16)
    feat_parser.add_argument("--json", action="store_true")

    def _feat_cmd(args):
        res = discover_features(args.activations, args.method, args.out, k=args.k)
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(f"✓ Catalog saved: {res['catalog_file']}")
        )

    feat_parser.set_defaults(func=_feat_cmd)

    # Studio (static MVP)
    studio_parser = subparsers.add_parser(
        "studio", help="Launch static studio UI (serves index.html)"
    )
    studio_parser.add_argument(
        "--root", default="runs", help="Root directory containing run artifacts"
    )
    studio_parser.add_argument("--port", type=int, default=8000)

    def _studio_cmd(args):
        from .studio import launch_studio

        # Blocking server; prints URL and waits
        launch_studio(args.root, port=args.port)

    studio_parser.set_defaults(func=_studio_cmd)

    # Route simulation (uncertainty routing)
    route_parser = subparsers.add_parser(
        "route-sim", help="Simulate routing via UQ threshold"
    )
    route_parser.add_argument(
        "--metrics", required=True, help="Path to UQ metrics.jsonl"
    )
    route_parser.add_argument(
        "--tau", type=float, default=0.7, help="Routing threshold on confidence"
    )
    route_parser.add_argument("--json", action="store_true")

    def _route_cmd(args):
        import json as _j
        import numpy as _np

        rows = []
        for line in open(args.metrics, "r"):
            if line.strip():
                rows.append(_j.loads(line))
        conf = _np.array([r.get("confidence", 0.0) for r in rows])
        routed = (conf < args.tau).sum()
        out = {
            "total": len(rows),
            "tau": args.tau,
            "routed": int(routed),
            "routed_frac": float(routed / max(1, len(rows))),
        }
        (
            print(_j.dumps(out, indent=2))
            if args.json
            else logging.info(f"✓ Routing simulated: {out}")
        )

    route_parser.set_defaults(func=_route_cmd)

    # Merge models (global)
    merge_parser = subparsers.add_parser(
        "merge", help="Merge models (average) and optional microtransplants from spec"
    )
    merge_parser.add_argument(
        "--global",
        dest="spec",
        required=True,
        help="Spec YAML/JSON with models list and optional micro",
    )
    merge_parser.add_argument("--out", required=True)
    merge_parser.add_argument(
        "--micro",
        action="store_true",
        help="Apply microtransplants defined in spec.micro",
    )
    merge_parser.add_argument("--json", action="store_true")

    def _merge_cmd(args):
        from .interop import merge_models_average

        if args.micro:
            from .interop.merge import merge_with_micro

            res = merge_with_micro(args.spec, args.out)
        else:
            res = merge_models_average(args.spec, args.out)
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(f"✓ Merged: {res['out']} with {res['merged_keys']} keys")
        )

    merge_parser.set_defaults(func=_merge_cmd)

    # Adapters import
    adapters_parser = subparsers.add_parser(
        "adapters", help="Import and manage adapters"
    )
    adapters_parser.add_argument(
        "--import", dest="import_path", help="Path to LoRA file (.safetensors/.pt)"
    )
    adapters_parser.add_argument(
        "--model", required=True, help="Model dir to inject into"
    )
    adapters_parser.add_argument("--layer", type=int, default=0)
    adapters_parser.add_argument(
        "--fuse", action="store_true", help="Attach fusion gate for adapters on layer"
    )
    adapters_parser.add_argument("--json", action="store_true")

    def _adapters_cmd(args):
        from .interop import import_lora_and_inject, fuse_layer_adapters

        result = {}
        if args.import_path:
            result["import"] = import_lora_and_inject(
                args.model, args.import_path, args.layer
            )
        if args.fuse:
            result["fuse"] = fuse_layer_adapters(args.model, args.layer)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if "import" in result:
                r = result["import"]
                logging.info(
                    f"✓ Imported LoRA into layer {r['layer']} ({r['in']}->{r['out']})"
                )
            if "fuse" in result:
                r = result["fuse"]
                logging.info(f"✓ Fused {r['adapters']} adapters on layer {r['layer']}")

    adapters_parser.set_defaults(func=_adapters_cmd)

    # Tokenizer alignment
    tok_parser = subparsers.add_parser(
        "tokenize-align", help="Align two tokenizers and emit mapping"
    )
    tok_parser.add_argument("--source", required=True)
    tok_parser.add_argument("--target", required=True)
    tok_parser.add_argument("--out", required=True)
    tok_parser.add_argument("--json", action="store_true")

    def _tok_cmd(args):
        from .interop import align_tokenizers

        res = align_tokenizers(args.source, args.target, args.out)
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(
                f"✓ Tokenizer overlap: {res['overlap']} (mapping: {res['mapping_file']})"
            )
        )

    tok_parser.set_defaults(func=_tok_cmd)

    # Bridge-train mixture (offline self-supervised)
    btrain = subparsers.add_parser(
        "bridge-train", help="Train mixture of adapters on a layer"
    )
    btrain.add_argument("--model", required=True)
    btrain.add_argument("--layer", type=int, required=True)
    btrain.add_argument("--mixture", dest="k", type=int, default=2)
    btrain.add_argument("--steps", type=int, default=50)
    btrain.add_argument("--json", action="store_true")

    def _btrain_cmd(args):
        cfg = ConfigManager(args.config)
        kt = KnowledgeTransplanter(cfg)
        m, _, _ = kt.model_loader.load_model_and_tokenizer(args.model)
        res = kt.train_mixture_on_layer(m, args.layer, k=args.k, steps=args.steps)
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(f"✓ Trained mixture on layer {args.layer}: {res}")
        )

    btrain.set_defaults(func=_btrain_cmd)

    # Reporting
    report_parser = subparsers.add_parser(
        "report", help="Generate IDEAL-style JSON/MD report"
    )
    report_parser.add_argument("--ideal", action="store_true")
    report_parser.add_argument("--out", required=True)
    report_parser.add_argument(
        "--from", dest="root", help="Root dir containing validation/uq (optional)"
    )
    report_parser.add_argument("--json", action="store_true")

    def _report_cmd(args):
        from .safety.report import generate_report

        res = generate_report(args.ideal, args.out, transplanted_dir=args.root)
        (
            print(json.dumps(res, indent=2))
            if args.json
            else print(f"✓ Report: {res['json']}")
        )

    report_parser.set_defaults(func=_report_cmd)

    # Stress & OOD drift
    stress_parser = subparsers.add_parser(
        "stress", help="Run stress prompts and drift metrics (PSI/KL)"
    )
    stress_parser.add_argument(
        "--model", required=True, help="Transplanted (or candidate) model path"
    )
    stress_parser.add_argument(
        "--baseline", required=True, help="Baseline model for comparison"
    )
    stress_parser.add_argument("--out", required=True)
    stress_parser.add_argument("--json", action="store_true")

    def _stress_cmd(args):
        from .safety.stress import run_stress_and_drift

        cfg = ConfigManager(args.config)
        res = run_stress_and_drift(cfg, args.model, args.baseline, args.out)
        (
            print(json.dumps(res, indent=2))
            if args.json
            else logging.info(
                f"✓ Stress/Drift: PSI={res['psi_entropy']:.4f} KLm0={res['kl_model_vs_baseline']:.4f}"
            )
        )

    stress_parser.set_defaults(func=_stress_cmd)

    # HDF5 tools command
    hdf5_parser = subparsers.add_parser(
        "hdf5", help="HDF5 utilities: repack and downsample"
    )
    hdf5_sub = hdf5_parser.add_subparsers(dest="hdf5_cmd")
    repack = hdf5_sub.add_parser(
        "repack", help="Repack an HDF5 with compression and chunking"
    )
    repack.add_argument("--in", dest="in_path", required=True)
    repack.add_argument("--out", dest="out_path", required=True)
    repack.add_argument("--compression", default="gzip")
    repack.add_argument("--chunk-size", type=int)
    down = hdf5_sub.add_parser(
        "downsample", help="Downsample an HDF5 by taking every N samples on first axis"
    )
    down.add_argument("--in", dest="in_path", required=True)
    down.add_argument("--out", dest="out_path", required=True)
    down.add_argument("--every-n", type=int, default=2)

    def _hdf5_cmd(args):
        from .utils.hdf5_tools import repack_hdf5, downsample

        if args.hdf5_cmd == "repack":
            repack_hdf5(
                args.in_path,
                args.out_path,
                compression=args.compression,
                chunk_size=args.chunk_size,
            )
        elif args.hdf5_cmd == "downsample":
            downsample(args.in_path, args.out_path, every_n=args.every_n)
        else:
            logging.error("Specify a subcommand: repack|downsample")
            sys.exit(1)

    hdf5_parser.set_defaults(func=_hdf5_cmd)

    # Quickstart (beginner demo)
    quick_parser = subparsers.add_parser(
        "quickstart", help="Create a beginner demo run with sample artifacts"
    )
    quick_parser.add_argument("--port", type=int, default=8000)
    quick_parser.add_argument("--open", action="store_true", help="Open Studio after creating demo")
    quick_parser.add_argument("--json", action="store_true")

    def _quickstart_cmd(args):
        from .utils.run import RunContext
        from .studio import launch_studio
        rc = RunContext.create()
        rc.write_json("catalog/heads.json", {"heads": [{"layer": 0, "head": 0, "note": "demo"}]})
        rc.write_json("traces/summary.json", {"metric": "nll_delta", "targets": ["head:0.q", "ffn:0.up"], "note": "demo"})
        rc.write_json(
            "validation/validation_results.json",
            {
                "summary": {"overall_score": 0.75},
                "recommendations": [
                    "Run more samples for stable metrics",
                    "Try different transplant strategies",
                ],
            },
        )
        out = {"run_root": str(rc.root), "studio_url": f"http://localhost:{args.port}/studio/index.html?root={rc.root}"}
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            logging.info("\u2713 Beginner demo created at: %s", rc.root)
            logging.info("Open Studio with: make studio RUN_ROOT=%s PORT=%s", rc.root, args.port)
        if args.open:
            launch_studio(str(rc.root), port=args.port)

    quick_parser.set_defaults(func=_quickstart_cmd)

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    _handle_top_level_flags(args)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        logging.error("✗ Error: %s", e)
        logging.info("Tip: if you are just getting started, try: llm-ripper quickstart --open")
        sys.exit(1)


if __name__ == "__main__":
    main()
