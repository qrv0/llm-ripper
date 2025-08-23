"""
Command-line interface for LLM Ripper.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from .utils.config import ConfigManager
from .core import KnowledgeExtractor, ActivationCapture, KnowledgeAnalyzer, KnowledgeTransplanter, ValidationSuite
from .core.transplant import TransplantConfig


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('llm_ripper.log')
        ]
    )


def extract_command(args):
    """Extract knowledge from a model."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    config.validate_config()
    config.create_directories()
    
    setup_logging(config.get("log_level"))
    
    extractor = KnowledgeExtractor(config)
    
    components = args.components.split(",") if args.components else None
    
    result = extractor.extract_model_components(
        model_name=args.model,
        output_dir=args.output_dir,
        components=components
    )
    
    print(f"✓ Extraction completed successfully!")
    print(f"  Source model: {result['source_model']}")
    print(f"  Components extracted: {list(result['extracted_components'].keys())}")
    print(f"  Output directory: {args.output_dir}")


def capture_command(args):
    """Capture activations from a model."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(config.get("log_level"))
    
    from datasets import load_dataset
    
    # Load corpus dataset
    if args.dataset:
        if args.dataset == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        elif args.dataset == "openwebtext":
            dataset = load_dataset("openwebtext", split="train")
        else:
            # Assume it's a local file or HuggingFace dataset
            dataset = load_dataset(args.dataset, split="train")
    else:
        # Create a simple dataset for testing
        from datasets import Dataset
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models have revolutionized AI applications."
        ]
        dataset = Dataset.from_dict({"text": texts})
    
    capture = ActivationCapture(config)
    
    result = capture.capture_model_activations(
        model_name=args.model,
        corpus_dataset=dataset,
        output_file=args.output_file,
        layers_to_capture=args.layers.split(",") if args.layers else None,
        max_samples=args.max_samples
    )
    
    print(f"✓ Activation capture completed successfully!")
    print(f"  Model: {result['model_name']}")
    print(f"  Samples processed: {result['num_samples']}")
    print(f"  Output file: {result['output_file']}")


def analyze_command(args):
    """Analyze extracted knowledge components."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(config.get("log_level"))
    
    analyzer = KnowledgeAnalyzer(config)
    
    result = analyzer.analyze_knowledge_bank(
        knowledge_bank_dir=args.knowledge_bank,
        activations_file=args.activations,
        output_dir=args.output_dir
    )
    
    print(f"✓ Analysis completed successfully!")
    print(f"  Source model: {result['source_model']}")
    print(f"  Components analyzed: {list(result['component_analysis'].keys())}")
    print(f"  Head catalog entries: {len(result.get('head_catalog', []))}")
    print(f"  Output directory: {args.output_dir}")


def transplant_command(args):
    """Transplant knowledge components to a target model."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(config.get("log_level"))
    
    transplanter = KnowledgeTransplanter(config)
    
    # Parse transplant configurations
    transplant_configs = []
    
    if args.config_file:
        import json
        with open(args.config_file, 'r') as f:
            configs_data = json.load(f)
        
        for config_data in configs_data:
            transplant_configs.append(TransplantConfig(**config_data))
    else:
        # Create a simple configuration from command line args
        transplant_configs.append(TransplantConfig(
            source_component=args.source_component,
            target_layer=args.target_layer,
            bridge_hidden_size=config.get("adapter_hidden_size", 64),
            freeze_donor=config.get("freeze_donor_weights", True),
            freeze_target=False,
            strategy=args.strategy
        ))
    
    result = transplanter.transplant_knowledge(
        source_knowledge_bank=args.source,
        target_model_name=args.target,
        transplant_configs=transplant_configs,
        output_dir=args.output_dir
    )
    
    print(f"✓ Transplantation completed successfully!")
    print(f"  Source: {result['source_model']}")
    print(f"  Target: {result['target_model']}")
    print(f"  Components transplanted: {len(result['transplanted_components'])}")
    print(f"  Output directory: {args.output_dir}")


def validate_command(args):
    """Validate a transplanted model."""
    config = ConfigManager(args.config)
    apply_cli_overrides(config, args)
    setup_logging(config.get("log_level"))
    
    validator = ValidationSuite(config)
    
    benchmarks = args.benchmarks.split(",") if args.benchmarks else None
    
    result = validator.validate_transplanted_model(
        transplanted_model_path=args.model,
        baseline_model_name=args.baseline,
        benchmarks=benchmarks,
        output_dir=args.output_dir
    )
    
    print(f"✓ Validation completed successfully!")
    print(f"  Model: {result['model_path']}")
    print(f"  Overall score: {result['summary']['overall_score']:.3f}")
    
    if result['summary']['recommendations']:
        print("  Recommendations:")
        for rec in result['summary']['recommendations']:
            print(f"    - {rec}")
    
    print(f"  Output directory: {args.output_dir}")


def apply_cli_overrides(config: ConfigManager, args):
    """Apply CLI flags to override config values."""
    if getattr(args, "device", None):
        config.set("device", args.device)
    if getattr(args, "load_in_8bit", False):
        config.set("load_in_8bit", True)
    if getattr(args, "load_in_4bit", False):
        config.set("load_in_4bit", True)
    if getattr(args, "trust_remote_code", False):
        config.set("trust_remote_code", True)


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
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract knowledge from a model")
    extract_parser.add_argument("--model", required=True, help="Model name or path")
    extract_parser.add_argument("--output-dir", required=True, help="Output directory for knowledge bank")
    extract_parser.add_argument("--components", help="Comma-separated list of components to extract")
    # Perf/loader flags
    extract_parser.add_argument("--device", choices=["auto","cuda","cpu","mps"], help="Device override")
    extract_parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    extract_parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    extract_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code")
    extract_parser.set_defaults(func=extract_command)
    
    # Capture command
    capture_parser = subparsers.add_parser("capture", help="Capture model activations")
    capture_parser.add_argument("--model", required=True, help="Model name or path")
    capture_parser.add_argument("--output-file", required=True, help="Output HDF5 file")
    capture_parser.add_argument("--dataset", help="Dataset to use for activation capture")
    capture_parser.add_argument("--layers", help="Comma-separated list of layers to capture")
    capture_parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    # Perf/loader flags
    capture_parser.add_argument("--device", choices=["auto","cuda","cpu","mps"], help="Device override")
    capture_parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    capture_parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    capture_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code")
    capture_parser.set_defaults(func=capture_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze extracted knowledge")
    analyze_parser.add_argument("--knowledge-bank", required=True, help="Path to knowledge bank directory")
    analyze_parser.add_argument("--activations", help="Path to activations HDF5 file")
    analyze_parser.add_argument("--output-dir", required=True, help="Output directory for analysis results")
    # Perf/loader flags (in case analyzer needs to load models)
    analyze_parser.add_argument("--device", choices=["auto","cuda","cpu","mps"], help="Device override")
    analyze_parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    analyze_parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    analyze_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code")
    analyze_parser.set_defaults(func=analyze_command)
    
    # Transplant command
    transplant_parser = subparsers.add_parser("transplant", help="Transplant knowledge components")
    transplant_parser.add_argument("--source", required=True, help="Source knowledge bank directory")
    transplant_parser.add_argument("--target", required=True, help="Target model name or path")
    transplant_parser.add_argument("--output-dir", required=True, help="Output directory for transplanted model")
    transplant_parser.add_argument("--config-file", help="JSON file with transplant configurations")
    transplant_parser.add_argument("--source-component", help="Source component to transplant")
    transplant_parser.add_argument("--target-layer", type=int, help="Target layer for transplantation")
    transplant_parser.add_argument("--strategy", choices=["embedding_init", "module_injection", "adapter_fusion"], 
                                 default="module_injection", help="Transplantation strategy")
    # Perf/loader flags
    transplant_parser.add_argument("--device", choices=["auto","cuda","cpu","mps"], help="Device override")
    transplant_parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    transplant_parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    transplant_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code")
    transplant_parser.set_defaults(func=transplant_command)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate transplanted model")
    validate_parser.add_argument("--model", required=True, help="Path to transplanted model")
    validate_parser.add_argument("--baseline", help="Baseline model for comparison")
    validate_parser.add_argument("--benchmarks", help="Comma-separated list of benchmarks to run")
    validate_parser.add_argument("--output-dir", required=True, help="Output directory for validation results")
    # Perf/loader flags
    validate_parser.add_argument("--device", choices=["auto","cuda","cpu","mps"], help="Device override")
    validate_parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    validate_parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    validate_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code")
    validate_parser.set_defaults(func=validate_command)
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()