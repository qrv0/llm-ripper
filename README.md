# LLM Ripper

A production-ready framework for modular deconstruction, analysis, and recomposition of knowledge in Transformer-based language models.

## Overview

LLM Ripper implements a comprehensive framework based on cutting-edge research for:
- **Part I**: Architectural dissection and knowledge extraction from donor models
- **Part II**: Analysis and synthesis of extracted knowledge components  
- **Part III**: Recomposition and validation in new architectures

This framework enables researchers and practitioners to systematically extract, analyze, and transplant knowledge components between different language models, opening new possibilities for model composition and knowledge transfer.

## Key Features

- **Modular Knowledge Extraction**: Extract embeddings, attention heads, FFN layers, and LM heads with architecture-sensitive handling
- **Dynamic Activity Capture**: Efficient activation capture using torch.fx for large-scale analysis
- **Comprehensive Analysis**: Semantic coverage, attention interpretability, and conceptual clustering metrics
- **Knowledge Transplantation**: Advanced transplantation using Bridge Networks and parameter-efficient adapters
- **Multi-level Validation**: Intrinsic and extrinsic validation protocols with comprehensive benchmarks

### Basic Usage Examples

**⚠️ Note: All model names below are examples only. Replace with your actual model paths or HuggingFace model identifiers.**

```bash
# Extract knowledge from a donor model (EXAMPLE)
llm-ripper extract --model "model-name-here" --output-dir ./knowledge_bank

# Capture dynamic activations (EXAMPLE)  
llm-ripper capture --model "model-name-here" --output-file ./activations.h5 --dataset wikitext

# Analyze extracted components
llm-ripper analyze --knowledge-bank ./knowledge_bank --activations ./activations.h5 --output-dir ./analysis

# Transplant components to a target model (EXAMPLE)
llm-ripper transplant --source ./knowledge_bank --target "model-name-here" --output-dir ./transplanted

# Validate transplanted model
llm-ripper validate --model ./transplanted --baseline "model-name-here" --output-dir ./validation_results
```

### Configuration

Device and performance flags can be set via environment variables or CLI flags.

Examples:
- DEVICE=auto|cuda|cpu|mps
- LOAD_IN_8BIT=true (requires bitsandbytes)
- LOAD_IN_4BIT=true (requires bitsandbytes)
- TRUST_REMOTE_CODE=true


Set environment variables for your specific models:

```bash
# EXAMPLES - Replace with your actual model identifiers
export DONOR_MODEL_NAME="your-donor-model-name-here"
export TARGET_MODEL_NAME="your-target-model-name-here" 
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"  # Optional for experiment tracking
```

## Framework Architecture

### Part I: Knowledge Extraction
- **Static Knowledge Extraction**: Comprehensive weight extraction with safetensors support
  - Input/output embeddings with weight-tying detection
  - Attention mechanisms (MHA, GQA, MQA) with architecture-sensitive handling  
  - Feed-forward networks with activation function detection
  - Language modeling heads
- **Dynamic Knowledge Capture**: Efficient activation capture using torch.fx
  - Hierarchical HDF5 storage for scalability
  - Configurable layer and component selection
  - Memory-efficient processing for large models

### Part II: Knowledge Analysis  
- **Semantic Coverage Analysis**: Perplexity-based embedding evaluation with downstream tasks
- **Attention Head Interpretability**: 
  - Syntactic Head Score (SHS) for grammatical dependency alignment
  - Factual Head Score (FHS) through ablation studies
  - Functional classification and head catalog generation
- **FFN Conceptual Clustering**:
  - PCA-based dimensionality analysis  
  - K-means clustering with purity metrics
  - Conceptual cluster extraction and visualization

### Part III: Knowledge Recomposition
- **Bridge Networks**: Parameter-efficient adapters for dimensional compatibility
  - Bottleneck architecture with residual connections
  - Configurable hidden dimensions and activation functions
- **Transplantation Strategies**:
  - Embedding initialization with dimensional bridging
  - Module injection with frozen donor weights
  - AdapterFusion for multi-component composition
- **Validation Protocols**:
  - Intrinsic validation: Component-level preservation checks
  - Extrinsic validation: Task-specific performance benchmarks
  - Comparative analysis with baseline models

## Advanced Usage

### Complete Pipeline Example
```python
from llm_ripper import KnowledgeExtractor, KnowledgeAnalyzer, KnowledgeTransplanter

# Initialize with configuration
config = ConfigManager("config.json")
extractor = KnowledgeExtractor(config)
analyzer = KnowledgeAnalyzer(config)  
transplanter = KnowledgeTransplanter(config)

# Extract knowledge (EXAMPLE model names)
extraction_result = extractor.extract_model_components(
    model_name="example-donor-model",
    output_dir="./knowledge_bank"
)

# Analyze components
analysis_result = analyzer.analyze_knowledge_bank(
    knowledge_bank_dir="./knowledge_bank",
    output_dir="./analysis"
)

# Transplant to target model (EXAMPLE)
transplant_result = transplanter.transplant_knowledge(
    source_knowledge_bank="./knowledge_bank",
    target_model_name="example-target-model", 
    transplant_configs=configs,
    output_dir="./transplanted"
)
```

### Custom Transplant Configurations
```json
[
  {
    "source_component": "layer_5_attention",
    "target_layer": 3,
    "bridge_hidden_size": 64,
    "freeze_donor": true,
    "strategy": "module_injection"
  },
  {
    "source_component": "embeddings", 
    "target_layer": 0,
    "bridge_hidden_size": 32,
    "strategy": "embedding_init"
  }
]
```

## Contributing

Maintainer: qrv0 (qorvuscompany@gmail.com)


We welcome contributions! Please see our contributing guidelines for:
- Code style and testing requirements
- Documentation standards
- Feature request and bug report processes

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Disclaimer

**Important**: All model names used in examples and documentation are for illustration purposes only. Users must:
- Replace example model names with their actual model identifiers
- Ensure proper licensing and permissions for any models used
- Verify compatibility with their specific use cases and requirements
- Follow all applicable terms of service for model providers
