# ⚡ Quick Start Guide

Get up and running with LLM Ripper in minutes!

## 🚀 Installation

=== "PyPI (Recommended)"

    ```bash
    pip install llm-ripper
    ```
    
    **✅ That's it! You're ready to go.**

=== "Development Install"

    ```bash
    git clone https://github.com/qrv0/LLM-Ripper.git
    cd LLM-Ripper
    pip install -e .
    ```

=== "With Optional Dependencies"

    ```bash
    # For all features
    pip install llm-ripper[viz,spacy,wandb,nlp]
    
    # Or pick what you need
    pip install llm-ripper[viz]  # Visualization
    pip install llm-ripper[wandb]  # W&B logging
    ```

## ✅ Verify Installation

```bash
# Check CLI is working
llm-ripper --help

# Verify Python import
python -c "import llm_ripper; print('✅ LLM Ripper ready!')"
```

## 🎯 Your First Analysis

Let's analyze a simple model:

### Step 1: Basic Analysis

```python
from llm_ripper import LLMRipper

# Initialize with a small model
ripper = LLMRipper("distilgpt2")

# Extract attention patterns
attention_data = ripper.extract_attention_patterns(
    text="The quick brown fox jumps over the lazy dog",
    layers=[0, 1, 2]  # Analyze first 3 layers
)

print(f"Extracted patterns from {len(attention_data)} layers")
```

### Step 2: Component Analysis

```python
# Analyze component importance
importance = ripper.analyze_importance(
    method="gradient",  # or "integrated_gradient", "attention"
    target_layer=6
)

# Get top components
top_components = importance.get_top_k(k=10)
print(f"Most important components: {top_components}")
```

### Step 3: Safety Report

```python
# Generate safety assessment
safety_report = ripper.generate_safety_report(
    dataset="path/to/eval_dataset.jsonl",  # Optional
    metrics=["perplexity", "bias", "toxicity"]
)

print(f"Safety score: {safety_report.overall_score}")
```

## 🛠️ CLI Usage

LLM Ripper provides powerful command-line tools:

### Basic Analysis

```bash
# Analyze any Hugging Face model
llm-ripper analyze --model gpt2 --output ./results

# With specific layers
llm-ripper analyze --model bert-base-uncased --layers 0,1,2,3 --output ./bert_analysis
```

### Extract Components

```bash
# Extract attention heads and FFN layers  
llm-ripper extract \
    --model microsoft/DialoGPT-medium \
    --components attention,ffn \
    --output ./extracted_components
```

### Knowledge Transfer

```bash
# Transfer knowledge between models
llm-ripper transplant \
    --source-model gpt2 \
    --target-model distilgpt2 \
    --components attention_heads \
    --layers 6-8 \
    --output ./transplanted_model
```

### Validation & Safety

```bash
# Validate model behavior
llm-ripper validate \
    --model ./my_modified_model \
    --baseline gpt2 \
    --dataset validation_set.jsonl \
    --output ./validation_report
```

## 🎨 Studio Interface

Launch the visual interface to explore your results:

```bash
# Start Studio server
llm-ripper studio --port 8000 --results ./analysis_results

# Open http://localhost:8000 in your browser
```

The Studio provides:

- 🔍 **Interactive visualizations** of attention patterns
- 📊 **Component importance graphs** 
- 🛡️ **Safety assessment dashboards**
- 📁 **Artifact browser** for all generated files

## 📖 Example Workflows

### Workflow 1: Model Interpretability

```python
# Complete interpretability analysis
ripper = LLMRipper("gpt2")

# 1. Extract all attention patterns
all_patterns = ripper.extract_all_patterns(
    texts=["Example text 1", "Example text 2", "Example text 3"]
)

# 2. Analyze component roles
roles = ripper.analyze_component_roles()

# 3. Generate interpretability report
report = ripper.generate_interpretability_report(
    patterns=all_patterns,
    roles=roles,
    output_format="html"
)
```

### Workflow 2: Model Optimization

```bash
# Find redundant components
llm-ripper analyze --model large-model --find-redundant --output ./analysis

# Remove identified components  
llm-ripper optimize --model large-model --config ./analysis/optimization.yaml --output ./optimized-model

# Validate optimized model
llm-ripper validate --model ./optimized-model --baseline large-model
```

## 🚨 Important Notes

!!! warning "Trust Remote Code"
    
    When working with models that require `trust_remote_code=True`, LLM Ripper will prompt you explicitly for confirmation. This ensures you're aware of potential security implications.

!!! tip "GPU Acceleration"
    
    For best performance, ensure you have CUDA installed:
    ```bash
    # Check CUDA availability
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```

!!! info "Model Compatibility"
    
    LLM Ripper works with most Transformer architectures:
    - ✅ GPT family (GPT-2, GPT-3, GPT-4, etc.)
    - ✅ BERT family (BERT, RoBERTa, DeBERTa, etc.) 
    - ✅ T5 family (T5, FLAN-T5, etc.)
    - ✅ LLaMA family (LLaMA, Alpaca, Vicuna, etc.)
    - ✅ And many more!

## 🆘 Need Help?

- 📖 **Documentation**: Browse the full docs in the sidebar
- 🐛 **Issues**: [Report bugs on GitHub](https://github.com/qrv0/LLM-Ripper/issues)
- 💬 **Discussions**: [Join the community](https://github.com/qrv0/LLM-Ripper/discussions)
- 📧 **Contact**: [qorvuscompany@gmail.com](mailto:qorvuscompany@gmail.com)

## ➡️ What's Next?

Now that you have LLM Ripper running, explore:

- 👶 **[Beginner's Guide](beginners.md)** - Gentle introduction with examples
- 🏗️ **[Architecture](architecture.md)** - Understand how LLM Ripper works
- 📚 **[User Guides](guides/end_to_end.md)** - Comprehensive tutorials
- 🔧 **[API Reference](api.md)** - Detailed function documentation

---

<div align="center" markdown="1">

**Ready for the next step?**

[Explore the Architecture :material-sitemap:](architecture.md){ .md-button .md-button--primary }
[Browse Examples :material-book-open:](guides/end_to_end.md){ .md-button }

</div>

