# LLM Ripper

<p align="center">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="status: alpha" />
  <img src="https://img.shields.io/github/license/qrv0/LLM-Ripper" alt="license: Apache-2.0" />
  <img src="https://img.shields.io/badge/python-3.8+-blue" alt="python 3.8+" />
</p>

**Modular surgery for Transformer LMs — extract, analyze, and transplant knowledge between models.**


---

## Why LLM Ripper?

Fine‑tuning is not the only way to reuse knowledge. LLM Ripper makes a model’s internals addressable: you can **extract** components (embeddings, attention heads, FFNs, LM head), **analyze** them, and **transplant** what you want into another architecture using bridge adapters — then **validate** the result.

---

## Install

> Python **3.8+** recommended (tested on 3.11). CUDA optional.

```bash
# Clone
git clone <your-fork-or-repo-url>
cd llm-ripper

# (Optional) create a virtualenv
python -m venv .venv && source .venv/bin/activate

# Dependencies + local package
pip install -r requirements.txt
pip install -e .
```

If you’ll load private/remote models, export your **HF token**:

```bash
export HF_TOKEN=hf_xxx
```

Optional performance flags require extra libs (e.g., 8‑bit/4‑bit via bitsandbytes).

---

## Quick Start (CLI)

1. **Set model identifiers** (via env or config):

```bash
export DONOR_MODEL_NAME="your-donor-model"
export TARGET_MODEL_NAME="your-target-model"
```

2. **Extract** static knowledge from the donor model:

```bash
llm-ripper extract \
  --model "$DONOR_MODEL_NAME" \
  --output-dir ./knowledge_bank \
  [--components embeddings,attention_heads,ffn_layers,lm_head] \
  [--device auto|cuda|cpu|mps] [--load-in-8bit] [--load-in-4bit] [--trust-remote-code]
```

3. **Capture** dynamic activations (optional, for richer analysis):

```bash
llm-ripper capture \
  --model "$DONOR_MODEL_NAME" \
  --output-file ./activations.h5 \
  [--dataset wikitext] [--layers layer_0,layer_5,...] [--max-samples 1000] \
  [--device auto|cuda|cpu|mps] [--load-in-8bit] [--load-in-4bit] [--trust-remote-code]
```

> By default, capture can register hooks for all layers (see config).

4. **Analyze** the knowledge bank (optionally using activations):

```bash
llm-ripper analyze \
  --knowledge-bank ./knowledge_bank \
  [--activations ./activations.h5] \
  --output-dir ./analysis \
  [--device auto|cuda|cpu|mps] [--load-in-8bit] [--load-in-4bit] [--trust-remote-code]
```

5. **Transplant** components into the target model:

```bash
# Using a config file (recommended)
llm-ripper transplant \
  --source ./knowledge_bank \
  --target "$TARGET_MODEL_NAME" \
  --output-dir ./transplanted \
  --config-file ./examples/transplant_config.json \
  [--device auto|cuda|cpu|mps] [--load-in-8bit] [--load-in-4bit] [--trust-remote-code]

# OR one-off via flags
llm-ripper transplant \
  --source ./knowledge_bank \
  --target "$TARGET_MODEL_NAME" \
  --output-dir ./transplanted \
  --source-component layer_5_attention \
  --target-layer 3 \
  --strategy module_injection
```

Supported strategies today:

* `module_injection` — inject donor module with **bridge** adapter & residual
* `embedding_init` — initialize target embeddings using donor space with bridging

6. **Validate** the transplanted model:

```bash
llm-ripper validate \
  --model ./transplanted \
  --baseline "$TARGET_MODEL_NAME" \
  --output-dir ./validation_results \
  [--benchmarks cola,stsb,mnli] \
  [--device auto|cuda|cpu|mps] [--load-in-8bit] [--load-in-4bit] [--trust-remote-code]
```

---

## Configuration

All knobs live in a JSON file (see `examples/config.json`) or env vars.

Common keys:

```json
{
  "model_cache_dir": "./models",
  "device": "auto",
  "knowledge_bank_dir": "./knowledge_bank",
  "corpus_dir": "./corpus",
  "output_dir": "./output",
  "hdf5_compression": "gzip",
  "batch_size": 8,
  "max_sequence_length": 512,
  "use_safetensors": true,
  "capture_all_layers": true,
  "pca_components": 50,
  "num_epochs": 3,
  "validation_datasets": ["cola", "stsb", "mnli"],
  "validation_batch_size": 16,
  "num_validation_samples": 1000,
  "log_level": "INFO",
  "wandb_project": "llm-ripper",
  "save_checkpoints": true,
  "num_workers": 4,
  "pin_memory": true,
  "mixed_precision": true
}
```

Also specify your models via env or config:

* Env: `DONOR_MODEL_NAME`, `TARGET_MODEL_NAME` (preferred)
* Config: `"donor_model_name"`, `"target_model_name"` (explicit)

> If you use custom model repos/classes, consider `--trust-remote-code`.

---

## What gets produced?

**Knowledge bank** (`./knowledge_bank/`):

```
embeddings/
heads/
  layer_0/  layer_1/  ...
ffns/
  layer_0/  layer_1/  ...
lm_head/
metadata.json
```

Each component includes weights (PT/safetensors) and a small `config.json`.

**Activations** (`activations.h5`):

```
/activations/<layer_name>/ ...
```

**Analysis** (`./analysis/`): metrics, summaries and plots.

**Transplanted model** (`./transplanted/`): target checkpoint with adapters/bridges.

**Validation results** (`./validation_results/`): task scores + recommendations.

---

## Python API

```python
from llm_ripper.utils import ConfigManager
from llm_ripper.core import (
    KnowledgeExtractor, ActivationCapture,
    KnowledgeAnalyzer, KnowledgeTransplanter, ValidationSuite
)

config = ConfigManager("examples/config.json")

# 1) Extract
extractor = KnowledgeExtractor(config)
extraction = extractor.extract_model_components(
    model_name=config.get("donor_model_name"),
    output_dir=config.get("knowledge_bank_dir")
)

# 2) Capture (optional)
capture = ActivationCapture(config)
capture.capture_model_activations(
    model_name=config.get("donor_model_name"),
    corpus_dataset=...,  # HF datasets.Dataset
    output_file="./activations.h5"
)

# 3) Analyze
analyzer = KnowledgeAnalyzer(config)
analysis = analyzer.analyze_knowledge_bank(
    knowledge_bank_dir=config.get("knowledge_bank_dir"),
    activations_file="./activations.h5",
    output_dir="./analysis"
)

# 4) Transplant
transplanter = KnowledgeTransplanter(config)
transplant = transplanter.transplant_knowledge(
    source_knowledge_bank=config.get("knowledge_bank_dir"),
    target_model_name=config.get("target_model_name"),
    transplant_configs=[
        {"source_component": "layer_5_attention", "target_layer": 3,
         "bridge_hidden_size": 64, "strategy": "module_injection"}
    ],
    output_dir="./transplanted"
)

# 5) Validate
validator = ValidationSuite(config)
report = validator.run_validation(
    transplanted_model_path="./transplanted",
    baseline_model_name=config.get("target_model_name"),
    benchmarks=["cola","stsb","mnli"],
    output_dir="./validation_results"
)
```

---

## Feature Highlights

* **Static knowledge extraction** with safetensors: embeddings, **attention\_heads**, **ffn\_layers**, LM head
* **Dynamic activation capture** to inform analysis (HDF5, compressed)
* **Analysis suite**: head catalogs & interpretability metrics, FFN clustering, embedding coverage, visualizations
* **Transplantation** via **Bridge Networks** (bottleneck MLP + residual) and adapter‑style injection
* **Validation** on standard tasks with summary + recommendations

---

## Examples & Scripts

* `examples/run_extraction_only.py` — minimal extraction flow
* `examples/run_complete_pipeline.py` — end‑to‑end pipeline (PT comments)
* `examples/transplant_config.json` — sample transplant plan
* `examples/config.json` — base configuration

Run locally:

```bash
python examples/run_extraction_only.py
# or
python examples/run_complete_pipeline.py
```

---

## Development

```bash
# Lint/tests
pytest -q
make lint  # if you add your own linters
```

Structure:

```
src/llm_ripper/        # package code
examples/              # configs & runnable scripts
paper/                 # draft write‑ups
tests/                 # unit tests
knowledge_bank/, output/, models/, corpus/  # runtime dirs
```

Contributions welcome — see **CONTRIBUTING.md** and **CODE\_OF\_CONDUCT.md**.

---

## Troubleshooting

* **bitsandbytes not installed** → avoid `--load-in-8bit/--load-in-4bit` or install bnb
* **remote model import** → use `--trust-remote-code` (only for sources you trust)
* **permission / private repos** → set `HF_TOKEN`
* **GPU not used** → set `--device cuda` and verify CUDA/PyTorch build

---

## Security & License

This project ships with **SECURITY.md** and **LICENSE** (Apache‑2.0). Use responsibly and ensure you have rights to any models you manipulate.
