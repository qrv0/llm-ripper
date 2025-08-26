# End-to-End Pipeline

This guide walks through an end-to-end run using the new features:

- Extraction (embeddings) → Analysis
- Causal Tracing → Alignment
- Transplant (embeddings init) → Mechanistic Validation (offline)
- UQ + Routing Simulation
- Counterfactuals (generate/evaluate)
- Stress/Drift (PSI/KL)
- Report (JSON/MD/PDF)
- Studio (MVP)

## Prerequisites

- A local model directory or a HuggingFace model id (e.g., `gpt2`).
- Optional: baseline model id/path for drift comparisons.

## One-File Example

Run the full pipeline example script:

```bash
python examples/run_full_pipeline.py --model gpt2 --baseline gpt2
```

Outputs are written to `runs/<stamp>/` in standardized folders.

## Manual CLI Path

```bash
# 1) Extract + Analyze
llm-ripper extract --model gpt2 --output-dir ./knowledge_bank
llm-ripper analyze --knowledge-bank ./knowledge_bank --output-dir ./analysis

# 2) Trace (impact ranking)
llm-ripper trace --model gpt2 --targets head:0.q,ffn:0.up --metric nll_delta --intervention zero --max-samples 64

# 3) Alignment
llm-ripper bridge-align --source ./knowledge_bank --target gpt2 --out ./transplants/W_align

# 4) Transplant (embeddings init)
llm-ripper transplant --source ./knowledge_bank --target gpt2 --output-dir ./transplants --strategy embedding_init --source-component embeddings

# 5) Mechanistic validation (offline)
llm-ripper validate --model ./transplants --mechanistic --output-dir ./validation

# 6) UQ + routing
llm-ripper uq --model ./transplants --samples 10 --max-texts 64
llm-ripper route-sim --metrics runs/<stamp>/uq/metrics.jsonl --tau 0.7

# 7) Counterfactuals
llm-ripper cfgen --task agreement --n 2000 --out ./counterfactuals/pairs.jsonl
llm-ripper cfeval --model ./transplants --pairs ./counterfactuals/pairs.jsonl --out ./counterfactuals/results.jsonl

# 8) Stress & Drift
llm-ripper stress --model ./transplants --baseline gpt2 --out ./reports

# 9) Report
llm-ripper report --ideal --out ./reports --from ./runs/<stamp>

# 10) Studio
llm-ripper studio --root ./runs/<stamp> --port 8000
```

## Tips

- Prefer local paths when running offline; pass `--offline` to subcommands that might download datasets.
- `merge --global spec.yaml --micro` supports global weight average and microtransplants in one step.
- `adapters --import <lora>` adds an extra adapter to a layer and `--fuse` attaches a fusion gate over all adapters.

