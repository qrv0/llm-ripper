# Uncertainty & Routing

The UQ module uses MC‑Dropout to estimate predictive uncertainty and calibration, and can simulate routing fallbacks.

## Metrics

- `predictive_entropy`: entropy of the mean posterior
- `mutual_info` (BALD): epistemic uncertainty proxy
- `ECE`: self-consistency calibration proxy (confidence vs agreement)

## CLI

```bash
llm-ripper uq --model <path_or_hf> --samples 20 --max-texts 128
llm-ripper route-sim --metrics runs/<stamp>/uq/metrics.jsonl --tau 0.7
```

Outputs:
- `runs/<stamp>/uq/metrics.jsonl`: per-example metrics
- `runs/<stamp>/uq/summary.json`: aggregate metrics
- routing sim prints `{routed, routed_frac}` for threshold τ.
