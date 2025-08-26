# Causal Tracing

Causal tracing runs controlled interventions on internal components (heads/FFN) and measures impact on a task metric.

## Targets

- Attention projections: `head:<layer>.<q|k|v|o>[:head_idx]`
- FFN parts: `ffn:<layer>.<gate|up|down>`

Examples: `head:0.q`, `head:12.o:3`, `ffn:7.up`.

## Interventions

- `zero`: zero out the selected subspace
- `noise`: add small Gaussian noise
- `mean-patch`: replace activations with the mean across batch/time

## Metrics

- `nll_delta`: difference in language-model loss (lower is better baseline)
- `logit_delta`: difference in target token logit (eos proxy)

## CLI

```bash
llm-ripper trace \
  --model <path_or_hf> \
  --targets head:12.q:0,ffn:7.up \
  --metric nll_delta \
  --intervention zero \
  --max-samples 64 --seed 42 --json
```

Outputs:
- `runs/<stamp>/traces/traces.jsonl`: per-target rows `{target, intervention, metric, baseline, intervened, delta}`
- `runs/<stamp>/traces/summary.json`: ranked by `delta` descending
