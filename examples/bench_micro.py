#!/usr/bin/env python3
import time
from pathlib import Path
import json
import torch
import torch.nn as nn


def run_benchmark(
    model: nn.Module, steps: int = 100, seq_len: int = 64, hidden: int = 512
) -> dict:
    x = torch.randn(1, seq_len, hidden)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            _ = model(x)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0 / steps
    tokens_per_s = (steps * seq_len) / (t1 - t0)
    return {"latency_ms": latency_ms, "tokens_per_s": tokens_per_s}


def main():
    # Synthetic baseline and transplanted (adapters mimic extra cost)
    class Base(nn.Module):
        def __init__(self, h=512):
            super().__init__()
            self.mlp = nn.Linear(h, h)

        def forward(self, x):
            return self.mlp(x)

    class WithAdapter(nn.Module):
        def __init__(self, h=512):
            super().__init__()
            self.mlp = nn.Linear(h, h)
            self.adapter = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, h))

        def forward(self, x):
            return self.mlp(x) + 0.1 * self.adapter(x)

    base = Base()
    transplanted = WithAdapter()
    out = Path("./output/bench")
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "baseline": run_benchmark(base),
        "transplanted": run_benchmark(transplanted),
    }
    (out / "bench_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
