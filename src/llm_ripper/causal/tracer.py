"""
Causal tracing with simple, pluggable interventions and impact metrics.

Implements a minimal but real tracer that:
 - Loads a (causal LM) model via ModelLoader
 - Runs a baseline NLL on a probing corpus
 - Applies interventions on specified targets (heads/projections or FFN parts)
 - Measures delta in NLL/logit for target tokens
 - Writes JSONL traces and a summary ranking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.config import ConfigManager
from ..utils.data_manager import DataManager
from ..utils.model_loader import ModelLoader
from ..utils.architecture import get_layer_module
from ..utils.run import RunContext

logger = logging.getLogger(__name__)


@dataclass
class TraceConfig:
    model: str
    targets: List[str]
    metric: str = "nll_delta"  # or "logit_delta"
    dataset: str = "diverse"
    intervention: str = "zero"  # zero|noise|mean-patch
    seed: int = 42
    max_samples: int = 64


@dataclass
class TraceResult:
    run_id: str
    summary_path: str
    traces_path: str


class Tracer:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.dm = DataManager(config)
        self.loader = ModelLoader(
            cache_dir=config.get("model_cache_dir"),
            device=config.get("device", "auto"),
        )

    def run(self, trace_cfg: TraceConfig, out_dir: str) -> TraceResult:
        torch.manual_seed(trace_cfg.seed)
        # Load model
        model, tokenizer, mcfg = self.loader.load_model_and_tokenizer(
            trace_cfg.model,
            model_type="causal_lm",
            load_in_8bit=self.config.get("load_in_8bit"),
            load_in_4bit=self.config.get("load_in_4bit"),
            trust_remote_code=self.config.get("trust_remote_code"),
        )
        model.eval()

        # Data
        ds = self.dm.load_probing_corpus(trace_cfg.dataset)
        texts = list(ds["text"])[: trace_cfg.max_samples]

        # Baseline metric
        with torch.no_grad():
            base_metric, base_stats = self._evaluate(
                model, tokenizer, texts, metric=trace_cfg.metric
            )

        # Prepare run context
        rc = RunContext.create(base="runs")
        traces_fp = rc.traces_dir() / "traces.jsonl"
        rows: List[Dict[str, Any]] = []

        # Iterate targets and apply interventions
        for tgt in trace_cfg.targets:
            try:
                layer_idx, sub, head_idx = self._parse_target(tgt)
            except Exception as e:
                logger.warning(f"Skipping target '{tgt}': {e}")
                continue

            with self._intervention(
                model, layer_idx, sub, head_idx, mcfg, mode=trace_cfg.intervention
            ):
                with torch.no_grad():
                    int_metric, int_stats = self._evaluate(
                        model, tokenizer, texts, metric=trace_cfg.metric
                    )

            delta = float(base_metric - int_metric)
            row = {
                "target": tgt,
                "layer": layer_idx,
                "sub": sub,
                "head": head_idx,
                "intervention": trace_cfg.intervention,
                "metric": trace_cfg.metric,
                "baseline": float(base_metric),
                "intervened": float(int_metric),
                "delta": delta,
                "seed": trace_cfg.seed,
                "samples": len(texts),
            }
            rows.append(row)

        # Save JSONL and summary
        rc.write_jsonl(traces_fp.relative_to(rc.root), rows)
        # Ranking by delta desc
        ranking = sorted(rows, key=lambda r: r["delta"], reverse=True)
        summary = {
            "model": trace_cfg.model,
            "metric": trace_cfg.metric,
            "intervention": trace_cfg.intervention,
            "dataset": trace_cfg.dataset,
            "seed": trace_cfg.seed,
            "baseline_stats": base_stats,
            "results": ranking,
        }
        summary_fp = rc.write_json("traces/summary.json", summary)
        return TraceResult(
            run_id=rc.stamp, summary_path=str(summary_fp), traces_path=str(traces_fp)
        )

    # ------------------ helpers ------------------
    def _evaluate(
        self, model: nn.Module, tokenizer, texts: List[str], metric: str
    ) -> Tuple[float, Dict[str, Any]]:
        # Build a small batch to compute LM loss
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}

        outputs = model(**enc, labels=enc["input_ids"])  # type: ignore[arg-type]
        loss = outputs.loss.detach().float().item()
        if metric == "nll_delta":
            return float(loss), {"loss": float(loss)}
        elif metric == "logit_delta":
            # Use eos token logits mean as proxy
            logits = outputs.logits.detach()
            eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
            if eos_id is None:
                eos_id = int(logits.shape[-1] - 1)
            last = logits[:, -1, :]
            score = float(last[:, eos_id].mean().item())
            return score, {"eos_logit": score}
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def _parse_target(self, spec: str) -> Tuple[int, str, Optional[int]]:
        # Formats: "head:<layer>.<q|k|v|o>[:head_idx]" or "ffn:<layer>.<gate|up|down>"
        if spec.startswith("head:"):
            rest = spec.split(":", 1)[1]
            layer_part, sub = rest.split(".")
            head_idx: Optional[int] = None
            if ":" in sub:
                sub, h = sub.split(":", 1)
                head_idx = int(h)
            return int(layer_part), sub, head_idx
        if spec.startswith("ffn:"):
            rest = spec.split(":", 1)[1]
            layer_part, sub = rest.split(".")
            return int(layer_part), f"ffn_{sub}", None
        raise ValueError(f"Unrecognized target: {spec}")

    from contextlib import contextmanager

    @contextmanager
    def _intervention(
        self,
        model: nn.Module,
        layer_idx: int,
        sub: str,
        head_idx: Optional[int],
        mcfg: Dict[str, Any],
        mode: str,
    ):
        hooks: List[Any] = []
        try:
            # Find module
            mt = getattr(getattr(model, "config", object()), "model_type", None)
            if sub.startswith("ffn_"):
                mod = get_layer_module(model, layer_idx, kind="mlp", model_type=mt)
                part = sub.replace("ffn_", "")
                lin = getattr(mod, f"{part}_proj", None) if mod is not None else None
                if isinstance(lin, nn.Linear):
                    hooks.append(
                        self._hook_linear_output(lin, head_idx=None, mode=mode)
                    )
            else:
                attn = get_layer_module(model, layer_idx, kind="attn", model_type=mt)
                if attn is not None and hasattr(attn, f"{sub}_proj"):
                    lin = getattr(attn, f"{sub}_proj")
                    hooks.append(
                        self._hook_linear_output(
                            lin, head_idx=head_idx, mode=mode, attn_module=attn
                        )
                    )
            yield
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

    def _hook_linear_output(
        self,
        lin: nn.Linear,
        head_idx: Optional[int],
        mode: str,
        attn_module: Optional[nn.Module] = None,
    ):
        # Determine head slicing if available
        head_dim = None
        num_heads = None
        if attn_module is not None:
            head_dim = getattr(attn_module, "head_dim", None)
            num_heads = getattr(attn_module, "num_heads", None)

        def fn(_mod, _inp, out):  # type: ignore[override]
            if not isinstance(out, torch.Tensor):
                return out
            x = out
            if mode == "zero":
                if head_idx is None or head_dim is None or num_heads is None:
                    return torch.zeros_like(x)
                # zero only selected head slice along last dim
                start = head_idx * head_dim
                end = start + head_dim
                x[..., start:end] = 0
                return x
            if mode == "noise":
                return x + 0.01 * torch.randn_like(x)
            if mode == "mean-patch":
                # replace with mean over batch/time dims
                mean = x.mean(dim=list(range(x.ndim - 1)), keepdim=True)
                return mean.expand_as(x)
            return x

        return lin.register_forward_hook(fn)
