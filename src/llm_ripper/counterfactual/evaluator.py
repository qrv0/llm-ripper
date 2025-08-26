"""
Evaluate counterfactual minimal pairs with delta metrics using a LM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader


class CounterfactualEvaluator:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.loader = ModelLoader(
            cache_dir=config.get("model_cache_dir"),
            device=config.get("device", "auto"),
        )

    def evaluate(
        self, model: str, pairs_glob: List[str], out_path: str
    ) -> Dict[str, Any]:
        model, tok, _ = self.loader.load_model_and_tokenizer(
            model,
            model_type="causal_lm",
            load_in_8bit=self.config.get("load_in_8bit"),
            load_in_4bit=self.config.get("load_in_4bit"),
            trust_remote_code=self.config.get("trust_remote_code"),
        )
        rows: List[Dict[str, Any]] = []
        # Load pairs
        import glob as _glob

        for pat in pairs_glob:
            for fp in _glob.glob(pat):
                for line in Path(fp).read_text().splitlines():
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
        # Evaluate per pair as delta in target token logits at last position
        out: List[Dict[str, Any]] = []
        _ = next(model.parameters()).device
        with torch.no_grad():
            for r in rows:
                toks = r.get("target_tokens") or []
                l_orig = self._score_token(model, tok, r["original"], toks)
                l_cf = self._score_token(model, tok, r["counterfactual"], toks)
                delta = float(l_orig - l_cf)
                out.append(
                    {
                        "id": r.get("id"),
                        "task": r.get("task"),
                        "delta": delta,
                        "orig_logit": float(l_orig),
                        "cf_logit": float(l_cf),
                        "tokens": toks,
                    }
                )
        # Save JSONL
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r))
                f.write("\n")
        # Summary
        avg_delta = float(sum(r["delta"] for r in out) / max(1, len(out)))
        summary = {"pairs": len(out), "avg_delta": avg_delta}
        return {"results_file": str(p), "summary": summary}

    def _score_token(self, model, tok, text: str, candidates: List[str]) -> float:
        enc = tok(text, return_tensors="pt")
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits[:, -1, :]
        if candidates:
            # pick the first candidate token id present in vocab; fallback to eos
            for c in candidates:
                try:
                    tid = tok.encode(c, add_special_tokens=False)
                    if len(tid) == 1:
                        return float(logits[0, tid[0]].item())
                except Exception:
                    continue
        eos_id = tok.eos_token_id or tok.pad_token_id or (logits.shape[-1] - 1)
        return float(logits[0, eos_id].item())
