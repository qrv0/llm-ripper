from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import ConfigManager
from ..utils.data_manager import DataManager
from ..utils.model_loader import ModelLoader
from ..utils.run import RunContext


@dataclass
class UQConfig:
    model: str
    tasks: str = "mechanistic"  # currently unused; we use probing corpus
    samples: int = 10
    max_texts: int = 64
    seed: int = 42


class UQRunner:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.dm = DataManager(config)
        self.loader = ModelLoader(
            cache_dir=config.get("model_cache_dir"),
            device=config.get("device", "auto"),
        )

    def run(self, uq_cfg: UQConfig, out_dir: str) -> Dict[str, Any]:
        torch.manual_seed(uq_cfg.seed)
        model, tok, _ = self.loader.load_model_and_tokenizer(
            uq_cfg.model,
            model_type="causal_lm",
            load_in_8bit=self.config.get("load_in_8bit"),
            load_in_4bit=self.config.get("load_in_4bit"),
            trust_remote_code=self.config.get("trust_remote_code"),
        )
        ds = self.dm.load_probing_corpus("diverse")
        texts = list(ds["text"])[: uq_cfg.max_texts]

        # Enable dropout at inference
        def _enable_dropout(m: nn.Module):
            if isinstance(m, (nn.Dropout,)):
                m.train()

        model.apply(_enable_dropout)
        device = next(model.parameters()).device

        # Collect MC samples of last-token probabilities
        probs: List[torch.Tensor] = []
        with torch.no_grad():
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            for _ in range(uq_cfg.samples):
                out = model(**enc)
                p = F.softmax(out.logits[:, -1, :].detach(), dim=-1)
                probs.append(p)
        P = torch.stack(probs, dim=0)  # [S, B, V]
        Pm = P.mean(dim=0)  # [B, V]
        # Predictive entropy H[p]
        pred_ent = -(Pm.clamp_min(1e-12) * Pm.clamp_min(1e-12).log()).sum(dim=-1)  # [B]
        # MI approx (BALD): H[p] - E[H[p_i]]
        ent_i = -(P.clamp_min(1e-12) * P.clamp_min(1e-12).log()).sum(dim=-1)  # [S, B]
        mi = pred_ent - ent_i.mean(dim=0)

        # ECE via self-consistency: confidence of mean argmax vs agreement rate across samples
        conf, acc = self._self_consistency(P)
        ece = self._ece(conf, acc, n_bins=10)

        rc = RunContext.create(base="runs")
        rows = []
        for i in range(Pm.shape[0]):
            rows.append(
                {
                    "idx": i,
                    "predictive_entropy": float(pred_ent[i].item()),
                    "mutual_info": float(mi[i].item()),
                    "confidence": float(conf[i]),
                    "agreement": float(acc[i]),
                }
            )
        rc.write_jsonl("uq/metrics.jsonl", rows)
        summary = {
            "samples": uq_cfg.samples,
            "texts": len(texts),
            "avg_predictive_entropy": float(pred_ent.mean().item()),
            "avg_mutual_info": float(mi.mean().item()),
            "ece": float(ece),
        }
        fp = rc.write_json("uq/summary.json", summary)
        return {"run_id": rc.stamp, "summary_file": str(fp)}

    def _self_consistency(self, P: torch.Tensor) -> Tuple[List[float], List[float]]:
        # P: [S, B, V]
        Pm = P.mean(dim=0)
        y = Pm.argmax(dim=-1)  # [B]
        conf = Pm[torch.arange(Pm.shape[0]), y]
        agree = (P.argmax(dim=-1) == y.unsqueeze(0)).float().mean(dim=0)
        return conf.cpu().tolist(), agree.cpu().tolist()

    def _ece(self, conf: List[float], acc: List[float], n_bins: int = 10) -> float:
        import numpy as np

        conf_a = np.array(conf)
        acc_a = np.array(acc)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            idx = (conf_a >= lo) & (conf_a < hi if i < n_bins - 1 else conf_a <= hi)
            if idx.any():
                ece += float(
                    abs(acc_a[idx].mean() - conf_a[idx].mean())
                    * (idx.sum() / len(conf_a))
                )
        return ece
