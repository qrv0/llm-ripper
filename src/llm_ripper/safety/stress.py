from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.config import ConfigManager
from ..utils.data_manager import DataManager
from ..utils.model_loader import ModelLoader


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def _hist(x: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    h, e = np.histogram(x, bins=bins, range=(x.min(), x.max() + 1e-9), density=True)
    # normalize to sum-1 across bins
    h = h / (h.sum() + 1e-12)
    return h, e


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = p + 1e-12
    q = q + 1e-12
    return float(np.sum(p * np.log(p / q)))


def _psi(p: np.ndarray, q: np.ndarray) -> float:
    # Population Stability Index across bins
    p = p + 1e-6
    q = q + 1e-6
    return float(np.sum((p - q) * np.log(p / q)))


def run_stress_and_drift(
    config: ConfigManager, model: str, baseline: str, out_dir: str
) -> Dict[str, Any]:
    dm = DataManager(config)
    texts = list(dm._create_diverse_corpus()["text"]) + [
        # lightweight adversarial-like prompts
        "aaaa aaaa aaaa aaaa aaaa aaaa aaaa",
        "!!!! ???? .... ,,,, ;;;; ::::",
        "number sequence: 1 2 3 4 5 6 7 8 9",
        "repeat repeat repeat repeat repeat repeat",
        "very very very very long long long long",
    ]
    texts = texts[:128]
    loader = ModelLoader(
        cache_dir=config.get("model_cache_dir"), device=config.get("device", "auto")
    )
    m1, t1, _ = loader.load_model_and_tokenizer(
        model, model_type="causal_lm", trust_remote_code=config.get("trust_remote_code")
    )
    m0, t0, _ = loader.load_model_and_tokenizer(
        baseline,
        model_type="causal_lm",
        trust_remote_code=config.get("trust_remote_code"),
    )
    device = next(m1.parameters()).device

    # Collect last-token distributions
    def collect(m, tok):
        with torch.no_grad():
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = m(**enc)
            p = F.softmax(out.logits[:, -1, :], dim=-1)
            return p.detach().cpu()

    P1 = collect(m1, t1)
    P0 = collect(m0, t0)
    # Entropy drift
    H1 = _entropy(P1).numpy()
    H0 = _entropy(P0).numpy()
    h1, edges = _hist(H1, bins=20)
    h0, _ = _hist(H0, bins=20)
    psi = _psi(h1, h0)
    # KL on averaged distributions
    p1 = P1.mean(dim=0).numpy()
    p0 = P0.mean(dim=0).numpy()
    kl10 = _kl(p1, p0)
    kl01 = _kl(p0, p1)
    res = {
        "psi_entropy": psi,
        "kl_model_vs_baseline": kl10,
        "kl_baseline_vs_model": kl01,
        "bins": edges.tolist(),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stress_drift.json").write_text(json.dumps(res, indent=2))
    return res
