from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader


def _load_kb_embeddings(kb_dir: str) -> torch.Tensor:
    p = Path(kb_dir) / "embeddings"
    # prefer safetensors
    st = p / "embeddings.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        d = load_file(str(st))
        return d["weight"].detach().cpu()
    pt = p / "embeddings.pt"
    if pt.exists():
        w = torch.load(str(pt), map_location="cpu")
        if isinstance(w, dict) and "weight" in w:
            return w["weight"].detach().cpu()
        if isinstance(w, torch.Tensor):
            return w.detach().cpu()
    # Sharded index
    idx = p / "embeddings.index.json"
    if idx.exists():
        meta = json.loads(idx.read_text())
        parts = []
        for name in meta.get("parts", []):
            t = torch.load(str(idx.parent / name), map_location="cpu")
            parts.append(t)
        return torch.cat(parts, dim=0)
    raise FileNotFoundError("Embeddings not found in knowledge bank")


def orthogonal_procrustes_align(
    cfg: ConfigManager,
    kb_dir: str,
    target_model: str,
    out_path: str,
) -> Dict[str, Any]:
    """Compute an orthogonal matrix W aligning donor->target embedding spaces.
    Saves W and reports cosine/MSE improvements.
    """
    donor = _load_kb_embeddings(kb_dir)  # [Vd, Hd]
    loader = ModelLoader(
        cache_dir=cfg.get("model_cache_dir"), device=cfg.get("device", "auto")
    )
    model, tok, _ = loader.load_model_and_tokenizer(
        target_model, model_type="base", trust_remote_code=cfg.get("trust_remote_code")
    )
    target = model.get_input_embeddings().weight.detach().cpu()  # [Vt, Ht]

    # Use overlapping vocab portion: min(Vd, Vt)
    n = min(donor.shape[0], target.shape[0])
    X = donor[:n].float().numpy()
    Y = target[:n].float().numpy()

    # Center
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # Compute orthogonal Procrustes via SVD of Yc^T Xc
    M = Yc.T @ Xc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt  # maps donor -> target space (Hd x Ht if dims equal)

    # If dims mismatch, project using least-squares to target dim
    if W.shape[0] != donor.shape[1] or W.shape[1] != target.shape[1]:
        # Fit linear map using ridge
        from numpy.linalg import lstsq

        W_ls, *_ = lstsq(Xc, Yc, rcond=None)  # [Hd, Ht]
        W = W_ls

    # Evaluate
    def _cos(a, b):
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return float((an * bn).sum(axis=1).mean())

    def _mse(a, b):
        d = a - b
        return float((d * d).mean())

    XW = Xc @ W
    cos_before = _cos(Xc, Yc)
    cos_after = _cos(XW, Yc)
    mse_before = _mse(Xc, Yc)
    mse_after = _mse(XW, Yc)

    # Save
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, W)
    return {
        "matrix_file": str(outp) + ".npy",
        "cosine_before": cos_before,
        "cosine_after": cos_after,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "rows_used": n,
    }
