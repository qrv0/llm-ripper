from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except Exception:
    h5py = None  # type: ignore


def _load_hdf5_matrix(path: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is required for feature discovery from HDF5")
    with h5py.File(path, "r") as f:
        # pick first dataset with 2D+ shape
        for name, ds in f.items():
            if isinstance(ds, h5py.Dataset) and ds.ndim >= 2:
                X = ds[...]
                # flatten time/seq dims; keep last as feature dim
                X = X.reshape(-1, X.shape[-1])
                return X
        raise RuntimeError("No suitable dataset found in HDF5")


def discover_features(
    activations_h5: str, method: str, out_dir: str, k: int = 16
) -> Dict[str, Any]:
    """Lightweight feature discovery: PCA-based prototype selection as SAE-lite.
    Saves a catalog JSON with top-k components and example activations.
    """
    X = _load_hdf5_matrix(activations_h5)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD for PCA directions
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:k]
    # Example indices with highest projection per component
    proj = X @ comps.T  # [N, k]
    examples: List[Tuple[int, float]] = []
    idx = np.argmax(proj, axis=0)
    scores = proj[idx, np.arange(k)]
    # Build catalog
    catalog = []
    for i in range(k):
        catalog.append(
            {
                "feature": int(i),
                "label": f"latent_feature_{i}",
                "example_index": int(idx[i]),
                "example_score": float(scores[i]),
            }
        )
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "heads.json").write_text(json.dumps({"features": catalog}, indent=2))
    return {"catalog_file": str(p / "heads.json"), "features": len(catalog)}
