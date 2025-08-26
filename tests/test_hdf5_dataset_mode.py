import tempfile
from pathlib import Path
import h5py
import numpy as np
import torch

from llm_ripper.core.activation_capture import ActivationCapture
from llm_ripper.utils.config import ConfigManager


def test_hdf5_dataset_mode_hidden_and_attention(tmp_path: Path):
    # Config with dataset mode
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}")
    cfg = ConfigManager(str(cfg_path))
    cfg.set("hdf5_store_mode", "dataset")
    cfg.set("max_sequence_length", 16)

    ac = ActivationCapture(cfg)
    out = tmp_path / "act.h5"
    layers = ["layer.hidden", "layer.attn"]
    meta = ac._setup_hdf5_storage(
        str(out), "dummy", {"num_hidden_layers": 1}, layers, 4
    )
    assert Path(meta["output_file"]).exists()

    # Build fake batch
    B, T, H = 2, 8, 4
    hidden = torch.randn(B, T, H)
    attn = torch.randn(B, 2, T, T)
    activations = {"layer.hidden": hidden, "layer.attn": attn}
    texts = ["a b c", "d e f"]
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    with h5py.File(str(out), "a") as f:
        ac._store_batch_activations(f, activations, texts, ids, 0, layers)

    with h5py.File(str(out), "r") as f:
        assert f["activations"]["layer_hidden"]["data"].shape[0] >= 2
        assert f["activations"]["layer_attn"]["attn"].shape[0] >= 2
        assert f["samples"]["texts"].shape[0] >= 2
