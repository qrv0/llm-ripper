from pathlib import Path
import h5py
import numpy as np
import torch

from llm_ripper.core.activation_capture import ActivationCapture
from llm_ripper.utils.config import ConfigManager


def test_dataset_slices_helper(tmp_path: Path):
    cfg = ConfigManager()
    cfg.set("hdf5_store_mode", "dataset")
    ac = ActivationCapture(cfg)
    out = tmp_path / "f.h5"
    layers = ["layer.hidden"]
    ac._setup_hdf5_storage(str(out), "m", {}, layers, 3)
    # write a small dataset-mode tensor
    with h5py.File(str(out), "a") as f:
        lg = f["activations"].create_group("layer_hidden")
        d = lg.create_dataset(
            "data", shape=(3, 2, 2), maxshape=(None, 2, 2), dtype="float32"
        )
        d[...] = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    arr = ac.load_dataset_slices(str(out), "layer.hidden", [0, 2])
    assert arr.shape == (2, 2, 2)
