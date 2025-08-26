from pathlib import Path
import numpy as np
import h5py

from llm_ripper.utils.hdf5_tools import repack_hdf5, downsample


def _make_h5(p: Path, shape=(200, 16), fill=0):
    with h5py.File(str(p), "w") as f:
        d = f.create_dataset("data", data=np.full(shape, fill, dtype=np.float32))


def test_repack_reduces_size_and_preserves(tmp_path: Path):
    src = tmp_path / "a.h5"
    dst = tmp_path / "b.h5"
    _make_h5(src, shape=(500, 64), fill=0)
    size_a = src.stat().st_size
    repack_hdf5(str(src), str(dst), compression="gzip", chunk_size=64)
    size_b = dst.stat().st_size
    assert size_b < size_a
    # data preserved
    with h5py.File(str(dst), "r") as f:
        arr = f["data"][...]
        assert arr.shape == (500, 64)
        assert (arr == 0).all()


def test_downsample_halves_first_axis(tmp_path: Path):
    src = tmp_path / "c.h5"
    dst = tmp_path / "d.h5"
    _make_h5(src, shape=(20, 8), fill=1)
    downsample(str(src), str(dst), every_n=2)
    with h5py.File(str(dst), "r") as f:
        arr = f["data"][...]
        assert arr.shape[0] == 10
        assert (arr == 1).all()
