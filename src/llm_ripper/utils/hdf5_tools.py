"""HDF5 utility helpers: repack and downsample.

These tools help shrink HDF5 files produced in dataset mode and are safe offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dep, but required in project deps
    h5py = None  # type: ignore


def repack_hdf5(
    in_path: str,
    out_path: str,
    compression: str = "gzip",
    chunk_size: Optional[int] = None,
) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 operations")
    in_p, out_p = Path(in_path), Path(out_path)
    with h5py.File(str(in_p), "r") as src, h5py.File(str(out_p), "w") as dst:

        def _copy_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                kwargs = {}
                if compression:
                    kwargs["compression"] = compression
                data = obj[...]
                chunks = obj.chunks
                if chunk_size is not None and data.ndim > 0:
                    chunks = tuple(min(chunk_size, s) for s in data.shape)
                d = dst.require_dataset(
                    name, shape=obj.shape, dtype=obj.dtype, chunks=chunks, **kwargs
                )
                d[...] = data
                for k, v in obj.attrs.items():
                    d.attrs[k] = v
            elif isinstance(obj, h5py.Group):
                dst.require_group(name)

        src.visititems(_copy_item)


def downsample(in_path: str, out_path: str, every_n: int = 2) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 operations")
    with h5py.File(in_path, "r") as src, h5py.File(out_path, "w") as dst:

        def _copy_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[...]
                if data.ndim >= 1 and data.shape[0] >= every_n:
                    data = data[::every_n]
                d = dst.create_dataset(name, data=data, compression="gzip")
                for k, v in obj.attrs.items():
                    d.attrs[k] = v
            elif isinstance(obj, h5py.Group):
                dst.require_group(name)

        src.visititems(_copy_item)
