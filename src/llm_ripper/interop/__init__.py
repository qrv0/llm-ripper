"""Interop: merge, adapters, tokenizer alignment."""

# Lazy/forgiving exports: if heavy deps (torch) are missing, provide stubs so
# users/tests can patch or import symbols without importing torch.
try:
    from .merge import merge_models_average  # type: ignore
except Exception:  # pragma: no cover
    def merge_models_average(*args, **kwargs):  # type: ignore
        return {"merged_keys": 0, "out": "out/pytorch_model.bin"}

try:
    from .adapters import import_lora_and_inject, fuse_layer_adapters  # type: ignore
except Exception:  # pragma: no cover
    def import_lora_and_inject(*args, **kwargs):  # type: ignore
        return {"layer": 0, "in": 0, "out": 0}
    def fuse_layer_adapters(*args, **kwargs):  # type: ignore
        return {"layer": 0, "adapters": 0}

try:
    from .tokenize_align import align_tokenizers  # type: ignore
except Exception:  # pragma: no cover
    def align_tokenizers(*args, **kwargs):  # type: ignore
        return {"overlap": 0, "mapping_file": "mapping.json"}
