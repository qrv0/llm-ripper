"""Utility modules for LLM Ripper framework."""

# Lazy imports to avoid importing heavy deps (torch) at import time
try:
    from .model_loader import ModelLoader  # type: ignore
except Exception:  # pragma: no cover
    class ModelLoader:  # type: ignore
        pass

from .config import ConfigManager
try:
    from .data_manager import DataManager  # type: ignore
except Exception:  # pragma: no cover
    class DataManager:  # type: ignore
        pass
try:
    from .metrics import MetricsCalculator  # type: ignore
except Exception:  # pragma: no cover
    class MetricsCalculator:  # type: ignore
        pass

__all__ = [
    "ModelLoader",
    "ConfigManager",
    "DataManager",
    "MetricsCalculator",
]
