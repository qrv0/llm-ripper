# ruff: noqa: E402
"""
LLM Ripper: A framework for modular deconstruction, analysis, and recomposition 
of knowledge in Transformer-based language models.

Warning hygiene:
- Suppress environment-related joblib UserWarnings seen in restricted sandboxes.
- Suppress CUDA initialization warnings when CUDA is unavailable or blocked.
These filters avoid noisy logs without hiding actionable errors.
"""

import warnings as _warnings

# Joblib multiprocessing warning in restricted environments
_warnings.filterwarnings(
    "ignore",
    message=".*joblib will operate in serial mode.*",
    category=UserWarning,
)
# Torch CUDA initialization warnings on non-GPU/blocked systems
_warnings.filterwarnings(
    "ignore",
    message=".*CUDA initialization: Unexpected error.*",
    category=UserWarning,
)
_warnings.filterwarnings(
    "ignore",
    module=r"torch\.cuda.*",
    category=UserWarning,
)

__version__ = "1.0.0"
__author__ = "LLM Ripper Team"

from .core import (
    KnowledgeExtractor,
    ActivationCapture,
    KnowledgeAnalyzer,
    KnowledgeTransplanter,
    ValidationSuite,
)

# Import light-weight utils directly to avoid importing optional heavy deps by default
from .utils.model_loader import ModelLoader
from .utils.config import ConfigManager
from .utils.data_manager import DataManager

__all__ = [
    "KnowledgeExtractor",
    "ActivationCapture",
    "KnowledgeAnalyzer",
    "KnowledgeTransplanter",
    "ValidationSuite",
    "ModelLoader",
    "ConfigManager",
    "DataManager",
]
