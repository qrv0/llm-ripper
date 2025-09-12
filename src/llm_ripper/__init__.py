# ruff: noqa: E402
"""
LLM Ripper: A framework for modular deconstruction, analysis, and recomposition
of knowledge in Transformer-based language models.

Import-light __init__:
- Avoid importing heavy dependencies at package import time (e.g., torch).
- Provide graceful fallbacks so `import llm_ripper` works in minimal environments.
"""

import warnings as _warnings

# Suppress noisy warnings in constrained environments (optional hygiene)
_warnings.filterwarnings(
    "ignore",
    message=".*joblib will operate in serial mode.*",
    category=UserWarning,
)
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

# Re-export interop package under llm_ripper.interop for tests/patching convenience
from . import interop  # type: ignore

# Try to import core classes; if unavailable (e.g., torch missing),
# expose lightweight placeholders so `hasattr` checks pass and CLI/tests
# that patch these symbols can still run.
try:
    from .core import (  # type: ignore
        KnowledgeExtractor,
        ActivationCapture,
        KnowledgeAnalyzer,
        KnowledgeTransplanter,
        ValidationSuite,
    )
except Exception:  # pragma: no cover - only in minimal envs
    class KnowledgeExtractor:  # type: ignore
        pass

    class ActivationCapture:  # type: ignore
        pass

    class KnowledgeAnalyzer:  # type: ignore
        pass

    class KnowledgeTransplanter:  # type: ignore
        pass

    class ValidationSuite:  # type: ignore
        pass

__all__ = [
    "KnowledgeExtractor",
    "ActivationCapture",
    "KnowledgeAnalyzer",
    "KnowledgeTransplanter",
    "ValidationSuite",
    "interop",
]
