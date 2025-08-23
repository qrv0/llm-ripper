"""
LLM Ripper: A framework for modular deconstruction, analysis, and recomposition 
of knowledge in Transformer-based language models.
"""

__version__ = "1.0.0"
__author__ = "LLM Ripper Team"

from .core import (
    KnowledgeExtractor,
    ActivationCapture,
    KnowledgeAnalyzer,
    KnowledgeTransplanter,
    ValidationSuite
)

from .utils import (
    ModelLoader,
    ConfigManager,
    DataManager
)

__all__ = [
    "KnowledgeExtractor",
    "ActivationCapture", 
    "KnowledgeAnalyzer",
    "KnowledgeTransplanter",
    "ValidationSuite",
    "ModelLoader",
    "ConfigManager",
    "DataManager"
]