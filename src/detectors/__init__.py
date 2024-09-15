"""Fall detection backends."""

from .base import BaseDetector, FallAnalysisResult, PhaseLabel
from .rules_detector import RulesDetector

# LSTM detector is optional (requires PyTorch)
try:
    from .lstm_detector import LSTMDetector
    LSTM_AVAILABLE = True
except ImportError:
    LSTMDetector = None
    LSTM_AVAILABLE = False

__all__ = [
    "BaseDetector",
    "FallAnalysisResult",
    "PhaseLabel",
    "RulesDetector",
    "LSTMDetector",
    "LSTM_AVAILABLE",
]
