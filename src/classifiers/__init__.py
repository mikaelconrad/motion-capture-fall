"""
Fall classifiers and specialized detectors.

- FallTypeClassifier: Classify falls as slip/trip/collapse
- NearFallDetector: Detect recovered balance losses
"""

from .fall_type import FallTypeClassifier, FallTypeResult, classify_fall_type
from .near_fall import NearFallDetector, NearFallEvent, detect_near_falls

__all__ = [
    "FallTypeClassifier",
    "FallTypeResult",
    "classify_fall_type",
    "NearFallDetector",
    "NearFallEvent",
    "detect_near_falls",
]
