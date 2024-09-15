"""
Fall Detection System

A biomechanically-informed fall detection and classification system
for 3D motion capture data (C3D format).
"""

from .config import DetectorConfig, BodyModelConfig, PhaseLabel, FallType
from .biomechanics import (
    calculate_whole_body_com,
    calculate_segment_com,
    calculate_xcom,
    calculate_base_of_support,
    calculate_margin_of_stability,
    calculate_trunk_tilt,
)
from .features import extract_features, MotionFeatures
from .annotations import (
    VideoAnnotation,
    PhaseAnnotation,
    EventAnnotation,
    load_annotation,
    save_annotation,
)
from .classifiers import (
    FallTypeClassifier,
    FallTypeResult,
    classify_fall_type,
    NearFallDetector,
    NearFallEvent,
    detect_near_falls,
)

__version__ = "2.0.0"
__all__ = [
    # Config
    "DetectorConfig",
    "BodyModelConfig",
    "PhaseLabel",
    "FallType",
    # Biomechanics
    "calculate_whole_body_com",
    "calculate_segment_com",
    "calculate_xcom",
    "calculate_base_of_support",
    "calculate_margin_of_stability",
    "calculate_trunk_tilt",
    # Features
    "extract_features",
    "MotionFeatures",
    # Annotations
    "VideoAnnotation",
    "PhaseAnnotation",
    "EventAnnotation",
    "load_annotation",
    "save_annotation",
    # Classifiers
    "FallTypeClassifier",
    "FallTypeResult",
    "classify_fall_type",
    "NearFallDetector",
    "NearFallEvent",
    "detect_near_falls",
]
