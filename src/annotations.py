"""
Annotation system for fall detection.

Provides data structures and utilities for temporal phase annotations
of motion capture data. Annotations mark the boundaries of fall phases
(pre_fall, initiation, descent, impact, post_fall) with frame-level precision.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import os
from datetime import datetime

from .config import PhaseLabel, FallType


@dataclass
class PhaseAnnotation:
    """A single phase annotation with frame boundaries."""
    phase: PhaseLabel
    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float
    confidence: float = 1.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'phase': self.phase.value,
            'frame_start': int(self.frame_start),
            'frame_end': int(self.frame_end),
            'time_start_s': round(float(self.time_start_s), 3),
            'time_end_s': round(float(self.time_end_s), 3),
            'confidence': float(self.confidence),
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhaseAnnotation':
        return cls(
            phase=PhaseLabel(data['phase']),
            frame_start=data['frame_start'],
            frame_end=data['frame_end'],
            time_start_s=data['time_start_s'],
            time_end_s=data['time_end_s'],
            confidence=data.get('confidence', 1.0),
            notes=data.get('notes', ''),
        )


@dataclass
class EventAnnotation:
    """A single event annotation (e.g., impact, recovery attempt)."""
    frame: int
    time_s: float
    event_type: str
    severity: str = "medium"  # low, medium, high
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame': int(self.frame),
            'time_s': round(float(self.time_s), 3),
            'event_type': self.event_type,
            'severity': self.severity,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventAnnotation':
        return cls(
            frame=data['frame'],
            time_s=data['time_s'],
            event_type=data['event_type'],
            severity=data.get('severity', 'medium'),
            notes=data.get('notes', ''),
        )


@dataclass
class VideoAnnotation:
    """Complete annotation for a single video/C3D file."""
    video_id: str
    annotator: str = "auto"
    annotation_date: str = ""
    frame_rate: float = 50.0
    total_frames: int = 0

    # Ground truth
    ground_truth_fall: bool = False
    fall_type: Optional[FallType] = None

    # Phase annotations
    phases: List[PhaseAnnotation] = field(default_factory=list)

    # Event annotations
    events: List[EventAnnotation] = field(default_factory=list)

    # Additional metadata
    notes: str = ""
    quality: str = "good"  # good, partial, poor

    def __post_init__(self):
        if not self.annotation_date:
            self.annotation_date = datetime.now().strftime("%Y-%m-%d")

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'video_id': self.video_id,
            'annotator': self.annotator,
            'annotation_date': self.annotation_date,
            'frame_rate': float(self.frame_rate),
            'total_frames': int(self.total_frames),
            'ground_truth_fall': bool(self.ground_truth_fall),
            'phases': [p.to_dict() for p in self.phases],
            'events': [e.to_dict() for e in self.events],
            'notes': self.notes,
            'quality': self.quality,
        }
        if self.fall_type:
            result['fall_type'] = self.fall_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoAnnotation':
        phases = [PhaseAnnotation.from_dict(p) for p in data.get('phases', [])]
        events = [EventAnnotation.from_dict(e) for e in data.get('events', [])]

        fall_type = None
        if 'fall_type' in data and data['fall_type']:
            fall_type = FallType(data['fall_type'])

        return cls(
            video_id=data['video_id'],
            annotator=data.get('annotator', 'unknown'),
            annotation_date=data.get('annotation_date', ''),
            frame_rate=data.get('frame_rate', 50.0),
            total_frames=data.get('total_frames', 0),
            ground_truth_fall=data.get('ground_truth_fall', False),
            fall_type=fall_type,
            phases=phases,
            events=events,
            notes=data.get('notes', ''),
            quality=data.get('quality', 'good'),
        )

    def get_phase_at_frame(self, frame: int) -> Optional[PhaseLabel]:
        """Get the phase label for a specific frame."""
        for phase in self.phases:
            if phase.frame_start <= frame <= phase.frame_end:
                return phase.phase
        return None

    def get_phase_labels(self) -> List[PhaseLabel]:
        """Get list of phase labels for each frame."""
        labels = [None] * self.total_frames
        for phase in self.phases:
            for f in range(phase.frame_start, min(phase.frame_end + 1, self.total_frames)):
                labels[f] = phase.phase
        return labels

    def validate(self) -> List[str]:
        """Validate annotation for consistency. Returns list of issues."""
        issues = []

        # Check for gaps
        covered = set()
        for phase in self.phases:
            for f in range(phase.frame_start, phase.frame_end + 1):
                if f in covered:
                    issues.append(f"Overlapping phases at frame {f}")
                covered.add(f)

        # Check for gaps in fall sequences
        if self.ground_truth_fall and self.phases:
            # Should have continuous coverage from first phase to last
            min_frame = min(p.frame_start for p in self.phases)
            max_frame = max(p.frame_end for p in self.phases)
            for f in range(min_frame, max_frame + 1):
                if f not in covered:
                    issues.append(f"Gap in annotation at frame {f}")

        # Check phase order for falls
        if self.ground_truth_fall:
            phase_order = [PhaseLabel.PRE_FALL, PhaseLabel.INITIATION,
                          PhaseLabel.DESCENT, PhaseLabel.IMPACT, PhaseLabel.POST_FALL]
            seen_phases = [p.phase for p in sorted(self.phases, key=lambda x: x.frame_start)]

            # Filter to only phases that should appear in order
            seen_ordered = [p for p in seen_phases if p in phase_order]
            expected_indices = [phase_order.index(p) for p in seen_ordered]
            if expected_indices != sorted(expected_indices):
                issues.append("Phase order violation")

        return issues


def save_annotation(annotation: VideoAnnotation, filepath: str) -> None:
    """Save annotation to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(annotation.to_dict(), f, indent=2)


def load_annotation(filepath: str) -> VideoAnnotation:
    """Load annotation from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return VideoAnnotation.from_dict(data)


def get_annotation_path(video_dir: str) -> str:
    """Get the annotation file path for a video directory."""
    return os.path.join(video_dir, 'annotations.json')


def load_all_annotations(data_dir: str) -> Dict[str, VideoAnnotation]:
    """Load all annotations from a data directory."""
    annotations = {}
    for video_name in os.listdir(data_dir):
        video_path = os.path.join(data_dir, video_name)
        if os.path.isdir(video_path):
            ann_path = get_annotation_path(video_path)
            if os.path.exists(ann_path):
                try:
                    annotations[video_name] = load_annotation(ann_path)
                except Exception as e:
                    print(f"Warning: Could not load {ann_path}: {e}")
    return annotations
