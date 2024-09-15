"""
Base detector interface and result types.

All fall detection backends should inherit from BaseDetector and
return FallAnalysisResult objects.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

from ..config import DetectorConfig, FallType, PhaseLabel, DEFAULT_CONFIG


@dataclass
class PhaseSegment:
    """A temporal segment labeled with a fall phase."""
    phase: PhaseLabel
    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float
    confidence: float = 1.0
    notes: str = ""


@dataclass
class TimelineData:
    """Time-series data for visualization."""
    # Time vectors (may have different lengths due to derivatives)
    time_pos: List[float]           # Time for position data
    time_vel: List[float]           # Time for velocity data
    time_acc: List[float]           # Time for acceleration data

    # Position series
    pelvis_height_m: List[float]
    head_height_m: List[float]
    com_height_m: Optional[List[float]] = None

    # Velocity series
    pelvis_vertical_velocity_ms: List[float] = field(default_factory=list)
    com_speed_ms: List[float] = field(default_factory=list)

    # Acceleration series
    acc_mag_g: List[float] = field(default_factory=list)

    # Orientation series
    trunk_tilt_deg: List[float] = field(default_factory=list)

    # Stability series (if calculated)
    margin_of_stability_m: Optional[List[float]] = None

    # Event markers
    impact_occurred: bool = False
    impact_frame: Optional[int] = None
    impact_time_s: Optional[float] = None
    impact_acc_g: Optional[float] = None


@dataclass
class FallAnalysisResult:
    """Complete result of fall analysis."""

    # Primary classification
    fall_detected: bool
    confidence: float  # 0-100 scale

    # Activity context
    activity_type: str  # 'rhythmic_motion', 'controlled_descent', 'general_motion', etc.

    # Fall details (if detected)
    fall_type: Optional[FallType] = None
    phase_timeline: List[PhaseSegment] = field(default_factory=list)

    # Characteristics that contributed to detection
    characteristics: List[str] = field(default_factory=list)

    # Quantitative metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Timestamp of key events
    fall_initiation_time_s: Optional[float] = None
    impact_time_s: Optional[float] = None

    # Time-series data for visualization
    timeline_data: Optional[TimelineData] = None

    # Near-fall events (if any detected)
    near_falls: List[Dict[str, Any]] = field(default_factory=list)

    # Raw data references (for debugging)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'fall_detected': self.fall_detected,
            'confidence': self.confidence,
            'activity_type': self.activity_type,
            'characteristics': self.characteristics,
            'metrics': self.metrics,
        }

        if self.fall_type:
            result['fall_type'] = self.fall_type.value

        if self.fall_initiation_time_s is not None:
            result['fall_initiation_time_s'] = self.fall_initiation_time_s

        if self.impact_time_s is not None:
            result['impact_time_s'] = self.impact_time_s

        if self.phase_timeline:
            result['phase_timeline'] = [
                {
                    'phase': seg.phase.value,
                    'frame_start': seg.frame_start,
                    'frame_end': seg.frame_end,
                    'time_start_s': seg.time_start_s,
                    'time_end_s': seg.time_end_s,
                    'confidence': seg.confidence,
                }
                for seg in self.phase_timeline
            ]

        if self.timeline_data:
            result['timeline_data'] = {
                'time_pos': self.timeline_data.time_pos,
                'time_vel': self.timeline_data.time_vel,
                'time_acc': self.timeline_data.time_acc,
                'pelvis_height_m': self.timeline_data.pelvis_height_m,
                'head_height_m': self.timeline_data.head_height_m,
                'pelvis_vertical_velocity_ms': self.timeline_data.pelvis_vertical_velocity_ms,
                'com_speed_ms': self.timeline_data.com_speed_ms,
                'acc_mag_g': self.timeline_data.acc_mag_g,
                'trunk_tilt_deg': self.timeline_data.trunk_tilt_deg,
                'impact_occurred': self.timeline_data.impact_occurred,
                'impact_frame': self.timeline_data.impact_frame,
                'impact_time_s': self.timeline_data.impact_time_s,
                'impact_acc_g': self.timeline_data.impact_acc_g,
            }

            if self.timeline_data.margin_of_stability_m:
                result['timeline_data']['margin_of_stability_m'] = \
                    self.timeline_data.margin_of_stability_m

        if self.near_falls:
            result['near_falls'] = self.near_falls

        return result


class BaseDetector(ABC):
    """
    Abstract base class for fall detection backends.

    All detector implementations should inherit from this class and
    implement the analyze() method.
    """

    def __init__(self, config: DetectorConfig = None):
        """
        Initialize detector with configuration.

        Args:
            config: Detector configuration. Uses default if not provided.
        """
        self.config = config or DEFAULT_CONFIG

    @abstractmethod
    def analyze(self, filepath: str) -> FallAnalysisResult:
        """
        Analyze a C3D file for fall detection.

        Args:
            filepath: Path to C3D file

        Returns:
            FallAnalysisResult with detection results
        """
        pass

    @abstractmethod
    def analyze_data(
        self,
        marker_positions: np.ndarray,
        marker_labels: List[str],
        frame_rate: float
    ) -> FallAnalysisResult:
        """
        Analyze pre-loaded motion data for fall detection.

        Args:
            marker_positions: Marker positions (N_frames, N_markers, 3)
            marker_labels: List of marker names
            frame_rate: Frame rate in Hz

        Returns:
            FallAnalysisResult with detection results
        """
        pass

    def _classify_activity(
        self,
        com_velocity: np.ndarray,
        vertical_positions: np.ndarray,
        frame_rate: float
    ) -> str:
        """
        Classify the type of activity based on motion patterns.

        Args:
            com_velocity: CoM velocity in m/s (N-1, 3)
            vertical_positions: Vertical position of CoM (N,)
            frame_rate: Frame rate in Hz

        Returns:
            Activity type string
        """
        cfg = self.config

        # Calculate speeds
        speeds = np.linalg.norm(com_velocity, axis=1)
        vertical_velocity = np.diff(vertical_positions) * frame_rate

        # Convert to m/s if in mm
        if cfg.units == 'mm':
            vertical_velocity = vertical_velocity / 1000.0

        # Check for static periods
        static_ratio = np.sum(speeds < cfg.static_speed_threshold_ms) / len(speeds)

        if static_ratio > cfg.static_ratio_threshold:
            # Mostly static - check if lying down or standing
            height_range = np.max(vertical_positions) - np.min(vertical_positions)
            final_height = vertical_positions[-1]
            initial_height = vertical_positions[0]

            # Convert to meters for comparison
            if cfg.units == 'mm':
                height_range = height_range / 1000.0
                height_diff = (initial_height - final_height) / 1000.0
            else:
                height_diff = initial_height - final_height

            # Check for slow descent to lying
            min_vert_vel = np.min(vertical_velocity) if len(vertical_velocity) > 0 else 0
            sustained_descent = np.sum(vertical_velocity < -0.1) / frame_rate

            if height_range > 0.8 and height_diff > 0.6:
                if min_vert_vel > -0.6 and sustained_descent > 1.0:
                    return "meditation_lying"

            if height_range < 0.3:
                return "static_low_position" if final_height < initial_height * 0.5 else "static_standing"

        # Check for rhythmic patterns (dance, exercise)
        if len(com_velocity) > 50:
            speed_signal = np.linalg.norm(com_velocity, axis=1)
            fft = np.fft.fft(speed_signal)
            freqs = np.fft.fftfreq(len(speed_signal), 1 / frame_rate)

            # Look for power in dance/exercise frequency band
            band = (freqs > cfg.rhythmic_motion_freq_min_hz) & \
                   (freqs < cfg.rhythmic_motion_freq_max_hz)

            if np.any(band):
                power_in_band = np.abs(fft[band]).max()
                total_power = np.abs(fft).max()

                if total_power > 0 and (power_in_band / total_power > cfg.rhythmic_motion_power_ratio):
                    return "rhythmic_motion"

        # Check for controlled descent
        if len(vertical_velocity) > 0:
            min_vel = np.min(vertical_velocity)
            if cfg.thresholds.vel_down_controlled_ms > min_vel > cfg.thresholds.vel_down_fall_ms:
                descent_duration = np.sum(vertical_velocity < -0.1) / frame_rate
                if descent_duration > 0.5:
                    return "controlled_descent"

        return "general_motion"

    def _detect_rhythmic_motion(
        self,
        velocities: np.ndarray,
        frame_rate: float
    ) -> bool:
        """
        Detect rhythmic motion patterns using FFT analysis.

        Args:
            velocities: Velocity vectors (N, 3)
            frame_rate: Frame rate in Hz

        Returns:
            True if rhythmic motion detected
        """
        if len(velocities) <= 50:
            return False

        cfg = self.config
        speed_signal = np.linalg.norm(velocities, axis=1)

        fft = np.fft.fft(speed_signal)
        freqs = np.fft.fftfreq(len(speed_signal), 1 / frame_rate)

        band = (freqs > cfg.rhythmic_motion_freq_min_hz) & \
               (freqs < cfg.rhythmic_motion_freq_max_hz)

        if not np.any(band):
            return False

        power_in_band = float(np.abs(fft[band]).max())
        total_power = float(np.abs(fft).max())

        return total_power > 0 and (power_in_band / total_power > cfg.rhythmic_motion_power_ratio)
