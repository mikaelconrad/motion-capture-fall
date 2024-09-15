"""
Unified configuration for fall detection system.

This module contains all configurable parameters organized into logical groups.
Thresholds are based on biomechanical literature where available.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class VerticalAxis(Enum):
    """Coordinate system vertical axis."""
    X = 0
    Y = 1
    Z = 2


class DetectorType(Enum):
    """Available detector backends."""
    RULES_V3 = "rules_v3"
    LSTM = "lstm"


class FallType(Enum):
    """Classification of fall types by initiation mechanism."""
    SLIP = "slip"           # Foot slides, backward fall
    TRIP = "trip"           # Swing foot impeded, forward fall
    COLLAPSE = "collapse"   # Vertical descent, knee buckling
    UNKNOWN = "unknown"


class PhaseLabel(Enum):
    """Temporal phases of a fall event."""
    PRE_FALL = "pre_fall"       # Normal activity before balance loss
    INITIATION = "initiation"   # Balance perturbation, compensatory attempts
    DESCENT = "descent"         # Uncontrolled falling (free fall)
    IMPACT = "impact"           # Ground contact
    POST_FALL = "post_fall"     # Lying on ground / recovery


@dataclass
class BodyModelConfig:
    """
    Anthropometric body model configuration.

    Based on Dempster (1955) and Winter (2009) biomechanics tables.
    Mass ratios are fractions of total body mass.
    CoM ratios are distance from proximal end as fraction of segment length.
    """

    # Segment mass as fraction of total body mass
    # Reference: Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement
    segment_mass_ratios: Dict[str, float] = field(default_factory=lambda: {
        'head_neck': 0.081,
        'trunk': 0.497,
        'upper_arm': 0.028,   # Per arm
        'forearm': 0.016,     # Per arm
        'hand': 0.006,        # Per hand
        'thigh': 0.100,       # Per leg
        'shank': 0.047,       # Per leg
        'foot': 0.014,        # Per foot
    })

    # Segment CoM location as fraction of segment length from proximal end
    segment_com_ratios: Dict[str, float] = field(default_factory=lambda: {
        'head_neck': 0.500,
        'trunk': 0.500,
        'upper_arm': 0.436,
        'forearm': 0.430,
        'hand': 0.506,
        'thigh': 0.433,
        'shank': 0.433,
        'foot': 0.500,
    })

    # Marker name patterns for segment identification
    # These are matched case-insensitively as substrings
    segment_marker_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        'head': ['head', 'q_head', 'skull', 'forehead'],
        'neck': ['neck', 'c7'],
        'trunk_upper': ['sternum', 'chest', 'clav', 'acromion'],
        'trunk_lower': ['t10', 't12', 'l3', 'l5'],
        'pelvis': ['pelvis', 'sacrum', 'asis', 'psis', 'waist', 'hip'],
        'shoulder_r': ['rshoulder', 'r_shoulder', 'rsho'],
        'shoulder_l': ['lshoulder', 'l_shoulder', 'lsho'],
        'elbow_r': ['relbow', 'r_elbow', 'relb'],
        'elbow_l': ['lelbow', 'l_elbow', 'lelb'],
        'wrist_r': ['rwrist', 'r_wrist', 'rwri', 'rhand'],
        'wrist_l': ['lwrist', 'l_wrist', 'lwri', 'lhand'],
        'hip_r': ['rhip', 'r_hip', 'rasi', 'rpsi'],
        'hip_l': ['lhip', 'l_hip', 'lasi', 'lpsi'],
        'knee_r': ['rknee', 'r_knee', 'rkne'],
        'knee_l': ['lknee', 'l_knee', 'lkne'],
        'ankle_r': ['rankle', 'r_ankle', 'rank'],
        'ankle_l': ['lankle', 'l_ankle', 'lank'],
        'foot_r': ['rfoot', 'r_foot', 'rtoe', 'rheel', 'rmt'],
        'foot_l': ['lfoot', 'l_foot', 'ltoe', 'lheel', 'lmt'],
    })

    # Default leg length in meters (used for MoS calculation)
    # Should be measured per subject when available
    default_leg_length_m: float = 0.90


@dataclass
class ThresholdConfig:
    """
    Detection thresholds based on biomechanical literature.

    References:
    - Pre-impact fall detection: Hu & Qu (2016), BioMedical Engineering OnLine
    - Real-world falls: Schonnop et al. (2013), BMC Geriatrics
    - Fall kinematics: Hsiao & Robinovitch (1998), J Biomech
    """

    # Vertical velocity thresholds (m/s)
    # Literature: -1.3 m/s cited for pre-impact detection
    # Using slightly less conservative value for better sensitivity
    vel_down_fall_ms: float = -0.8           # Threshold for fall detection
    vel_down_controlled_ms: float = -0.3     # Max for controlled descent

    # Descent duration (seconds)
    # Literature: Real-world falls average 583±255 ms descent duration
    descent_min_s: float = 0.15              # Minimum for fall (not just bump)
    descent_max_s: float = 1.2               # Maximum for fall (vs controlled)
    descent_controlled_s: float = 1.5        # Longer = likely controlled

    # Height drop thresholds (meters)
    height_drop_fall_m: float = 0.5          # Significant height change
    height_drop_hard_fall_m: float = 1.0     # Large height change

    # Trunk tilt thresholds (degrees from vertical)
    # Literature: >90° indicates horizontal orientation
    trunk_tilt_warning_deg: float = 45.0     # Early warning
    trunk_tilt_fall_deg: float = 75.0        # Likely falling
    trunk_tilt_horizontal_deg: float = 85.0  # Nearly horizontal

    # Impact detection (z-score of acceleration magnitude)
    impact_z_min: float = 3.0                # Minimum for impact detection
    impact_z_severe: float = 5.0             # Severe impact

    # Freefall detection (acceleration in g)
    # During freefall, acceleration magnitude approaches 0g
    freefall_g_max: float = 0.6              # Below this = freefall

    # Post-impact rest (m/s)
    post_impact_rest_speed_ms: float = 0.15  # Low motion after impact

    # Margin of Stability thresholds (meters)
    mos_warning_m: float = 0.05              # Getting close to boundary
    mos_critical_m: float = 0.0              # At or past boundary

    # Final posture thresholds (meters)
    head_final_lying_m: float = 0.8          # Head height when lying (generous for different body sizes)
    head_final_hard_fall_m: float = 0.6      # Head height for hard fall path
    head_pelvis_delta_upright_m: float = 0.65 # Min difference when upright (guard threshold)


@dataclass
class DetectorConfig:
    """
    Main configuration for fall detection system.

    Combines all sub-configurations into a single entry point.
    """

    # Coordinate system
    vertical_axis: VerticalAxis = VerticalAxis.Y
    units: str = 'mm'  # Input data units (mm or m)

    # Detection backend
    detector_type: DetectorType = DetectorType.RULES_V3

    # Use advanced biomechanical calculations
    use_mass_weighted_com: bool = True
    use_margin_of_stability: bool = True

    # Sub-configurations
    body_model: BodyModelConfig = field(default_factory=BodyModelConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    # Activity classification
    rhythmic_motion_freq_min_hz: float = 0.5   # Min frequency for dance/exercise
    rhythmic_motion_freq_max_hz: float = 3.0   # Max frequency for dance/exercise
    rhythmic_motion_power_ratio: float = 0.2   # Power ratio threshold

    # Static detection
    static_speed_threshold_ms: float = 0.05    # Below this = static
    static_ratio_threshold: float = 0.3        # Fraction of frames static

    def get_vertical_axis_index(self) -> int:
        """Get the vertical axis as an integer index."""
        return self.vertical_axis.value

    def get_horizontal_axes(self) -> List[int]:
        """Get horizontal axis indices."""
        return [i for i in range(3) if i != self.vertical_axis.value]

    def to_meters(self, value: float) -> float:
        """Convert value from input units to meters."""
        if self.units == 'mm':
            return value / 1000.0
        return value

    def from_meters(self, value: float) -> float:
        """Convert value from meters to input units."""
        if self.units == 'mm':
            return value * 1000.0
        return value


# Default configuration instance
DEFAULT_CONFIG = DetectorConfig()
