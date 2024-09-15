"""
Fall type classifier.

Classifies falls into categories based on kinematic signatures:
- SLIP: Backward fall caused by foot sliding forward
- TRIP: Forward fall caused by swing foot being impeded
- COLLAPSE: Vertical fall due to loss of muscle tone or syncope

References:
- Hsiao & Robinovitch (1998). Common protective movements govern unexpected falls
- Smeesters et al. (2001). The pivoting of slips and trips
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np

from ..config import FallType, DetectorConfig, DEFAULT_CONFIG
from ..features import MotionFeatures


@dataclass
class FallTypeResult:
    """Result of fall type classification."""
    fall_type: FallType
    confidence: float  # 0-1
    evidence: List[str]

    # Detailed metrics
    horizontal_velocity_direction: Optional[Tuple[float, float]] = None  # (X, Z) mean
    horizontal_vertical_ratio: float = 0.0
    trunk_rotation_direction: Optional[str] = None  # 'forward', 'backward', 'lateral'


class FallTypeClassifier:
    """
    Classify fall type based on kinematic features.

    The classification is based on the predominant direction of motion
    during the descent phase, along with trunk orientation changes.

    Slip characteristics:
    - CoM moves backward (negative X if forward is positive)
    - Foot slides forward
    - Often ends with posterior impact (buttocks/back)

    Trip characteristics:
    - CoM moves forward (positive X)
    - Swing foot suddenly stops
    - Often involves forward rotation
    - Hands may reach forward for protection

    Collapse characteristics:
    - Primarily vertical descent
    - Minimal horizontal momentum
    - Often associated with knee buckling
    - May be slower than slip/trip
    """

    def __init__(self, config: DetectorConfig = None):
        self.config = config or DEFAULT_CONFIG

    def classify(
        self,
        features: MotionFeatures,
        descent_start_frame: Optional[int] = None,
        descent_end_frame: Optional[int] = None
    ) -> FallTypeResult:
        """
        Classify the fall type from motion features.

        Args:
            features: Extracted motion features
            descent_start_frame: Start of descent phase (auto-detected if None)
            descent_end_frame: End of descent phase (auto-detected if None)

        Returns:
            FallTypeResult with classification and evidence
        """
        # Auto-detect descent phase if not provided
        if descent_start_frame is None or descent_end_frame is None:
            descent_start_frame, descent_end_frame = self._find_descent_phase(features)

        if descent_end_frame <= descent_start_frame:
            return FallTypeResult(
                fall_type=FallType.UNKNOWN,
                confidence=0.0,
                evidence=["Could not identify descent phase"]
            )

        # Extract descent-phase data
        vel_start = max(0, descent_start_frame - 1)
        vel_end = min(len(features.com_velocity), descent_end_frame)

        if vel_end <= vel_start:
            return FallTypeResult(
                fall_type=FallType.UNKNOWN,
                confidence=0.0,
                evidence=["Insufficient velocity data"]
            )

        com_vel = features.com_velocity[vel_start:vel_end]
        vert_vel = features.vertical_velocity[vel_start:vel_end]

        # Analyze horizontal motion
        horiz_result = self._analyze_horizontal_motion(com_vel, vert_vel)

        # Analyze trunk rotation
        trunk_result = self._analyze_trunk_rotation(
            features.trunk_tilt_deg,
            descent_start_frame,
            descent_end_frame
        )

        # Combine evidence for classification
        return self._make_classification(horiz_result, trunk_result)

    def _find_descent_phase(self, features: MotionFeatures) -> Tuple[int, int]:
        """Auto-detect descent phase from velocity profile."""
        vert_vel = features.vertical_velocity
        threshold = -0.3  # m/s

        # Find first significant downward velocity
        start = 0
        for i, vel in enumerate(vert_vel):
            if vel < threshold:
                start = i
                break

        # Find end (impact or velocity recovery)
        end = len(vert_vel)
        if len(features.com_acceleration) > 0:
            # Use impact as end point
            acc_mag = np.linalg.norm(features.com_acceleration, axis=1)
            impact_frame = np.argmax(acc_mag) + 2
            end = min(impact_frame, end)

        return start, end

    def _analyze_horizontal_motion(
        self,
        com_vel: np.ndarray,
        vert_vel: np.ndarray
    ) -> dict:
        """Analyze horizontal motion during descent."""
        vaxis = self.config.get_vertical_axis_index()
        horiz_axes = self.config.get_horizontal_axes()

        # Get horizontal velocity components
        horiz_vel = com_vel[:, horiz_axes]
        mean_horiz = np.mean(horiz_vel, axis=0)
        horiz_speed = np.linalg.norm(mean_horiz)
        mean_vert = np.mean(vert_vel) if len(vert_vel) > 0 else 0

        # Calculate ratio
        if abs(mean_vert) > 0.01:
            horiz_vert_ratio = horiz_speed / abs(mean_vert)
        else:
            horiz_vert_ratio = float('inf') if horiz_speed > 0.01 else 0

        # Determine direction (assuming first horizontal axis is forward/backward)
        forward_vel = mean_horiz[0] if len(mean_horiz) > 0 else 0
        lateral_vel = mean_horiz[1] if len(mean_horiz) > 1 else 0

        direction = "neutral"
        if abs(forward_vel) > 0.1:
            direction = "forward" if forward_vel > 0 else "backward"
        elif abs(lateral_vel) > 0.1:
            direction = "lateral"

        return {
            'mean_horizontal': mean_horiz.tolist(),
            'horizontal_speed': horiz_speed,
            'mean_vertical': mean_vert,
            'horiz_vert_ratio': horiz_vert_ratio,
            'direction': direction,
            'forward_velocity': forward_vel,
            'lateral_velocity': lateral_vel,
        }

    def _analyze_trunk_rotation(
        self,
        trunk_tilt: np.ndarray,
        start: int,
        end: int
    ) -> dict:
        """Analyze trunk rotation during descent."""
        if len(trunk_tilt) == 0 or end <= start:
            return {
                'tilt_change': 0,
                'max_tilt': 0,
                'rotation_rate': 0,
            }

        start = min(start, len(trunk_tilt) - 1)
        end = min(end, len(trunk_tilt))

        tilt_segment = trunk_tilt[start:end]
        if len(tilt_segment) == 0:
            return {
                'tilt_change': 0,
                'max_tilt': 0,
                'rotation_rate': 0,
            }

        tilt_change = tilt_segment[-1] - tilt_segment[0]
        max_tilt = np.max(tilt_segment)

        # Rotation rate (degrees per second)
        duration = (end - start) / 50.0  # Assume 50 fps if not specified
        rotation_rate = abs(tilt_change) / duration if duration > 0 else 0

        return {
            'tilt_change': float(tilt_change),
            'max_tilt': float(max_tilt),
            'rotation_rate': float(rotation_rate),
        }

    def _make_classification(
        self,
        horiz_result: dict,
        trunk_result: dict
    ) -> FallTypeResult:
        """Make final classification based on analyzed features."""
        evidence = []
        scores = {
            FallType.SLIP: 0.0,
            FallType.TRIP: 0.0,
            FallType.COLLAPSE: 0.0,
        }

        ratio = horiz_result['horiz_vert_ratio']
        direction = horiz_result['direction']
        forward_vel = horiz_result['forward_velocity']
        rotation_rate = trunk_result['rotation_rate']

        # Horizontal motion analysis
        if ratio < 0.3:
            # Primarily vertical - collapse
            scores[FallType.COLLAPSE] += 0.4
            evidence.append(f"Primarily vertical motion (ratio={ratio:.2f})")
        elif ratio > 0.6:
            # Significant horizontal component
            if direction == "backward":
                scores[FallType.SLIP] += 0.4
                evidence.append(f"Backward horizontal motion ({forward_vel:.2f} m/s)")
            elif direction == "forward":
                scores[FallType.TRIP] += 0.4
                evidence.append(f"Forward horizontal motion ({forward_vel:.2f} m/s)")
            else:
                scores[FallType.COLLAPSE] += 0.2
                evidence.append(f"Lateral or ambiguous direction")

        # Trunk rotation analysis
        if rotation_rate > 150:
            # Fast trunk rotation suggests trip or slip
            if direction == "forward":
                scores[FallType.TRIP] += 0.3
                evidence.append(f"Fast forward trunk rotation ({rotation_rate:.0f} deg/s)")
            elif direction == "backward":
                scores[FallType.SLIP] += 0.3
                evidence.append(f"Fast backward trunk rotation ({rotation_rate:.0f} deg/s)")
        elif rotation_rate < 50:
            # Slow rotation more consistent with collapse
            scores[FallType.COLLAPSE] += 0.2
            evidence.append(f"Slow trunk rotation ({rotation_rate:.0f} deg/s)")

        # Max trunk tilt
        if trunk_result['max_tilt'] > 80:
            evidence.append(f"High trunk tilt ({trunk_result['max_tilt']:.0f} deg)")

        # Determine winner
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Calculate confidence
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            confidence = 0.33  # Uniform if no evidence
            best_type = FallType.UNKNOWN

        # Low confidence -> unknown
        if confidence < 0.4:
            return FallTypeResult(
                fall_type=FallType.UNKNOWN,
                confidence=confidence,
                evidence=evidence,
                horizontal_velocity_direction=tuple(horiz_result['mean_horizontal']),
                horizontal_vertical_ratio=ratio,
            )

        return FallTypeResult(
            fall_type=best_type,
            confidence=confidence,
            evidence=evidence,
            horizontal_velocity_direction=tuple(horiz_result['mean_horizontal']),
            horizontal_vertical_ratio=ratio,
            trunk_rotation_direction=direction,
        )


def classify_fall_type(
    features: MotionFeatures,
    descent_start: Optional[int] = None,
    descent_end: Optional[int] = None,
    config: DetectorConfig = None
) -> FallTypeResult:
    """
    Convenience function for fall type classification.

    Args:
        features: Extracted motion features
        descent_start: Start frame of descent phase
        descent_end: End frame of descent phase
        config: Detector configuration

    Returns:
        FallTypeResult with classification
    """
    classifier = FallTypeClassifier(config)
    return classifier.classify(features, descent_start, descent_end)
