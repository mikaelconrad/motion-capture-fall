"""
Near-fall (stumble/recovery) detector.

Detects near-falls based on the presence of compensatory mechanisms that
indicate a loss of balance that was successfully recovered.

A near-fall is defined as an event where:
1. Balance is perturbed (similar to fall initiation)
2. Compensatory mechanisms are activated
3. Recovery is successful (person doesn't fall)

According to Maidan et al., a near-fall requires at least 2 of 5
compensatory mechanisms to be observed.

References:
- Maidan et al. (2014). Introducing a new definition of a near fall
- Srygley et al. (2009). Self-report of missteps in older adults
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from ..config import DetectorConfig, DEFAULT_CONFIG
from ..features import MotionFeatures


class CompensatoryMechanism:
    """Types of compensatory mechanisms that indicate near-fall."""
    ARM_FLAILING = "arm_flailing"
    RECOVERY_STEP = "recovery_step"
    TRUNK_COUNTER_ROTATION = "trunk_counter_rotation"
    COM_LOWERING = "com_lowering"
    SUDDEN_SPEED_CHANGE = "sudden_speed_change"


@dataclass
class NearFallEvent:
    """A detected near-fall event."""
    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float
    severity: float  # 0-1, how close to actual fall
    confidence: float  # Detection confidence
    compensatory_mechanisms: List[str]
    peak_frame: int  # Frame of maximum instability
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            'frame_start': self.frame_start,
            'frame_end': self.frame_end,
            'time_start_s': round(self.time_start_s, 3),
            'time_end_s': round(self.time_end_s, 3),
            'severity': round(self.severity, 2),
            'confidence': round(self.confidence, 2),
            'compensatory_mechanisms': self.compensatory_mechanisms,
            'peak_frame': self.peak_frame,
            'notes': self.notes,
        }


@dataclass
class NearFallDetectorConfig:
    """Configuration for near-fall detection thresholds."""
    # Minimum compensatory mechanisms required
    min_mechanisms: int = 2

    # Arm flailing detection
    arm_angular_velocity_threshold: float = 200.0  # deg/s

    # Recovery step detection
    step_velocity_threshold: float = 0.5  # m/s sudden BoS change

    # Trunk counter-rotation
    trunk_angular_velocity_threshold: float = 80.0  # deg/s

    # CoM lowering
    com_lowering_threshold: float = -0.3  # m/s vertical velocity
    com_lowering_duration_min: float = 0.1  # seconds

    # Speed change
    speed_change_threshold: float = 0.5  # m/s change

    # Window parameters
    detection_window_s: float = 1.0  # seconds
    min_event_duration_s: float = 0.2
    max_event_duration_s: float = 2.0

    # Recovery verification
    recovery_window_s: float = 1.5  # Time to verify recovery after event


class NearFallDetector:
    """
    Detect near-falls (recovered losses of balance).

    Uses a sliding window approach to identify periods where
    compensatory mechanisms are activated, then verifies that
    recovery was successful (no fall occurred).
    """

    def __init__(
        self,
        config: DetectorConfig = None,
        near_fall_config: NearFallDetectorConfig = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.nf_config = near_fall_config or NearFallDetectorConfig()

    def detect(self, features: MotionFeatures) -> List[NearFallEvent]:
        """
        Detect near-fall events in motion sequence.

        Args:
            features: Extracted motion features

        Returns:
            List of detected near-fall events
        """
        events = []
        frame_rate = features.frame_rate
        n_frames = features.n_frames

        window_frames = int(self.nf_config.detection_window_s * frame_rate)
        step_frames = max(1, window_frames // 4)

        # Slide window across sequence
        for start in range(0, n_frames - window_frames, step_frames):
            end = start + window_frames

            # Check for compensatory mechanisms in this window
            mechanisms = self._detect_mechanisms(features, start, end)

            # Need at least min_mechanisms to be a near-fall candidate
            if len(mechanisms) >= self.nf_config.min_mechanisms:
                # Verify recovery (person didn't fall)
                if self._verify_recovery(features, end):
                    # Calculate severity and peak
                    severity, peak_frame = self._calculate_severity(features, start, end)

                    # Refine event boundaries
                    event_start, event_end = self._refine_boundaries(
                        features, start, end, mechanisms
                    )

                    # Check duration
                    duration = (event_end - event_start) / frame_rate
                    if (self.nf_config.min_event_duration_s <= duration <=
                        self.nf_config.max_event_duration_s):

                        event = NearFallEvent(
                            frame_start=event_start,
                            frame_end=event_end,
                            time_start_s=event_start / frame_rate,
                            time_end_s=event_end / frame_rate,
                            severity=severity,
                            confidence=len(mechanisms) / 5.0,  # Max 5 mechanisms
                            compensatory_mechanisms=mechanisms,
                            peak_frame=peak_frame,
                            notes=f"Detected {len(mechanisms)} compensatory mechanisms"
                        )

                        # Avoid overlapping events
                        if not self._overlaps_existing(event, events):
                            events.append(event)

        return events

    def _detect_mechanisms(
        self,
        features: MotionFeatures,
        start: int,
        end: int
    ) -> List[str]:
        """Detect which compensatory mechanisms are present in window."""
        mechanisms = []

        # 1. Check for sudden speed change
        if self._detect_speed_change(features, start, end):
            mechanisms.append(CompensatoryMechanism.SUDDEN_SPEED_CHANGE)

        # 2. Check for CoM lowering
        if self._detect_com_lowering(features, start, end):
            mechanisms.append(CompensatoryMechanism.COM_LOWERING)

        # 3. Check for trunk counter-rotation
        if self._detect_trunk_rotation(features, start, end):
            mechanisms.append(CompensatoryMechanism.TRUNK_COUNTER_ROTATION)

        # Note: Arm flailing and recovery step detection require additional
        # marker data (arm segments, foot positions) that may not be available
        # in all datasets. These could be added when such data is present.

        return mechanisms

    def _detect_speed_change(
        self,
        features: MotionFeatures,
        start: int,
        end: int
    ) -> bool:
        """Detect sudden change in movement speed."""
        if len(features.com_speed) == 0:
            return False

        speed_start = max(0, start - 1)
        speed_end = min(len(features.com_speed), end - 1)

        if speed_end <= speed_start:
            return False

        speeds = features.com_speed[speed_start:speed_end]
        speed_diff = np.diff(speeds)

        # Look for sudden acceleration or deceleration
        max_change = np.max(np.abs(speed_diff)) * features.frame_rate
        return max_change > self.nf_config.speed_change_threshold

    def _detect_com_lowering(
        self,
        features: MotionFeatures,
        start: int,
        end: int
    ) -> bool:
        """Detect rapid lowering of center of mass."""
        if len(features.vertical_velocity) == 0:
            return False

        vel_start = max(0, start - 1)
        vel_end = min(len(features.vertical_velocity), end - 1)

        if vel_end <= vel_start:
            return False

        vert_vel = features.vertical_velocity[vel_start:vel_end]

        # Check for sustained downward velocity
        below_threshold = vert_vel < self.nf_config.com_lowering_threshold
        min_frames = int(self.nf_config.com_lowering_duration_min * features.frame_rate)

        # Count consecutive frames below threshold
        count = 0
        max_count = 0
        for below in below_threshold:
            if below:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0

        return max_count >= min_frames

    def _detect_trunk_rotation(
        self,
        features: MotionFeatures,
        start: int,
        end: int
    ) -> bool:
        """Detect rapid trunk rotation (counter-balance attempt)."""
        if len(features.trunk_angular_velocity) == 0:
            return False

        ang_vel_start = max(0, start - 1)
        ang_vel_end = min(len(features.trunk_angular_velocity), end - 1)

        if ang_vel_end <= ang_vel_start:
            return False

        ang_vel = features.trunk_angular_velocity[ang_vel_start:ang_vel_end]
        max_rotation = np.max(np.abs(ang_vel))

        return max_rotation > self.nf_config.trunk_angular_velocity_threshold

    def _verify_recovery(self, features: MotionFeatures, event_end: int) -> bool:
        """
        Verify that recovery occurred (person didn't fall).

        Checks that after the event, the person returns to stable posture.
        """
        frame_rate = features.frame_rate
        recovery_frames = int(self.nf_config.recovery_window_s * frame_rate)
        check_start = event_end
        check_end = min(len(features.trunk_tilt_deg), event_end + recovery_frames)

        if check_end <= check_start:
            return True  # Can't verify, assume recovery

        # Check trunk returns to upright
        trunk_tilt = features.trunk_tilt_deg[check_start:check_end]
        final_tilt = np.mean(trunk_tilt[-min(10, len(trunk_tilt)):])

        # Check speed stabilizes
        if check_end - 1 < len(features.com_speed):
            speed_end = min(len(features.com_speed), check_end - 1)
            final_speeds = features.com_speed[check_start:speed_end]
            if len(final_speeds) > 0:
                speed_stable = np.std(final_speeds[-min(10, len(final_speeds)):]) < 0.3
            else:
                speed_stable = True
        else:
            speed_stable = True

        # Recovery criteria: relatively upright and stable
        return final_tilt < 45 and speed_stable

    def _calculate_severity(
        self,
        features: MotionFeatures,
        start: int,
        end: int
    ) -> Tuple[float, int]:
        """
        Calculate severity of near-fall (how close to actual fall).

        Returns (severity_score, peak_frame)
        """
        scores = []

        # Trunk tilt contribution
        if len(features.trunk_tilt_deg) > 0:
            tilt = features.trunk_tilt_deg[start:min(end, len(features.trunk_tilt_deg))]
            if len(tilt) > 0:
                max_tilt = np.max(tilt)
                tilt_score = min(max_tilt / 90.0, 1.0)
                scores.append(tilt_score)

        # Vertical velocity contribution
        if len(features.vertical_velocity) > 0:
            vel_start = max(0, start - 1)
            vel_end = min(len(features.vertical_velocity), end - 1)
            if vel_end > vel_start:
                vert_vel = features.vertical_velocity[vel_start:vel_end]
                min_vel = np.min(vert_vel)
                vel_score = min(abs(min_vel) / 1.5, 1.0)  # 1.5 m/s is severe
                scores.append(vel_score)

        # Speed change contribution
        if len(features.com_speed) > 0:
            speed_start = max(0, start - 1)
            speed_end = min(len(features.com_speed), end - 1)
            if speed_end > speed_start:
                speeds = features.com_speed[speed_start:speed_end]
                speed_range = np.max(speeds) - np.min(speeds)
                speed_score = min(speed_range / 2.0, 1.0)
                scores.append(speed_score)

        severity = np.mean(scores) if scores else 0.5

        # Find peak frame (maximum instability)
        peak_frame = start
        if len(features.trunk_tilt_deg) > start:
            tilt = features.trunk_tilt_deg[start:min(end, len(features.trunk_tilt_deg))]
            if len(tilt) > 0:
                peak_frame = start + np.argmax(tilt)

        return float(severity), int(peak_frame)

    def _refine_boundaries(
        self,
        features: MotionFeatures,
        start: int,
        end: int,
        mechanisms: List[str]
    ) -> Tuple[int, int]:
        """Refine event boundaries based on detected mechanisms."""
        # For now, use the window boundaries
        # Could be improved by looking for specific onset/offset markers
        return start, end

    def _overlaps_existing(
        self,
        event: NearFallEvent,
        existing: List[NearFallEvent]
    ) -> bool:
        """Check if event overlaps with already detected events."""
        for existing_event in existing:
            if (event.frame_start <= existing_event.frame_end and
                event.frame_end >= existing_event.frame_start):
                return True
        return False


def detect_near_falls(
    features: MotionFeatures,
    config: DetectorConfig = None
) -> List[NearFallEvent]:
    """
    Convenience function for near-fall detection.

    Args:
        features: Extracted motion features
        config: Detector configuration

    Returns:
        List of detected near-fall events
    """
    detector = NearFallDetector(config)
    return detector.detect(features)
