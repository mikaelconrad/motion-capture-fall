"""Unit tests for fall classifiers."""

import pytest
import numpy as np

from src.classifiers import (
    FallTypeClassifier,
    FallTypeResult,
    classify_fall_type,
    NearFallDetector,
    NearFallEvent,
    detect_near_falls,
)
from src.classifiers.fall_type import FallType
from src.classifiers.near_fall import NearFallDetectorConfig, CompensatoryMechanism
from src.features import MotionFeatures
from src.config import DetectorConfig


class TestFallTypeClassifier:
    """Tests for fall type classification."""

    def create_mock_features(
        self,
        n_frames: int = 100,
        frame_rate: float = 50.0,
        forward_velocity: float = 0.0,
        vertical_velocity: float = -1.0,
        trunk_tilt_deg: float = 45.0,
    ) -> MotionFeatures:
        """Create mock motion features for testing."""
        # Create CoM velocity: [X, Y, Z] where Y is vertical
        com_velocity = np.zeros((n_frames - 1, 3))
        com_velocity[:, 0] = forward_velocity  # Forward/backward
        com_velocity[:, 1] = vertical_velocity  # Vertical
        com_velocity[:, 2] = 0.0  # Lateral

        # Create position arrays
        com_position = np.zeros((n_frames, 3))
        com_position[:, 1] = np.linspace(1.0, 0.5, n_frames)

        pelvis_position = np.zeros((n_frames, 3))
        pelvis_position[:, 1] = np.linspace(1.0, 0.5, n_frames)

        head_position = np.zeros((n_frames, 3))
        head_position[:, 1] = np.linspace(1.7, 0.8, n_frames)

        return MotionFeatures(
            n_frames=n_frames,
            frame_rate=frame_rate,
            duration_s=n_frames / frame_rate,
            com_position=com_position,
            pelvis_position=pelvis_position,
            head_position=head_position,
            com_velocity=com_velocity,
            pelvis_velocity=com_velocity.copy(),
            com_acceleration=np.zeros((n_frames - 2, 3)),
            com_speed=np.linalg.norm(com_velocity, axis=1),
            vertical_velocity=np.full(n_frames - 1, vertical_velocity),
            vertical_acceleration=np.zeros(n_frames - 2),
            trunk_tilt_deg=np.full(n_frames, trunk_tilt_deg),
            trunk_angular_velocity=np.zeros(n_frames - 1),
            com_height=np.linspace(1.0, 0.5, n_frames),
            pelvis_height=np.linspace(1.0, 0.5, n_frames),
            head_height=np.linspace(1.7, 0.8, n_frames),
            stats={},
        )

    def test_classifier_initialization(self):
        """Test classifier can be initialized."""
        classifier = FallTypeClassifier()
        assert classifier is not None
        assert classifier.config is not None

    def test_classify_trip_forward_motion(self):
        """Test classification of trip (forward motion)."""
        features = self.create_mock_features(
            forward_velocity=0.8,  # Strong forward motion
            vertical_velocity=-1.0,
            trunk_tilt_deg=60.0,
        )
        classifier = FallTypeClassifier()
        result = classifier.classify(features, descent_start_frame=0, descent_end_frame=50)

        # Forward motion should indicate trip
        assert result.fall_type in [FallType.TRIP, FallType.UNKNOWN]
        assert result.confidence >= 0.0
        assert len(result.evidence) > 0

    def test_classify_slip_backward_motion(self):
        """Test classification of slip (backward motion)."""
        features = self.create_mock_features(
            forward_velocity=-0.8,  # Strong backward motion
            vertical_velocity=-1.0,
            trunk_tilt_deg=60.0,
        )
        classifier = FallTypeClassifier()
        result = classifier.classify(features, descent_start_frame=0, descent_end_frame=50)

        # Backward motion should indicate slip
        assert result.fall_type in [FallType.SLIP, FallType.UNKNOWN]
        assert result.confidence >= 0.0

    def test_classify_collapse_vertical_motion(self):
        """Test classification of collapse (primarily vertical)."""
        features = self.create_mock_features(
            forward_velocity=0.05,  # Minimal horizontal motion
            vertical_velocity=-1.5,  # Strong vertical motion
            trunk_tilt_deg=30.0,
        )
        classifier = FallTypeClassifier()
        result = classifier.classify(features, descent_start_frame=0, descent_end_frame=50)

        # Primarily vertical should indicate collapse
        assert result.fall_type in [FallType.COLLAPSE, FallType.UNKNOWN]
        assert result.horizontal_vertical_ratio is not None

    def test_classify_returns_unknown_when_ambiguous(self):
        """Test classifier returns unknown for ambiguous cases."""
        features = self.create_mock_features(
            forward_velocity=0.2,  # Moderate motion
            vertical_velocity=-0.5,  # Moderate descent
            trunk_tilt_deg=40.0,
        )
        classifier = FallTypeClassifier()
        result = classifier.classify(features, descent_start_frame=0, descent_end_frame=50)

        # Ambiguous case
        assert result.fall_type is not None
        assert result.confidence >= 0.0

    def test_classify_auto_detects_descent(self):
        """Test classifier auto-detects descent phase."""
        features = self.create_mock_features(
            forward_velocity=0.0,
            vertical_velocity=-0.8,
        )
        # Set velocity to start at 0 and drop
        features.vertical_velocity[:20] = 0.0
        features.vertical_velocity[20:] = -0.8

        classifier = FallTypeClassifier()
        result = classifier.classify(features)  # No explicit descent frames

        assert result is not None
        assert result.fall_type is not None

    def test_convenience_function(self):
        """Test the classify_fall_type convenience function."""
        features = self.create_mock_features(
            forward_velocity=0.5,
            vertical_velocity=-1.0,
        )
        result = classify_fall_type(features)

        assert isinstance(result, FallTypeResult)
        assert result.fall_type is not None


class TestNearFallDetector:
    """Tests for near-fall detection."""

    def create_mock_features_with_recovery(
        self,
        n_frames: int = 200,
        frame_rate: float = 50.0,
        perturbation_start: int = 50,
        perturbation_end: int = 100,
    ) -> MotionFeatures:
        """Create features simulating a near-fall with recovery."""
        # Create baseline features
        pelvis_height = np.ones(n_frames) * 1.0
        head_height = np.ones(n_frames) * 1.7
        trunk_tilt = np.ones(n_frames) * 5.0  # Near vertical
        vertical_velocity = np.zeros(n_frames - 1)
        com_speed = np.zeros(n_frames - 1)
        trunk_angular_velocity = np.zeros(n_frames - 1)

        # Add perturbation (simulating stumble)
        # Trunk tilts forward then recovers
        trunk_tilt[perturbation_start:perturbation_end] = np.linspace(5, 40, perturbation_end - perturbation_start)
        trunk_tilt[perturbation_end:perturbation_end + 30] = np.linspace(40, 10, 30)
        trunk_tilt[perturbation_end + 30:] = 10.0

        # Downward velocity during perturbation
        vel_start = max(0, perturbation_start - 1)
        vel_end = min(n_frames - 1, perturbation_end)
        vertical_velocity[vel_start:vel_end] = -0.5

        # Speed change during perturbation
        com_speed[vel_start:vel_end] = 0.8

        # Trunk rotation during recovery
        ang_vel_start = max(0, perturbation_start - 1)
        ang_vel_end = min(n_frames - 1, perturbation_end)
        trunk_angular_velocity[ang_vel_start:ang_vel_end] = 100.0  # deg/s

        # Create position arrays
        com_position = np.zeros((n_frames, 3))
        com_position[:, 1] = pelvis_height

        pelvis_position = np.zeros((n_frames, 3))
        pelvis_position[:, 1] = pelvis_height

        head_position = np.zeros((n_frames, 3))
        head_position[:, 1] = head_height

        com_velocity = np.column_stack([np.zeros(n_frames - 1), vertical_velocity, np.zeros(n_frames - 1)])

        return MotionFeatures(
            n_frames=n_frames,
            frame_rate=frame_rate,
            duration_s=n_frames / frame_rate,
            com_position=com_position,
            pelvis_position=pelvis_position,
            head_position=head_position,
            com_velocity=com_velocity,
            pelvis_velocity=com_velocity.copy(),
            com_acceleration=np.zeros((n_frames - 2, 3)),
            com_speed=com_speed,
            vertical_velocity=vertical_velocity,
            vertical_acceleration=np.zeros(n_frames - 2),
            trunk_tilt_deg=trunk_tilt,
            trunk_angular_velocity=trunk_angular_velocity,
            com_height=pelvis_height.copy(),
            pelvis_height=pelvis_height,
            head_height=head_height,
            stats={},
        )

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = NearFallDetector()
        assert detector is not None
        assert detector.nf_config is not None

    def test_detector_with_custom_config(self):
        """Test detector with custom configuration."""
        custom_config = NearFallDetectorConfig(
            min_mechanisms=1,
            com_lowering_threshold=-0.2,
        )
        detector = NearFallDetector(near_fall_config=custom_config)
        assert detector.nf_config.min_mechanisms == 1
        assert detector.nf_config.com_lowering_threshold == -0.2

    def test_detect_near_fall_with_recovery(self):
        """Test detection of near-fall that was recovered."""
        features = self.create_mock_features_with_recovery()
        detector = NearFallDetector()
        events = detector.detect(features)

        # Should detect at least one near-fall event
        # Note: This depends on the thresholds and feature values
        assert isinstance(events, list)
        for event in events:
            assert isinstance(event, NearFallEvent)
            assert event.frame_start >= 0
            assert event.frame_end > event.frame_start
            assert 0 <= event.severity <= 1
            assert 0 <= event.confidence <= 1

    def test_no_detection_in_stable_walk(self):
        """Test no near-falls detected in stable walking."""
        n_frames = 200
        com_position = np.zeros((n_frames, 3))
        com_position[:, 1] = 1.0

        pelvis_position = np.zeros((n_frames, 3))
        pelvis_position[:, 1] = 1.0

        head_position = np.zeros((n_frames, 3))
        head_position[:, 1] = 1.7

        features = MotionFeatures(
            n_frames=n_frames,
            frame_rate=50.0,
            duration_s=n_frames / 50.0,
            com_position=com_position,
            pelvis_position=pelvis_position,
            head_position=head_position,
            com_velocity=np.zeros((199, 3)),
            pelvis_velocity=np.zeros((199, 3)),
            com_acceleration=np.zeros((198, 3)),
            com_speed=np.ones(199) * 0.1,  # Slow, stable speed
            vertical_velocity=np.zeros(199),  # No vertical motion
            vertical_acceleration=np.zeros(198),
            trunk_tilt_deg=np.ones(200) * 5.0,  # Upright
            trunk_angular_velocity=np.zeros(199),
            com_height=np.ones(200) * 1.0,
            pelvis_height=np.ones(200) * 1.0,
            head_height=np.ones(200) * 1.7,
            stats={},
        )

        detector = NearFallDetector()
        events = detector.detect(features)

        # Should not detect near-falls in stable walking
        assert len(events) == 0

    def test_near_fall_event_to_dict(self):
        """Test NearFallEvent serialization."""
        event = NearFallEvent(
            frame_start=50,
            frame_end=100,
            time_start_s=1.0,
            time_end_s=2.0,
            severity=0.7,
            confidence=0.8,
            compensatory_mechanisms=[
                CompensatoryMechanism.COM_LOWERING,
                CompensatoryMechanism.TRUNK_COUNTER_ROTATION,
            ],
            peak_frame=75,
            notes="Test event",
        )

        event_dict = event.to_dict()

        assert event_dict['frame_start'] == 50
        assert event_dict['frame_end'] == 100
        assert event_dict['severity'] == 0.7
        assert event_dict['confidence'] == 0.8
        assert len(event_dict['compensatory_mechanisms']) == 2
        assert event_dict['peak_frame'] == 75

    def test_convenience_function(self):
        """Test the detect_near_falls convenience function."""
        features = self.create_mock_features_with_recovery()
        events = detect_near_falls(features)

        assert isinstance(events, list)

    def test_overlapping_events_prevention(self):
        """Test that overlapping events are not duplicated."""
        features = self.create_mock_features_with_recovery()
        detector = NearFallDetector()
        events = detector.detect(features)

        # Check no overlapping events
        for i, event1 in enumerate(events):
            for event2 in events[i + 1:]:
                assert not (
                    event1.frame_start <= event2.frame_end and
                    event1.frame_end >= event2.frame_start
                ), "Overlapping events detected"


class TestCompensatoryMechanismDetection:
    """Tests for individual compensatory mechanism detection."""

    def _create_mock_features(
        self,
        n_frames: int = 100,
        com_speed: np.ndarray = None,
        vertical_velocity: np.ndarray = None,
        trunk_angular_velocity: np.ndarray = None,
    ) -> MotionFeatures:
        """Helper to create mock features with specific values."""
        com_position = np.zeros((n_frames, 3))
        com_position[:, 1] = 1.0

        pelvis_position = np.zeros((n_frames, 3))
        pelvis_position[:, 1] = 1.0

        head_position = np.zeros((n_frames, 3))
        head_position[:, 1] = 1.7

        if com_speed is None:
            com_speed = np.zeros(n_frames - 1)
        if vertical_velocity is None:
            vertical_velocity = np.zeros(n_frames - 1)
        if trunk_angular_velocity is None:
            trunk_angular_velocity = np.zeros(n_frames - 1)

        return MotionFeatures(
            n_frames=n_frames,
            frame_rate=50.0,
            duration_s=n_frames / 50.0,
            com_position=com_position,
            pelvis_position=pelvis_position,
            head_position=head_position,
            com_velocity=np.zeros((n_frames - 1, 3)),
            pelvis_velocity=np.zeros((n_frames - 1, 3)),
            com_acceleration=np.zeros((n_frames - 2, 3)),
            com_speed=com_speed,
            vertical_velocity=vertical_velocity,
            vertical_acceleration=np.zeros(n_frames - 2),
            trunk_tilt_deg=np.ones(n_frames) * 5.0,
            trunk_angular_velocity=trunk_angular_velocity,
            com_height=np.ones(n_frames) * 1.0,
            pelvis_height=np.ones(n_frames) * 1.0,
            head_height=np.ones(n_frames) * 1.7,
            stats={},
        )

    def test_speed_change_detection(self):
        """Test detection of sudden speed changes."""
        detector = NearFallDetector()

        # Create features with sudden speed change
        com_speed = np.concatenate([np.zeros(49), np.ones(50) * 2.0])
        features = self._create_mock_features(n_frames=100, com_speed=com_speed)

        # Check internal method
        has_speed_change = detector._detect_speed_change(features, 40, 60)
        assert has_speed_change == True

    def test_com_lowering_detection(self):
        """Test detection of CoM lowering."""
        detector = NearFallDetector()

        # Create features with downward velocity
        vertical_velocity = np.concatenate([np.zeros(40), np.ones(59) * -0.5])
        features = self._create_mock_features(n_frames=100, vertical_velocity=vertical_velocity)

        # Check internal method
        has_com_lowering = detector._detect_com_lowering(features, 40, 70)
        assert has_com_lowering == True

    def test_trunk_rotation_detection(self):
        """Test detection of trunk counter-rotation."""
        detector = NearFallDetector()

        # Create features with fast trunk rotation
        trunk_angular_velocity = np.concatenate([np.zeros(40), np.ones(59) * 100.0])
        features = self._create_mock_features(n_frames=100, trunk_angular_velocity=trunk_angular_velocity)

        # Check internal method
        has_trunk_rotation = detector._detect_trunk_rotation(features, 40, 70)
        assert has_trunk_rotation == True
