"""
Integration tests for the fall detection pipeline.

Tests the complete pipeline on actual C3D files from the test dataset.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.detectors.rules_detector import RulesDetector, analyze_c3d_for_fall
from src.utils.c3d_reader import read_c3d
from src.features import extract_features
from src.config import DetectorConfig


# Ground truth for test files
GROUND_TRUTH = {
    'video-02-epic-fall': True,
    'video-04-short-fall-standing-up-try': True,
    'video-05-turn': False,
    'video-06-under-the-mattress': False,
    'video-09-meditation': False,
    'video-11-dance': False,
}

# Path to test data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'new_data')


def get_test_file_path(video_name: str) -> str:
    """Get path to test C3D file."""
    return os.path.join(DATA_DIR, video_name, f'{video_name}.c3d')


class TestC3DReader:
    """Tests for C3D file reading."""

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-02-epic-fall')),
        reason="Test data not available"
    )
    def test_read_c3d_metadata(self):
        """Test that C3D metadata is correctly read."""
        filepath = get_test_file_path('video-02-epic-fall')
        data = read_c3d(filepath)

        assert data.frame_rate > 0
        assert data.n_frames > 0
        assert data.n_markers > 0
        assert len(data.marker_labels) == data.n_markers
        assert data.marker_positions.shape == (data.n_frames, data.n_markers, 3)

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-02-epic-fall')),
        reason="Test data not available"
    )
    def test_read_c3d_marker_access(self):
        """Test marker access by label."""
        filepath = get_test_file_path('video-02-epic-fall')
        data = read_c3d(filepath)

        # Try to get a marker by pattern
        pelvis_markers = data.get_markers_by_pattern(['pelvis', 'waist', 'sacrum'])
        assert pelvis_markers.shape[0] == data.n_frames


class TestFeatureExtraction:
    """Tests for feature extraction."""

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-02-epic-fall')),
        reason="Test data not available"
    )
    def test_extract_features(self):
        """Test feature extraction from C3D data."""
        filepath = get_test_file_path('video-02-epic-fall')
        data = read_c3d(filepath)

        features = extract_features(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

        assert features.n_frames == data.n_frames
        assert features.frame_rate == data.frame_rate
        assert len(features.com_position) == data.n_frames
        assert len(features.trunk_tilt_deg) == data.n_frames

        # Check stats are computed
        assert 'min_vertical_velocity_ms' in features.stats
        assert 'height_drop_m' in features.stats

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-02-epic-fall')),
        reason="Test data not available"
    )
    def test_fall_features(self):
        """Test that fall video has expected feature patterns."""
        filepath = get_test_file_path('video-02-epic-fall')
        data = read_c3d(filepath)

        features = extract_features(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

        # Fall should have significant height drop
        assert features.stats['height_drop_m'] > 0.5

        # Fall should have rapid downward velocity
        assert features.stats['min_vertical_velocity_ms'] < -0.5


class TestRulesDetector:
    """Tests for the rules-based detector."""

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-02-epic-fall')),
        reason="Test data not available"
    )
    def test_detect_fall(self):
        """Test detection of known fall."""
        filepath = get_test_file_path('video-02-epic-fall')
        detector = RulesDetector()
        result = detector.analyze(filepath)

        assert result.fall_detected == True
        assert result.confidence > 50
        assert len(result.characteristics) > 0
        assert 'height_drop_m' in result.metrics

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-09-meditation')),
        reason="Test data not available"
    )
    def test_detect_non_fall_meditation(self):
        """Test that meditation is not detected as fall."""
        filepath = get_test_file_path('video-09-meditation')
        detector = RulesDetector()
        result = detector.analyze(filepath)

        assert result.fall_detected == False

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-11-dance')),
        reason="Test data not available"
    )
    def test_detect_non_fall_dance(self):
        """Test that dancing is not detected as fall."""
        filepath = get_test_file_path('video-11-dance')
        detector = RulesDetector()
        result = detector.analyze(filepath)

        assert result.fall_detected == False

    @pytest.mark.skipif(
        not os.path.exists(DATA_DIR),
        reason="Test data not available"
    )
    def test_all_ground_truth(self):
        """Test all videos against ground truth."""
        detector = RulesDetector()
        results = {}

        for video_name, expected in GROUND_TRUTH.items():
            filepath = get_test_file_path(video_name)
            if os.path.exists(filepath):
                result = detector.analyze(filepath)
                results[video_name] = {
                    'expected': expected,
                    'predicted': result.fall_detected,
                    'correct': result.fall_detected == expected,
                }

        # Calculate accuracy
        if results:
            correct = sum(1 for r in results.values() if r['correct'])
            accuracy = correct / len(results)
            print(f"\nAccuracy: {accuracy:.1%} ({correct}/{len(results)})")

            for name, r in results.items():
                status = "PASS" if r['correct'] else "FAIL"
                print(f"  {status}: {name} (expected={r['expected']}, predicted={r['predicted']})")

            assert accuracy >= 0.8, f"Accuracy {accuracy:.1%} below threshold"


class TestBackwardCompatibility:
    """Test backward compatibility with original API."""

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-02-epic-fall')),
        reason="Test data not available"
    )
    def test_analyze_c3d_for_fall_function(self):
        """Test the convenience function matches original API."""
        filepath = get_test_file_path('video-02-epic-fall')
        result = analyze_c3d_for_fall(filepath)

        # Check expected keys exist
        assert 'fall_detected' in result
        assert 'confidence' in result
        assert 'activity_type' in result
        assert 'characteristics' in result
        assert 'metrics' in result

        # Check types
        assert isinstance(result['fall_detected'], bool)
        assert isinstance(result['confidence'], (int, float))


class TestTimelineData:
    """Test timeline data generation for visualization."""

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-02-epic-fall')),
        reason="Test data not available"
    )
    def test_timeline_data_present(self):
        """Test that timeline data is generated."""
        filepath = get_test_file_path('video-02-epic-fall')
        detector = RulesDetector()
        result = detector.analyze(filepath)

        assert result.timeline_data is not None
        assert len(result.timeline_data.time_pos) > 0
        assert len(result.timeline_data.pelvis_height_m) > 0
        assert len(result.timeline_data.head_height_m) > 0

    @pytest.mark.skipif(
        not os.path.exists(get_test_file_path('video-02-epic-fall')),
        reason="Test data not available"
    )
    def test_impact_marked_in_fall(self):
        """Test that impact is marked in fall timeline."""
        filepath = get_test_file_path('video-02-epic-fall')
        detector = RulesDetector()
        result = detector.analyze(filepath)

        # Fall should have impact marked
        assert result.timeline_data.impact_occurred == True
        assert result.timeline_data.impact_frame is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
