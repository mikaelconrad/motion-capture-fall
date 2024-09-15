"""
Unit tests for biomechanics module.

Tests the core biomechanical calculations including:
- Center of Mass calculation
- Extrapolated Center of Mass
- Base of Support
- Margin of Stability
- Trunk tilt
- Impact detection
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.biomechanics import (
    calculate_segment_com,
    calculate_geometric_com,
    calculate_whole_body_com,
    calculate_xcom,
    calculate_base_of_support,
    calculate_margin_of_stability,
    calculate_trunk_tilt,
    calculate_trunk_angular_velocity,
    calculate_com_velocity,
    calculate_com_acceleration,
    calculate_impact_zscore,
    detect_freefall,
    find_marker_indices,
)
from src.config import BodyModelConfig


class TestFindMarkerIndices:
    """Tests for marker index finding."""

    def test_find_single_marker(self):
        labels = ['head', 'pelvis', 'lfoot', 'rfoot']
        indices = find_marker_indices(labels, ['head'])
        assert indices == [0]

    def test_find_multiple_markers(self):
        labels = ['head', 'pelvis', 'lfoot', 'rfoot']
        indices = find_marker_indices(labels, ['foot'])
        assert set(indices) == {2, 3}

    def test_case_insensitive(self):
        labels = ['HEAD', 'Pelvis', 'LFoot']
        indices = find_marker_indices(labels, ['head', 'pelvis'])
        assert set(indices) == {0, 1}

    def test_no_match(self):
        labels = ['head', 'pelvis']
        indices = find_marker_indices(labels, ['knee'])
        assert indices == []

    def test_partial_match(self):
        labels = ['r_shoulder_marker', 'l_shoulder_marker']
        indices = find_marker_indices(labels, ['shoulder'])
        assert len(indices) == 2


class TestSegmentCom:
    """Tests for segment CoM calculation."""

    def test_midpoint(self):
        proximal = np.array([0, 0, 0])
        distal = np.array([0, 0, 1])
        com = calculate_segment_com(proximal, distal, 0.5)
        np.testing.assert_array_almost_equal(com, [0, 0, 0.5])

    def test_custom_ratio(self):
        proximal = np.array([0, 0, 0])
        distal = np.array([0, 0, 1])
        com = calculate_segment_com(proximal, distal, 0.3)
        np.testing.assert_array_almost_equal(com, [0, 0, 0.3])

    def test_time_series(self):
        proximal = np.array([[0, 0, 0], [0, 0, 0]])
        distal = np.array([[0, 0, 1], [0, 0, 2]])
        com = calculate_segment_com(proximal, distal, 0.5)
        expected = np.array([[0, 0, 0.5], [0, 0, 1.0]])
        np.testing.assert_array_almost_equal(com, expected)


class TestGeometricCom:
    """Tests for geometric center calculation."""

    def test_simple_case(self):
        # 2 frames, 3 markers, 3 coordinates
        markers = np.array([
            [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
        ])
        com = calculate_geometric_com(markers)
        expected = np.array([[0.5, 1/3, 0], [0.5, 1/3, 0]])
        np.testing.assert_array_almost_equal(com, expected)

    def test_single_marker(self):
        markers = np.array([[[1, 2, 3]]])
        com = calculate_geometric_com(markers)
        np.testing.assert_array_almost_equal(com, [[1, 2, 3]])


class TestXcom:
    """Tests for Extrapolated Center of Mass."""

    def test_stationary(self):
        """XCoM equals CoM when velocity is zero."""
        com_pos = np.array([[0, 1, 0], [0, 1, 0]])
        com_vel = np.array([[0, 0, 0]])
        xcom = calculate_xcom(com_pos, com_vel, leg_length=1.0, vertical_axis=1)
        # XCoM should equal horizontal CoM components when velocity is 0
        np.testing.assert_array_almost_equal(xcom[:1], [[0, 0]])

    def test_forward_velocity(self):
        """XCoM is ahead of CoM when moving forward."""
        com_pos = np.array([[0, 1, 0], [0.1, 1, 0]])
        com_vel = np.array([[1, 0, 0]])  # 1 m/s in X direction
        xcom = calculate_xcom(com_pos, com_vel, leg_length=1.0, vertical_axis=1)
        # XCoM should be ahead of CoM in X direction
        assert xcom[0, 0] > com_pos[0, 0]


class TestBaseOfSupport:
    """Tests for Base of Support calculation."""

    def test_triangle(self):
        foot_markers = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0, 1],
        ])
        bos = calculate_base_of_support(foot_markers, vertical_axis=1)
        # Should return convex hull vertices
        assert len(bos) == 3

    def test_two_points(self):
        """Falls back to bounding box with < 3 points."""
        foot_markers = np.array([
            [0, 0, 0],
            [1, 0, 0],
        ])
        bos = calculate_base_of_support(foot_markers, vertical_axis=1)
        assert len(bos) >= 2


class TestMarginOfStability:
    """Tests for Margin of Stability calculation."""

    def test_stable_center(self):
        """Point at center of BoS should have positive MoS."""
        xcom = np.array([0.0, 0.0])
        bos = np.array([
            [-0.2, -0.2],
            [0.2, -0.2],
            [0.2, 0.2],
            [-0.2, 0.2],
        ])
        mos = calculate_margin_of_stability(xcom, bos)
        assert mos > 0

    def test_unstable_outside(self):
        """Point outside BoS should have negative MoS."""
        xcom = np.array([0.5, 0.0])
        bos = np.array([
            [-0.2, -0.2],
            [0.2, -0.2],
            [0.2, 0.2],
            [-0.2, 0.2],
        ])
        mos = calculate_margin_of_stability(xcom, bos)
        assert mos < 0

    def test_on_boundary(self):
        """Point on boundary should have MoS near zero."""
        xcom = np.array([0.2, 0.0])
        bos = np.array([
            [-0.2, -0.2],
            [0.2, -0.2],
            [0.2, 0.2],
            [-0.2, 0.2],
        ])
        mos = calculate_margin_of_stability(xcom, bos)
        assert abs(mos) < 0.01


class TestTrunkTilt:
    """Tests for trunk tilt calculation."""

    def test_upright(self):
        """Upright trunk should have ~0 degree tilt."""
        upper = np.array([[0, 2, 0]])
        lower = np.array([[0, 1, 0]])
        tilt = calculate_trunk_tilt(upper, lower, vertical_axis=1)
        assert tilt[0] < 5  # Within 5 degrees of vertical

    def test_horizontal(self):
        """Horizontal trunk should have ~90 degree tilt."""
        upper = np.array([[1, 1, 0]])
        lower = np.array([[0, 1, 0]])
        tilt = calculate_trunk_tilt(upper, lower, vertical_axis=1)
        assert 85 < tilt[0] < 95  # Near 90 degrees

    def test_45_degrees(self):
        """45 degree trunk tilt."""
        upper = np.array([[1, 2, 0]])
        lower = np.array([[0, 1, 0]])
        tilt = calculate_trunk_tilt(upper, lower, vertical_axis=1)
        assert 40 < tilt[0] < 50


class TestTrunkAngularVelocity:
    """Tests for trunk angular velocity calculation."""

    def test_constant_angle(self):
        """No change in angle should give zero velocity."""
        tilt = np.array([45.0, 45.0, 45.0])
        ang_vel = calculate_trunk_angular_velocity(tilt, frame_rate=50)
        np.testing.assert_array_almost_equal(ang_vel, [0, 0])

    def test_rotation(self):
        """Rotating trunk should have non-zero velocity."""
        tilt = np.array([0.0, 45.0, 90.0])
        ang_vel = calculate_trunk_angular_velocity(tilt, frame_rate=50)
        assert np.all(ang_vel > 0)


class TestComVelocity:
    """Tests for CoM velocity calculation."""

    def test_stationary(self):
        com_pos = np.array([[0, 0, 0], [0, 0, 0]])
        vel = calculate_com_velocity(com_pos, frame_rate=50, units='m')
        np.testing.assert_array_almost_equal(vel, [[0, 0, 0]])

    def test_constant_velocity_mm(self):
        """Moving 50mm per frame at 50Hz = 2.5 m/s."""
        com_pos = np.array([[0, 0, 0], [50, 0, 0]])
        vel = calculate_com_velocity(com_pos, frame_rate=50, units='mm')
        assert vel[0, 0] == pytest.approx(2.5, rel=0.01)


class TestImpactZscore:
    """Tests for impact detection."""

    def test_no_impact(self):
        """Constant acceleration should not be detected as impact."""
        acc = np.ones((100, 3)) * 9.81  # 1g constant
        detected, z, idx = calculate_impact_zscore(acc)
        assert not detected  # z-score should be near 0

    def test_clear_impact(self):
        """Sharp spike should be detected."""
        acc = np.ones((100, 3)) * 9.81
        acc[50, :] = 50 * 9.81  # 50g spike
        detected, z, idx = calculate_impact_zscore(acc)
        assert detected
        assert idx == 50
        assert z > 3


class TestFreefall:
    """Tests for freefall detection."""

    def test_no_freefall(self):
        """Normal acceleration (~1g) should not be freefall."""
        acc = np.ones((100, 3)) * 9.81 / np.sqrt(3)  # ~1g magnitude
        detected, frames, mask = detect_freefall(acc)
        assert not detected

    def test_freefall_detected(self):
        """Near-zero acceleration should be detected as freefall."""
        acc = np.ones((100, 3)) * 0.1  # Very low acceleration
        detected, frames, mask = detect_freefall(acc, threshold_g=0.6)
        assert detected
        assert frames > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
