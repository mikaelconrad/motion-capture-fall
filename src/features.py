"""
Feature extraction for fall detection.

This module provides utilities for extracting biomechanically meaningful
features from motion capture data for use in fall detection algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import DetectorConfig, DEFAULT_CONFIG
from .biomechanics import (
    calculate_whole_body_com,
    calculate_geometric_com,
    calculate_trunk_tilt,
    calculate_trunk_angular_velocity,
    calculate_com_velocity,
    calculate_com_acceleration,
    calculate_impact_zscore,
    calculate_xcom,
    calculate_base_of_support,
    calculate_margin_of_stability,
    find_marker_indices,
)


@dataclass
class MotionFeatures:
    """
    Extracted motion features for fall detection.

    Contains all biomechanically relevant features computed from
    raw marker data.
    """

    # Metadata
    n_frames: int
    frame_rate: float
    duration_s: float

    # Position features (in meters)
    com_position: np.ndarray           # (N, 3)
    pelvis_position: np.ndarray        # (N, 3)
    head_position: np.ndarray          # (N, 3)

    # Velocity features (in m/s)
    com_velocity: np.ndarray           # (N-1, 3)
    pelvis_velocity: np.ndarray        # (N-1, 3)

    # Acceleration features (in m/s²)
    com_acceleration: np.ndarray       # (N-2, 3)

    # Derived scalar features
    com_speed: np.ndarray              # (N-1,) magnitude of velocity
    vertical_velocity: np.ndarray      # (N-1,) vertical component
    vertical_acceleration: np.ndarray  # (N-2,)

    # Trunk features
    trunk_tilt_deg: np.ndarray         # (N,)
    trunk_angular_velocity: np.ndarray # (N-1,)

    # Height features (meters)
    com_height: np.ndarray             # (N,)
    pelvis_height: np.ndarray          # (N,)
    head_height: np.ndarray            # (N,)

    # Stability features (optional, requires foot markers)
    margin_of_stability: Optional[np.ndarray] = None  # (N,)
    xcom: Optional[np.ndarray] = None                 # (N, 2)

    # Summary statistics
    stats: Dict[str, float] = field(default_factory=dict)

    def compute_stats(self) -> Dict[str, float]:
        """Compute summary statistics from time series."""
        stats = {}

        # Velocity stats
        if len(self.vertical_velocity) > 0:
            stats['min_vertical_velocity_ms'] = float(np.min(self.vertical_velocity))
            stats['max_vertical_velocity_ms'] = float(np.max(self.vertical_velocity))
            stats['mean_speed_ms'] = float(np.mean(self.com_speed))

        # Height stats
        stats['initial_com_height_m'] = float(self.com_height[0])
        stats['final_com_height_m'] = float(self.com_height[-1])
        stats['min_com_height_m'] = float(np.min(self.com_height))
        stats['max_com_height_m'] = float(np.max(self.com_height))
        stats['height_drop_m'] = float(np.max(self.com_height) - np.min(self.com_height))

        # Using percentiles for robustness
        stats['height_p90_m'] = float(np.percentile(self.pelvis_height, 90))
        stats['height_p10_m'] = float(np.percentile(self.pelvis_height, 10))
        stats['height_drop_robust_m'] = stats['height_p90_m'] - stats['height_p10_m']

        # Final posture
        win = max(1, int(self.frame_rate))
        stats['head_final_m'] = float(np.median(self.head_height[-win:]))
        stats['pelvis_final_m'] = float(np.median(self.pelvis_height[-win:]))
        stats['head_pelvis_delta_m'] = stats['head_final_m'] - stats['pelvis_final_m']

        # Trunk stats
        if len(self.trunk_tilt_deg) > 0:
            stats['trunk_tilt_max_deg'] = float(np.max(self.trunk_tilt_deg))
            stats['trunk_tilt_final_deg'] = float(np.median(self.trunk_tilt_deg[-win:]))

        if len(self.trunk_angular_velocity) > 0:
            stats['trunk_angular_velocity_max'] = float(np.max(np.abs(self.trunk_angular_velocity)))

        # Acceleration stats
        if len(self.com_acceleration) > 0:
            acc_mag = np.linalg.norm(self.com_acceleration, axis=1)
            stats['acc_max_g'] = float(np.max(acc_mag) / 9.81)
            stats['acc_mean_g'] = float(np.mean(acc_mag) / 9.81)

            # Impact z-score
            impact_detected, impact_z, impact_idx = calculate_impact_zscore(self.com_acceleration)
            stats['impact_z_score'] = impact_z
            stats['impact_detected'] = float(impact_detected)
            stats['impact_frame'] = float(impact_idx)

        # Descent duration
        if len(self.vertical_velocity) > 0:
            descent_mask = self.vertical_velocity < -0.1
            stats['descent_duration_s'] = float(np.sum(descent_mask) / self.frame_rate)
            stats['descent_frames'] = float(np.sum(descent_mask))

        # Static ratio
        if len(self.com_speed) > 0:
            static_mask = self.com_speed < 0.05  # < 5 cm/s
            stats['static_ratio'] = float(np.mean(static_mask))

        # Stability stats
        if self.margin_of_stability is not None:
            stats['mos_min_m'] = float(np.min(self.margin_of_stability))
            stats['mos_mean_m'] = float(np.mean(self.margin_of_stability))
            stats['mos_negative_ratio'] = float(np.mean(self.margin_of_stability < 0))

        self.stats = stats
        return stats


def extract_features(
    marker_positions: np.ndarray,
    marker_labels: List[str],
    frame_rate: float,
    config: DetectorConfig = None
) -> MotionFeatures:
    """
    Extract motion features from marker data.

    Args:
        marker_positions: Marker positions (N_frames, N_markers, 3)
        marker_labels: List of marker names
        frame_rate: Frame rate in Hz
        config: Detector configuration

    Returns:
        MotionFeatures object containing all extracted features
    """
    if config is None:
        config = DEFAULT_CONFIG

    n_frames = marker_positions.shape[0]
    vaxis = config.get_vertical_axis_index()
    patterns = config.body_model.segment_marker_patterns

    # Find anatomical markers
    pelvis_idx = find_marker_indices(marker_labels, patterns.get('pelvis', []))
    head_idx = find_marker_indices(marker_labels, patterns.get('head', []))
    trunk_upper_idx = find_marker_indices(marker_labels, patterns.get('trunk_upper', []))
    foot_r_idx = find_marker_indices(marker_labels, patterns.get('foot_r', []))
    foot_l_idx = find_marker_indices(marker_labels, patterns.get('foot_l', []))

    # Calculate positions
    if pelvis_idx:
        pelvis_pos = np.mean(marker_positions[:, pelvis_idx, :], axis=1)
    else:
        pelvis_pos = np.mean(marker_positions, axis=1)

    if head_idx:
        head_pos = np.mean(marker_positions[:, head_idx, :], axis=1)
    else:
        highest_idx = np.argmax(np.mean(marker_positions[:, :, vaxis], axis=0))
        head_pos = marker_positions[:, highest_idx, :]

    # Calculate CoM
    if config.use_mass_weighted_com:
        com_pos = calculate_whole_body_com(marker_positions, marker_labels, config.body_model)
    else:
        com_pos = calculate_geometric_com(marker_positions)

    # Convert to meters if needed
    scale = 1.0 / 1000.0 if config.units == 'mm' else 1.0
    com_pos_m = com_pos * scale
    pelvis_pos_m = pelvis_pos * scale
    head_pos_m = head_pos * scale

    # Calculate velocities
    com_vel = calculate_com_velocity(com_pos, frame_rate, config.units)
    pelvis_vel = calculate_com_velocity(pelvis_pos, frame_rate, config.units)

    # Calculate acceleration
    com_acc = calculate_com_acceleration(com_vel, frame_rate)

    # Calculate trunk tilt
    if head_idx and (pelvis_idx or trunk_upper_idx):
        lower_pos = pelvis_pos_m if pelvis_idx else np.mean(marker_positions[:, trunk_upper_idx, :], axis=1) * scale
        trunk_tilt = calculate_trunk_tilt(head_pos_m, lower_pos, vaxis)
    else:
        trunk_tilt = np.zeros(n_frames)

    trunk_ang_vel = calculate_trunk_angular_velocity(trunk_tilt, frame_rate)

    # Extract height components
    com_height = com_pos_m[:, vaxis]
    pelvis_height = pelvis_pos_m[:, vaxis]
    head_height = head_pos_m[:, vaxis]

    # Vertical velocity and acceleration
    vertical_vel = com_vel[:, vaxis]
    vertical_acc = com_acc[:, vaxis] if com_acc.size else np.array([])

    # Speed (magnitude)
    com_speed = np.linalg.norm(com_vel, axis=1)

    # Calculate MoS if foot markers available
    margin_of_stability = None
    xcom = None

    if (foot_r_idx or foot_l_idx) and config.use_margin_of_stability:
        foot_idx = foot_r_idx + foot_l_idx
        leg_length = config.body_model.default_leg_length_m

        # Calculate XCoM
        xcom = calculate_xcom(com_pos_m, com_vel, leg_length, vaxis)

        # Calculate MoS for each frame
        mos_values = []
        for i in range(n_frames):
            foot_markers = marker_positions[i, foot_idx, :] * scale
            bos = calculate_base_of_support(foot_markers, vaxis)

            if i < len(xcom):
                mos = calculate_margin_of_stability(xcom[i], bos)
            else:
                mos = calculate_margin_of_stability(xcom[-1], bos)
            mos_values.append(mos)

        margin_of_stability = np.array(mos_values)

    features = MotionFeatures(
        n_frames=n_frames,
        frame_rate=frame_rate,
        duration_s=n_frames / frame_rate,
        com_position=com_pos_m,
        pelvis_position=pelvis_pos_m,
        head_position=head_pos_m,
        com_velocity=com_vel,
        pelvis_velocity=pelvis_vel,
        com_acceleration=com_acc,
        com_speed=com_speed,
        vertical_velocity=vertical_vel,
        vertical_acceleration=vertical_acc,
        trunk_tilt_deg=trunk_tilt,
        trunk_angular_velocity=trunk_ang_vel,
        com_height=com_height,
        pelvis_height=pelvis_height,
        head_height=head_height,
        margin_of_stability=margin_of_stability,
        xcom=xcom,
    )

    features.compute_stats()
    return features


def extract_windowed_features(
    features: MotionFeatures,
    window_size_s: float = 1.0,
    step_size_s: float = 0.5
) -> List[Dict[str, float]]:
    """
    Extract features from sliding windows for ML models.

    Args:
        features: MotionFeatures object
        window_size_s: Window size in seconds
        step_size_s: Step size in seconds

    Returns:
        List of feature dictionaries, one per window
    """
    window_frames = int(window_size_s * features.frame_rate)
    step_frames = int(step_size_s * features.frame_rate)

    windowed_features = []

    for start in range(0, features.n_frames - window_frames + 1, step_frames):
        end = start + window_frames
        window_feat = {}

        # Position features
        window_feat['com_height_mean'] = float(np.mean(features.com_height[start:end]))
        window_feat['com_height_std'] = float(np.std(features.com_height[start:end]))
        window_feat['com_height_min'] = float(np.min(features.com_height[start:end]))
        window_feat['com_height_max'] = float(np.max(features.com_height[start:end]))

        # Velocity features (account for shorter arrays)
        vel_start = max(0, start - 1)
        vel_end = min(len(features.vertical_velocity), end - 1)
        if vel_end > vel_start:
            window_feat['vertical_velocity_mean'] = float(np.mean(features.vertical_velocity[vel_start:vel_end]))
            window_feat['vertical_velocity_min'] = float(np.min(features.vertical_velocity[vel_start:vel_end]))
            window_feat['vertical_velocity_max'] = float(np.max(features.vertical_velocity[vel_start:vel_end]))
            window_feat['speed_mean'] = float(np.mean(features.com_speed[vel_start:vel_end]))
            window_feat['speed_max'] = float(np.max(features.com_speed[vel_start:vel_end]))

        # Trunk features
        window_feat['trunk_tilt_mean'] = float(np.mean(features.trunk_tilt_deg[start:end]))
        window_feat['trunk_tilt_max'] = float(np.max(features.trunk_tilt_deg[start:end]))

        # Acceleration features
        acc_start = max(0, start - 2)
        acc_end = min(len(features.com_acceleration), end - 2)
        if acc_end > acc_start:
            acc_mag = np.linalg.norm(features.com_acceleration[acc_start:acc_end], axis=1)
            window_feat['acc_mag_mean'] = float(np.mean(acc_mag) / 9.81)
            window_feat['acc_mag_max'] = float(np.max(acc_mag) / 9.81)

        # MoS features
        if features.margin_of_stability is not None:
            window_feat['mos_mean'] = float(np.mean(features.margin_of_stability[start:end]))
            window_feat['mos_min'] = float(np.min(features.margin_of_stability[start:end]))

        # Temporal position
        window_feat['time_start_s'] = start / features.frame_rate
        window_feat['time_end_s'] = end / features.frame_rate
        window_feat['relative_position'] = start / features.n_frames

        windowed_features.append(window_feat)

    return windowed_features
