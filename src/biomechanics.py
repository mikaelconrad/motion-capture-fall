"""
Biomechanical calculations for fall detection.

This module provides functions for calculating:
- Center of Mass (CoM) using mass-weighted segmental model
- Extrapolated Center of Mass (XCoM) for dynamic balance
- Base of Support (BoS) from foot markers
- Margin of Stability (MoS) for balance assessment
- Trunk orientation and angular velocity

References:
- Dempster, W.T. (1955). Space requirements of the seated operator
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement
- Hof, A.L. et al. (2005). The condition for dynamic stability. J Biomech
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import BodyModelConfig, DetectorConfig, DEFAULT_CONFIG


def find_marker_indices(labels: List[str], patterns: List[str]) -> List[int]:
    """
    Find marker indices matching any of the given patterns.

    Args:
        labels: List of marker labels from C3D file
        patterns: List of substring patterns to match (case-insensitive)

    Returns:
        List of indices where labels match any pattern
    """
    indices = []
    labels_lower = [lbl.strip().lower() for lbl in labels]
    patterns_lower = [p.lower() for p in patterns]

    for i, label in enumerate(labels_lower):
        for pattern in patterns_lower:
            if pattern in label:
                indices.append(i)
                break

    return indices


def calculate_segment_com(
    proximal: np.ndarray,
    distal: np.ndarray,
    com_ratio: float = 0.5
) -> np.ndarray:
    """
    Calculate segment center of mass as weighted point between joints.

    The CoM is located at a fixed fraction of the segment length from
    the proximal end, based on anthropometric data.

    Args:
        proximal: Proximal joint position (N, 3) or (3,)
        distal: Distal joint position (N, 3) or (3,)
        com_ratio: Fraction of segment length from proximal end (default 0.5)

    Returns:
        Segment CoM position with same shape as input
    """
    return proximal + com_ratio * (distal - proximal)


def calculate_whole_body_com(
    marker_positions: np.ndarray,
    marker_labels: List[str],
    config: BodyModelConfig = None,
    fallback_to_geometric: bool = True
) -> np.ndarray:
    """
    Calculate mass-weighted whole-body center of mass.

    Uses Dempster's anthropometric model when segment markers are available,
    falls back to geometric mean (average of all markers) when they aren't.

    Args:
        marker_positions: Marker positions (N_frames, N_markers, 3)
        marker_labels: List of marker names
        config: Body model configuration
        fallback_to_geometric: If True, use geometric mean when segments not found

    Returns:
        Whole-body CoM trajectory (N_frames, 3)
    """
    if config is None:
        config = BodyModelConfig()

    n_frames = marker_positions.shape[0]

    # Try to find key segments
    patterns = config.segment_marker_patterns

    # Find marker indices for each segment
    pelvis_idx = find_marker_indices(marker_labels, patterns.get('pelvis', []))
    trunk_upper_idx = find_marker_indices(marker_labels, patterns.get('trunk_upper', []))
    head_idx = find_marker_indices(marker_labels, patterns.get('head', []))

    # Check if we have enough markers for mass-weighted calculation
    has_core_segments = len(pelvis_idx) > 0 and (len(trunk_upper_idx) > 0 or len(head_idx) > 0)

    if not has_core_segments and fallback_to_geometric:
        # Fallback to geometric mean
        return np.mean(marker_positions, axis=1)

    # Build weighted CoM from available segments
    total_mass = 0.0
    weighted_sum = np.zeros((n_frames, 3))

    mass_ratios = config.segment_mass_ratios
    com_ratios = config.segment_com_ratios

    # Pelvis/trunk (largest mass contribution)
    if pelvis_idx:
        pelvis_pos = np.mean(marker_positions[:, pelvis_idx, :], axis=1)

        if trunk_upper_idx:
            trunk_upper_pos = np.mean(marker_positions[:, trunk_upper_idx, :], axis=1)
            trunk_com = calculate_segment_com(
                pelvis_pos, trunk_upper_pos, com_ratios.get('trunk', 0.5)
            )
            trunk_mass = mass_ratios.get('trunk', 0.497)
            weighted_sum += trunk_mass * trunk_com
            total_mass += trunk_mass

        # Add pelvis contribution (approximated)
        pelvis_mass = 0.142  # Approximate pelvis segment mass
        weighted_sum += pelvis_mass * pelvis_pos
        total_mass += pelvis_mass

    # Head
    if head_idx:
        head_pos = np.mean(marker_positions[:, head_idx, :], axis=1)
        head_mass = mass_ratios.get('head_neck', 0.081)
        weighted_sum += head_mass * head_pos
        total_mass += head_mass

    # Arms (if available)
    for side in ['r', 'l']:
        shoulder_idx = find_marker_indices(marker_labels, patterns.get(f'shoulder_{side}', []))
        elbow_idx = find_marker_indices(marker_labels, patterns.get(f'elbow_{side}', []))
        wrist_idx = find_marker_indices(marker_labels, patterns.get(f'wrist_{side}', []))

        if shoulder_idx and elbow_idx:
            shoulder_pos = np.mean(marker_positions[:, shoulder_idx, :], axis=1)
            elbow_pos = np.mean(marker_positions[:, elbow_idx, :], axis=1)
            upper_arm_com = calculate_segment_com(
                shoulder_pos, elbow_pos, com_ratios.get('upper_arm', 0.436)
            )
            arm_mass = mass_ratios.get('upper_arm', 0.028)
            weighted_sum += arm_mass * upper_arm_com
            total_mass += arm_mass

        if elbow_idx and wrist_idx:
            elbow_pos = np.mean(marker_positions[:, elbow_idx, :], axis=1)
            wrist_pos = np.mean(marker_positions[:, wrist_idx, :], axis=1)
            forearm_com = calculate_segment_com(
                elbow_pos, wrist_pos, com_ratios.get('forearm', 0.430)
            )
            forearm_mass = mass_ratios.get('forearm', 0.016)
            weighted_sum += forearm_mass * forearm_com
            total_mass += forearm_mass

    # Legs (if available)
    for side in ['r', 'l']:
        hip_idx = find_marker_indices(marker_labels, patterns.get(f'hip_{side}', []))
        knee_idx = find_marker_indices(marker_labels, patterns.get(f'knee_{side}', []))
        ankle_idx = find_marker_indices(marker_labels, patterns.get(f'ankle_{side}', []))

        if not hip_idx and pelvis_idx:
            # Use pelvis as hip approximation
            hip_idx = pelvis_idx

        if hip_idx and knee_idx:
            hip_pos = np.mean(marker_positions[:, hip_idx, :], axis=1)
            knee_pos = np.mean(marker_positions[:, knee_idx, :], axis=1)
            thigh_com = calculate_segment_com(
                hip_pos, knee_pos, com_ratios.get('thigh', 0.433)
            )
            thigh_mass = mass_ratios.get('thigh', 0.100)
            weighted_sum += thigh_mass * thigh_com
            total_mass += thigh_mass

        if knee_idx and ankle_idx:
            knee_pos = np.mean(marker_positions[:, knee_idx, :], axis=1)
            ankle_pos = np.mean(marker_positions[:, ankle_idx, :], axis=1)
            shank_com = calculate_segment_com(
                knee_pos, ankle_pos, com_ratios.get('shank', 0.433)
            )
            shank_mass = mass_ratios.get('shank', 0.047)
            weighted_sum += shank_mass * shank_com
            total_mass += shank_mass

    if total_mass > 0:
        return weighted_sum / total_mass
    elif fallback_to_geometric:
        return np.mean(marker_positions, axis=1)
    else:
        raise ValueError("Could not calculate CoM: insufficient markers and fallback disabled")


def calculate_geometric_com(marker_positions: np.ndarray) -> np.ndarray:
    """
    Calculate simple geometric center of all markers.

    This is the fallback method when segment information is unavailable.

    Args:
        marker_positions: Marker positions (N_frames, N_markers, 3)

    Returns:
        Geometric center trajectory (N_frames, 3)
    """
    return np.mean(marker_positions, axis=1)


def calculate_xcom(
    com_pos: np.ndarray,
    com_vel: np.ndarray,
    leg_length: float,
    vertical_axis: int = 1
) -> np.ndarray:
    """
    Calculate Extrapolated Center of Mass (XCoM).

    The XCoM accounts for the body's momentum and predicts where the CoM
    will be if no corrective action is taken. This is fundamental for
    assessing dynamic stability.

    XCoM = CoM + CoM_velocity / ω₀
    where ω₀ = sqrt(g/l) is the eigenfrequency of an inverted pendulum

    Reference: Hof, A.L. et al. (2005). J Biomech, 38(1), 1-8.

    Args:
        com_pos: Center of mass positions (N, 3) in meters
        com_vel: Center of mass velocities (N-1, 3) or (N, 3) in m/s
        leg_length: Leg length in meters
        vertical_axis: Index of vertical axis (0=X, 1=Y, 2=Z)

    Returns:
        Extrapolated CoM in horizontal plane (N, 2)
    """
    g = 9.81  # m/s²
    omega_0 = np.sqrt(g / leg_length)

    # Get horizontal axes
    horiz_axes = [i for i in range(3) if i != vertical_axis]

    com_horiz = com_pos[:, horiz_axes]

    # Handle velocity array length mismatch
    if com_vel.shape[0] < com_pos.shape[0]:
        # Pad velocity with last value
        vel_padded = np.vstack([com_vel[:, horiz_axes],
                                com_vel[-1:, horiz_axes]])
    else:
        vel_padded = com_vel[:, horiz_axes]

    xcom = com_horiz + vel_padded / omega_0
    return xcom


def calculate_base_of_support(
    foot_markers: np.ndarray,
    vertical_axis: int = 1
) -> np.ndarray:
    """
    Calculate convex hull of foot contact points as Base of Support.

    The BoS is the polygon formed by the outer boundary of all ground
    contact points. For standing, this is typically the area under and
    between the feet.

    Args:
        foot_markers: Foot marker positions (N_markers, 3) for single frame
        vertical_axis: Index of vertical axis

    Returns:
        Vertices of BoS polygon in horizontal plane (M, 2)
    """
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        raise ImportError("scipy required for BoS calculation")

    # Get horizontal components
    horiz_axes = [i for i in range(3) if i != vertical_axis]
    foot_horiz = foot_markers[:, horiz_axes]

    if len(foot_horiz) < 3:
        # Not enough points for convex hull, return bounding box
        return np.array([
            [foot_horiz[:, 0].min(), foot_horiz[:, 1].min()],
            [foot_horiz[:, 0].max(), foot_horiz[:, 1].min()],
            [foot_horiz[:, 0].max(), foot_horiz[:, 1].max()],
            [foot_horiz[:, 0].min(), foot_horiz[:, 1].max()],
        ])

    try:
        hull = ConvexHull(foot_horiz)
        return foot_horiz[hull.vertices]
    except Exception:
        # Degenerate case (collinear points)
        return np.array([
            [foot_horiz[:, 0].min(), foot_horiz[:, 1].min()],
            [foot_horiz[:, 0].max(), foot_horiz[:, 1].max()],
        ])


def calculate_margin_of_stability(
    xcom: np.ndarray,
    bos_polygon: np.ndarray
) -> float:
    """
    Calculate Margin of Stability (MoS).

    MoS is the minimum distance from the XCoM to the BoS boundary.
    - MoS > 0: Stable (XCoM inside BoS)
    - MoS < 0: Unstable (XCoM outside BoS, recovery action needed)
    - MoS << 0: Fall likely imminent

    Args:
        xcom: Extrapolated CoM position in horizontal plane (2,)
        bos_polygon: Vertices of Base of Support polygon (N, 2)

    Returns:
        Signed distance: positive if inside, negative if outside
    """
    try:
        from shapely.geometry import Point, Polygon
    except ImportError:
        # Fallback to simple bounding box check
        return _margin_of_stability_simple(xcom, bos_polygon)

    point = Point(xcom)
    polygon = Polygon(bos_polygon)

    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # Fix invalid polygon

    if polygon.contains(point):
        return float(point.distance(polygon.boundary))
    else:
        return -float(point.distance(polygon.boundary))


def _margin_of_stability_simple(
    xcom: np.ndarray,
    bos_polygon: np.ndarray
) -> float:
    """
    Simple MoS calculation without shapely (bounding box only).
    """
    x_min, y_min = bos_polygon.min(axis=0)
    x_max, y_max = bos_polygon.max(axis=0)

    x, y = xcom

    # Check if inside bounding box
    inside_x = x_min <= x <= x_max
    inside_y = y_min <= y <= y_max

    if inside_x and inside_y:
        # Distance to nearest edge
        return min(x - x_min, x_max - x, y - y_min, y_max - y)
    else:
        # Distance to nearest edge (negative)
        dx = max(x_min - x, 0, x - x_max)
        dy = max(y_min - y, 0, y - y_max)
        return -np.sqrt(dx**2 + dy**2)


def calculate_trunk_tilt(
    upper_trunk_pos: np.ndarray,
    lower_trunk_pos: np.ndarray,
    vertical_axis: int = 1
) -> np.ndarray:
    """
    Calculate trunk tilt angle from vertical.

    Args:
        upper_trunk_pos: Upper trunk (sternum/chest) position (N, 3)
        lower_trunk_pos: Lower trunk (pelvis/sacrum) position (N, 3)
        vertical_axis: Index of vertical axis

    Returns:
        Trunk tilt angle in degrees (N,)
        0° = perfectly upright, 90° = horizontal
    """
    trunk_vec = upper_trunk_pos - lower_trunk_pos

    # Create vertical reference vector
    vertical = np.zeros(3)
    vertical[vertical_axis] = 1.0

    # Calculate angle from vertical
    trunk_norm = np.linalg.norm(trunk_vec, axis=1, keepdims=True)
    trunk_norm = np.maximum(trunk_norm, 1e-9)  # Avoid division by zero

    trunk_unit = trunk_vec / trunk_norm
    cos_theta = np.abs(trunk_unit @ vertical)  # Absolute value for angle from vertical
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    tilt_rad = np.arccos(cos_theta)
    tilt_deg = np.degrees(tilt_rad)

    return tilt_deg.flatten()


def calculate_trunk_angular_velocity(
    trunk_tilt_deg: np.ndarray,
    frame_rate: float
) -> np.ndarray:
    """
    Calculate trunk angular velocity from tilt angle time series.

    Args:
        trunk_tilt_deg: Trunk tilt angle in degrees (N,)
        frame_rate: Frame rate in Hz

    Returns:
        Angular velocity in degrees per second (N-1,)
    """
    dt = 1.0 / frame_rate
    angular_vel = np.diff(trunk_tilt_deg) / dt
    return angular_vel


def calculate_com_velocity(
    com_pos: np.ndarray,
    frame_rate: float,
    units: str = 'mm'
) -> np.ndarray:
    """
    Calculate CoM velocity from position time series.

    Args:
        com_pos: CoM positions (N, 3)
        frame_rate: Frame rate in Hz
        units: Input units ('mm' or 'm')

    Returns:
        CoM velocity in m/s (N-1, 3)
    """
    dt = 1.0 / frame_rate
    vel = np.diff(com_pos, axis=0) / dt

    if units == 'mm':
        vel = vel / 1000.0  # Convert mm/s to m/s

    return vel


def calculate_com_acceleration(
    com_vel: np.ndarray,
    frame_rate: float
) -> np.ndarray:
    """
    Calculate CoM acceleration from velocity time series.

    Args:
        com_vel: CoM velocities in m/s (N, 3)
        frame_rate: Frame rate in Hz

    Returns:
        CoM acceleration in m/s² (N-1, 3)
    """
    dt = 1.0 / frame_rate
    acc = np.diff(com_vel, axis=0) / dt
    return acc


def calculate_impact_zscore(acceleration: np.ndarray) -> Tuple[bool, float, int]:
    """
    Detect impact as statistical outlier in acceleration magnitude.

    Uses z-score to identify peaks that are significantly above the mean,
    which is more robust than absolute thresholds.

    Args:
        acceleration: Acceleration vectors in m/s² (N, 3)

    Returns:
        Tuple of (impact_detected, peak_zscore, peak_frame_index)
    """
    if acceleration.size == 0:
        return False, 0.0, 0

    acc_mag = np.linalg.norm(acceleration, axis=1)
    mu = float(np.mean(acc_mag))
    sigma = float(np.std(acc_mag))

    if sigma < 1e-6:
        return False, 0.0, 0

    peak_idx = int(np.argmax(acc_mag))
    peak_z = float((acc_mag[peak_idx] - mu) / sigma)

    impact_detected = peak_z >= 3.0  # 3 standard deviations

    return impact_detected, peak_z, peak_idx


def detect_freefall(
    acceleration: np.ndarray,
    threshold_g: float = 0.6
) -> Tuple[bool, float, np.ndarray]:
    """
    Detect freefall phase where acceleration magnitude drops toward 0g.

    During freefall, the body experiences near-weightlessness as it
    accelerates under gravity alone.

    Args:
        acceleration: Acceleration vectors in m/s² (N, 3)
        threshold_g: Threshold below which is considered freefall (in g)

    Returns:
        Tuple of (freefall_detected, freefall_duration_s, freefall_mask)
    """
    g = 9.81
    acc_mag = np.linalg.norm(acceleration, axis=1)
    acc_g = acc_mag / g

    freefall_mask = acc_g <= threshold_g
    freefall_detected = np.any(freefall_mask)

    # Duration would need frame_rate, return frame count instead
    freefall_frames = np.sum(freefall_mask)

    return freefall_detected, float(freefall_frames), freefall_mask
