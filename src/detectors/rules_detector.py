"""
Rules-based fall detector.

This detector uses biomechanically-informed thresholds and heuristics
to detect falls. It is based on the v3 rules detector with improvements:

- Proper mass-weighted CoM calculation
- Margin of Stability when foot markers available
- Improved threshold values from literature
- Better activity classification

Scientific References:
    Velocity thresholds:
        Hu, X., & Qu, X. (2016). Pre-impact fall detection.
        BioMedical Engineering OnLine, 15, 87.
        DOI: 10.1186/s12938-016-0194-x

    Fall timing (descent duration):
        Schonnop, R., et al. (2013). Prevalence of head impact during falls.
        CMAJ, 185(17), E803-E810. (Real-world falls: 583±255 ms)
        DOI: 10.1503/cmaj.130498

    Fall kinematics and trunk orientation:
        Hsiao, E.T., & Robinovitch, S.N. (1998). Common protective movements.
        Journal of Biomechanics, 31(1), 1-9.
        DOI: 10.1016/S0021-9290(97)00114-0

    Impact detection methodology:
        Bourke, A.K., et al. (2007). Evaluation of threshold-based fall detection.
        Gait & Posture, 26(2), 194-199.
        DOI: 10.1016/j.gaitpost.2006.09.012

See docs/REFERENCES.md for complete bibliography.
See docs/THRESHOLDS.md for threshold justifications.
"""

from typing import List, Optional, Tuple
import numpy as np

from .base import BaseDetector, FallAnalysisResult, TimelineData, PhaseSegment
from ..config import DetectorConfig, FallType, PhaseLabel, DEFAULT_CONFIG
from ..biomechanics import (
    calculate_whole_body_com,
    calculate_geometric_com,
    calculate_trunk_tilt,
    calculate_com_velocity,
    calculate_com_acceleration,
    calculate_impact_zscore,
    find_marker_indices,
)
from ..utils.c3d_reader import read_c3d


class RulesDetector(BaseDetector):
    """
    Rules-based fall detection using biomechanical thresholds.

    Detection is based on multiple criteria:
    1. Rapid vertical descent of pelvis/CoM
    2. Descent duration within fall-like window
    3. Impact acceleration spike
    4. Trunk tilt approaching horizontal
    5. Final posture (head height, head-pelvis delta)
    6. Context (rhythmic motion, controlled descent)

    Multiple fall "paths" are checked:
    - Core fall: rapid descent + impact/trunk tilt + low final head
    - Hard fall: extreme trunk tilt + high impact + large height drop
    - Short fall: low final head + high impact + sufficient height drop
    """

    def __init__(self, config: DetectorConfig = None):
        super().__init__(config)

    def analyze(self, filepath: str) -> FallAnalysisResult:
        """Analyze a C3D file for fall detection."""
        data = read_c3d(filepath)
        return self.analyze_data(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

    def analyze_data(
        self,
        marker_positions: np.ndarray,
        marker_labels: List[str],
        frame_rate: float
    ) -> FallAnalysisResult:
        """Analyze motion data for fall detection."""

        cfg = self.config
        thresholds = cfg.thresholds
        vaxis = cfg.get_vertical_axis_index()

        n_frames = marker_positions.shape[0]

        # Find anatomical markers
        patterns = cfg.body_model.segment_marker_patterns
        pelvis_idx = find_marker_indices(marker_labels, patterns.get('pelvis', []))
        head_idx = find_marker_indices(marker_labels, patterns.get('head', []))
        trunk_upper_idx = find_marker_indices(marker_labels, patterns.get('trunk_upper', []))

        # Calculate pelvis position
        if pelvis_idx:
            pelvis_pos = np.mean(marker_positions[:, pelvis_idx, :], axis=1)
        else:
            pelvis_pos = np.mean(marker_positions, axis=1)

        # Calculate head position
        if head_idx:
            head_pos = np.mean(marker_positions[:, head_idx, :], axis=1)
        else:
            # Fallback: use highest markers
            highest_idx = np.argmax(np.mean(marker_positions[:, :, vaxis], axis=0))
            head_pos = marker_positions[:, highest_idx, :]

        # Calculate CoM
        if cfg.use_mass_weighted_com:
            com_pos = calculate_whole_body_com(
                marker_positions, marker_labels, cfg.body_model
            )
        else:
            com_pos = calculate_geometric_com(marker_positions)

        # Extract vertical positions
        pelvis_h = pelvis_pos[:, vaxis]
        head_h = head_pos[:, vaxis]
        com_h = com_pos[:, vaxis]

        # Calculate velocities (in input units per frame, then convert)
        pelvis_vel = np.diff(pelvis_pos, axis=0)
        com_vel = np.diff(com_pos, axis=0)

        # Convert to m/s
        pelvis_vel_ms = pelvis_vel * (frame_rate / 1000.0) if cfg.units == 'mm' else pelvis_vel * frame_rate
        com_vel_ms = com_vel * (frame_rate / 1000.0) if cfg.units == 'mm' else com_vel * frame_rate

        pelvis_vert_vel_ms = pelvis_vel_ms[:, vaxis]

        # Calculate acceleration
        com_acc = np.diff(com_vel_ms, axis=0) * frame_rate  # m/s²

        # Calculate trunk tilt
        if head_idx and (pelvis_idx or trunk_upper_idx):
            lower_pos = pelvis_pos if pelvis_idx else np.mean(marker_positions[:, trunk_upper_idx, :], axis=1)
            trunk_tilt_deg = calculate_trunk_tilt(head_pos, lower_pos, vaxis)
        else:
            trunk_tilt_deg = np.zeros(n_frames)

        # Final posture analysis (median over ~1 second window)
        win = max(1, int(round(frame_rate)))
        head_final_m = float(np.median(head_h[-win:]) / 1000.0) if cfg.units == 'mm' else float(np.median(head_h[-win:]))
        pelvis_final_m = float(np.median(pelvis_h[-win:]) / 1000.0) if cfg.units == 'mm' else float(np.median(pelvis_h[-win:]))
        head_pelvis_delta = head_final_m - pelvis_final_m

        # Descent analysis
        min_pelvis_vel = float(np.min(pelvis_vert_vel_ms)) if len(pelvis_vert_vel_ms) > 0 else 0.0
        descent_mask = pelvis_vert_vel_ms < -0.1
        descent_dur = float(np.sum(descent_mask) / frame_rate)

        descent_ok = thresholds.descent_min_s <= descent_dur <= thresholds.descent_max_s
        controlled_descent = descent_dur > thresholds.descent_controlled_s

        # Height drop (using percentiles for robustness)
        p90 = float(np.percentile(pelvis_h, 90))
        p10 = float(np.percentile(pelvis_h, 10))
        height_drop_m = (p90 - p10) / 1000.0 if cfg.units == 'mm' else (p90 - p10)

        # Impact detection
        impact_detected, impact_z, impact_idx = calculate_impact_zscore(com_acc)
        impact_hit = impact_z >= thresholds.impact_z_min

        # Trunk metrics
        trunk_tilt_max = float(np.max(trunk_tilt_deg)) if len(trunk_tilt_deg) > 0 else 0.0

        # Post-impact rest check
        com_speed_ms = np.linalg.norm(com_vel_ms, axis=1)
        rest_win = max(1, int(round(frame_rate * 0.6)))
        start = min(max(0, impact_idx + 1), max(0, len(com_speed_ms) - 1))
        end = min(len(com_speed_ms), start + rest_win)
        if end > start:
            post_rest_speed = float(np.mean(com_speed_ms[start:end]))
        else:
            post_rest_speed = float(np.mean(com_speed_ms[-rest_win:])) if len(com_speed_ms) >= rest_win else 0.0

        # Activity classification
        rhythmic = self._detect_rhythmic_motion(com_vel_ms, frame_rate)
        activity_type = self._classify_activity(com_vel_ms, com_h, frame_rate)

        # === Fall Detection Paths ===

        # Core fall path: rapid descent within typical fall duration window
        fall_core = (
            (min_pelvis_vel < thresholds.vel_down_fall_ms) and
            descent_ok and  # Descent duration within 0.2-1.2s window
            (impact_hit or (trunk_tilt_max >= thresholds.trunk_tilt_fall_deg)) and
            (head_final_m < thresholds.head_final_lying_m)
        )

        # Hard fall path: extreme indicators with proper posture change
        # Requires significant head-pelvis separation (person lying down, not standing bent over)
        hard_fall = (
            (trunk_tilt_max >= thresholds.trunk_tilt_horizontal_deg) and
            (impact_z >= thresholds.impact_z_severe) and
            (height_drop_m >= 1.5) and  # Stricter height drop for hard fall
            (head_final_m < thresholds.head_final_hard_fall_m) and
            (abs(head_pelvis_delta) >= 0.2)  # Head and pelvis at different heights (lying down)
        )

        # Short/fast fall path: quick fall with clear impact
        short_fall = (
            (head_final_m < 0.45) and
            (impact_z >= thresholds.impact_z_severe) and
            (height_drop_m >= 1.0) and
            (descent_dur <= 1.8)  # Short descent duration
        )

        # Extended fall path: for falls with longer descent but clear final lying posture
        # Requires very low pelvis AND positive head-pelvis delta (head above pelvis = lying on back/side)
        extended_fall = (
            (min_pelvis_vel < thresholds.vel_down_fall_ms) and
            (impact_z >= thresholds.impact_z_severe) and
            (trunk_tilt_max >= thresholds.trunk_tilt_horizontal_deg) and
            (height_drop_m >= thresholds.height_drop_hard_fall_m) and
            (pelvis_final_m >= 0) and (pelvis_final_m < 0.15) and  # Pelvis near ground, not negative
            (head_pelvis_delta > 0.2)  # Head above pelvis (lying position)
        )

        # Combine paths with context
        if rhythmic:
            # More strict for rhythmic motion - require height drop
            fall = (
                (fall_core and (height_drop_m > thresholds.height_drop_fall_m)) or
                (hard_fall and (height_drop_m > thresholds.height_drop_fall_m)) or
                (short_fall and (height_drop_m > thresholds.height_drop_fall_m))
            )
        else:
            # Normal detection: any path triggers, but controlled descent blocks core path
            fall = (fall_core and not controlled_descent) or hard_fall or short_fall or extended_fall

        # === Guard conditions ===

        # Upright guard: if head remains well above pelvis AND pelvis is high, not a fall
        if fall and head_pelvis_delta > thresholds.head_pelvis_delta_upright_m and pelvis_final_m > 0.5:
            fall = False

        # Similar height guard: head and pelvis at similar heights suggests not lying down
        # This catches cases like bending over, turning, etc.
        if fall and (abs(head_pelvis_delta) < 0.15) and (descent_dur > 2.0):
            fall = False

        # Turn/jump guard: dynamic spike but upright end posture
        if fall and (abs(head_pelvis_delta) < 0.1) and (height_drop_m > 1.2) and (impact_z > 8.0) and not rhythmic and (head_final_m >= 0.45):
            fall = False

        # Post-impact rest guard: no quiescent period after impact in ambiguous cases
        if fall and (abs(head_pelvis_delta) < 0.1) and (descent_dur > 2.5) and (post_rest_speed > 0.3):
            fall = False

        # === Build characteristics list ===
        characteristics = []
        if min_pelvis_vel < thresholds.vel_down_fall_ms:
            characteristics.append('rapid_pelvis_descent')
        if descent_ok:
            characteristics.append('descent_duration_ok')
        if impact_hit:
            characteristics.append('impact_spike')
        if trunk_tilt_max >= thresholds.trunk_tilt_fall_deg:
            characteristics.append('trunk_tilt_high')
        if hard_fall:
            characteristics.append('hard_fall_path')
        if short_fall:
            characteristics.append('short_fall_ext')
        if extended_fall:
            characteristics.append('extended_fall_path')
        if rhythmic:
            characteristics.append('rhythmic_motion')
        if controlled_descent:
            characteristics.append('controlled_descent')

        # === Build metrics ===
        metrics = {
            'min_pelvis_vertical_velocity_ms': round(min_pelvis_vel, 3),
            'descent_duration_s': round(descent_dur, 2),
            'height_drop_m': round(height_drop_m, 3),
            'head_final_height_m': round(head_final_m, 3),
            'pelvis_final_height_m': round(pelvis_final_m, 3),
            'head_pelvis_delta_m': round(head_pelvis_delta, 3),
            'trunk_tilt_max_deg': round(trunk_tilt_max, 1),
            'impact_z_score': round(impact_z, 2),
            'post_impact_rest_speed_ms': round(post_rest_speed, 3),
            'vertical_axis': ['X', 'Y', 'Z'][vaxis],
        }

        # === Build timeline data ===
        time_pos = (np.arange(n_frames) / frame_rate).tolist()
        time_vel = (np.arange(max(0, n_frames - 1)) / frame_rate + 0.5 / frame_rate).tolist()
        time_acc = (np.arange(max(0, n_frames - 2)) / frame_rate + 1.0 / frame_rate).tolist()

        pelvis_h_m = (pelvis_h / 1000.0).tolist() if cfg.units == 'mm' else pelvis_h.tolist()
        head_h_m = (head_h / 1000.0).tolist() if cfg.units == 'mm' else head_h.tolist()

        acc_mag = np.linalg.norm(com_acc, axis=1) if com_acc.size else np.array([])
        acc_mag_g = (acc_mag / 9.81).tolist() if acc_mag.size else []

        timeline_data = TimelineData(
            time_pos=time_pos,
            time_vel=time_vel,
            time_acc=time_acc,
            pelvis_height_m=pelvis_h_m,
            head_height_m=head_h_m,
            pelvis_vertical_velocity_ms=pelvis_vert_vel_ms.tolist() if len(pelvis_vert_vel_ms) else [],
            com_speed_ms=com_speed_ms.tolist() if len(com_speed_ms) else [],
            acc_mag_g=acc_mag_g,
            trunk_tilt_deg=trunk_tilt_deg.tolist(),
            impact_occurred=impact_hit,
            impact_frame=impact_idx if impact_hit else None,
            impact_time_s=(impact_idx / frame_rate) if impact_hit else None,
            impact_acc_g=(acc_mag_g[impact_idx] if impact_hit and impact_idx < len(acc_mag_g) else None),
        )

        # === Calculate confidence ===
        # More nuanced confidence based on evidence strength
        confidence = self._calculate_confidence(
            fall, fall_core, hard_fall, short_fall,
            min_pelvis_vel, impact_z, trunk_tilt_max, height_drop_m,
            rhythmic, controlled_descent
        )

        return FallAnalysisResult(
            fall_detected=fall,
            confidence=confidence,
            activity_type=activity_type,
            characteristics=characteristics,
            metrics=metrics,
            timeline_data=timeline_data,
            impact_time_s=(impact_idx / frame_rate) if impact_hit else None,
        )

    def _calculate_confidence(
        self,
        fall_detected: bool,
        fall_core: bool,
        hard_fall: bool,
        short_fall: bool,
        min_pelvis_vel: float,
        impact_z: float,
        trunk_tilt_max: float,
        height_drop_m: float,
        rhythmic: bool,
        controlled_descent: bool
    ) -> float:
        """
        Calculate confidence score based on evidence strength.

        Returns a value 0-100 representing confidence in the detection.
        """
        if not fall_detected:
            # Even for non-falls, calculate how close it was
            score = 0.0
            thresholds = self.config.thresholds

            if min_pelvis_vel < thresholds.vel_down_fall_ms:
                score += 15
            if impact_z >= thresholds.impact_z_min:
                score += 15
            if trunk_tilt_max >= thresholds.trunk_tilt_fall_deg:
                score += 10
            if height_drop_m >= thresholds.height_drop_fall_m:
                score += 10

            # Reduce for mitigating factors
            if rhythmic:
                score *= 0.5
            if controlled_descent:
                score *= 0.3

            return min(max(score, 0), 49)  # Cap at 49 for non-falls

        # For detected falls, score based on path and evidence strength
        base_score = 60.0

        # Bonus for multiple paths
        paths_matched = sum([fall_core, hard_fall, short_fall])
        base_score += paths_matched * 10

        # Bonus for strong indicators
        thresholds = self.config.thresholds
        if min_pelvis_vel < thresholds.vel_down_fall_ms * 1.5:
            base_score += 5
        if impact_z >= thresholds.impact_z_severe:
            base_score += 10
        if trunk_tilt_max >= thresholds.trunk_tilt_horizontal_deg:
            base_score += 5
        if height_drop_m >= thresholds.height_drop_hard_fall_m:
            base_score += 5

        # Penalty for ambiguous context
        if rhythmic:
            base_score -= 10
        if controlled_descent:
            base_score -= 15

        return min(max(base_score, 50), 100)


def analyze_c3d_for_fall(filepath: str, config: DetectorConfig = None) -> dict:
    """
    Convenience function matching the original API.

    Args:
        filepath: Path to C3D file
        config: Optional detector configuration

    Returns:
        Dictionary with detection results (for backward compatibility)
    """
    detector = RulesDetector(config)
    result = detector.analyze(filepath)
    return result.to_dict()
