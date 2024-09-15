#!/usr/bin/env python3
"""
Annotation helper tool for fall detection.

This tool assists with creating temporal phase annotations for C3D files.
It uses the existing detector to auto-suggest phase boundaries, which
can then be refined manually.

Usage:
    python tools/annotate.py <video_dir>
    python tools/annotate.py new_data/video-02-epic-fall

Features:
    - Auto-suggest phase boundaries from motion analysis
    - Detect key events (impacts, velocity peaks)
    - Output JSON annotations
"""

import sys
import os
import argparse
import json
from typing import List, Tuple, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.c3d_reader import read_c3d
from src.features import extract_features
from src.config import PhaseLabel, FallType, DetectorConfig
from src.annotations import (
    VideoAnnotation, PhaseAnnotation, EventAnnotation,
    save_annotation, get_annotation_path
)
from src.biomechanics import calculate_impact_zscore


def find_phase_boundaries(
    features,
    is_fall: bool,
    config: DetectorConfig = None
) -> List[PhaseAnnotation]:
    """
    Auto-detect phase boundaries from motion features.

    Heuristics:
    - pre_fall ends: significant trunk tilt or velocity change
    - initiation: rapid velocity increase begins
    - descent: sustained downward velocity
    - impact: acceleration peak
    - post_fall: low motion period after impact
    """
    if config is None:
        config = DetectorConfig()

    frame_rate = features.frame_rate
    n_frames = features.n_frames
    phases = []

    if not is_fall:
        # Non-fall: entire sequence is pre_fall or general activity
        phases.append(PhaseAnnotation(
            phase=PhaseLabel.PRE_FALL,
            frame_start=0,
            frame_end=n_frames - 1,
            time_start_s=0.0,
            time_end_s=(n_frames - 1) / frame_rate,
            notes="Non-fall activity"
        ))
        return phases

    # For falls, detect key transition points
    vert_vel = features.vertical_velocity
    trunk_tilt = features.trunk_tilt_deg
    com_speed = features.com_speed

    # Find impact frame (peak acceleration)
    if len(features.com_acceleration) > 0:
        _, impact_z, impact_frame = calculate_impact_zscore(features.com_acceleration)
        # Adjust for acceleration array offset
        impact_frame = impact_frame + 2
    else:
        impact_frame = n_frames // 2

    # Find descent start: first frame with significant downward velocity
    descent_threshold = -0.3  # m/s
    descent_start = None
    for i, vel in enumerate(vert_vel):
        if vel < descent_threshold:
            descent_start = i + 1  # +1 for velocity array offset
            break

    if descent_start is None:
        descent_start = max(0, impact_frame - int(0.5 * frame_rate))

    # Find initiation: look for trunk tilt increase or velocity change before descent
    initiation_start = None
    trunk_threshold = 20.0  # degrees

    # Look backwards from descent start for initiation signs
    search_start = max(0, descent_start - int(1.0 * frame_rate))
    for i in range(descent_start, search_start, -1):
        if i < len(trunk_tilt) and trunk_tilt[i] > trunk_threshold:
            initiation_start = i
        elif i - 1 < len(vert_vel) and vert_vel[i - 1] < -0.1:
            initiation_start = i
            break

    if initiation_start is None:
        initiation_start = max(0, descent_start - int(0.3 * frame_rate))

    # Build phases
    # Pre-fall: 0 to initiation_start
    if initiation_start > 0:
        phases.append(PhaseAnnotation(
            phase=PhaseLabel.PRE_FALL,
            frame_start=0,
            frame_end=initiation_start - 1,
            time_start_s=0.0,
            time_end_s=(initiation_start - 1) / frame_rate,
            notes="Normal activity before fall"
        ))

    # Initiation: initiation_start to descent_start
    if descent_start > initiation_start:
        phases.append(PhaseAnnotation(
            phase=PhaseLabel.INITIATION,
            frame_start=initiation_start,
            frame_end=descent_start - 1,
            time_start_s=initiation_start / frame_rate,
            time_end_s=(descent_start - 1) / frame_rate,
            notes="Balance loss / compensatory movements"
        ))

    # Descent: descent_start to impact
    phases.append(PhaseAnnotation(
        phase=PhaseLabel.DESCENT,
        frame_start=descent_start,
        frame_end=impact_frame - 1,
        time_start_s=descent_start / frame_rate,
        time_end_s=(impact_frame - 1) / frame_rate,
        notes="Uncontrolled falling"
    ))

    # Impact: short window around impact frame
    impact_duration = int(0.1 * frame_rate)
    impact_end = min(impact_frame + impact_duration, n_frames - 1)
    phases.append(PhaseAnnotation(
        phase=PhaseLabel.IMPACT,
        frame_start=impact_frame,
        frame_end=impact_end,
        time_start_s=impact_frame / frame_rate,
        time_end_s=impact_end / frame_rate,
        notes="Ground contact"
    ))

    # Post-fall: immediately after impact to end (no gap)
    post_fall_start = impact_end + 1
    if post_fall_start < n_frames:
        phases.append(PhaseAnnotation(
            phase=PhaseLabel.POST_FALL,
            frame_start=post_fall_start,
            frame_end=n_frames - 1,
            time_start_s=post_fall_start / frame_rate,
            time_end_s=(n_frames - 1) / frame_rate,
            notes="Lying on ground"
        ))

    return phases


def find_events(features, phases: List[PhaseAnnotation]) -> List[EventAnnotation]:
    """Find key events (impacts, velocity peaks, etc.)."""
    events = []
    frame_rate = features.frame_rate

    # Impact event from acceleration peak
    if len(features.com_acceleration) > 0:
        _, impact_z, impact_frame = calculate_impact_zscore(features.com_acceleration)
        impact_frame += 2  # Adjust for offset

        acc_mag = np.linalg.norm(features.com_acceleration, axis=1)
        impact_g = acc_mag[impact_frame - 2] / 9.81 if impact_frame - 2 < len(acc_mag) else 0

        severity = "high" if impact_z > 5 else ("medium" if impact_z > 3 else "low")

        events.append(EventAnnotation(
            frame=impact_frame,
            time_s=impact_frame / frame_rate,
            event_type="impact_primary",
            severity=severity,
            notes=f"Impact z-score: {impact_z:.1f}, {impact_g:.1f}g"
        ))

    # Velocity minimum (fastest descent)
    if len(features.vertical_velocity) > 0:
        min_vel_frame = np.argmin(features.vertical_velocity) + 1
        min_vel = features.vertical_velocity[min_vel_frame - 1]

        events.append(EventAnnotation(
            frame=min_vel_frame,
            time_s=min_vel_frame / frame_rate,
            event_type="max_descent_velocity",
            severity="medium",
            notes=f"Vertical velocity: {min_vel:.2f} m/s"
        ))

    # Maximum trunk tilt
    if len(features.trunk_tilt_deg) > 0:
        max_tilt_frame = np.argmax(features.trunk_tilt_deg)
        max_tilt = features.trunk_tilt_deg[max_tilt_frame]

        if max_tilt > 60:
            events.append(EventAnnotation(
                frame=max_tilt_frame,
                time_s=max_tilt_frame / frame_rate,
                event_type="max_trunk_tilt",
                severity="medium" if max_tilt < 80 else "high",
                notes=f"Trunk tilt: {max_tilt:.1f} degrees"
            ))

    return events


def infer_fall_type(features, phases: List[PhaseAnnotation]) -> Optional[FallType]:
    """
    Infer fall type from kinematic patterns.

    - Slip: backward CoM movement, foot slides forward
    - Trip: forward CoM movement, swing foot stops
    - Collapse: primarily vertical, minimal horizontal momentum
    """
    # Get horizontal velocity components during descent
    descent_phases = [p for p in phases if p.phase == PhaseLabel.DESCENT]
    if not descent_phases:
        return FallType.UNKNOWN

    descent = descent_phases[0]
    start = max(0, descent.frame_start - 1)
    end = min(len(features.com_velocity), descent.frame_end)

    if end <= start:
        return FallType.UNKNOWN

    # Get horizontal components (assume Y is vertical, so X and Z are horizontal)
    horiz_vel = features.com_velocity[start:end, [0, 2]]
    vert_vel = features.vertical_velocity[start:end] if end <= len(features.vertical_velocity) else np.array([])

    if len(horiz_vel) == 0 or len(vert_vel) == 0:
        return FallType.UNKNOWN

    # Calculate average horizontal velocity during descent
    mean_horiz = np.mean(horiz_vel, axis=0)
    horiz_speed = np.linalg.norm(mean_horiz)
    mean_vert = np.mean(vert_vel)

    # Ratio of horizontal to vertical motion
    if abs(mean_vert) < 0.1:
        return FallType.UNKNOWN

    horiz_vert_ratio = horiz_speed / abs(mean_vert)

    # Classify based on ratio
    if horiz_vert_ratio < 0.3:
        # Primarily vertical - collapse
        return FallType.COLLAPSE
    elif horiz_vert_ratio > 0.7:
        # Significant horizontal component
        # Check direction: positive X often means forward, negative means backward
        if mean_horiz[0] > 0.1:
            return FallType.TRIP  # Forward
        elif mean_horiz[0] < -0.1:
            return FallType.SLIP  # Backward
        else:
            return FallType.COLLAPSE  # Lateral or ambiguous

    return FallType.UNKNOWN


def create_annotation(
    video_dir: str,
    is_fall: bool,
    annotator: str = "auto"
) -> VideoAnnotation:
    """
    Create annotation for a video directory.

    Args:
        video_dir: Path to video directory containing .c3d file
        is_fall: Ground truth - whether this is a fall
        annotator: Name of annotator

    Returns:
        VideoAnnotation object
    """
    video_name = os.path.basename(video_dir)
    c3d_path = os.path.join(video_dir, f"{video_name}.c3d")

    if not os.path.exists(c3d_path):
        # Try to find any .c3d file
        c3d_files = [f for f in os.listdir(video_dir) if f.endswith('.c3d')]
        if c3d_files:
            c3d_path = os.path.join(video_dir, c3d_files[0])
        else:
            raise FileNotFoundError(f"No C3D file found in {video_dir}")

    # Load and analyze
    data = read_c3d(c3d_path)
    features = extract_features(data.marker_positions, data.marker_labels, data.frame_rate)

    # Find phases
    phases = find_phase_boundaries(features, is_fall)

    # Find events
    events = find_events(features, phases)

    # Infer fall type
    fall_type = infer_fall_type(features, phases) if is_fall else None

    # Create annotation
    annotation = VideoAnnotation(
        video_id=video_name,
        annotator=annotator,
        frame_rate=data.frame_rate,
        total_frames=data.n_frames,
        ground_truth_fall=is_fall,
        fall_type=fall_type,
        phases=phases,
        events=events,
        notes=f"Auto-generated from {os.path.basename(c3d_path)}"
    )

    return annotation


def main():
    parser = argparse.ArgumentParser(description='Create fall phase annotations')
    parser.add_argument('video_dir', help='Path to video directory')
    parser.add_argument('--fall', action='store_true', help='Mark as fall (ground truth)')
    parser.add_argument('--no-fall', action='store_true', help='Mark as non-fall (ground truth)')
    parser.add_argument('--annotator', default='auto', help='Annotator name')
    parser.add_argument('--output', help='Output file (default: annotations.json in video_dir)')
    parser.add_argument('--print', action='store_true', help='Print annotation to stdout')

    args = parser.parse_args()

    if not os.path.isdir(args.video_dir):
        print(f"Error: {args.video_dir} is not a directory")
        sys.exit(1)

    # Determine if fall
    if args.fall:
        is_fall = True
    elif args.no_fall:
        is_fall = False
    else:
        # Try to infer from directory name
        video_name = os.path.basename(args.video_dir).lower()
        is_fall = 'fall' in video_name and 'non' not in video_name
        print(f"Inferred is_fall={is_fall} from directory name")

    # Create annotation
    print(f"Analyzing {args.video_dir}...")
    annotation = create_annotation(args.video_dir, is_fall, args.annotator)

    # Validate
    issues = annotation.validate()
    if issues:
        print("Validation warnings:")
        for issue in issues:
            print(f"  - {issue}")

    # Output
    if args.print:
        print(json.dumps(annotation.to_dict(), indent=2))

    output_path = args.output or get_annotation_path(args.video_dir)
    save_annotation(annotation, output_path)
    print(f"Saved annotation to {output_path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Video: {annotation.video_id}")
    print(f"  Frames: {annotation.total_frames} @ {annotation.frame_rate} fps")
    print(f"  Duration: {annotation.total_frames / annotation.frame_rate:.1f}s")
    print(f"  Fall: {annotation.ground_truth_fall}")
    if annotation.fall_type:
        print(f"  Fall type: {annotation.fall_type.value}")
    print(f"  Phases: {len(annotation.phases)}")
    for phase in annotation.phases:
        duration = phase.time_end_s - phase.time_start_s
        print(f"    {phase.phase.value}: frames {phase.frame_start}-{phase.frame_end} ({duration:.2f}s)")
    print(f"  Events: {len(annotation.events)}")
    for event in annotation.events:
        print(f"    {event.event_type} @ frame {event.frame} ({event.time_s:.2f}s)")


if __name__ == '__main__':
    main()
