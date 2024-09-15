"""
Visualization utilities for fall detection.

Provides functions for plotting motion data, detection results,
and phase annotations.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from ..features import MotionFeatures
from ..annotations import VideoAnnotation, PhaseAnnotation
from ..config import PhaseLabel
from ..detectors.base import FallAnalysisResult


# Phase colors for visualization
PHASE_COLORS = {
    PhaseLabel.PRE_FALL: '#4CAF50',      # Green
    PhaseLabel.INITIATION: '#FFC107',    # Amber
    PhaseLabel.DESCENT: '#FF5722',       # Deep Orange
    PhaseLabel.IMPACT: '#F44336',        # Red
    PhaseLabel.POST_FALL: '#9C27B0',     # Purple
}

PHASE_NAMES = {
    PhaseLabel.PRE_FALL: 'Pre-fall',
    PhaseLabel.INITIATION: 'Initiation',
    PhaseLabel.DESCENT: 'Descent',
    PhaseLabel.IMPACT: 'Impact',
    PhaseLabel.POST_FALL: 'Post-fall',
}


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")


def plot_motion_timeline(
    features: MotionFeatures,
    result: Optional[FallAnalysisResult] = None,
    annotation: Optional[VideoAnnotation] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot comprehensive motion timeline with detection results.

    Args:
        features: Extracted motion features
        result: Detection result (optional)
        annotation: Phase annotations (optional)
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib figure object
    """
    _check_matplotlib()

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    time = np.arange(features.n_frames) / features.frame_rate

    # 1. Height plot
    ax = axes[0]
    ax.plot(time, features.pelvis_height, label='Pelvis', color='#2196F3', linewidth=1.5)
    ax.plot(time, features.head_height, label='Head', color='#4CAF50', linewidth=1.5)
    if features.com_height is not None:
        ax.plot(time, features.com_height, label='CoM', color='#FF9800', linewidth=1.5, linestyle='--')
    ax.set_ylabel('Height (m)')
    ax.legend(loc='upper right')
    ax.set_title('Vertical Position')
    ax.grid(True, alpha=0.3)

    # 2. Velocity plot
    ax = axes[1]
    time_vel = np.arange(len(features.vertical_velocity)) / features.frame_rate
    ax.plot(time_vel, features.vertical_velocity, color='#E91E63', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.8, color='red', linestyle=':', alpha=0.7, label='Fall threshold')
    ax.set_ylabel('Vertical Velocity (m/s)')
    ax.set_title('Vertical Velocity')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Trunk tilt plot
    ax = axes[2]
    ax.plot(time, features.trunk_tilt_deg, color='#9C27B0', linewidth=1.5)
    ax.axhline(y=75, color='orange', linestyle=':', alpha=0.7, label='Warning')
    ax.axhline(y=85, color='red', linestyle=':', alpha=0.7, label='Fall threshold')
    ax.set_ylabel('Trunk Tilt (deg)')
    ax.set_title('Trunk Orientation')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 4. Acceleration plot
    ax = axes[3]
    if len(features.com_acceleration) > 0:
        time_acc = np.arange(len(features.com_acceleration)) / features.frame_rate
        acc_mag = np.linalg.norm(features.com_acceleration, axis=1) / 9.81
        ax.plot(time_acc, acc_mag, color='#FF5722', linewidth=1.5)
        ax.axhline(y=2.0, color='red', linestyle=':', alpha=0.7, label='Impact threshold')
    ax.set_ylabel('Acceleration (g)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Acceleration Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add phase annotations if provided
    if annotation and annotation.phases:
        _add_phase_regions(axes, annotation.phases, features.frame_rate)

    # Add detection result markers if provided
    if result:
        _add_result_markers(axes, result, features)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def _add_phase_regions(
    axes: List[Any],
    phases: List[PhaseAnnotation],
    frame_rate: float
):
    """Add colored regions for phases to all axes."""
    for phase in phases:
        color = PHASE_COLORS.get(phase.phase, '#CCCCCC')
        for ax in axes:
            ax.axvspan(
                phase.time_start_s,
                phase.time_end_s,
                alpha=0.15,
                color=color,
                label=PHASE_NAMES.get(phase.phase, phase.phase.value)
            )


def _add_result_markers(
    axes: List[Any],
    result: FallAnalysisResult,
    features: MotionFeatures
):
    """Add markers for detected events."""
    # Mark impact if detected
    if result.timeline_data and result.timeline_data.impact_occurred:
        impact_time = result.timeline_data.impact_time_s
        if impact_time:
            for ax in axes:
                ax.axvline(x=impact_time, color='red', linestyle='-', alpha=0.8, linewidth=2)

    # Add detection result text
    detection_text = "FALL DETECTED" if result.fall_detected else "No fall"
    axes[0].text(
        0.02, 0.98, f"{detection_text} (conf: {result.confidence:.0f}%)",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top',
        color='red' if result.fall_detected else 'green',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )


def plot_feature_comparison(
    features_list: List[Tuple[str, MotionFeatures]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot comparison of features across multiple videos.

    Args:
        features_list: List of (name, features) tuples
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib figure object
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(features_list)))

    for idx, (name, features) in enumerate(features_list):
        color = colors[idx]
        time = np.arange(features.n_frames) / features.frame_rate

        # Height
        axes[0, 0].plot(time, features.pelvis_height, label=name, color=color, alpha=0.8)

        # Vertical velocity
        time_vel = np.arange(len(features.vertical_velocity)) / features.frame_rate
        axes[0, 1].plot(time_vel, features.vertical_velocity, label=name, color=color, alpha=0.8)

        # Trunk tilt
        axes[1, 0].plot(time, features.trunk_tilt_deg, label=name, color=color, alpha=0.8)

        # Speed
        time_speed = np.arange(len(features.com_speed)) / features.frame_rate
        axes[1, 1].plot(time_speed, features.com_speed, label=name, color=color, alpha=0.8)

    axes[0, 0].set_title('Pelvis Height')
    axes[0, 0].set_ylabel('Height (m)')
    axes[0, 0].legend(loc='best', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Vertical Velocity')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Trunk Tilt')
    axes[1, 0].set_ylabel('Tilt (deg)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('CoM Speed')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_phase_distribution(
    annotations: List[VideoAnnotation],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot distribution of phase durations across videos.

    Args:
        annotations: List of video annotations
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib figure object
    """
    _check_matplotlib()

    phase_durations = {phase: [] for phase in PhaseLabel}

    for ann in annotations:
        if ann.ground_truth_fall:
            for phase in ann.phases:
                duration = phase.time_end_s - phase.time_start_s
                phase_durations[phase.phase].append(duration)

    fig, ax = plt.subplots(figsize=figsize)

    phases = [p for p in PhaseLabel if phase_durations[p]]
    positions = range(len(phases))
    data = [phase_durations[p] for p in phases]

    bp = ax.boxplot(data, positions=positions, patch_artist=True)

    for idx, (patch, phase) in enumerate(zip(bp['boxes'], phases)):
        patch.set_facecolor(PHASE_COLORS.get(phase, '#CCCCCC'))
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([PHASE_NAMES.get(p, p.value) for p in phases])
    ax.set_ylabel('Duration (s)')
    ax.set_title('Fall Phase Durations')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_report(
    features: MotionFeatures,
    result: FallAnalysisResult,
    annotation: Optional[VideoAnnotation] = None
) -> str:
    """
    Create a text summary report of the analysis.

    Args:
        features: Extracted motion features
        result: Detection result
        annotation: Phase annotations (optional)

    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("FALL DETECTION ANALYSIS REPORT")
    lines.append("=" * 60)

    # Detection result
    lines.append(f"\nDETECTION RESULT: {'FALL' if result.fall_detected else 'NO FALL'}")
    lines.append(f"Confidence: {result.confidence:.1f}%")
    lines.append(f"Activity Type: {result.activity_type}")

    if result.characteristics:
        lines.append(f"Characteristics: {', '.join(result.characteristics)}")

    # Key metrics
    lines.append("\n" + "-" * 40)
    lines.append("KEY METRICS")
    lines.append("-" * 40)

    stats = features.stats
    if stats:
        lines.append(f"Min Vertical Velocity: {stats.get('min_vertical_velocity_ms', 'N/A'):.2f} m/s")
        lines.append(f"Height Drop: {stats.get('height_drop_robust_m', 'N/A'):.2f} m")
        lines.append(f"Max Trunk Tilt: {stats.get('trunk_tilt_max_deg', 'N/A'):.1f} deg")
        lines.append(f"Impact Z-score: {stats.get('impact_z_score', 'N/A'):.2f}")
        lines.append(f"Descent Duration: {stats.get('descent_duration_s', 'N/A'):.2f} s")

        lines.append(f"\nFinal Posture:")
        lines.append(f"  Head Height: {stats.get('head_final_m', 'N/A'):.2f} m")
        lines.append(f"  Pelvis Height: {stats.get('pelvis_final_m', 'N/A'):.2f} m")
        lines.append(f"  Head-Pelvis Delta: {stats.get('head_pelvis_delta_m', 'N/A'):.2f} m")

    # Phase annotations
    if annotation and annotation.phases:
        lines.append("\n" + "-" * 40)
        lines.append("PHASE ANNOTATIONS")
        lines.append("-" * 40)

        for phase in annotation.phases:
            duration = phase.time_end_s - phase.time_start_s
            lines.append(f"{PHASE_NAMES.get(phase.phase, phase.phase.value):12s}: "
                        f"{phase.time_start_s:.2f}s - {phase.time_end_s:.2f}s "
                        f"({duration:.2f}s)")

    # Events
    if annotation and annotation.events:
        lines.append("\n" + "-" * 40)
        lines.append("EVENTS")
        lines.append("-" * 40)

        for event in annotation.events:
            lines.append(f"{event.event_type}: {event.time_s:.2f}s (frame {event.frame})")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
