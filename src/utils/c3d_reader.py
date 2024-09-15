"""
C3D file reading utilities.

Provides a clean interface for reading motion capture data from C3D files.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class C3DData:
    """Container for C3D file data."""

    # Metadata
    frame_rate: float
    n_frames: int
    n_markers: int
    first_frame: int
    last_frame: int

    # Marker data
    marker_positions: np.ndarray  # (N_frames, N_markers, 3)
    marker_labels: List[str]

    # Computed properties
    duration: float

    @property
    def time_vector(self) -> np.ndarray:
        """Get time vector in seconds."""
        return np.arange(self.n_frames) / self.frame_rate

    def get_marker(self, label: str) -> Optional[np.ndarray]:
        """
        Get position data for a specific marker by label.

        Args:
            label: Marker label (case-insensitive substring match)

        Returns:
            Marker positions (N_frames, 3) or None if not found
        """
        label_lower = label.lower()
        for i, lbl in enumerate(self.marker_labels):
            if label_lower in lbl.lower():
                return self.marker_positions[:, i, :]
        return None

    def get_markers_by_pattern(self, patterns: List[str]) -> np.ndarray:
        """
        Get positions for markers matching any of the patterns.

        Args:
            patterns: List of substring patterns (case-insensitive)

        Returns:
            Marker positions (N_frames, N_matched, 3)
        """
        indices = []
        patterns_lower = [p.lower() for p in patterns]

        for i, lbl in enumerate(self.marker_labels):
            lbl_lower = lbl.lower()
            for pattern in patterns_lower:
                if pattern in lbl_lower:
                    indices.append(i)
                    break

        if not indices:
            return np.array([]).reshape(self.n_frames, 0, 3)

        return self.marker_positions[:, indices, :]


def read_c3d(filepath: str) -> C3DData:
    """
    Read a C3D file and return structured data.

    Args:
        filepath: Path to .c3d file

    Returns:
        C3DData object containing marker positions and metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or corrupted
    """
    try:
        import c3d
    except ImportError:
        raise ImportError("c3d library required: pip install c3d")

    with open(filepath, 'rb') as f:
        reader = c3d.Reader(f)

        # Extract metadata
        frame_rate = float(reader.header.frame_rate)
        first_frame = reader.header.first_frame
        last_frame = reader.header.last_frame
        n_frames = last_frame - first_frame + 1
        n_markers = reader.header.point_count

        # Extract marker labels
        marker_labels = []
        try:
            for i in range(n_markers):
                label = reader.point_labels[i].strip()
                marker_labels.append(label)
        except Exception:
            # Fallback to generic labels
            marker_labels = [f"marker_{i}" for i in range(n_markers)]

        # Read frame data
        frames = []
        for _, points, _ in reader.read_frames():
            # Extract only XYZ coordinates (ignore residuals)
            frames.append(points[:, :3])

        if not frames:
            raise ValueError(f"No frame data found in {filepath}")

        marker_positions = np.array(frames)

        return C3DData(
            frame_rate=frame_rate,
            n_frames=n_frames,
            n_markers=n_markers,
            first_frame=first_frame,
            last_frame=last_frame,
            marker_positions=marker_positions,
            marker_labels=marker_labels,
            duration=n_frames / frame_rate
        )


def get_marker_info(filepath: str) -> dict:
    """
    Get basic information about markers in a C3D file.

    Args:
        filepath: Path to .c3d file

    Returns:
        Dictionary with marker labels and statistics
    """
    data = read_c3d(filepath)

    info = {
        'n_markers': data.n_markers,
        'n_frames': data.n_frames,
        'frame_rate': data.frame_rate,
        'duration': data.duration,
        'labels': data.marker_labels,
    }

    # Add per-marker statistics
    marker_stats = {}
    for i, label in enumerate(data.marker_labels):
        positions = data.marker_positions[:, i, :]
        marker_stats[label] = {
            'mean_position': positions.mean(axis=0).tolist(),
            'std_position': positions.std(axis=0).tolist(),
            'range': (positions.max(axis=0) - positions.min(axis=0)).tolist(),
        }

    info['marker_stats'] = marker_stats

    return info
