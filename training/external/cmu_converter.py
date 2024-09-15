#!/usr/bin/env python3
"""
Convert CMU Motion Capture BVH files to training features.

CMU mocap provides diverse motion data (walking, running, dancing, etc.)
that can be used as negative samples (non-fall activities) for training.

Usage:
    python training/external/cmu_converter.py \
        --input data/external/cmu_mocap \
        --output data/external/cmu_features.npz
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.dataset import save_features, FeatureConfig


def parse_bvh_file(filepath: str) -> Optional[np.ndarray]:
    """
    Parse a BVH file and extract joint positions.

    BVH format:
    - HIERARCHY section: bone structure
    - MOTION section: frame data

    Args:
        filepath: Path to BVH file

    Returns:
        Joint positions (n_frames, n_joints, 3) or None if parsing fails
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Find MOTION section
        if 'MOTION' not in content:
            return None

        motion_start = content.index('MOTION')
        motion_section = content[motion_start:]
        lines = motion_section.strip().split('\n')

        # Parse header
        n_frames = int(lines[1].split(':')[1].strip())
        frame_time = float(lines[2].split(':')[1].strip())
        frame_rate = 1.0 / frame_time if frame_time > 0 else 120.0

        # Parse frame data
        frames = []
        for line in lines[3:]:
            if line.strip():
                values = [float(v) for v in line.strip().split()]
                frames.append(values)

        if not frames:
            return None

        motion_data = np.array(frames)

        # BVH typically has: root pos (3) + rotations for each joint
        # Extract positions for key joints
        # This is simplified - full implementation would use forward kinematics

        # For now, use root position as CoM approximation
        root_pos = motion_data[:, :3]  # Root X, Y, Z

        return root_pos, frame_rate

    except Exception as e:
        print(f"  Error parsing {filepath}: {e}")
        return None


def extract_features_from_bvh(
    positions: np.ndarray,
    frame_rate: float
) -> np.ndarray:
    """
    Extract training features from BVH position data.

    Args:
        positions: Position data (n_frames, 3)
        frame_rate: Frame rate in Hz

    Returns:
        Feature array (n_frames, n_features)
    """
    n_frames = positions.shape[0]

    # Convert to meters (BVH typically in cm)
    pos_m = positions / 100.0

    # Calculate velocity
    velocity = np.diff(pos_m, axis=0) * frame_rate
    velocity = np.vstack([velocity, velocity[-1:]])  # Pad

    # Calculate acceleration
    acceleration = np.diff(velocity, axis=0) * frame_rate
    acceleration = np.vstack([acceleration, acceleration[-1:], acceleration[-1:]])  # Pad
    acc_magnitude = np.linalg.norm(acceleration, axis=1) / 9.81  # In g

    # Height (Y typically up in BVH)
    height = pos_m[:, 1]

    # Vertical velocity
    vertical_velocity = velocity[:, 1]

    # For CMU data, we don't have trunk markers, so use placeholders
    trunk_tilt = np.zeros(n_frames)  # Unknown
    mos = np.zeros(n_frames)  # Unknown
    angular_velocity = np.zeros(n_frames)

    # Build feature array (same structure as C3D features)
    features = np.column_stack([
        velocity[:, 0],      # com_velocity_x
        velocity[:, 1],      # com_velocity_y
        velocity[:, 2],      # com_velocity_z
        trunk_tilt,          # trunk_tilt_deg (placeholder)
        mos,                 # margin_of_stability (placeholder)
        acc_magnitude,       # acceleration_magnitude
        height,              # pelvis_height (using root as proxy)
        height,              # head_height (using root as proxy)
        vertical_velocity,   # vertical_velocity
        angular_velocity,    # angular_velocity (placeholder)
    ])

    return features.astype(np.float32)


def create_sequences_from_bvh(
    features: np.ndarray,
    seq_length: int = 50,
    overlap: int = 25
) -> np.ndarray:
    """Create sequences from BVH features (all labeled as non-fall)."""
    n_frames = features.shape[0]
    step = seq_length - overlap

    sequences = []
    for start in range(0, n_frames - seq_length + 1, step):
        end = start + seq_length
        sequences.append(features[start:end])

    if not sequences and n_frames >= seq_length:
        sequences.append(features[:seq_length])

    return np.array(sequences) if sequences else None


def convert_cmu_dataset(
    input_dir: str,
    output_path: str,
    max_files: Optional[int] = None,
    seq_length: int = 50
):
    """
    Convert CMU BVH files to training features.

    Args:
        input_dir: Directory containing BVH files
        output_path: Output .npz file path
        max_files: Maximum files to process
        seq_length: Sequence length
    """
    input_dir = Path(input_dir)

    # Find all BVH files
    bvh_files = list(input_dir.rglob("*.bvh"))

    if not bvh_files:
        print(f"No BVH files found in {input_dir}")
        return

    print(f"Found {len(bvh_files)} BVH files")

    if max_files:
        bvh_files = bvh_files[:max_files]
        print(f"Processing first {max_files} files")

    all_sequences = []
    processed = 0
    errors = 0

    for i, bvh_file in enumerate(bvh_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(bvh_files)}...")

        result = parse_bvh_file(str(bvh_file))
        if result is None:
            errors += 1
            continue

        positions, frame_rate = result
        features = extract_features_from_bvh(positions, frame_rate)
        sequences = create_sequences_from_bvh(features, seq_length)

        if sequences is not None and len(sequences) > 0:
            all_sequences.append(sequences)
            processed += 1

    if not all_sequences:
        print("No sequences extracted.")
        return

    X = np.concatenate(all_sequences, axis=0)
    y = np.zeros(len(X), dtype=np.float32)  # All non-falls

    print(f"\nProcessed {processed} files ({errors} errors)")
    print(f"Total sequences: {len(X)}")
    print(f"Shape: X={X.shape}, y={y.shape}")

    # Save
    metadata = {
        'source': 'CMU Motion Capture Database',
        'format': 'BVH',
        'label': 'non_fall',
        'n_files': processed,
        'sequence_length': seq_length,
    }
    save_features(X, y, output_path, metadata)


def main():
    parser = argparse.ArgumentParser(
        description='Convert CMU BVH files to training features'
    )
    parser.add_argument('--input', required=True, help='Input directory with BVH files')
    parser.add_argument('--output', required=True, help='Output .npz file')
    parser.add_argument('--max-files', type=int, help='Max files to process')
    parser.add_argument('--seq-length', type=int, default=50, help='Sequence length')

    args = parser.parse_args()
    convert_cmu_dataset(args.input, args.output, args.max_files, args.seq_length)


if __name__ == '__main__':
    main()
