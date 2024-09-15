"""
Dataset loading and preparation for fall detection training.

This module handles:
- Loading C3D files and extracting features
- Creating sliding window sequences
- Preparing PyTorch datasets
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    sequence_length: int = 50  # frames per sample
    overlap: int = 25  # sliding window overlap
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = [
                'com_velocity_x',
                'com_velocity_y',
                'com_velocity_z',
                'trunk_tilt_deg',
                'margin_of_stability',
                'acceleration_magnitude',
                'pelvis_height',
                'head_height',
                'vertical_velocity',
                'angular_velocity',
            ]


def extract_features_from_c3d(
    filepath: str,
    config: FeatureConfig = None
) -> np.ndarray:
    """
    Extract training features from a C3D file.

    Args:
        filepath: Path to C3D file
        config: Feature extraction configuration

    Returns:
        Feature array (n_frames, n_features)
    """
    from src.utils.c3d_reader import read_c3d
    from src.biomechanics import (
        calculate_whole_body_com,
        calculate_trunk_tilt,
        calculate_com_velocity,
        calculate_com_acceleration,
        find_marker_indices,
    )
    from src.config import DetectorConfig

    if config is None:
        config = FeatureConfig()

    # Load C3D data
    data = read_c3d(filepath)
    detector_cfg = DetectorConfig()

    n_frames = data.marker_positions.shape[0]
    vaxis = detector_cfg.get_vertical_axis_index()

    # Find markers
    patterns = detector_cfg.body_model.segment_marker_patterns
    pelvis_idx = find_marker_indices(data.marker_labels, patterns.get('pelvis', []))
    head_idx = find_marker_indices(data.marker_labels, patterns.get('head', []))

    # Calculate positions
    if pelvis_idx:
        pelvis_pos = np.mean(data.marker_positions[:, pelvis_idx, :], axis=1)
    else:
        pelvis_pos = np.mean(data.marker_positions, axis=1)

    if head_idx:
        head_pos = np.mean(data.marker_positions[:, head_idx, :], axis=1)
    else:
        highest_idx = np.argmax(np.mean(data.marker_positions[:, :, vaxis], axis=0))
        head_pos = data.marker_positions[:, highest_idx, :]

    # Calculate CoM
    com_pos = calculate_whole_body_com(
        data.marker_positions,
        data.marker_labels,
        detector_cfg.body_model
    )

    # Calculate velocities (m/s)
    com_vel = calculate_com_velocity(com_pos, data.frame_rate, 'mm')
    com_vel = np.vstack([com_vel, com_vel[-1:]])  # Pad to match length (N-1 -> N)

    # Calculate acceleration
    com_acc = calculate_com_acceleration(com_vel, data.frame_rate)
    com_acc = np.vstack([com_acc, com_acc[-1:]])  # Pad to match length (N-1 -> N)

    # Calculate trunk tilt
    trunk_tilt = calculate_trunk_tilt(head_pos, pelvis_pos, vaxis)

    # Convert heights to meters
    pelvis_h_m = pelvis_pos[:, vaxis] / 1000.0
    head_h_m = head_pos[:, vaxis] / 1000.0

    # Calculate derived features
    vertical_velocity = com_vel[:, vaxis]
    acc_magnitude = np.linalg.norm(com_acc, axis=1) / 9.81  # In g
    angular_velocity = np.gradient(trunk_tilt) * data.frame_rate

    # Margin of stability (simplified - use 0 if not calculated)
    mos = np.zeros(n_frames)

    # Build feature array
    features = np.column_stack([
        com_vel[:, 0],       # com_velocity_x
        com_vel[:, 1],       # com_velocity_y
        com_vel[:, 2],       # com_velocity_z
        trunk_tilt,          # trunk_tilt_deg
        mos,                 # margin_of_stability
        acc_magnitude,       # acceleration_magnitude
        pelvis_h_m,          # pelvis_height
        head_h_m,            # head_height
        vertical_velocity,   # vertical_velocity
        angular_velocity,    # angular_velocity
    ])

    return features.astype(np.float32)


def create_sequences(
    features: np.ndarray,
    label: int,
    seq_length: int = 50,
    overlap: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from feature array.

    Args:
        features: Feature array (n_frames, n_features)
        label: Label for this recording (0 or 1)
        seq_length: Sequence length
        overlap: Overlap between sequences

    Returns:
        Tuple of (sequences, labels)
    """
    n_frames = features.shape[0]
    step = seq_length - overlap

    sequences = []
    labels = []

    for start in range(0, n_frames - seq_length + 1, step):
        end = start + seq_length
        seq = features[start:end]
        sequences.append(seq)
        labels.append(label)

    if not sequences:
        # If recording is shorter than sequence length, pad it
        padded = np.zeros((seq_length, features.shape[1]), dtype=np.float32)
        padded[:n_frames] = features
        sequences.append(padded)
        labels.append(label)

    return np.array(sequences), np.array(labels)


def prepare_training_data_with_groups(
    labels_path: str,
    data_dir: str,
    config: FeatureConfig = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare training data with recording group labels.

    Args:
        labels_path: Path to labels.json
        data_dir: Base directory for data files
        config: Feature extraction configuration

    Returns:
        Tuple of (X, y, recording_ids, recording_names)
    """
    if config is None:
        config = FeatureConfig()

    # Load labels
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)

    data_dir = Path(data_dir)
    all_sequences = []
    all_labels = []
    all_recording_ids = []
    recording_names = []

    for video in labels_data['videos']:
        # Construct path
        file_path = data_dir.parent / video['file']
        if not file_path.exists():
            # Try relative to data_dir
            file_path = data_dir / Path(video['file']).name
            if not file_path.exists():
                print(f"Warning: Could not find {video['file']}, skipping")
                continue

        # Extract features
        try:
            features = extract_features_from_c3d(str(file_path), config)
        except Exception as e:
            print(f"Warning: Error processing {video['file']}: {e}")
            continue

        # Create sequences
        label = 1 if video['fall'] else 0
        seqs, labs = create_sequences(
            features,
            label,
            seq_length=config.sequence_length,
            overlap=config.overlap
        )

        group_id = len(recording_names)
        recording_name = video.get('id') or Path(video['file']).stem
        recording_names.append(recording_name)

        all_sequences.append(seqs)
        all_labels.append(labs)
        all_recording_ids.append(
            np.full(len(labs), group_id, dtype=np.int32)
        )

    if not all_sequences:
        raise ValueError("No data loaded. Check file paths.")

    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_labels, axis=0)
    recording_ids = np.concatenate(all_recording_ids, axis=0)

    return X, y, recording_ids, recording_names


def prepare_training_data(
    labels_path: str,
    data_dir: str,
    config: FeatureConfig = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from labels file.

    Args:
        labels_path: Path to labels.json
        data_dir: Base directory for data files
        config: Feature extraction configuration

    Returns:
        Tuple of (X, y) arrays
    """
    X, y, _, _ = prepare_training_data_with_groups(
        labels_path, data_dir, config
    )
    return X, y


if TORCH_AVAILABLE:
    class FallDataset(Dataset):
        """
        PyTorch dataset for fall detection.
        """

        def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            transform=None
        ):
            """
            Initialize dataset.

            Args:
                X: Feature sequences (n_samples, seq_len, n_features)
                y: Labels (n_samples,)
                transform: Optional transform to apply
            """
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.transform = transform

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            x = self.X[idx]
            y = self.y[idx]

            if self.transform:
                x = self.transform(x)

            return x, y

    def create_data_loaders(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders.

        Args:
            X: Feature sequences
            y: Labels
            batch_size: Batch size
            train_ratio: Fraction for training
            shuffle: Shuffle data

        Returns:
            Tuple of (train_loader, val_loader)
        """
        n_samples = len(y)
        indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)

        split_idx = int(n_samples * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_dataset = FallDataset(X[train_indices], y[train_indices])
        val_dataset = FallDataset(X[val_indices], y[val_indices])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader


def save_features(
    X: np.ndarray,
    y: np.ndarray,
    output_path: str,
    metadata: Dict = None,
    recording_ids: Optional[np.ndarray] = None,
    recording_names: Optional[List[str]] = None
):
    """
    Save extracted features to .npz file.

    Args:
        X: Feature sequences
        y: Labels
        output_path: Output file path
        metadata: Optional metadata dictionary
        recording_ids: Optional recording group IDs per sample
        recording_names: Optional list of recording names by group ID
    """
    save_dict = {
        'X': X,
        'y': y,
    }
    if recording_ids is not None:
        save_dict['recording_ids'] = recording_ids
    if recording_names is not None:
        save_dict['recording_names'] = np.array(recording_names)
    if metadata:
        save_dict['metadata'] = np.array([json.dumps(metadata)])

    np.savez_compressed(output_path, **save_dict)
    print(f"Saved features to {output_path}")
    print(f"  Shape: X={X.shape}, y={y.shape}")


def load_features(
    filepath: str,
    return_groups: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Load features from .npz file.

    Args:
        filepath: Path to .npz file
        return_groups: Return recording group info if present

    Returns:
        Tuple of (X, y, metadata) or (X, y, metadata, recording_ids, recording_names)
    """
    data = np.load(filepath, allow_pickle=True)
    X = data['X']
    y = data['y']

    metadata = None
    if 'metadata' in data:
        metadata = json.loads(str(data['metadata'][0]))

    recording_ids = data['recording_ids'] if 'recording_ids' in data else None
    recording_names = None
    if 'recording_names' in data:
        recording_names = data['recording_names'].tolist()

    if return_groups:
        return X, y, metadata, recording_ids, recording_names

    return X, y, metadata
