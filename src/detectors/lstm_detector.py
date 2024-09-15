"""
LSTM-based fall detector.

This detector uses a trained LSTM neural network for fall detection,
providing an alternative to the rules-based detector.

Usage:
    from src.detectors.lstm_detector import LSTMDetector

    detector = LSTMDetector(model_path='models/lstm_fall_v1.pt')
    result = detector.analyze('path/to/file.c3d')

References:
    Hu, X., & Qu, X. (2016). Pre-impact fall detection.
    BioMedical Engineering OnLine, 15, 87.
    DOI: 10.1186/s12938-016-0194-x

See docs/TRAINING.md for training instructions.
"""

from typing import List, Optional
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import BaseDetector, FallAnalysisResult, TimelineData
from ..config import DetectorConfig, FallType, DEFAULT_CONFIG
from ..biomechanics import (
    calculate_whole_body_com,
    calculate_trunk_tilt,
    calculate_com_velocity,
    calculate_com_acceleration,
    find_marker_indices,
)
from ..utils.c3d_reader import read_c3d

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Placeholder classes when PyTorch not available
FallDetectorLSTM = None
LSTMDetector = None


# Only define torch-dependent classes if PyTorch is available
if TORCH_AVAILABLE:

    class FallDetectorLSTM(nn.Module):
        """
        LSTM model architecture for fall detection.

        This is a copy of the training model to allow loading weights
        without depending on the training package.
        """

        def __init__(
            self,
            input_size: int = 10,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = True
        ):
            super().__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size

            self.lstm1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0
            )

            lstm1_output_size = hidden_size * (2 if bidirectional else 1)
            self.dropout1 = nn.Dropout(dropout)

            self.lstm2 = nn.LSTM(
                input_size=lstm1_output_size,
                hidden_size=hidden_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                dropout=0
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            lstm1_out, _ = self.lstm1(x)
            lstm1_out = self.dropout1(lstm1_out)
            lstm2_out, _ = self.lstm2(lstm1_out)
            last_output = lstm2_out[:, -1, :]
            return self.classifier(last_output)


    class LSTMDetector(BaseDetector):
        """
        LSTM-based fall detector.

        Uses a trained LSTM neural network to classify motion sequences
        as falls or non-falls. Provides probability-based confidence scores.

        Attributes:
            model: Loaded PyTorch LSTM model
            model_path: Path to model weights file
            device: PyTorch device (cpu/cuda/mps)
            seq_length: Expected sequence length
            threshold: Classification threshold (default 0.5)
        """

        def __init__(
            self,
            config: DetectorConfig = None,
            model_path: str = None,
            threshold: float = 0.5
        ):
            """
            Initialize LSTM detector.

            Args:
                config: Detector configuration
                model_path: Path to trained model weights (.pt file)
                threshold: Classification threshold (0-1)
            """
            super().__init__(config)

            self.threshold = threshold
            self.seq_length = 50  # Default, can be updated from model config

            # Determine device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

            # Load model
            if model_path:
                self.model = self._load_model(model_path)
            else:
                self.model = None

        def _load_model(self, model_path: str) -> nn.Module:
            """Load trained model from file."""
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Get model config from checkpoint
            model_config = checkpoint.get('config', {})
            input_size = model_config.get('input_size', 10)
            hidden_size = model_config.get('hidden_size', 64)
            dropout = model_config.get('dropout', 0.3)

            # Create model
            model = FallDetectorLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout
            ).to(self.device)

            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            return model

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
            """Analyze motion data using LSTM model."""

            if self.model is None:
                raise ValueError("No model loaded. Provide model_path to constructor.")

            cfg = self.config
            vaxis = cfg.get_vertical_axis_index()
            n_frames = marker_positions.shape[0]

            # Extract features
            features = self._extract_features(
                marker_positions, marker_labels, frame_rate
            )

            # Create sequences
            sequences = self._create_sequences(features)

            if len(sequences) == 0:
                # Recording too short
                return FallAnalysisResult(
                    fall_detected=False,
                    confidence=0.0,
                    activity_type='unknown',
                    characteristics=['recording_too_short'],
                    metrics={'error': 'Recording shorter than sequence length'}
                )

            # Run inference
            with torch.no_grad():
                X = torch.FloatTensor(sequences).to(self.device)
                probs = self.model(X).cpu().numpy().flatten()

            # Aggregate predictions
            max_prob = float(np.max(probs))
            mean_prob = float(np.mean(probs))

            # Use max probability as fall indicator
            fall_detected = max_prob >= self.threshold

            # Convert to 0-100 confidence
            confidence = max_prob * 100

            # Build metrics
            metrics = {
                'max_fall_probability': round(max_prob, 4),
                'mean_fall_probability': round(mean_prob, 4),
                'n_sequences': len(sequences),
                'threshold': self.threshold,
                'model_type': 'LSTM',
            }

            # Classify activity
            com_vel = self._extract_velocity(marker_positions, marker_labels, frame_rate)
            if com_vel is not None:
                patterns = self.config.body_model.segment_marker_patterns
                pelvis_idx = find_marker_indices(marker_labels, patterns.get('pelvis', []))
                if pelvis_idx:
                    pelvis_pos = np.mean(marker_positions[:, pelvis_idx, :], axis=1)
                    pelvis_h = pelvis_pos[:, vaxis]
                else:
                    pelvis_h = np.mean(marker_positions[:, :, vaxis], axis=1)
                activity_type = self._classify_activity(com_vel, pelvis_h, frame_rate)
            else:
                activity_type = 'unknown'

            # Build characteristics
            characteristics = ['lstm_detection']
            if fall_detected:
                characteristics.append('high_fall_probability')
            if max_prob > 0.9:
                characteristics.append('very_high_confidence')
            elif max_prob > 0.7:
                characteristics.append('moderate_confidence')

            # Build timeline data
            patterns = cfg.body_model.segment_marker_patterns
            pelvis_idx = find_marker_indices(marker_labels, patterns.get('pelvis', []))
            head_idx = find_marker_indices(marker_labels, patterns.get('head', []))

            if pelvis_idx:
                pelvis_h = np.mean(marker_positions[:, pelvis_idx, vaxis], axis=1)
            else:
                pelvis_h = np.mean(marker_positions[:, :, vaxis], axis=1)

            if head_idx:
                head_h = np.mean(marker_positions[:, head_idx, vaxis], axis=1)
            else:
                head_h = pelvis_h

            pelvis_h_m = (pelvis_h / 1000.0).tolist() if cfg.units == 'mm' else pelvis_h.tolist()
            head_h_m = (head_h / 1000.0).tolist() if cfg.units == 'mm' else head_h.tolist()

            time_pos = (np.arange(n_frames) / frame_rate).tolist()

            timeline_data = TimelineData(
                time_pos=time_pos,
                time_vel=[],
                time_acc=[],
                pelvis_height_m=pelvis_h_m,
                head_height_m=head_h_m,
            )

            return FallAnalysisResult(
                fall_detected=fall_detected,
                confidence=confidence,
                activity_type=activity_type,
                characteristics=characteristics,
                metrics=metrics,
                timeline_data=timeline_data,
            )

        def _extract_features(
            self,
            marker_positions: np.ndarray,
            marker_labels: List[str],
            frame_rate: float
        ) -> np.ndarray:
            """Extract features for LSTM input."""
            cfg = self.config
            vaxis = cfg.get_vertical_axis_index()
            n_frames = marker_positions.shape[0]

            # Find markers
            patterns = cfg.body_model.segment_marker_patterns
            pelvis_idx = find_marker_indices(marker_labels, patterns.get('pelvis', []))
            head_idx = find_marker_indices(marker_labels, patterns.get('head', []))

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
            com_pos = calculate_whole_body_com(
                marker_positions, marker_labels, cfg.body_model
            )

            # Calculate velocities
            com_vel = calculate_com_velocity(com_pos, frame_rate, cfg.units)
            com_vel = np.vstack([com_vel, com_vel[-1:]])

            # Calculate acceleration
            com_acc = calculate_com_acceleration(com_vel, frame_rate)
            com_acc = np.vstack([com_acc, com_acc[-1:]])

            # Calculate trunk tilt
            trunk_tilt = calculate_trunk_tilt(head_pos, pelvis_pos, vaxis)

            # Convert heights
            pelvis_h_m = pelvis_pos[:, vaxis] / 1000.0 if cfg.units == 'mm' else pelvis_pos[:, vaxis]
            head_h_m = head_pos[:, vaxis] / 1000.0 if cfg.units == 'mm' else head_pos[:, vaxis]

            # Calculate derived features
            vertical_velocity = com_vel[:, vaxis]
            acc_magnitude = np.linalg.norm(com_acc, axis=1) / 9.81
            angular_velocity = np.gradient(trunk_tilt) * frame_rate

            # MoS placeholder
            mos = np.zeros(n_frames)

            # Build feature array
            features = np.column_stack([
                com_vel[:, 0],
                com_vel[:, 1],
                com_vel[:, 2],
                trunk_tilt,
                mos,
                acc_magnitude,
                pelvis_h_m,
                head_h_m,
                vertical_velocity,
                angular_velocity,
            ])

            return features.astype(np.float32)

        def _extract_velocity(
            self,
            marker_positions: np.ndarray,
            marker_labels: List[str],
            frame_rate: float
        ) -> Optional[np.ndarray]:
            """Extract CoM velocity for activity classification."""
            try:
                com_pos = calculate_whole_body_com(
                    marker_positions, marker_labels, self.config.body_model
                )
                return calculate_com_velocity(com_pos, frame_rate, self.config.units)
            except:
                return None

        def _create_sequences(
            self,
            features: np.ndarray,
            overlap: int = 25
        ) -> np.ndarray:
            """Create sliding window sequences."""
            n_frames = features.shape[0]
            step = self.seq_length - overlap

            sequences = []
            for start in range(0, n_frames - self.seq_length + 1, step):
                end = start + self.seq_length
                sequences.append(features[start:end])

            return np.array(sequences) if sequences else np.array([])


    def analyze_c3d_with_lstm(
        filepath: str,
        model_path: str,
        config: DetectorConfig = None
    ) -> dict:
        """
        Convenience function for LSTM analysis.

        Args:
            filepath: Path to C3D file
            model_path: Path to trained model
            config: Optional detector configuration

        Returns:
            Dictionary with detection results
        """
        detector = LSTMDetector(config, model_path)
        result = detector.analyze(filepath)
        return result.to_dict()
