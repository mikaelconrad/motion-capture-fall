"""
LSTM-based fall detection model.

This module implements a bidirectional LSTM network for fall detection
from motion capture time series data.

Architecture inspired by:
    Hu, X., & Qu, X. (2016). Pre-impact fall detection.
    BioMedical Engineering OnLine, 15, 87.
"""

import torch
import torch.nn as nn
from typing import Optional


class FallDetectorLSTM(nn.Module):
    """
    LSTM-based fall detector.

    Architecture:
        Input: (batch, seq_len, n_features)
        → LSTM(hidden_size, bidirectional=True)
        → Dropout
        → LSTM(hidden_size // 2)
        → Dense(hidden_size // 2, ReLU)
        → Dropout
        → Dense(1, Sigmoid)
        Output: fall_probability (batch, 1)

    Features (10 per frame):
        - com_velocity_x, com_velocity_y, com_velocity_z (m/s)
        - trunk_tilt_deg (degrees)
        - margin_of_stability (m)
        - acceleration_magnitude (g)
        - pelvis_height (m)
        - head_height (m)
        - vertical_velocity (m/s)
        - angular_velocity (deg/s)
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize the LSTM fall detector.

        Args:
            input_size: Number of features per timestep
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # First LSTM layer (bidirectional)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0  # No dropout for single layer
        )

        lstm1_output_size = hidden_size * (2 if bidirectional else 1)

        # Dropout after first LSTM
        self.dropout1 = nn.Dropout(dropout)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=lstm1_output_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            Fall probability (batch, 1)
        """
        # First LSTM
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # Second LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)

        # Use last timestep output
        last_output = lstm2_out[:, -1, :]

        # Classify
        return self.classifier(last_output)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary prediction.

        Args:
            x: Input tensor (batch, seq_len, input_size)
            threshold: Classification threshold

        Returns:
            Binary predictions (batch,)
        """
        with torch.no_grad():
            probs = self.forward(x)
            return (probs >= threshold).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            Probabilities (batch,)
        """
        with torch.no_grad():
            return self.forward(x).squeeze(-1)


class FallDetectorLSTMWithAttention(nn.Module):
    """
    LSTM with attention mechanism for fall detection.

    Adds temporal attention to focus on critical moments (e.g., impact).
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

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.

        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            Fall probability (batch, 1)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_size)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, lstm_output_size)

        # Classify
        return self.classifier(context)


def create_model(
    model_type: str = 'lstm',
    input_size: int = 10,
    hidden_size: int = 64,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fall detection models.

    Args:
        model_type: 'lstm' or 'lstm_attention'
        input_size: Number of input features
        hidden_size: LSTM hidden size
        **kwargs: Additional model arguments

    Returns:
        Initialized model
    """
    if model_type == 'lstm':
        return FallDetectorLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs
        )
    elif model_type == 'lstm_attention':
        return FallDetectorLSTMWithAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
