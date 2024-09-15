"""
Data augmentation for fall detection training.

Provides augmentation techniques suitable for time series motion data,
helping to expand small datasets for neural network training.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    time_warp_enabled: bool = True
    time_warp_sigma: float = 0.2
    time_warp_knots: int = 4

    noise_enabled: bool = True
    noise_std: float = 0.02

    random_crop_enabled: bool = True
    random_crop_min_ratio: float = 0.8

    channel_dropout_enabled: bool = True
    channel_dropout_prob: float = 0.1

    magnitude_warp_enabled: bool = True
    magnitude_warp_sigma: float = 0.1


def time_warp(
    x: np.ndarray,
    sigma: float = 0.2,
    knots: int = 4
) -> np.ndarray:
    """
    Apply time warping augmentation.

    Randomly stretches and compresses different parts of the sequence
    while maintaining the overall length.

    Args:
        x: Input sequence (seq_len, n_features)
        sigma: Standard deviation for warp magnitude
        knots: Number of warp knots

    Returns:
        Time-warped sequence
    """
    seq_len = x.shape[0]

    # Generate random warp path
    orig_steps = np.arange(seq_len)

    # Create knot points with random offsets
    knot_positions = np.linspace(0, seq_len - 1, knots + 2)
    warp_offsets = np.random.normal(0, sigma, knots + 2)
    warp_offsets[0] = 0  # Fix endpoints
    warp_offsets[-1] = 0

    warped_positions = knot_positions + warp_offsets * (seq_len / knots)
    warped_positions = np.clip(warped_positions, 0, seq_len - 1)
    warped_positions = np.sort(warped_positions)

    # Interpolate to get mapping
    warp_map = np.interp(
        orig_steps,
        warped_positions,
        np.linspace(0, seq_len - 1, knots + 2)
    )

    # Apply warp
    warped = np.zeros_like(x)
    for i, t in enumerate(warp_map):
        t_floor = int(np.floor(t))
        t_ceil = min(t_floor + 1, seq_len - 1)
        alpha = t - t_floor
        warped[i] = (1 - alpha) * x[t_floor] + alpha * x[t_ceil]

    return warped


def add_gaussian_noise(
    x: np.ndarray,
    std: float = 0.02
) -> np.ndarray:
    """
    Add Gaussian noise to sequence.

    Args:
        x: Input sequence (seq_len, n_features)
        std: Noise standard deviation (relative to data std)

    Returns:
        Noisy sequence
    """
    # Scale noise by feature standard deviation
    feature_std = np.std(x, axis=0, keepdims=True) + 1e-8
    noise = np.random.normal(0, std, x.shape) * feature_std
    return x + noise


def random_crop(
    x: np.ndarray,
    min_ratio: float = 0.8
) -> np.ndarray:
    """
    Randomly crop and resize sequence.

    Args:
        x: Input sequence (seq_len, n_features)
        min_ratio: Minimum crop ratio

    Returns:
        Cropped and resized sequence
    """
    seq_len = x.shape[0]
    target_len = seq_len

    # Random crop ratio
    crop_ratio = np.random.uniform(min_ratio, 1.0)
    crop_len = int(seq_len * crop_ratio)

    # Random start position
    max_start = seq_len - crop_len
    start = np.random.randint(0, max_start + 1)
    end = start + crop_len

    # Crop
    cropped = x[start:end]

    # Resize back to original length using interpolation
    orig_indices = np.linspace(0, crop_len - 1, target_len)
    resized = np.zeros((target_len, x.shape[1]))
    for i, t in enumerate(orig_indices):
        t_floor = int(np.floor(t))
        t_ceil = min(t_floor + 1, crop_len - 1)
        alpha = t - t_floor
        resized[i] = (1 - alpha) * cropped[t_floor] + alpha * cropped[t_ceil]

    return resized


def channel_dropout(
    x: np.ndarray,
    prob: float = 0.1
) -> np.ndarray:
    """
    Randomly zero out feature channels.

    Args:
        x: Input sequence (seq_len, n_features)
        prob: Probability of dropping each channel

    Returns:
        Sequence with dropped channels
    """
    mask = np.random.binomial(1, 1 - prob, x.shape[1])
    return x * mask


def magnitude_warp(
    x: np.ndarray,
    sigma: float = 0.1,
    knots: int = 4
) -> np.ndarray:
    """
    Randomly scale magnitude over time.

    Args:
        x: Input sequence (seq_len, n_features)
        sigma: Standard deviation for scale factors
        knots: Number of knot points

    Returns:
        Magnitude-warped sequence
    """
    seq_len = x.shape[0]

    # Generate smooth scaling curve
    knot_positions = np.linspace(0, seq_len - 1, knots + 2)
    scale_factors = np.random.normal(1.0, sigma, knots + 2)

    # Ensure reasonable scaling
    scale_factors = np.clip(scale_factors, 0.5, 1.5)

    # Interpolate to full sequence
    full_scale = np.interp(
        np.arange(seq_len),
        knot_positions,
        scale_factors
    )

    return x * full_scale[:, np.newaxis]


def speed_change(
    x: np.ndarray,
    speed_factor: float = None
) -> np.ndarray:
    """
    Change playback speed of sequence.

    Args:
        x: Input sequence (seq_len, n_features)
        speed_factor: Speed multiplier (None for random)

    Returns:
        Speed-modified sequence
    """
    if speed_factor is None:
        speed_factor = np.random.uniform(0.8, 1.2)

    seq_len = x.shape[0]
    new_len = int(seq_len / speed_factor)
    new_len = max(new_len, 2)

    # Resample
    orig_indices = np.linspace(0, seq_len - 1, new_len)
    resampled = np.zeros((new_len, x.shape[1]))
    for i, t in enumerate(orig_indices):
        t_floor = int(np.floor(t))
        t_ceil = min(t_floor + 1, seq_len - 1)
        alpha = t - t_floor
        resampled[i] = (1 - alpha) * x[t_floor] + alpha * x[t_ceil]

    # Resize back to original length
    target_indices = np.linspace(0, new_len - 1, seq_len)
    resized = np.zeros((seq_len, x.shape[1]))
    for i, t in enumerate(target_indices):
        t_floor = int(np.floor(t))
        t_ceil = min(t_floor + 1, new_len - 1)
        alpha = t - t_floor
        resized[i] = (1 - alpha) * resampled[t_floor] + alpha * resampled[t_ceil]

    return resized


class AugmentationPipeline:
    """
    Pipeline for applying multiple augmentations.

    Example:
        pipeline = AugmentationPipeline([
            TimeWarpAugmentation(sigma=0.2),
            NoiseAugmentation(std=0.02),
        ])
        X_aug, y_aug = pipeline.transform(X, y, target_multiplier=50)
    """

    def __init__(self, config: AugmentationConfig = None):
        """
        Initialize augmentation pipeline.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to a single sequence.

        Args:
            x: Input sequence (seq_len, n_features)

        Returns:
            Augmented sequence
        """
        result = x.copy()

        if self.config.time_warp_enabled and np.random.random() < 0.5:
            result = time_warp(
                result,
                sigma=self.config.time_warp_sigma,
                knots=self.config.time_warp_knots
            )

        if self.config.noise_enabled and np.random.random() < 0.7:
            result = add_gaussian_noise(
                result,
                std=self.config.noise_std
            )

        if self.config.random_crop_enabled and np.random.random() < 0.3:
            result = random_crop(
                result,
                min_ratio=self.config.random_crop_min_ratio
            )

        if self.config.channel_dropout_enabled and np.random.random() < 0.2:
            result = channel_dropout(
                result,
                prob=self.config.channel_dropout_prob
            )

        if self.config.magnitude_warp_enabled and np.random.random() < 0.3:
            result = magnitude_warp(
                result,
                sigma=self.config.magnitude_warp_sigma
            )

        return result

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_multiplier: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset to target size.

        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Labels (n_samples,)
            target_multiplier: Target size multiplier

        Returns:
            Augmented (X, y) tuple
        """
        n_samples = X.shape[0]
        target_size = n_samples * target_multiplier

        X_aug = [X]  # Include original
        y_aug = [y]

        while sum(len(arr) for arr in X_aug) < target_size:
            # Sample random index
            idx = np.random.randint(0, n_samples)

            # Augment
            x_new = self.augment_single(X[idx])
            X_aug.append(x_new[np.newaxis, ...])
            y_aug.append(y[idx:idx+1])

        X_result = np.concatenate(X_aug, axis=0)
        y_result = np.concatenate(y_aug, axis=0)

        # Shuffle
        indices = np.random.permutation(len(X_result))
        return X_result[indices], y_result[indices]


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    time_warp: bool = True,
    add_noise: bool = True,
    random_crop: bool = True,
    target_multiplier: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to augment a dataset.

    Args:
        X: Input sequences (n_samples, seq_len, n_features)
        y: Labels (n_samples,)
        time_warp: Enable time warping
        add_noise: Enable noise addition
        random_crop: Enable random cropping
        target_multiplier: Target size multiplier

    Returns:
        Augmented (X, y) tuple
    """
    config = AugmentationConfig(
        time_warp_enabled=time_warp,
        noise_enabled=add_noise,
        random_crop_enabled=random_crop,
    )
    pipeline = AugmentationPipeline(config)
    return pipeline.transform(X, y, target_multiplier)
