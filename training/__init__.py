"""
ML Training Infrastructure for Fall Detection.

This package provides tools for training LSTM-based fall detectors,
including data loading, augmentation, and AWS GPU training support.
"""

from .dataset import prepare_training_data, save_features, load_features, FeatureConfig

# PyTorch-dependent classes (optional)
try:
    from .dataset import FallDataset, create_data_loaders
    from .augmentation import AugmentationPipeline
    TORCH_AVAILABLE = True
except ImportError:
    FallDataset = None
    create_data_loaders = None
    AugmentationPipeline = None
    TORCH_AVAILABLE = False

__all__ = [
    'prepare_training_data',
    'save_features',
    'load_features',
    'FeatureConfig',
    'FallDataset',
    'create_data_loaders',
    'AugmentationPipeline',
    'TORCH_AVAILABLE',
]
