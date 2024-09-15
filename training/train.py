#!/usr/bin/env python3
"""
Local training script for fall detection LSTM model.

Usage:
    python training/train.py --data-dir data/internal --labels data/labels.json
    python training/train.py --features training/data/features.npz --epochs 100
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

from training.dataset import (
    prepare_training_data,
    prepare_training_data_with_groups,
    save_features,
    load_features,
    FeatureConfig,
)
from training.augmentation import augment_dataset, AugmentationConfig
from training.models.lstm import FallDetectorLSTM, create_model


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def split_by_recording(
    X: np.ndarray,
    y: np.ndarray,
    recording_ids: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = None
) -> tuple:
    """Split train/val sets by recording IDs."""
    unique_ids = np.unique(recording_ids)
    if len(unique_ids) < 2:
        raise ValueError("Need at least 2 recordings to split by recording.")

    rng = np.random.default_rng(seed)
    unique_ids = np.array(unique_ids, copy=True)
    rng.shuffle(unique_ids)

    split_idx = int(len(unique_ids) * train_ratio)
    split_idx = max(1, min(split_idx, len(unique_ids) - 1))

    train_ids = unique_ids[:split_idx]
    val_ids = unique_ids[split_idx:]

    train_mask = np.isin(recording_ids, train_ids)
    val_mask = ~train_mask

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    return X_train, y_train, X_val, y_val, train_ids, val_ids


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = (outputs >= 0.5).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    avg_loss = total_loss / len(val_loader)

    return avg_loss, accuracy


def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    dropout: float = 0.3,
    augment: bool = True,
    augment_multiplier: int = 50,
    output_path: str = 'models/lstm_fall_v1.pt',
    early_stopping_patience: int = 10,
    device: torch.device = None,
    recording_ids: np.ndarray = None,
    recording_names: list = None,
    train_ratio: float = 0.8,
    split_seed: int = None
):
    """
    Train the fall detection model.

    Args:
        X: Feature sequences (n_samples, seq_len, n_features)
        y: Labels (n_samples,)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_size: LSTM hidden size
        dropout: Dropout probability
        augment: Apply data augmentation
        augment_multiplier: Augmentation multiplier
        output_path: Path to save model
        early_stopping_patience: Early stopping patience
        device: Device to use
    """
    if device is None:
        device = get_device()

    print(f"Using device: {device}")
    print(f"Original dataset: {len(y)} samples ({sum(y)} falls, {len(y) - sum(y)} non-falls)")

    # Split into train/val before augmentation
    if recording_ids is not None:
        try:
            X_train, y_train, X_val, y_val, train_ids, val_ids = split_by_recording(
                X, y, recording_ids, train_ratio=train_ratio, seed=split_seed
            )
            if recording_names:
                train_names = [recording_names[i] for i in train_ids]
                val_names = [recording_names[i] for i in val_ids]
                print(f"Train recordings: {train_names}")
                print(f"Val recordings: {val_names}")
        except ValueError as e:
            print(f"Warning: {e} Falling back to sequence-level split.")
            recording_ids = None

    if recording_ids is None:
        n_samples = len(y)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

    # Augment training data only
    if augment:
        print(f"Augmenting training data with {augment_multiplier}x multiplier...")
        X_train, y_train = augment_dataset(
            X_train, y_train,
            time_warp=True,
            add_noise=True,
            random_crop=True,
            target_multiplier=augment_multiplier
        )
        print(f"Augmented training samples: {len(y_train)}")

    print(f"Train samples: {len(y_train)} | Val samples: {len(y_val)}")

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_size = X.shape[2]
    model = create_model(
        model_type='lstm',
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=dropout
    ).to(device)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'dropout': dropout,
                }
            }, output_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("-" * 60)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train fall detection LSTM model')

    # Data options
    parser.add_argument('--data-dir', type=str, default='data/internal',
                        help='Directory with C3D files')
    parser.add_argument('--labels', type=str, default='data/labels.json',
                        help='Path to labels.json')
    parser.add_argument('--features', type=str,
                        help='Path to pre-extracted features.npz (overrides --data-dir)')

    # Training options
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    # Model options
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='LSTM hidden size')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='Sequence length')

    # Augmentation
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--augment-multiplier', type=int, default=50,
                        help='Augmentation multiplier')

    # Output
    parser.add_argument('--output', type=str, default='models/lstm_fall_v1.pt',
                        help='Output model path')

    # Device
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps', 'auto'],
                        default='auto', help='Device to use')

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    # Get device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)

    # Load data
    if args.features:
        print(f"Loading features from {args.features}")
        X, y, _, recording_ids, recording_names = load_features(
            args.features, return_groups=True
        )
    else:
        print(f"Extracting features from {args.data_dir}")
        config = FeatureConfig(sequence_length=args.seq_length)
        X, y, recording_ids, recording_names = prepare_training_data_with_groups(
            args.labels, args.data_dir, config
        )

    # Train
    train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        augment=not args.no_augment,
        augment_multiplier=args.augment_multiplier,
        output_path=args.output,
        device=device,
        recording_ids=recording_ids,
        recording_names=recording_names
    )


if __name__ == '__main__':
    main()
