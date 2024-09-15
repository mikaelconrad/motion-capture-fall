#!/usr/bin/env python3
"""
AWS GPU training script for fall detection model.

This script handles:
- Data preparation and upload to S3
- EC2 GPU instance training
- Model download from S3

Usage:
    # Prepare and upload data
    python training/train_aws.py prepare --data-dir data/internal --labels data/labels.json

    # Start training on AWS
    python training/train_aws.py train --epochs 100 --instance-type g4dn.xlarge

    # Download trained model
    python training/train_aws.py download --model-name lstm_fall_v1.pt

Environment Variables Required:
    AWS_PROFILE=crypto-ml (or set in ~/.aws/credentials)
    AWS_REGION=eu-central-1 (optional, defaults from config)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
except ImportError:
    print("Note: python-dotenv not installed. Using system environment variables.")
    print("Install with: pip install python-dotenv")

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("Warning: boto3 not available. Install with: pip install boto3")

import numpy as np

from training.dataset import prepare_training_data_with_groups, save_features, FeatureConfig


# AWS Configuration
AWS_CONFIG = {
    'profile': os.environ.get('AWS_PROFILE', 'crypto-ml'),
    'region': os.environ.get('AWS_REGION', 'eu-central-1'),
    'bucket': os.environ.get('AWS_S3_BUCKET', 'crypto-ml-565319458516-eu-central-1'),
    's3_prefix': 'fall-detection',
}


def get_s3_client():
    """Get S3 client with configured profile."""
    session = boto3.Session(profile_name=AWS_CONFIG['profile'])
    return session.client('s3', region_name=AWS_CONFIG['region'])


def get_ec2_client():
    """Get EC2 client with configured profile."""
    session = boto3.Session(profile_name=AWS_CONFIG['profile'])
    return session.client('ec2', region_name=AWS_CONFIG['region'])


def upload_to_s3(local_path: str, s3_key: str):
    """Upload file to S3."""
    s3 = get_s3_client()
    bucket = AWS_CONFIG['bucket']

    print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Upload complete: s3://{bucket}/{s3_key}")


def download_from_s3(s3_key: str, local_path: str):
    """Download file from S3."""
    s3 = get_s3_client()
    bucket = AWS_CONFIG['bucket']

    print(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, s3_key, local_path)
    print(f"Download complete: {local_path}")


def prepare_data(args):
    """Prepare and upload training data to S3."""
    print("=" * 60)
    print("STEP 1: Extracting features from C3D files")
    print("=" * 60)

    config = FeatureConfig(sequence_length=args.seq_length)
    X, y, recording_ids, recording_names = prepare_training_data_with_groups(
        args.labels, args.data_dir, config
    )

    print(f"\nExtracted features:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Falls: {sum(y)}, Non-falls: {len(y) - sum(y)}")

    # Save to local file
    local_path = 'training/data/features.npz'
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'created': datetime.now().isoformat(),
        'source_dir': args.data_dir,
        'labels_file': args.labels,
        'sequence_length': args.seq_length,
    }
    save_features(
        X, y, local_path, metadata,
        recording_ids=recording_ids,
        recording_names=recording_names
    )

    # Upload to S3 if boto3 available
    if BOTO3_AVAILABLE and args.upload:
        print("\n" + "=" * 60)
        print("STEP 2: Uploading to S3")
        print("=" * 60)

        s3_key = f"{AWS_CONFIG['s3_prefix']}/training/features.npz"
        upload_to_s3(local_path, s3_key)

        # Upload training config
        train_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'hidden_size': args.hidden_size,
            'dropout': args.dropout,
            'augment_multiplier': args.augment_multiplier,
        }
        config_path = 'training/data/train_config.json'
        with open(config_path, 'w') as f:
            json.dump(train_config, f, indent=2)

        s3_config_key = f"{AWS_CONFIG['s3_prefix']}/training/train_config.json"
        upload_to_s3(config_path, s3_config_key)

        print("\nData uploaded successfully!")
        print(f"  Features: s3://{AWS_CONFIG['bucket']}/{s3_key}")
        print(f"  Config: s3://{AWS_CONFIG['bucket']}/{s3_config_key}")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


def generate_training_script():
    """Generate the training script to run on EC2."""
    return '''#!/bin/bash
set -e

echo "Installing dependencies..."
pip install torch numpy

echo "Downloading data from S3..."
aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/training/features.npz /tmp/features.npz
aws s3 cp s3://${S3_BUCKET}/${S3_PREFIX}/training/train_config.json /tmp/train_config.json

echo "Running training..."
python3 << 'PYTHON_SCRIPT'
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load config
with open('/tmp/train_config.json') as f:
    config = json.load(f)

# Load data
data = np.load('/tmp/features.npz')
X = data['X']
y = data['y']

print(f"Loaded data: X={X.shape}, y={y.shape}")

# Simple LSTM model
class FallDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FallDetectorLSTM(
    input_size=X.shape[2],
    hidden_size=config.get('hidden_size', 64),
    dropout=config.get('dropout', 0.3)
).to(device)

# Split data (by recording if available)
recording_ids = data['recording_ids'] if 'recording_ids' in data else None
if recording_ids is not None and len(np.unique(recording_ids)) > 1:
    unique_ids = np.unique(recording_ids)
    np.random.shuffle(unique_ids)
    split = int(len(unique_ids) * 0.8)
    split = max(1, min(split, len(unique_ids) - 1))
    train_ids = unique_ids[:split]
    val_ids = unique_ids[split:]
    train_mask = np.isin(recording_ids, train_ids)
    val_mask = ~train_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Split by recording: {len(train_ids)} train, {len(val_ids)} val")
else:
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_val, y_val = X[idx[split:]], y[idx[split:]]

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=config.get('batch_size', 32), shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
    batch_size=config.get('batch_size', 32)
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

best_loss = float('inf')
for epoch in range(config.get('epochs', 100)):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            val_loss += criterion(model(X_batch), y_batch).item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), '/tmp/model.pt')

print(f"Training complete. Best val_loss: {best_loss:.4f}")
PYTHON_SCRIPT

echo "Uploading model to S3..."
aws s3 cp /tmp/model.pt s3://${S3_BUCKET}/${S3_PREFIX}/models/lstm_fall_v1.pt

echo "Training complete!"
'''


def start_training(args):
    """Start training on AWS EC2."""
    if not BOTO3_AVAILABLE:
        print("Error: boto3 is required for AWS training. Install with: pip install boto3")
        sys.exit(1)

    print("=" * 60)
    print("Starting AWS GPU Training")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Instance type: {args.instance_type}")
    print(f"  Region: {AWS_CONFIG['region']}")
    print(f"  S3 Bucket: {AWS_CONFIG['bucket']}")
    print(f"  Spot instance: {args.spot}")

    # This is a simplified version - full implementation would:
    # 1. Launch EC2 instance with user data script
    # 2. Wait for training to complete
    # 3. Terminate instance

    print("\n[Simplified Mode]")
    print("For full AWS training, you can:")
    print("1. Launch an EC2 GPU instance (g4dn.xlarge recommended)")
    print("2. SSH into the instance")
    print("3. Run the following commands:")
    print()
    print("  # Set environment")
    print(f"  export S3_BUCKET={AWS_CONFIG['bucket']}")
    print(f"  export S3_PREFIX={AWS_CONFIG['s3_prefix']}")
    print()
    print("  # Download and run training")
    print("  aws s3 cp s3://$S3_BUCKET/$S3_PREFIX/training/features.npz .")
    print("  python training/train.py --features features.npz --epochs 100")
    print()
    print("  # Upload model")
    print("  aws s3 cp models/lstm_fall_v1.pt s3://$S3_BUCKET/$S3_PREFIX/models/")

    # Save training script for reference
    script_path = 'training/aws_train_script.sh'
    with open(script_path, 'w') as f:
        f.write(generate_training_script())
    print(f"\nTraining script saved to: {script_path}")


def download_model(args):
    """Download trained model from S3."""
    if not BOTO3_AVAILABLE:
        print("Error: boto3 is required. Install with: pip install boto3")
        sys.exit(1)

    s3_key = f"{AWS_CONFIG['s3_prefix']}/models/{args.model_name}"
    local_path = f"models/{args.model_name}"

    download_from_s3(s3_key, local_path)
    print(f"\nModel downloaded to: {local_path}")


def list_models(args):
    """List available models in S3."""
    if not BOTO3_AVAILABLE:
        print("Error: boto3 is required. Install with: pip install boto3")
        sys.exit(1)

    s3 = get_s3_client()
    bucket = AWS_CONFIG['bucket']
    prefix = f"{AWS_CONFIG['s3_prefix']}/models/"

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            print("Available models:")
            for obj in response['Contents']:
                key = obj['Key']
                size = obj['Size'] / 1024  # KB
                modified = obj['LastModified']
                name = key.replace(prefix, '')
                print(f"  {name:30s} {size:8.1f} KB  {modified}")
        else:
            print("No models found in S3.")
    except ClientError as e:
        print(f"Error listing models: {e}")


def main():
    parser = argparse.ArgumentParser(description='AWS GPU training for fall detection')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare and upload training data')
    prepare_parser.add_argument('--data-dir', default='data/internal')
    prepare_parser.add_argument('--labels', default='data/labels.json')
    prepare_parser.add_argument('--seq-length', type=int, default=50)
    prepare_parser.add_argument('--epochs', type=int, default=100)
    prepare_parser.add_argument('--batch-size', type=int, default=32)
    prepare_parser.add_argument('--lr', type=float, default=0.001)
    prepare_parser.add_argument('--hidden-size', type=int, default=64)
    prepare_parser.add_argument('--dropout', type=float, default=0.3)
    prepare_parser.add_argument('--augment-multiplier', type=int, default=50)
    prepare_parser.add_argument('--upload', action='store_true', help='Upload to S3')

    # Train command
    train_parser = subparsers.add_parser('train', help='Start training on AWS')
    train_parser.add_argument('--instance-type', default='g4dn.xlarge')
    train_parser.add_argument('--spot', action='store_true', help='Use spot instances')
    train_parser.add_argument('--epochs', type=int, default=100)

    # Download command
    download_parser = subparsers.add_parser('download', help='Download model from S3')
    download_parser.add_argument('--model-name', default='lstm_fall_v1.pt')

    # List command
    list_parser = subparsers.add_parser('list', help='List models in S3')

    args = parser.parse_args()

    if args.command == 'prepare':
        prepare_data(args)
    elif args.command == 'train':
        start_training(args)
    elif args.command == 'download':
        download_model(args)
    elif args.command == 'list':
        list_models(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
