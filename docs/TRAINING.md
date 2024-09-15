# ML Training Guide

Train the LSTM fall detector on windowed kinematic features extracted from C3D data.

## Dependencies

```bash
pip install torch torchvision tensorboard
pip install boto3 python-dotenv  # Optional, for AWS training
```

## Data Preparation

- Labels: `data/labels.json`
- C3D files: `data/internal`

```python
from training.dataset import prepare_training_data

X, y = prepare_training_data(
    labels_path="data/labels.json",
    data_dir="data/internal",
    sequence_length=50,
    overlap=25
)

print(X.shape)  # (n_samples, seq_len, n_features)
```

Augmentations (see `training/augmentation.py`): time warp, noise, random crop, channel dropout.

## Local Training

```bash
python training/train.py \
    --data-dir data/internal \
    --labels data/labels.json \
    --epochs 100 \
    --batch-size 32 \
    --output models/lstm_fall_v1.pt
```

Notes:
- Train/val splits are by recording to avoid leakage.
- Augmentation is applied to the training split only.

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `data/internal` | Directory with C3D files |
| `--labels` | `data/labels.json` | Labels file path |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--seq-length` | 50 | Sequence length (frames) |
| `--hidden-size` | 64 | LSTM hidden size |
| `--dropout` | 0.3 | Dropout rate |
| `--output` | `models/model.pt` | Output model path |
| `--device` | `auto` | Device (cpu, cuda, mps, auto) |

## Model Summary

- Two-layer LSTM (first layer bidirectional), followed by a small MLP classifier.
- Sequence length: 50 frames, sliding window overlap: 25 frames.
- Output: fall probability per window; max probability is used for detection.

### Input Features (10 per frame)

| Index | Feature | Unit | Description |
|-------|---------|------|-------------|
| 0 | com_velocity_x | m/s | CoM velocity X |
| 1 | com_velocity_y | m/s | CoM velocity Y (vertical) |
| 2 | com_velocity_z | m/s | CoM velocity Z |
| 3 | trunk_tilt | deg | Trunk angle from vertical |
| 4 | margin_of_stability | m | MoS value |
| 5 | acceleration_mag | g | Acceleration magnitude |
| 6 | pelvis_height | m | Pelvis height |
| 7 | head_height | m | Head height |
| 8 | vertical_velocity | m/s | Downward velocity |
| 9 | angular_velocity | rad/s | Body rotation rate |

## Optional: AWS Training

```bash
python training/train_aws.py prepare \
    --data-dir data/internal \
    --labels data/labels.json \
    --output training/data/features.npz

python training/train_aws.py upload \
    --local-path training/data/features.npz \
    --s3-path s3://<bucket>/training/

python training/train_aws.py train \
    --instance-type g4dn.xlarge \
    --s3-data s3://<bucket>/training/features.npz \
    --epochs 100 \
    --spot

python training/train_aws.py download \
    --s3-path s3://<bucket>/models/lstm_fall_v1.pt \
    --local-path models/lstm_fall_v1.pt
```

## Optional: External Datasets

- CMU mocap: `tools/download_datasets.py cmu --output data/external/cmu_mocap`
- Converters: `training/external/cmu_converter.py`, `training/external/sisfall_converter.py`

## Evaluation

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred > 0.5))
```

## Tips

- Out of memory: reduce `--batch-size` or `--seq-length`.
- Overfitting: increase `--dropout` or add more augmentation.
