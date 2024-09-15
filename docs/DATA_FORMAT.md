# Data Format & Ground Truth

This document describes the data formats, labeling conventions, and ground truth schema used in the fall detection system.

---

## Directory Structure

```
data/
├── labels.json              # Ground truth labels for all videos
├── internal/                # Your motion capture data
│   ├── falls/              # C3D files containing falls
│   │   ├── video-02-epic-fall.c3d
│   │   └── video-04-short-fall-standing-up-try.c3d
│   └── non_falls/          # C3D files without falls
│       ├── video-05-turn.c3d
│       ├── video-06-under-the-mattress.c3d
│       ├── video-09-meditation.c3d
│       └── video-11-dance.c3d
└── external/               # Downloaded external datasets
    ├── cmu_mocap/          # CMU Motion Capture Database
    └── sisfall/            # SisFall dataset (if available)
```

---

## Labels File Schema (labels.json)

```json
{
  "version": "1.0",
  "created": "2026-01-02",
  "description": "Ground truth labels for fall detection evaluation",
  "videos": [
    {
      "id": "video-02-epic-fall",
      "file": "internal/falls/video-02-epic-fall.c3d",
      "fall": true,
      "fall_type": "collapse",
      "confidence": 1.0,
      "notes": "Clear collapse from standing position",
      "phases": {
        "pre_fall": [0, 120],
        "initiation": [121, 145],
        "descent": [146, 185],
        "impact": [186, 195],
        "post_fall": [196, 500]
      }
    },
    {
      "id": "video-04-short-fall-standing-up-try",
      "file": "internal/falls/video-04-short-fall-standing-up-try.c3d",
      "fall": true,
      "fall_type": "unknown",
      "confidence": 0.9,
      "notes": "Short fall during standing attempt"
    },
    {
      "id": "video-05-turn",
      "file": "internal/non_falls/video-05-turn.c3d",
      "fall": false,
      "activity": "turning",
      "notes": "Normal turning motion"
    },
    {
      "id": "video-06-under-the-mattress",
      "file": "internal/non_falls/video-06-under-the-mattress.c3d",
      "fall": false,
      "activity": "controlled_descent",
      "notes": "Intentional descent onto mattress"
    },
    {
      "id": "video-09-meditation",
      "file": "internal/non_falls/video-09-meditation.c3d",
      "fall": false,
      "activity": "meditation_lying",
      "notes": "Slow controlled transition to lying"
    },
    {
      "id": "video-11-dance",
      "file": "internal/non_falls/video-11-dance.c3d",
      "fall": false,
      "activity": "rhythmic_motion",
      "notes": "Dancing with rapid movements"
    }
  ]
}
```

---

## Field Definitions

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the video |
| `file` | string | Path to C3D file relative to data/ |
| `fall` | boolean | Ground truth: does this video contain a fall? |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `fall_type` | string | Type of fall: "slip", "trip", "collapse", "unknown" |
| `activity` | string | Activity type for non-falls |
| `confidence` | float | Labeler confidence (0.0-1.0) |
| `notes` | string | Human-readable description |
| `phases` | object | Frame ranges for each fall phase |

### Fall Types

| Type | Description | Kinematic Signature |
|------|-------------|---------------------|
| `slip` | Foot slides, backward fall | Foot acceleration, backward CoM |
| `trip` | Foot catches, forward fall | Sudden foot stop, forward CoM |
| `collapse` | Vertical descent | Knee flexion, minimal horizontal motion |
| `unknown` | Cannot determine | Mixed or unclear pattern |

### Activity Types (Non-Falls)

| Activity | Description |
|----------|-------------|
| `turning` | Rotational movement while standing |
| `controlled_descent` | Intentional movement to lower position |
| `meditation_lying` | Slow transition to lying position |
| `rhythmic_motion` | Dancing or exercise movements |
| `walking` | Normal gait |
| `sitting` | Transition to seated position |

---

## Phase Annotations

For labeled falls, phase annotations specify frame ranges:

```json
"phases": {
  "pre_fall": [start_frame, end_frame],
  "initiation": [start_frame, end_frame],
  "descent": [start_frame, end_frame],
  "impact": [start_frame, end_frame],
  "post_fall": [start_frame, end_frame]
}
```

### Phase Definitions

| Phase | Description | Detection Criteria |
|-------|-------------|---------------------|
| `pre_fall` | Normal activity before fall | MoS stable, normal motion |
| `initiation` | Balance lost, fall begins | MoS < 0, trunk tilt increasing |
| `descent` | Uncontrolled falling | High downward velocity |
| `impact` | Ground contact | Acceleration spike |
| `post_fall` | After impact | Low velocity, final posture |

---

## Original Format (Legacy)

The original `results_v3.txt` format:

```
Run: 2025-08-17 16:29:38

video-02-epic-fall: expected=True, predicted=True
video-04-short-fall-standing-up-try: expected=True, predicted=True
video-05-turn: expected=False, predicted=False
video-06-under-the-mattress: expected=False, predicted=False
video-09-meditation: expected=False, predicted=False
video-11-dance: expected=False, predicted=False

TP=2 TN=4 FP=0 FN=0
Accuracy=1.000
```

This format is converted to `labels.json` for structured access.

---

## Evaluation Metrics

| Metric | Reward/Penalty | Rationale |
|--------|----------------|-----------|
| True Positive (TP) | +2 | Correctly detected fall (most important) |
| True Negative (TN) | +1 | Correctly detected non-fall |
| False Positive (FP) | -3 | False alarm (disrupts clinical workflow) |
| False Negative (FN) | -2 | Missed fall (dangerous for patient) |

### Score Calculation

```python
score = 2 * TP + 1 * TN - 3 * FP - 2 * FN
```

### Why Asymmetric Penalties?

- **FP (-3)**: False alarms cause alarm fatigue in clinical settings
- **FN (-2)**: Missed falls are dangerous but less disruptive than constant alarms
- **TP (+2)**: Detecting falls is the primary goal
- **TN (+1)**: Correctly ignoring non-falls is expected behavior

---

## C3D File Format

### Structure

C3D files are binary files containing:
- **Header**: Metadata (frame rate, marker count, etc.)
- **Parameters**: Detailed configuration
- **Point Data**: 3D marker positions for each frame

### Reading C3D Files

```python
from src.utils.c3d_reader import read_c3d

data = read_c3d("path/to/file.c3d")

print(f"Frame rate: {data.frame_rate} Hz")
print(f"Duration: {data.n_frames / data.frame_rate:.2f} seconds")
print(f"Markers: {data.marker_labels}")
print(f"Positions shape: {data.positions.shape}")  # (n_frames, n_markers, 3)
```

### Coordinate System

Default coordinate system (may vary by capture system):
- X: Forward/Backward (anterior/posterior)
- Y: Up/Down (vertical)
- Z: Left/Right (medial/lateral)

The system auto-detects the vertical axis based on data range.

---

## Adding New Data

### Step 1: Place C3D File

```bash
# For falls
cp new_recording.c3d data/internal/falls/

# For non-falls
cp new_recording.c3d data/internal/non_falls/
```

### Step 2: Update labels.json

```json
{
  "id": "new_recording",
  "file": "internal/falls/new_recording.c3d",
  "fall": true,
  "fall_type": "trip",
  "notes": "Trip over obstacle"
}
```

### Step 3: Validate

```bash
python tools/evaluate.py --validate
```

---

## Updates

| Date | Change |
|------|--------|
| 2026-01-02 | Initial documentation created |
