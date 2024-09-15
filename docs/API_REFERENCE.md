# API Reference

This document describes the REST API endpoints and programmatic usage of the Fall Detection System.

---

## REST API

The Flask server runs on port 5003 by default.

### Base URL

```
http://localhost:5003/api/v2
```

---

## Endpoints

### POST /api/v2/analyze

Analyze a C3D file for fall detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body: C3D file upload (field name: `file`)
- Query Parameters:
  - `detector`: Detection method - `rules` (default) or `lstm`

**Examples:**
```bash
# Using rules-based detector (default)
curl -X POST \
  -F "file=@video-02-epic-fall.c3d" \
  http://localhost:5003/api/v2/analyze

# Using LSTM detector (requires PyTorch)
curl -X POST \
  -F "file=@video-02-epic-fall.c3d" \
  "http://localhost:5003/api/v2/analyze?detector=lstm"
```

**Response:**
```json
{
  "fall_detected": true,
  "confidence": 92.5,
  "activity_type": "falling",
  "characteristics": [
    "rapid_descent",
    "high_impact",
    "horizontal_trunk"
  ],
  "metrics": {
    "max_velocity_ms": -2.3,
    "min_vertical_velocity_ms": -2.8,
    "height_drop_m": 1.2,
    "max_trunk_tilt_deg": 87.5,
    "impact_zscore": 4.2,
    "descent_duration_s": 0.45
  },
  "timeline_data": {
    "pelvis_height": [0.95, 0.94, 0.92, ...],
    "head_height": [1.65, 1.64, 1.60, ...],
    "vertical_velocity": [-0.1, -0.3, -0.8, ...],
    "trunk_tilt_deg": [5, 8, 15, ...],
    "frame_rate": 50.0,
    "n_frames": 500,
    "impact_occurred": true,
    "impact_time_s": 3.2,
    "descent_start_time_s": 2.9,
    "descent_end_time_s": 3.7
  }
}
```

---

### POST /api/v2/classify-fall-type

Classify the type of a detected fall.

**Request:**
- Content-Type: `multipart/form-data`
- Body: C3D file upload

**Response:**
```json
{
  "success": true,
  "fall_type": "slip",
  "confidence": 0.85,
  "indicators": {
    "foot_slide_detected": true,
    "backward_momentum": true,
    "forward_momentum": false
  }
}
```

---

### POST /api/v2/detect-near-falls

Detect near-fall events (recovered balance losses).

**Request:**
- Content-Type: `multipart/form-data`
- Body: C3D file upload

**Response:**
```json
{
  "success": true,
  "near_falls": [
    {
      "frame_start": 250,
      "frame_end": 280,
      "time_start_s": 5.0,
      "time_end_s": 5.6,
      "severity": 0.7,
      "recovered": true,
      "compensatory_mechanisms": [
        "arm_flailing",
        "recovery_step"
      ]
    }
  ]
}
```

---

### GET /api/v2/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "detectors": {
    "rules": true,
    "lstm": true
  }
}
```

Note: `lstm` will be `false` if PyTorch is not installed or the model file is missing.

---

### GET /api/v2/annotations/{video_id}

Retrieve saved annotations for a video.

**Response:**
```json
{
  "success": true,
  "video_id": "video-02-epic-fall",
  "phases": [...],
  "events": [...],
  "metadata": {...}
}
```

---

### POST /api/v2/annotations

Save phase annotations.

**Request:**
```json
{
  "video_id": "video-02-epic-fall",
  "phases": [
    {
      "phase": "pre_fall",
      "frame_start": 0,
      "frame_end": 120
    }
  ]
}
```

---

## Programmatic Usage (Python)

### Basic Usage

```python
from src.utils.c3d_reader import read_c3d
from src.features import extract_features
from src.detectors.rules_detector import RulesDetector
from src.config import DetectorConfig

# Load C3D file
data = read_c3d("path/to/file.c3d")

# Extract features
config = DetectorConfig()
features = extract_features(
    data.marker_positions,
    data.marker_labels,
    data.frame_rate,
    config
)

# Run detection
detector = RulesDetector(config)
result = detector.analyze_data(
    data.marker_positions,
    data.marker_labels,
    data.frame_rate
)

# Check result
if result.fall_detected:
    print(f"Fall detected with {result.confidence:.1f}% confidence")
    print(f"Fall type: {result.fall_type}")
    print(f"Metrics: {result.metrics}")
```

### Using LSTM Detector

```python
from src.detectors.lstm_detector import LSTMDetector

# Load trained model
detector = LSTMDetector(
    config=config,
    model_path="models/lstm_fall_v1.pt"
)

# Run inference
result = detector.analyze_data(
    data.marker_positions,
    data.marker_labels,
    data.frame_rate
)
```

### Batch Processing

```python
from pathlib import Path

def process_directory(dir_path: str):
    """Process all C3D files in a directory."""
    results = []

    for c3d_file in Path(dir_path).glob("*.c3d"):
        data = read_c3d(str(c3d_file))
        features = extract_features(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate,
            config
        )
        result = detector.analyze_data(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

        results.append({
            "file": c3d_file.name,
            "fall_detected": result.fall_detected,
            "confidence": result.confidence
        })

    return results
```

### Custom Configuration

```python
from src.config import DetectorConfig, ThresholdConfig

# Create custom thresholds
custom_thresholds = ThresholdConfig(
    vel_down_fall_ms=-1.0,  # More conservative
    trunk_tilt_fall_deg=80.0,
    impact_z_min=4.0
)

# Create config with custom thresholds
config = DetectorConfig(
    thresholds=custom_thresholds,
    detector_type=DetectorType.RULES_V3
)
```

---

## Data Classes

### FallAnalysisResult

```python
@dataclass
class FallAnalysisResult:
    fall_detected: bool
    confidence: float  # 0-100
    fall_type: FallType  # SLIP, TRIP, COLLAPSE, UNKNOWN
    characteristics: List[str]
    metrics: Dict[str, float]
    phase_segments: List[PhaseSegment]
    timeline_data: Optional[Dict[str, List[float]]]
```

### PhaseSegment

```python
@dataclass
class PhaseSegment:
    phase: PhaseLabel  # PRE_FALL, INITIATION, DESCENT, IMPACT, POST_FALL
    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float
    confidence: float
```

### MotionFeatures

```python
@dataclass
class MotionFeatures:
    time: np.ndarray
    frame_rate: float
    com_position: np.ndarray  # (n_frames, 3)
    com_velocity: np.ndarray  # (n_frames, 3)
    pelvis_height: np.ndarray
    head_height: np.ndarray
    trunk_tilt: np.ndarray
    margin_of_stability: np.ndarray
    acceleration_magnitude: np.ndarray
```

---

## Error Responses

### File Not Found

```json
{
  "success": false,
  "error": "File not found",
  "error_code": "FILE_NOT_FOUND"
}
```

### Invalid File Format

```json
{
  "success": false,
  "error": "Invalid C3D file format",
  "error_code": "INVALID_FORMAT"
}
```

### Processing Error

```json
{
  "success": false,
  "error": "Error extracting features: insufficient markers",
  "error_code": "PROCESSING_ERROR"
}
```

---

## Rate Limits

The API does not currently implement rate limiting. For production deployment, consider adding:

- Request rate limiting per IP
- File size limits (current: 64MB)
- Concurrent request limits

---

## Updates

| Date | Change |
|------|--------|
| 2026-01-02 | Initial documentation created |
