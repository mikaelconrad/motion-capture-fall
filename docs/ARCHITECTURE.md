# System Architecture

This document describes the architecture of the Fall Detection System, including module dependencies, design patterns, and data flow.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Fall Detection System                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   C3D File   │───►│   Features   │───►│   Detector (Rules/ML)    │  │
│  │   Reader     │    │  Extraction  │    │                          │  │
│  └──────────────┘    └──────────────┘    └────────────┬─────────────┘  │
│         │                   │                         │                 │
│         │                   │                         ▼                 │
│         │                   │            ┌──────────────────────────┐  │
│         │                   │            │   Fall Analysis Result   │  │
│         │                   │            │   - fall_detected        │  │
│         │                   │            │   - confidence           │  │
│         │                   │            │   - fall_type            │  │
│         │                   │            │   - phase_timeline       │  │
│         │                   │            └──────────────────────────┘  │
│         │                   │                         │                 │
│         ▼                   ▼                         ▼                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Biomechanics Module                          │   │
│  │  - Center of Mass (Dempster 1955)                               │   │
│  │  - Margin of Stability (Hof 2005)                               │   │
│  │  - Trunk Tilt Calculation                                        │   │
│  │  - Impact Detection                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/
├── __init__.py              # Package entry point, version info
├── config.py                # Configuration dataclasses
├── biomechanics.py          # Biomechanical calculations
├── features.py              # Feature extraction
├── annotations.py           # Phase annotation schema
├── detectors/
│   ├── __init__.py
│   ├── base.py              # Abstract base detector
│   ├── rules_detector.py    # Rules-based detector
│   └── lstm_detector.py     # LSTM-based detector (ML)
├── classifiers/
│   ├── __init__.py
│   ├── fall_type.py         # Fall type classification
│   └── near_fall.py         # Near-fall detection
└── utils/
    ├── __init__.py
    ├── c3d_reader.py        # C3D file I/O
    └── visualization.py     # Plotting utilities
```

---

## Module Dependencies

```
                    ┌─────────────┐
                    │   config    │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
    ┌──────────────┐ ┌───────────┐ ┌──────────────┐
    │ biomechanics │ │  features │ │  annotations │
    └──────┬───────┘ └─────┬─────┘ └──────────────┘
           │               │
           └───────┬───────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │           detectors/base             │
    └──────────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│ rules_detector│    │ lstm_detector  │
└───────────────┘    └────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │            classifiers/              │
    │  ┌─────────────┐  ┌──────────────┐  │
    │  │  fall_type  │  │  near_fall   │  │
    │  └─────────────┘  └──────────────┘  │
    └──────────────────────────────────────┘
```

---

## Core Modules

### src/config.py

Central configuration using Python dataclasses.

```python
@dataclass
class DetectorConfig:
    """Main configuration for fall detection."""

    # Detector selection
    detector_type: DetectorType = DetectorType.RULES_V3

    # Coordinate system
    vertical_axis: VerticalAxis = VerticalAxis.Y

    # Body model
    body_model: BodyModelConfig = field(default_factory=BodyModelConfig)

    # Thresholds
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

@dataclass
class BodyModelConfig:
    """Anthropometric body model (Dempster 1955, Winter 2009)."""

    segment_masses: Dict[str, float]  # Fraction of total body mass
    segment_com_ratios: Dict[str, float]  # CoM location (proximal to distal)

@dataclass
class ThresholdConfig:
    """Detection thresholds with literature sources."""

    vel_down_fall_ms: float = -0.8  # Hu & Qu (2016)
    trunk_tilt_fall_deg: float = 75.0  # Hsiao & Robinovitch (1998)
    # ... etc
```

### src/biomechanics.py

Biomechanical calculations based on scientific literature.

**Key Functions:**

| Function | Description | Reference |
|----------|-------------|-----------|
| `calculate_whole_body_com()` | Mass-weighted CoM | Dempster (1955) |
| `calculate_segment_com()` | Individual segment CoM | Winter (2009) |
| `calculate_xcom()` | Extrapolated CoM | Hof et al. (2005) |
| `calculate_base_of_support()` | Foot contact polygon | Hof et al. (2005) |
| `calculate_margin_of_stability()` | Dynamic balance metric | Hof et al. (2005) |
| `calculate_trunk_tilt()` | Upper body orientation | Geometric |
| `calculate_impact_zscore()` | Impact acceleration detection | Statistical |

### src/features.py

Feature extraction from marker data.

```python
@dataclass
class MotionFeatures:
    """Features extracted from motion capture data."""

    # Time series
    time: np.ndarray
    frame_rate: float

    # Position features
    com_position: np.ndarray  # (n_frames, 3)
    pelvis_height: np.ndarray
    head_height: np.ndarray

    # Velocity features
    com_velocity: np.ndarray  # (n_frames, 3)
    vertical_velocity: np.ndarray

    # Stability features
    margin_of_stability: np.ndarray
    trunk_tilt: np.ndarray

    # Impact features
    acceleration_magnitude: np.ndarray
    impact_zscore: np.ndarray
```

### src/detectors/base.py

Abstract base class for detectors.

```python
class BaseDetector(ABC):
    """Abstract base class for fall detectors."""

    @abstractmethod
    def analyze(self, features: MotionFeatures) -> FallAnalysisResult:
        """Analyze motion features and return fall detection result."""
        pass

    @abstractmethod
    def get_config(self) -> DetectorConfig:
        """Return detector configuration."""
        pass
```

### src/detectors/rules_detector.py

Rules-based detector using threshold cascade.

**Detection Paths:**

1. **Core Fall Path**: Rapid descent + impact + trunk tilt + low final position
2. **Hard Fall Path**: Extreme trunk tilt + high impact + large height drop
3. **Short Fall Path**: Rapid impact despite small drop

### src/detectors/lstm_detector.py

LSTM-based detector for ML inference.

```python
class LSTMDetector(BaseDetector):
    """LSTM-based fall detector."""

    def __init__(self, model_path: str = None):
        self.model = self._load_model(model_path)

    def analyze(self, features: MotionFeatures) -> FallAnalysisResult:
        # Extract sequence features
        X = self._prepare_features(features)

        # Run inference
        predictions = self.model.predict(X)

        # Convert to result
        return self._decode_predictions(predictions)
```

---

## Data Flow

### Detection Pipeline

```
1. Input: C3D file path
   │
   ▼
2. C3D Reader: Parse binary file
   │  └─► C3DData(positions, frame_rate, marker_labels)
   │
   ▼
3. Feature Extraction: Calculate biomechanical features
   │  └─► MotionFeatures(com, velocity, trunk_tilt, mos, ...)
   │
   ▼
4. Detection: Apply rules or ML model
   │  ├─► RulesDetector: Threshold cascade
   │  └─► LSTMDetector: Neural network inference
   │
   ▼
5. Classification: Determine fall type (if fall detected)
   │  └─► FallTypeClassifier: slip/trip/collapse
   │
   ▼
6. Output: FallAnalysisResult
   └─► {fall_detected, confidence, fall_type, phases, metrics}
```

### API Pipeline

```
HTTP POST /api/v2/analyze
   │
   ▼
Flask Server (api/server.py)
   │
   ├─► Parse multipart/form-data
   │     └─► Extract C3D file
   │
   ├─► Load configuration
   │     └─► DetectorConfig
   │
   ├─► Run detection pipeline
   │     └─► FallAnalysisResult
   │
   └─► Return JSON response
         └─► {fall_detected, confidence, metrics, timeline_data, ...}
```

---

## Design Patterns

### Strategy Pattern (Detectors)

```python
# Select detector based on configuration
if config.detector_type == DetectorType.RULES_V3:
    detector = RulesDetector(config)
elif config.detector_type == DetectorType.LSTM:
    detector = LSTMDetector(config, model_path)

# Use detector (same interface)
result = detector.analyze_data(
    data.marker_positions,
    data.marker_labels,
    data.frame_rate
)
```

### Factory Pattern (Feature Extraction)

```python
# Extract features based on available markers
features = extract_features(
    data.marker_positions,
    data.marker_labels,
    data.frame_rate,
    config
)
```

### Dataclass Pattern (Configuration)

All configuration uses frozen dataclasses for immutability and type safety:

```python
@dataclass(frozen=True)
class ThresholdConfig:
    vel_down_fall_ms: float = -0.8
    # ...
```

---

## Error Handling

### Input Validation

```python
def read_c3d(path: str) -> C3DData:
    if not os.path.exists(path):
        raise FileNotFoundError(f"C3D file not found: {path}")

    if not path.endswith('.c3d'):
        raise ValueError(f"Expected .c3d file, got: {path}")

    # ... parse file
```

### Graceful Degradation

```python
def calculate_whole_body_com(markers, config):
    """Calculate CoM with fallback to geometric mean."""
    try:
        return _mass_weighted_com(markers, config)
    except MissingMarkersError:
        logger.warning("Falling back to geometric CoM")
        return _geometric_com(markers)
```

---

## Performance Considerations

### Memory Efficiency

- C3D files loaded on-demand (not cached by default)
- Feature extraction operates on numpy arrays
- Large datasets processed in batches

### Computation

| Operation | Complexity | Notes |
|-----------|------------|-------|
| C3D parsing | O(n) | n = number of frames |
| CoM calculation | O(n × m) | m = number of segments |
| MoS calculation | O(n) | Per-frame |
| LSTM inference | O(n × seq_len) | Sliding window |

---

## Extension Points

### Adding a New Detector

1. Create class inheriting from `BaseDetector`
2. Implement `analyze()` method
3. Add to `DetectorType` enum
4. Register in detector factory

### Adding New Features

1. Add to `MotionFeatures` dataclass
2. Implement calculation in `features.py`
3. Update `extract_features()` function

### Adding New Thresholds

1. Add field to `ThresholdConfig` with default value
2. Document source in `docs/THRESHOLDS.md`
3. Use in detection logic

---

## Updates

| Date | Change |
|------|--------|
| 2026-01-02 | Initial documentation created |
