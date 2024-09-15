# Detection Thresholds

This document details all detection thresholds used in the fall detection system, their values, units, and scientific sources.

---

## Overview

All thresholds are defined in `src/config.py` within the `ThresholdConfig` dataclass. Thresholds are organized by detection phase and purpose.

---

## Velocity Thresholds

| Parameter | Value | Unit | Source | Description |
|-----------|-------|------|--------|-------------|
| `vel_down_fall_ms` | -0.8 | m/s | Hu & Qu (2016), adjusted | Vertical velocity indicating potential fall |
| `vel_down_controlled_ms` | -0.3 | m/s | Empirical | Distinguishes controlled descent from fall |
| `vel_min_significant_ms` | 0.5 | m/s | Empirical | Minimum velocity for meaningful motion |

### Notes
- Literature reports -1.3 m/s as fall threshold (Hu & Qu, 2016)
- We use -0.8 m/s for earlier detection (more sensitive)
- Controlled descents (sitting, lying down) typically < -0.3 m/s

**Code location**: `src/config.py:ThresholdConfig`

---

## Temporal Thresholds

| Parameter | Value | Unit | Source | Description |
|-----------|-------|------|--------|-------------|
| `descent_min_s` | 0.15 | s | Schonnop et al. (2013) | Minimum duration for fall descent phase |
| `descent_max_s` | 1.2 | s | Schonnop et al. (2013) | Maximum duration for fall descent phase |
| `impact_duration_max_s` | 0.2 | s | Empirical | Maximum duration of impact phase |
| `post_impact_rest_s` | 0.5 | s | Empirical | Time to assess post-impact rest state |

### Notes
- Real-world falls average 583 ± 255 ms (Schonnop et al., 2013)
- descent_min_s (150ms) catches fast falls
- descent_max_s (1200ms) = mean + 2.4 standard deviations

**Code location**: `src/config.py:ThresholdConfig`

---

## Postural Thresholds

| Parameter | Value | Unit | Source | Description |
|-----------|-------|------|--------|-------------|
| `trunk_tilt_warning_deg` | 45.0 | deg | Empirical | Early warning for balance loss |
| `trunk_tilt_fall_deg` | 75.0 | deg | Hsiao & Robinovitch (1998) | Trunk angle indicating fall |
| `trunk_tilt_horiz_deg` | 85.0 | deg | Geometric | Near-horizontal orientation |
| `head_final_max_m` | 0.8 | m | Winter (2009) | Maximum head height for "lying" posture |

### Notes
- 90° = perfectly horizontal; we use 75° to allow margin
- Head height threshold accounts for anthropometric variation
- Typical lying head height: 0.1-0.3m depending on position

**Code location**: `src/config.py:ThresholdConfig`

---

## Impact Detection Thresholds

| Parameter | Value | Unit | Source | Description |
|-----------|-------|------|--------|-------------|
| `impact_z_min` | 3.0 | σ | Statistical | Minimum z-score for impact detection |
| `impact_z_severe` | 5.0 | σ | Statistical | Severe impact threshold |
| `freefall_g_max` | 0.6 | g | Hof et al. (2005) | Maximum acceleration during freefall |

### Notes
- Impact z-score is computed relative to baseline acceleration
- z-score > 3.0 = ~0.3% probability of normal motion
- Freefall: < 1g indicates loss of ground support; 0.6g threshold accounts for noise

**Calculation**:
```python
z_score = (acceleration - baseline_mean) / baseline_std
```

**Code location**: `src/config.py:ThresholdConfig`, `src/biomechanics.py:calculate_impact_zscore()`

---

## Margin of Stability Thresholds

| Parameter | Value | Unit | Source | Description |
|-----------|-------|------|--------|-------------|
| `mos_warning_m` | 0.05 | m | Hof et al. (2005) | MoS below this triggers warning |
| `mos_critical_m` | 0.0 | m | Hof et al. (2005) | XCoM at BoS boundary |

### Notes
- MoS > 0: Stable (XCoM inside Base of Support)
- MoS < 0: Unstable (requires corrective action)
- MoS << 0: Fall likely without intervention

**Calculation**:
```python
XCoM = CoM + CoM_velocity / sqrt(g / leg_length)
MoS = signed_distance(XCoM, BoS_boundary)
```

**Code location**: `src/biomechanics.py:calculate_margin_of_stability()`

---

## Height Thresholds

| Parameter | Value | Unit | Source | Description |
|-----------|-------|------|--------|-------------|
| `height_drop_significant_m` | 0.5 | m | Empirical | Significant height drop |
| `height_drop_hard_fall_m` | 1.0 | m | Empirical | Indicates hard/fast fall |
| `pelvis_sitting_max_m` | 0.6 | m | Anthropometry | Maximum pelvis height when sitting |

### Notes
- Height drop = initial_height - final_height
- Typical standing CoM height: ~1.0m
- Sitting CoM height: ~0.4-0.5m

**Code location**: `src/config.py:ThresholdConfig`

---

## Post-Fall Assessment Thresholds

| Parameter | Value | Unit | Source | Description |
|-----------|-------|------|--------|-------------|
| `post_impact_speed_max_ms` | 0.15 | m/s | Empirical | Maximum speed indicating "at rest" |
| `recovery_time_min_s` | 1.0 | s | Empirical | Minimum time for recovery assessment |

### Notes
- Low post-impact velocity indicates person is lying still
- Used to distinguish fall from controlled movement completion

**Code location**: `src/config.py:ThresholdConfig`

---

## Threshold Tuning Guidelines

### Increasing Sensitivity (fewer missed falls)
- Decrease `vel_down_fall_ms` (e.g., -0.6 m/s)
- Decrease `trunk_tilt_fall_deg` (e.g., 60°)
- Decrease `impact_z_min` (e.g., 2.5)

### Increasing Specificity (fewer false alarms)
- Increase `vel_down_fall_ms` (e.g., -1.0 m/s)
- Increase `trunk_tilt_fall_deg` (e.g., 85°)
- Increase `impact_z_min` (e.g., 4.0)

### Clinical vs. Research Settings
- **Clinical**: Prioritize sensitivity (catch all falls)
- **Research**: Balance sensitivity and specificity

---

## Validation Status

| Threshold | Validation Status | Notes |
|-----------|-------------------|-------|
| Velocity | Literature-based | Adjusted from Hu & Qu (2016) |
| Temporal | Literature-based | From Schonnop et al. (2013) |
| Trunk tilt | Literature-based | From Hsiao & Robinovitch (1998) |
| Impact z-score | Empirical | Needs validation study |
| MoS | Literature-based | From Hof et al. (2005) |
| Height | Empirical | Based on anthropometry |
| Post-fall | Empirical | Needs validation study |

---

## Updates

| Date | Change |
|------|--------|
| 2026-01-02 | Initial documentation created |
