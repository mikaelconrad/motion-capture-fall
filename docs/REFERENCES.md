# Scientific References

This document provides complete citations for all scientific literature used in the fall detection system.

---

## Biomechanics Foundations

### 1. Dempster, W.T. (1955)
**Space requirements of the seated operator: Geometrical, kinematic, and mechanical aspects of the body with special reference to the limbs.**
WADC Technical Report 55-159. Wright-Patterson Air Force Base, OH: Wright Air Development Center.

*Used for:*
- Segment mass ratios (Table in `src/config.py:BodyModelConfig`)
- Segment CoM locations relative to proximal/distal joints

*Code locations:*
- `src/config.py`: `DEMPSTER_MASS_RATIOS`
- `src/biomechanics.py`: `calculate_segment_com()`, `calculate_whole_body_com()`

---

### 2. Winter, D.A. (2009)
**Biomechanics and Motor Control of Human Movement.** 4th Edition.
John Wiley & Sons, Hoboken, NJ.
ISBN: 978-0470398180

*Used for:*
- Anthropometric data and segment parameters
- CoM calculation methodology
- Gait and balance analysis principles

*Code locations:*
- `src/config.py`: `BodyModelConfig.segment_com_ratios`
- `src/biomechanics.py`: CoM calculation functions

---

## Dynamic Balance & Stability

### 3. Hof, A.L., Gazendam, M.G.J., & Sinke, W.E. (2005)
**The condition for dynamic stability.**
Journal of Biomechanics, 38(1), 1-8.
DOI: [10.1016/j.jbiomech.2004.03.025](https://doi.org/10.1016/j.jbiomech.2004.03.025)

*Used for:*
- Extrapolated Center of Mass (XCoM) concept
- Margin of Stability (MoS) calculation
- Dynamic balance assessment during movement

*Key equations:*
```
XCoM = CoM + CoM_velocity / omega_0
omega_0 = sqrt(g / leg_length)
MoS = distance(XCoM, BoS_boundary)
```

*Code locations:*
- `src/biomechanics.py`: `calculate_xcom()`, `calculate_margin_of_stability()`
- `src/config.py`: `ThresholdConfig.mos_warning_m`, `ThresholdConfig.mos_critical_m`

---

## Fall Detection & Kinematics

### 4. Hu, X., & Qu, X. (2016)
**Pre-impact fall detection.**
BioMedical Engineering OnLine, 15, 87.
DOI: [10.1186/s12938-016-0194-x](https://doi.org/10.1186/s12938-016-0194-x)

*Used for:*
- Pre-impact detection timing requirements
- Velocity thresholds for fall detection
- Lead time requirements for protective systems

*Key findings:*
- Vertical velocity threshold: -1.3 m/s (we use -0.8 m/s for earlier detection)
- Required lead time: >300ms for airbag deployment

*Code locations:*
- `src/config.py`: `ThresholdConfig.vel_down_fall_ms`
- `src/detectors/rules_detector.py`: velocity-based detection logic

---

### 5. Hsiao, E.T., & Robinovitch, S.N. (1998)
**Common protective movements govern unexpected falls from standing height.**
Journal of Biomechanics, 31(1), 1-9.
DOI: [10.1016/S0021-9290(97)00114-0](https://doi.org/10.1016/S0021-9290(97)00114-0)

*Used for:*
- Fall type classification (forward vs. backward falls)
- Protective response patterns
- Trunk orientation during falls

*Key findings:*
- Forward falls: hands contact ground first
- Backward falls: hip impact common
- Trunk angle >90° indicates horizontal orientation

*Code locations:*
- `src/classifiers/fall_type.py`: Fall direction classification
- `src/config.py`: `ThresholdConfig.trunk_tilt_fall_deg`

---

### 6. Schonnop, R., Yang, Y., Feldman, F., Robinson, E., Loughin, M., & Robinovitch, S.N. (2013)
**Prevalence of and factors associated with head impact during falls in older adults in long-term care.**
CMAJ, 185(17), E803-E810.
DOI: [10.1503/cmaj.130498](https://doi.org/10.1503/cmaj.130498)

*Used for:*
- Real-world fall duration statistics
- Fall timing parameters

*Key findings:*
- Mean fall duration: 583 ± 255 ms
- Used to set descent_min_s (0.15s) and descent_max_s (1.2s)

*Code locations:*
- `src/config.py`: `ThresholdConfig.descent_min_s`, `ThresholdConfig.descent_max_s`

---

## Near-Fall Detection

### 7. Maidan, I., Freedman, T., Galperin, I., Gazit, E., & Hausdorff, J.M. (2014)
**Altered brain activation in complex walking conditions in patients with Parkinson's disease.**
Parkinsonism & Related Disorders, 20(9), 952-956.
DOI: [10.1016/j.parkreldis.2014.05.016](https://doi.org/10.1016/j.parkreldis.2014.05.016)

*Used for:*
- Near-fall definition
- Compensatory mechanism identification
- Recovery behavior patterns

*Near-fall criteria (2+ of 5 mechanisms):*
1. Arm flailing
2. Recovery stepping
3. Trunk counter-rotation
4. Rapid CoM lowering
5. Abrupt speed change

*Code locations:*
- `src/classifiers/near_fall.py`: `NearFallDetector`

---

## External Datasets

### 8. Sucerquia, A., López, J.D., & Vargas-Bonilla, J.F. (2017)
**SisFall: A Fall and Movement Dataset.**
Sensors, 17(1), 198.
DOI: [10.3390/s17010198](https://doi.org/10.3390/s17010198)

*Dataset characteristics:*
- 1798 falls + 2706 ADLs
- Accelerometer and gyroscope data
- 19 ADL types, 15 fall types
- 23 young adults + 14 elderly participants

*Used for:*
- External training data (after feature alignment)
- Fall pattern validation

---

### 9. CMU Graphics Lab Motion Capture Database
**Carnegie Mellon University Motion Capture Database.**
URL: [https://mocap.cs.cmu.edu/](https://mocap.cs.cmu.edu/)

*Dataset characteristics:*
- 2500+ motion sequences
- C3D and BVH formats
- Various activities (walking, running, dancing, sports)

*Used for:*
- Non-fall motion samples
- Data augmentation source

---

## Additional References

### 10. Bourke, A.K., O'Brien, J.V., & Lyons, G.M. (2007)
**Evaluation of a threshold-based tri-axial accelerometer fall detection algorithm.**
Gait & Posture, 26(2), 194-199.
DOI: [10.1016/j.gaitpost.2006.09.012](https://doi.org/10.1016/j.gaitpost.2006.09.012)

*Used for:*
- Acceleration threshold methodology
- Impact detection approach

---

### 11. Kangas, M., Konttila, A., Lindgren, P., Winblad, I., & Jämsä, T. (2008)
**Comparison of low-complexity fall detection algorithms for body attached accelerometers.**
Gait & Posture, 28(2), 285-291.
DOI: [10.1016/j.gaitpost.2008.01.003](https://doi.org/10.1016/j.gaitpost.2008.01.003)

*Used for:*
- Fall detection algorithm comparison
- Sensitivity/specificity benchmarks

---

## Citation Format

When citing this system in academic work:

```bibtex
@software{fall_detection_mocap,
  title = {Motion Capture Fall Detection System},
  year = {2025},
  note = {Biomechanically-informed fall detection using marker-based motion capture},
  url = {https://github.com/your-repo/motion-capture-fall}
}
```

