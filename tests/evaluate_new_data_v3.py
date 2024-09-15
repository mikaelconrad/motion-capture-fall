#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.c3d_reader import read_c3d
from src.features import extract_features
from src.detectors.rules_detector import RulesDetector


def analyze_c3d_for_fall(path: str) -> dict:
    """Analyze a C3D file for fall detection using RulesDetector."""
    data = read_c3d(path)
    features = extract_features(
        data.marker_positions,
        data.marker_labels,
        data.frame_rate
    )
    detector = RulesDetector()
    result = detector.analyze_data(
        data.marker_positions,
        data.marker_labels,
        data.frame_rate
    )
    return {
        'fall_detected': result.fall_detected,
        'confidence': result.confidence,
        'characteristics': result.characteristics,
        'metrics': features.stats,
    }

GROUND_TRUTH = {
    'video-02-epic-fall': True,
    'video-04-short-fall-standing-up-try': True,
    'video-05-turn': False,
    'video-06-under-the-mattress': False,
    'video-09-meditation': False,
    'video-11-dance': False,
}

FILES = {name: os.path.join('new_data', name, f'{name}.c3d') for name in GROUND_TRUTH}

REPORTS_DIR = os.path.join('tests', 'reports_v3')
RESULTS_PATH = os.path.join('tests', 'results_v3.txt')


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def format_report(name: str, gt: bool, result: dict) -> str:
    m = result.get('metrics', {})
    lines = []
    lines.append(f"File: {name}")
    lines.append(f"Expected fall: {gt}")
    lines.append(f"Predicted fall: {result.get('fall_detected')}")
    lines.append(f"Characteristics: {', '.join(result.get('characteristics', []))}")
    if m:
        lines.append("Metrics:")
        for k, v in m.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def main():
    ensure_dirs()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total = len(FILES)
    tp = tn = fp = fn = 0
    lines = [f"Run: {ts}", ""]

    for name, path in FILES.items():
        try:
            result = analyze_c3d_for_fall(path)
        except Exception as e:
            result = {'error': str(e)}
        gt = GROUND_TRUTH[name]
        pred = result.get('fall_detected') if isinstance(result, dict) else None

        if isinstance(pred, bool):
            if gt and pred:
                tp += 1
            elif (not gt) and (not pred):
                tn += 1
            elif (not gt) and pred:
                fp += 1
            elif gt and (not pred):
                fn += 1

        with open(os.path.join(REPORTS_DIR, f"{name}.txt"), 'w') as f:
            if isinstance(result, dict):
                f.write(format_report(name, gt, result))
            else:
                f.write(f"File: {name}\nExpected fall: {gt}\nError: {result}\n")
        lines.append(f"{name}: expected={gt}, predicted={pred}")

    acc = (tp + tn) / total if total else 0.0
    lines.extend(["", f"TP={tp} TN={tn} FP={fp} FN={fn}", f"Accuracy={acc:.3f}"])

    with open(RESULTS_PATH, 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    main()
