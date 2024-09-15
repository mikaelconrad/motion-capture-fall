#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.lstm_detector import LSTMDetector, TORCH_AVAILABLE


def load_ground_truth(labels_path: str) -> dict:
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    return {video['id']: bool(video['fall']) for video in labels_data['videos']}


def analyze_c3d_for_fall(path: str, detector: LSTMDetector) -> dict:
    """Analyze a C3D file for fall detection using LSTMDetector."""
    result = detector.analyze(path)
    return {
        'fall_detected': result.fall_detected,
        'confidence': result.confidence,
        'activity_type': result.activity_type,
        'characteristics': result.characteristics,
        'metrics': result.metrics,
    }


LABELS_PATH = os.path.join('data', 'labels.json')
GROUND_TRUTH = load_ground_truth(LABELS_PATH)
FILES = {name: os.path.join('new_data', name, f'{name}.c3d') for name in GROUND_TRUTH}

REPORTS_DIR = os.path.join('tests', 'reports_lstm_v1')
RESULTS_PATH = os.path.join('tests', 'results_lstm_v1.txt')


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def format_report(name: str, gt: bool, result: dict) -> str:
    m = result.get('metrics', {})
    lines = []
    lines.append(f"File: {name}")
    lines.append(f"Expected fall: {gt}")
    lines.append(f"Predicted fall: {result.get('fall_detected')}")
    lines.append(f"Confidence: {result.get('confidence'):.2f}")
    lines.append(f"Activity type: {result.get('activity_type')}")
    lines.append(f"Characteristics: {', '.join(result.get('characteristics', []))}")
    if m:
        lines.append("Metrics:")
        for k, v in m.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def main():
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for LSTM evaluation. Install with: pip install torch")
        sys.exit(1)

    ensure_dirs()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total = len(FILES)
    tp = tn = fp = fn = 0
    lines = [f"Run: {ts}", ""]

    model_path = os.environ.get('LSTM_MODEL_PATH', os.path.join('models', 'lstm_fall_v1.pt'))
    threshold = float(os.environ.get('LSTM_THRESHOLD', '0.5'))

    detector = LSTMDetector(model_path=model_path, threshold=threshold)

    for name, path in FILES.items():
        try:
            result = analyze_c3d_for_fall(path, detector)
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
            if isinstance(result, dict) and 'error' not in result:
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
