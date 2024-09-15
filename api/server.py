#!/usr/bin/env python3
"""
Fall Detection API Server

Comprehensive REST API for fall detection and analysis of motion capture data.

Endpoints:
- POST /api/v2/analyze - Full fall analysis
- POST /api/v2/classify-fall-type - Classify fall type (slip/trip/collapse)
- POST /api/v2/detect-near-falls - Detect near-fall events
- GET /api/v2/annotations/<video_id> - Get annotations for a video
- POST /api/v2/annotations - Save annotations
- GET /api/v2/health - Health check

Usage:
    python api/server.py
    # or
    flask --app api.server run --port 5003
"""

import os
import sys
import tempfile
import json
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.c3d_reader import read_c3d
from src.features import extract_features
from src.detectors.rules_detector import RulesDetector
from src.classifiers import (
    FallTypeClassifier,
    classify_fall_type,
    NearFallDetector,
    detect_near_falls,
)
from src.annotations import (
    VideoAnnotation,
    load_annotation,
    save_annotation,
    get_annotation_path,
)
from src.config import DetectorConfig

# Try to import LSTM detector (requires PyTorch)
try:
    from src.detectors.lstm_detector import LSTMDetector, TORCH_AVAILABLE
except ImportError:
    LSTMDetector = None
    TORCH_AVAILABLE = False

# Default model path for LSTM detector
DEFAULT_LSTM_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'models', 'lstm_fall_v1.pt'
)


app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'c3d'}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new_data')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_uploaded_file(request_obj) -> tuple:
    """Extract and validate uploaded file from request.

    Returns:
        (filepath, error_response) - filepath if successful, None and error response if failed
    """
    if 'file' not in request_obj.files:
        return None, (jsonify({'error': 'No file uploaded'}), 400)

    file = request_obj.files['file']
    if file.filename == '':
        return None, (jsonify({'error': 'No file selected'}), 400)

    if not allowed_file(file.filename):
        return None, (jsonify({'error': 'Invalid file type. Please upload a .c3d file'}), 400)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return filepath, None


def cleanup_file(filepath: str) -> None:
    """Remove temporary file."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except:
            pass


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.route('/')
def index():
    """Redirect to API documentation."""
    from flask import redirect, url_for
    return redirect(url_for('api_docs'))


@app.route('/frontend/<path:filename>')
def serve_frontend(filename: str):
    """Serve frontend assets."""
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend'), filename)


@app.route('/api/docs')
def api_docs():
    """API documentation page."""
    return jsonify({
        'name': 'Fall Detection API',
        'version': '2.0.0',
        'endpoints': {
            'POST /api/v2/analyze': 'Full fall analysis with detection and metrics',
            'POST /api/v2/classify-fall-type': 'Classify fall type (slip/trip/collapse)',
            'POST /api/v2/detect-near-falls': 'Detect near-fall events',
            'GET /api/v2/annotations/<video_id>': 'Get annotations for a video',
            'POST /api/v2/annotations': 'Save annotations',
            'GET /api/v2/health': 'Health check',
            'POST /api/analyze_v3': 'Legacy endpoint (compatible with v3 UI)',
        }
    })


@app.route('/api/v2/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'detectors': {
            'rules': True,
            'lstm': TORCH_AVAILABLE and os.path.exists(DEFAULT_LSTM_MODEL),
        },
    })


@app.route('/api/v2/analyze', methods=['POST'])
def analyze():
    """
    Full fall analysis endpoint.

    Request:
        - file: C3D file (multipart/form-data)
        - detector (query param): 'rules' (default) or 'lstm'

    Response:
        {
            "fall_detected": bool,
            "confidence": float,
            "activity_type": str,
            "characteristics": [str],
            "timeline_data": {...},
            "metrics": {...}
        }
    """
    filepath, error = get_uploaded_file(request)
    if error:
        return error

    # Get detector type from query parameter
    detector_type = request.args.get('detector', 'rules').lower()

    try:
        # Load and process data
        data = read_c3d(filepath)
        features = extract_features(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

        # Select and run detector
        if detector_type == 'lstm':
            if not TORCH_AVAILABLE or LSTMDetector is None:
                cleanup_file(filepath)
                return jsonify({
                    'error': 'LSTM detector not available. Install PyTorch: pip install torch'
                }), 400

            if not os.path.exists(DEFAULT_LSTM_MODEL):
                cleanup_file(filepath)
                return jsonify({
                    'error': f'LSTM model not found at {DEFAULT_LSTM_MODEL}'
                }), 400

            detector = LSTMDetector(model_path=DEFAULT_LSTM_MODEL)
            result = detector.analyze_data(
                data.marker_positions,
                data.marker_labels,
                data.frame_rate
            )
        else:
            # Default: rules-based detector
            detector = RulesDetector()
            result = detector.analyze_data(
                data.marker_positions,
                data.marker_labels,
                data.frame_rate
            )

        # Build response
        response = {
            'fall_detected': result.fall_detected,
            'confidence': float(result.confidence),
            'activity_type': result.activity_type,
            'characteristics': result.characteristics,
            'metrics': features.stats,
        }

        # Add timeline data
        if result.timeline_data:
            td = result.timeline_data
            descent_start_time_s = None
            descent_end_time_s = None
            for i, vel in enumerate(features.vertical_velocity):
                if vel < -0.1:
                    if descent_start_time_s is None:
                        descent_start_time_s = float((i + 1) / features.frame_rate)
                    descent_end_time_s = float((i + 1) / features.frame_rate)
            response['timeline_data'] = {
                'pelvis_height': features.pelvis_height.tolist() if len(features.pelvis_height) > 0 else [],
                'head_height': features.head_height.tolist() if len(features.head_height) > 0 else [],
                'vertical_velocity': features.vertical_velocity.tolist() if len(features.vertical_velocity) > 0 else [],
                'trunk_tilt_deg': features.trunk_tilt_deg.tolist() if len(features.trunk_tilt_deg) > 0 else [],
                'frame_rate': float(features.frame_rate),
                'n_frames': int(features.n_frames),
                'impact_occurred': td.impact_occurred,
                'impact_time_s': td.impact_time_s,
                'descent_start_time_s': descent_start_time_s,
                'descent_end_time_s': descent_end_time_s,
            }

        cleanup_file(filepath)
        return jsonify(response)

    except Exception as e:
        cleanup_file(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze_v3', methods=['POST'])
def analyze_v3_legacy():
    """
    Legacy endpoint compatible with the v3 UI.

    Returns data in the format expected by fall-detector-v3.html
    """
    filepath, error = get_uploaded_file(request)
    if error:
        return error

    try:
        import numpy as np

        # Load and process data
        data = read_c3d(filepath)
        features = extract_features(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

        # Run detector
        detector = RulesDetector()
        result = detector.analyze_data(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

        # Build timeline data in legacy format
        n_frames = features.n_frames
        frame_rate = features.frame_rate

        time_pos = (np.arange(n_frames) / frame_rate).tolist()
        time_vel = (np.arange(max(0, n_frames - 1)) / frame_rate + (0.5 / frame_rate)).tolist()
        time_acc = (np.arange(max(0, n_frames - 2)) / frame_rate + (1.0 / frame_rate)).tolist()

        # Calculate acceleration magnitude in g
        acc_mag_g = []
        if len(features.com_acceleration) > 0:
            acc_mag = np.linalg.norm(features.com_acceleration, axis=1)
            acc_mag_g = (acc_mag / 9.81).tolist()

        # Find impact indices
        impact_occurred = result.timeline_data.impact_occurred if result.timeline_data else False
        impact_indices = None

        if impact_occurred and len(acc_mag_g) > 0:
            peak_idx = int(np.argmax(acc_mag_g))
            impact_indices = {
                'pos': min(peak_idx + 2, len(time_pos) - 1),
                'vel': min(peak_idx + 1, len(time_vel) - 1),
                'acc': peak_idx,
            }

        timeline = {
            'time_pos': time_pos,
            'time_vel': time_vel,
            'time_acc': time_acc,
            'pelvis_height_m': features.pelvis_height.tolist(),
            'head_height_m': features.head_height.tolist(),
            'pelvis_vertical_velocity_ms': features.vertical_velocity.tolist(),
            'com_speed_ms': features.com_speed.tolist(),
            'acc_mag_g': acc_mag_g,
            'trunk_tilt_deg': features.trunk_tilt_deg.tolist(),
            'impact_occurred': impact_occurred,
            'impact_indices': impact_indices,
        }

        response = {
            'fall_detected': result.fall_detected,
            'characteristics': result.characteristics,
            'metrics': features.stats,
            'timeline_data': timeline,
        }

        cleanup_file(filepath)
        return jsonify(response)

    except Exception as e:
        cleanup_file(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/classify-fall-type', methods=['POST'])
def classify_fall():
    """
    Classify fall type (slip, trip, collapse).

    Request:
        - file: C3D file (multipart/form-data)
        - descent_start (optional): Frame number for descent start
        - descent_end (optional): Frame number for descent end

    Response:
        {
            "fall_type": str,
            "confidence": float,
            "evidence": [str],
            "horizontal_vertical_ratio": float
        }
    """
    filepath, error = get_uploaded_file(request)
    if error:
        return error

    try:
        # Get optional parameters
        descent_start = request.form.get('descent_start', type=int)
        descent_end = request.form.get('descent_end', type=int)

        # Load and process data
        data = read_c3d(filepath)
        features = extract_features(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

        # Classify fall type
        classifier = FallTypeClassifier()
        result = classifier.classify(features, descent_start, descent_end)

        response = {
            'fall_type': result.fall_type.value,
            'confidence': float(result.confidence),
            'evidence': result.evidence,
            'horizontal_vertical_ratio': float(result.horizontal_vertical_ratio),
        }

        if result.horizontal_velocity_direction:
            response['horizontal_velocity_direction'] = list(result.horizontal_velocity_direction)

        if result.trunk_rotation_direction:
            response['trunk_rotation_direction'] = result.trunk_rotation_direction

        cleanup_file(filepath)
        return jsonify(response)

    except Exception as e:
        cleanup_file(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/detect-near-falls', methods=['POST'])
def near_falls():
    """
    Detect near-fall events in motion data.

    Request:
        - file: C3D file (multipart/form-data)

    Response:
        {
            "events": [
                {
                    "frame_start": int,
                    "frame_end": int,
                    "time_start_s": float,
                    "time_end_s": float,
                    "severity": float,
                    "confidence": float,
                    "compensatory_mechanisms": [str],
                    "peak_frame": int
                }
            ],
            "count": int
        }
    """
    filepath, error = get_uploaded_file(request)
    if error:
        return error

    try:
        # Load and process data
        data = read_c3d(filepath)
        features = extract_features(
            data.marker_positions,
            data.marker_labels,
            data.frame_rate
        )

        # Detect near-falls
        detector = NearFallDetector()
        events = detector.detect(features)

        response = {
            'events': [e.to_dict() for e in events],
            'count': len(events),
        }

        cleanup_file(filepath)
        return jsonify(response)

    except Exception as e:
        cleanup_file(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/annotations/<video_id>', methods=['GET'])
def get_annotations(video_id: str):
    """
    Get annotations for a video.

    Args:
        video_id: Video directory name

    Response:
        VideoAnnotation as JSON
    """
    video_dir = os.path.join(DATA_DIR, secure_filename(video_id))
    ann_path = get_annotation_path(video_dir)

    if not os.path.exists(ann_path):
        return jsonify({'error': f'No annotations found for {video_id}'}), 404

    try:
        annotation = load_annotation(ann_path)
        return jsonify(annotation.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/annotations', methods=['POST'])
def save_annotations():
    """
    Save annotations for a video.

    Request (JSON):
        {
            "video_id": str,
            "annotation": {...}
        }

    Response:
        {"success": true, "path": str}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        video_id = data.get('video_id')
        annotation_data = data.get('annotation')

        if not video_id or not annotation_data:
            return jsonify({'error': 'Missing video_id or annotation'}), 400

        video_dir = os.path.join(DATA_DIR, secure_filename(video_id))
        if not os.path.isdir(video_dir):
            return jsonify({'error': f'Video directory not found: {video_id}'}), 404

        annotation = VideoAnnotation.from_dict(annotation_data)
        ann_path = get_annotation_path(video_dir)
        save_annotation(annotation, ann_path)

        return jsonify({
            'success': True,
            'path': ann_path,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/videos', methods=['GET'])
def list_videos():
    """
    List available videos with their annotation status.

    Response:
        {
            "videos": [
                {
                    "id": str,
                    "has_annotation": bool,
                    "ground_truth_fall": bool (if annotated)
                }
            ]
        }
    """
    try:
        videos = []

        if os.path.isdir(DATA_DIR):
            for name in sorted(os.listdir(DATA_DIR)):
                video_dir = os.path.join(DATA_DIR, name)
                if os.path.isdir(video_dir):
                    video_info = {
                        'id': name,
                        'has_annotation': False,
                        'ground_truth_fall': None,
                    }

                    ann_path = get_annotation_path(video_dir)
                    if os.path.exists(ann_path):
                        try:
                            ann = load_annotation(ann_path)
                            video_info['has_annotation'] = True
                            video_info['ground_truth_fall'] = ann.ground_truth_fall
                            if ann.fall_type:
                                video_info['fall_type'] = ann.fall_type.value
                        except:
                            pass

                    videos.append(video_info)

        return jsonify({'videos': videos})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v2/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Batch analyze all videos in the data directory.

    Response:
        {
            "results": [
                {
                    "video_id": str,
                    "fall_detected": bool,
                    "confidence": float,
                    "ground_truth": bool (if available),
                    "correct": bool (if ground truth available)
                }
            ],
            "summary": {
                "total": int,
                "falls_detected": int,
                "accuracy": float (if ground truth available)
            }
        }
    """
    try:
        results = []
        correct_count = 0
        has_ground_truth = 0

        if os.path.isdir(DATA_DIR):
            detector = RulesDetector()

            for name in sorted(os.listdir(DATA_DIR)):
                video_dir = os.path.join(DATA_DIR, name)
                if not os.path.isdir(video_dir):
                    continue

                # Find C3D file
                c3d_files = [f for f in os.listdir(video_dir) if f.endswith('.c3d')]
                if not c3d_files:
                    continue

                c3d_path = os.path.join(video_dir, c3d_files[0])

                try:
                    # Analyze
                    data = read_c3d(c3d_path)
                    result = detector.analyze_data(
                        data.marker_positions,
                        data.marker_labels,
                        data.frame_rate
                    )

                    video_result = {
                        'video_id': name,
                        'fall_detected': result.fall_detected,
                        'confidence': float(result.confidence),
                    }

                    # Check against ground truth
                    ann_path = get_annotation_path(video_dir)
                    if os.path.exists(ann_path):
                        try:
                            ann = load_annotation(ann_path)
                            video_result['ground_truth'] = ann.ground_truth_fall
                            video_result['correct'] = (result.fall_detected == ann.ground_truth_fall)
                            has_ground_truth += 1
                            if video_result['correct']:
                                correct_count += 1
                        except:
                            pass

                    results.append(video_result)

                except Exception as e:
                    results.append({
                        'video_id': name,
                        'error': str(e),
                    })

        summary = {
            'total': len(results),
            'falls_detected': sum(1 for r in results if r.get('fall_detected', False)),
        }

        if has_ground_truth > 0:
            summary['accuracy'] = round(correct_count / has_ground_truth * 100, 1)
            summary['evaluated'] = has_ground_truth

        return jsonify({
            'results': results,
            'summary': summary,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('=' * 50)
    print('Fall Detection API Server v2.0')
    print('=' * 50)
    print(f'Data directory: {DATA_DIR}')
    print('Endpoints:')
    print('  - POST /api/v2/analyze')
    print('  - POST /api/v2/classify-fall-type')
    print('  - POST /api/v2/detect-near-falls')
    print('  - GET  /api/v2/annotations/<video_id>')
    print('  - POST /api/v2/annotations')
    print('  - GET  /api/v2/videos')
    print('  - POST /api/v2/batch-analyze')
    print('=' * 50)
    print('Starting server at http://localhost:5003')
    app.run(debug=True, host='0.0.0.0', port=5003)
