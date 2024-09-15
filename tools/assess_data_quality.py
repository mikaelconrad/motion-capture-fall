#!/usr/bin/env python3
"""
Analyze new Qualisys datasets in `new_data/` to assess completeness/quality.

For each subfolder containing `unknown.c3d`, we compute:
- frame_rate, n_frames, point_count
- overall valid sample ratio (non-zero XYZ)
- per-frame valid marker ratio and proportion of frames with >=80% valid markers
- presence of head markers and their coverage across frames

Outputs a concise, human-readable summary and a JSON-like block for downstream use.
"""

import os
import json
from typing import Dict, Any, List

import numpy as np
import c3d


# Point to repo root new_data directory (one level up from tools/)
NEW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new_data')


def is_point_valid(point_xyz: np.ndarray) -> bool:
    """Return True if a (3,) array has any non-zero coordinate."""
    # Many C3D writers encode missing markers as (0, 0, 0)
    return point_xyz.shape == (3,) and not np.allclose(point_xyz, 0.0)


def detect_head_marker_indices(point_labels: List[str]) -> List[int]:
    """Find indices of markers that likely correspond to the head."""
    head_keywords = ['head', 'q_head', 'headtop', 'helmet']
    indices: List[int] = []
    for idx, raw_label in enumerate(point_labels):
        label = (raw_label or '').strip().lower()
        if any(key in label for key in head_keywords):
            indices.append(idx)
    return indices


def analyze_c3d_file(c3d_path: str) -> Dict[str, Any]:
    with open(c3d_path, 'rb') as f:
        reader = c3d.Reader(f)

        frame_rate = float(reader.header.frame_rate)
        first_frame = int(reader.header.first_frame)
        last_frame = int(reader.header.last_frame)
        n_frames = last_frame - first_frame + 1 if last_frame >= first_frame else 0
        point_count = int(reader.header.point_count)

        # Gather point labels if available
        point_labels: List[str] = []
        try:
            for i in range(point_count):
                point_labels.append(reader.point_labels[i].strip())
        except Exception:
            point_labels = []

        head_indices = detect_head_marker_indices(point_labels) if point_labels else []

        total_pairs = 0
        valid_pairs = 0
        frames_with_ge80pct_valid = 0
        head_frames_valid = 0
        frames_processed = 0

        for _, points, _ in reader.read_frames():
            # points shape: (point_count, 5) -> use first 3 for XYZ
            xyz = points[:, :3]

            # Validity per marker for this frame
            valid_mask = np.any(xyz != 0.0, axis=1)

            frames_processed += 1
            total_pairs += xyz.shape[0]
            valid_pairs += int(np.count_nonzero(valid_mask))

            # Frame-level valid ratio
            if xyz.shape[0] > 0:
                frame_valid_ratio = float(np.count_nonzero(valid_mask)) / float(xyz.shape[0])
                if frame_valid_ratio >= 0.80:
                    frames_with_ge80pct_valid += 1

            # Head coverage this frame
            if head_indices:
                head_valid = any(valid_mask[idx] for idx in head_indices if idx < valid_mask.shape[0])
                if head_valid:
                    head_frames_valid += 1

        overall_valid_ratio = (valid_pairs / total_pairs) if total_pairs > 0 else 0.0
        ge80pct_frames_ratio = (frames_with_ge80pct_valid / frames_processed) if frames_processed > 0 else 0.0
        head_coverage_ratio = (head_frames_valid / frames_processed) if frames_processed > 0 else 0.0

        return {
            'c3d_path': c3d_path,
            'frame_rate': frame_rate,
            'n_frames': frames_processed if frames_processed else n_frames,
            'point_count': point_count,
            'has_point_labels': bool(point_labels),
            'head_marker_indices': head_indices,
            'overall_valid_ratio': round(overall_valid_ratio, 3),
            'ge80pct_frames_ratio': round(ge80pct_frames_ratio, 3),
            'head_coverage_ratio': round(head_coverage_ratio, 3),
        }


def summarize_quality(metrics: Dict[str, Any]) -> str:
    """Heuristic quality classification based on validity and head coverage."""
    valid = metrics['overall_valid_ratio']
    frame_ok = metrics['ge80pct_frames_ratio']
    head_ok = metrics['head_coverage_ratio']
    has_labels = metrics['has_point_labels']

    if valid >= 0.9 and frame_ok >= 0.9 and (head_ok >= 0.6 or not metrics['head_marker_indices']):
        return 'excellent'
    if valid >= 0.8 and frame_ok >= 0.8:
        return 'good'
    if valid >= 0.6 and frame_ok >= 0.6:
        return 'fair'
    return 'poor'


def main() -> None:
    if not os.path.isdir(NEW_DATA_DIR):
        print(f"No 'new_data' directory found at: {NEW_DATA_DIR}")
        return

    results: List[Dict[str, Any]] = []

    for entry in sorted(os.listdir(NEW_DATA_DIR)):
        subdir = os.path.join(NEW_DATA_DIR, entry)
        if not os.path.isdir(subdir):
            continue

        # Find any .c3d file in the subdirectory. Prefer 'unknown.c3d' if present.
        c3d_candidates = [f for f in os.listdir(subdir) if f.lower().endswith('.c3d')]
        chosen_c3d = None
        if c3d_candidates:
            if 'unknown.c3d' in c3d_candidates:
                chosen_c3d = 'unknown.c3d'
            else:
                chosen_c3d = sorted(c3d_candidates)[0]
        c3d_path = os.path.join(subdir, chosen_c3d) if chosen_c3d else ''
        has_c3d = bool(chosen_c3d and os.path.isfile(c3d_path))

        # Collect presence of auxiliary files
        aux_files = {
            'unknown_csv': os.path.isfile(os.path.join(subdir, 'unknown.csv')),
            'projected_csvs': sum(1 for f in os.listdir(subdir) if f.startswith('unknown.projected') and f.endswith('.csv')),
            'stream_videos': sum(1 for f in os.listdir(subdir) if f.startswith('stream') and f.endswith('.avi')),
        }

        if not has_c3d:
            results.append({
                'folder': entry,
                'has_c3d': False,
                'aux': aux_files,
                'note': 'No C3D file present (cannot run detectors)'
            })
            continue

        try:
            metrics = analyze_c3d_file(c3d_path)
            quality = summarize_quality(metrics)
            results.append({
                'folder': entry,
                'has_c3d': True,
                'c3d_filename': chosen_c3d,
                'quality': quality,
                'metrics': metrics,
                'aux': aux_files,
            })
        except Exception as exc:
            results.append({
                'folder': entry,
                'has_c3d': True,
                'c3d_filename': chosen_c3d,
                'error': str(exc),
                'aux': aux_files,
            })

    # Sort by quality and completeness heuristics
    def sort_key(item: Dict[str, Any]):
        order = {'excellent': 3, 'good': 2, 'fair': 1, 'poor': 0}
        q = order.get(item.get('quality', ''), -1)
        if 'metrics' in item:
            m = item['metrics']
            return (q, m.get('overall_valid_ratio', 0.0), m.get('ge80pct_frames_ratio', 0.0), m.get('head_coverage_ratio', 0.0))
        return (q, 0.0, 0.0, 0.0)

    # Human-readable report
    print("New Data Quality Report")
    print("========================\n")
    for item in sorted(results, key=sort_key, reverse=True):
        print(f"Folder: {item['folder']}")
        if not item.get('has_c3d'):
            print("  C3D: MISSING")
            print(f"  Aux: videos={item['aux']['stream_videos']}, projected_csvs={item['aux']['projected_csvs']}, unknown.csv={item['aux']['unknown_csv']}")
            if 'note' in item:
                print(f"  Note: {item['note']}")
            print()
            continue

        if 'error' in item:
            print(f"  C3D: ERROR reading ({item.get('c3d_filename', 'unknown')})")
            print(f"  Error: {item['error']}")
            print()
            continue

        m = item['metrics']
        print(f"  Quality: {item['quality']}")
        print(f"  C3D file: {item.get('c3d_filename', 'unknown')}")
        print(f"  Frames: {m['n_frames']}, Frame rate: {m['frame_rate']} Hz, Markers: {m['point_count']}")
        print(f"  Valid ratio (overall): {m['overall_valid_ratio']:.3f}")
        print(f"  Frames with >=80% valid markers: {m['ge80pct_frames_ratio']:.3f}")
        if m['head_marker_indices']:
            print(f"  Head markers found: {len(m['head_marker_indices'])} (coverage: {m['head_coverage_ratio']:.3f})")
        else:
            print("  Head markers found: 0")
        print(f"  Aux: videos={item['aux']['stream_videos']}, projected_csvs={item['aux']['projected_csvs']}, unknown.csv={item['aux']['unknown_csv']}")
        print()

    # Machine-readable block (single line JSON for easy copy/paste)
    print("JSON_SUMMARY_START")
    print(json.dumps(results))
    print("JSON_SUMMARY_END")


if __name__ == '__main__':
    main()


