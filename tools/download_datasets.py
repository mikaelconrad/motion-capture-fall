#!/usr/bin/env python3
"""
Download external datasets for fall detection training.

Supports:
- CMU Motion Capture Database (C3D/BVH format)
- SisFall dataset (accelerometer data - requires manual download)

Usage:
    python tools/download_datasets.py cmu --output data/external/cmu_mocap
    python tools/download_datasets.py sisfall --help
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import List, Optional

# CMU Mocap categories that may contain useful data
CMU_CATEGORIES = {
    'walking': ['01', '02', '07', '08', '35', '36', '37', '38', '39'],
    'running': ['09', '16'],
    'jumping': ['13', '93'],
    'balance': ['91'],  # Balance beam
    'physical': ['49', '55', '56'],  # Various physical activities
}

# CMU BVH mirror (GitHub)
CMU_BVH_URL = "https://github.com/una-dinosauria/cmu-mocap/archive/refs/heads/master.zip"

# CMU original site (may have SSL issues)
CMU_ORIGINAL_URL = "http://mocap.cs.cmu.edu/subjects/{subject:02d}/{subject:02d}_{trial:02d}.c3d"


def download_file(url: str, output_path: str, show_progress: bool = True):
    """Download a file with progress indication."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"Downloading: {url}")
        print(f"       To: {output_path}")

    def progress_hook(block_num, block_size, total_size):
        if show_progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            print(f"\r  Progress: {percent}% ({downloaded // 1024}KB / {total_size // 1024}KB)", end='')

    try:
        urllib.request.urlretrieve(url, output_path, progress_hook)
        if show_progress:
            print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_cmu_bvh(output_dir: str, max_files: Optional[int] = None):
    """
    Download CMU Mocap dataset in BVH format from GitHub mirror.

    Args:
        output_dir: Output directory
        max_files: Maximum files to download (None for all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CMU Motion Capture Database (BVH format)")
    print("=" * 60)
    print()
    print("Source: https://github.com/una-dinosauria/cmu-mocap")
    print("License: Free for research use")
    print()

    zip_path = output_dir / "cmu-mocap.zip"

    if not download_file(CMU_BVH_URL, str(zip_path)):
        print("Failed to download CMU dataset.")
        return

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Count BVH files
        bvh_files = [f for f in zf.namelist() if f.endswith('.bvh')]
        print(f"  Found {len(bvh_files)} BVH files")

        if max_files:
            bvh_files = bvh_files[:max_files]
            print(f"  Extracting first {max_files} files")

        # Extract BVH files only
        for i, f in enumerate(bvh_files):
            zf.extract(f, output_dir)
            if (i + 1) % 100 == 0:
                print(f"  Extracted {i + 1}/{len(bvh_files)} files")

    # Clean up zip
    zip_path.unlink()

    print()
    print(f"CMU dataset downloaded to: {output_dir}")
    print()
    print("Note: These are general motion files (walking, running, etc.)")
    print("      Use them as negative samples (non-fall activities).")
    print()
    print("To convert to features:")
    print("  python training/external/cmu_converter.py \\")
    print(f"    --input {output_dir} \\")
    print("    --output data/external/cmu_features.npz")


def download_sisfall_info():
    """Show information about downloading SisFall dataset."""
    print("=" * 60)
    print("SisFall Dataset")
    print("=" * 60)
    print()
    print("Reference:")
    print("  Sucerquia, A., et al. (2017). SisFall: A Fall and Movement Dataset.")
    print("  Sensors, 17(1), 198. DOI: 10.3390/s17010198")
    print()
    print("Dataset Info:")
    print("  - 1798 falls + 2706 activities of daily living (ADL)")
    print("  - Accelerometer and gyroscope data (not C3D)")
    print("  - 19 ADL types, 15 fall types")
    print("  - 23 young adults + 14 elderly participants")
    print()
    print("Download:")
    print("  The original download link is reportedly broken.")
    print("  Contact the authors or check the MDPI journal page:")
    print("  https://www.mdpi.com/1424-8220/17/1/198")
    print()
    print("  Alternative sources:")
    print("  1. Kaggle: Search for 'SisFall' dataset")
    print("  2. IEEE DataPort: May have mirror")
    print("  3. Direct author contact")
    print()
    print("After downloading, place files in: data/external/sisfall/")
    print()
    print("To convert to features:")
    print("  python training/external/sisfall_converter.py \\")
    print("    --input data/external/sisfall \\")
    print("    --output data/external/sisfall_features.npz")
    print()
    print("Note: SisFall uses accelerometer data, not motion capture.")
    print("      Feature alignment is required for compatibility.")


def list_available_datasets():
    """List available datasets and their status."""
    print("=" * 60)
    print("Available External Datasets")
    print("=" * 60)
    print()

    print("1. CMU Motion Capture Database")
    print("   Format: C3D, BVH")
    print("   Content: Walking, running, dancing, sports (2500+ sequences)")
    print("   Use: Negative samples (non-fall activities)")
    print("   Download: python tools/download_datasets.py cmu")
    print()

    print("2. SisFall Dataset")
    print("   Format: CSV (accelerometer/gyroscope)")
    print("   Content: Falls (1798) + ADL (2706)")
    print("   Use: Fall pattern learning (requires feature alignment)")
    print("   Download: Manual (see instructions)")
    print()

    print("3. University of Bath Motion Capture (2024)")
    print("   Format: C3D")
    print("   Content: Walking, running (motion validation)")
    print("   Use: Additional non-fall samples")
    print("   Download: https://www.nature.com/articles/s41597-024-04077-3")
    print()

    # Check what's already downloaded
    print("-" * 60)
    print("Currently downloaded:")
    data_dir = Path("data/external")
    if data_dir.exists():
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                n_files = len(list(subdir.rglob("*")))
                print(f"  {subdir.name}: {n_files} files")
    else:
        print("  None (data/external/ does not exist)")


def main():
    parser = argparse.ArgumentParser(
        description='Download external datasets for fall detection'
    )
    subparsers = parser.add_subparsers(dest='dataset', help='Dataset to download')

    # CMU parser
    cmu_parser = subparsers.add_parser('cmu', help='CMU Motion Capture Database')
    cmu_parser.add_argument('--output', default='data/external/cmu_mocap',
                           help='Output directory')
    cmu_parser.add_argument('--max-files', type=int,
                           help='Maximum files to download')

    # SisFall parser
    sisfall_parser = subparsers.add_parser('sisfall', help='SisFall Dataset')

    # List parser
    list_parser = subparsers.add_parser('list', help='List available datasets')

    args = parser.parse_args()

    if args.dataset == 'cmu':
        download_cmu_bvh(args.output, args.max_files)
    elif args.dataset == 'sisfall':
        download_sisfall_info()
    elif args.dataset == 'list':
        list_available_datasets()
    else:
        parser.print_help()
        print()
        list_available_datasets()


if __name__ == '__main__':
    main()
