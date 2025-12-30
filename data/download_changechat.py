#!/usr/bin/env python3
"""
Download Script for ChangeChat Dataset

This script downloads and prepares the ChangeChat dataset for DeltaVLM training.

Usage:
    python data/download_changechat.py --output_dir ./data/changechat

Note: You may need to provide authentication or download manually from the official source.

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Download ChangeChat Dataset")
    parser.add_argument("--output_dir", type=str, default="./data/changechat",
                       help="Output directory for dataset")
    parser.add_argument("--url", type=str, default=None,
                       help="Custom download URL")
    parser.add_argument("--skip_download", action="store_true",
                       help="Skip download, only verify existing files")
    return parser.parse_args()


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        import requests
        from tqdm import tqdm
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_archive(archive_path: str, output_dir: str) -> bool:
    """Extract zip or tar archive."""
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(output_dir)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def verify_dataset(data_dir: str) -> bool:
    """Verify dataset structure."""
    required = [
        "images/train",
        "images/val", 
        "images/test",
        "annotations/train.json",
        "annotations/val.json",
        "annotations/test.json",
    ]
    
    missing = []
    for item in required:
        path = os.path.join(data_dir, item)
        if not os.path.exists(path):
            missing.append(item)
    
    if missing:
        print(f"Missing items: {missing}")
        return False
    
    # Count samples
    for split in ["train", "val", "test"]:
        ann_path = os.path.join(data_dir, "annotations", f"{split}.json")
        with open(ann_path, 'r') as f:
            data = json.load(f)
        print(f"  {split}: {len(data)} samples")
    
    return True


def create_dummy_dataset(output_dir: str):
    """Create dummy dataset for testing (when download is not available)."""
    print("\nCreating dummy dataset structure...")
    print("NOTE: This is for testing only. Replace with actual data.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    # Create dummy annotations
    for split, n_samples in [("train", 10), ("val", 2), ("test", 2)]:
        samples = []
        for i in range(n_samples):
            samples.append({
                "id": f"{split}_{i:05d}",
                "image": [
                    f"{split}/pair_{i:05d}_A.png",
                    f"{split}/pair_{i:05d}_B.png"
                ],
                "conversations": [
                    {"from": "human", "value": "<image>Describe the changes."},
                    {"from": "gpt", "value": "Some buildings have changed."}
                ]
            })
        
        ann_path = os.path.join(output_dir, "annotations", f"{split}.json")
        with open(ann_path, 'w') as f:
            json.dump(samples, f, indent=2)
    
    print(f"Dummy dataset created at {output_dir}")
    print("Please replace with actual ChangeChat data.")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("ChangeChat Dataset Download")
    print("=" * 60)
    
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset already exists
    if verify_dataset(output_dir):
        print("\nDataset already exists and is valid!")
        return
    
    if args.skip_download:
        print("\nSkipping download. Dataset verification failed.")
        return
    
    if args.url:
        # Download from provided URL
        print(f"\nDownloading from {args.url}...")
        archive_path = os.path.join(output_dir, "changechat.zip")
        
        if download_file(args.url, archive_path):
            print("Extracting archive...")
            extract_archive(archive_path, output_dir)
            os.remove(archive_path)
            
            if verify_dataset(output_dir):
                print("\nDataset download complete!")
            else:
                print("\nWarning: Dataset structure may be incorrect.")
        else:
            print("\nDownload failed.")
    else:
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 60)
        print("""
The ChangeChat dataset requires manual download.

Steps:
1. Visit the official ChangeChat repository/website
2. Request access or download the dataset
3. Extract to: {output_dir}

Expected structure:
{output_dir}/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json

For testing purposes, we'll create a dummy dataset structure.
        """.format(output_dir=output_dir))
        
        create_dummy_dataset(output_dir)


if __name__ == "__main__":
    main()


