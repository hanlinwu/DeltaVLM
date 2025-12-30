#!/usr/bin/env python3
"""
Download Script for LEVIR-MCI Dataset

This script downloads and prepares the LEVIR-MCI dataset for mask training.

Usage:
    python data/download_levir_mci.py --output_dir ./data/levir_mci

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download LEVIR-MCI Dataset")
    parser.add_argument("--output_dir", type=str, default="./data/levir_mci",
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


def verify_dataset(data_dir: str) -> bool:
    """Verify dataset structure."""
    images_dir = os.path.join(data_dir, "images")
    
    required = []
    for split in ["train", "val", "test"]:
        required.extend([
            f"images/{split}/A",
            f"images/{split}/B",
            f"images/{split}/label",
        ])
    
    missing = []
    for item in required:
        path = os.path.join(data_dir, item)
        if not os.path.exists(path):
            missing.append(item)
    
    if missing:
        print(f"Missing directories: {missing}")
        return False
    
    # Count samples
    print("\nDataset statistics:")
    for split in ["train", "val", "test"]:
        label_dir = os.path.join(images_dir, split, "label")
        if os.path.exists(label_dir):
            n_samples = len([f for f in os.listdir(label_dir) if f.endswith('.png')])
            print(f"  {split}: {n_samples} samples")
    
    return True


def create_dummy_dataset(output_dir: str):
    """Create dummy dataset structure for testing."""
    print("\nCreating dummy dataset structure...")
    print("NOTE: This is for testing only. Replace with actual data.")
    
    import numpy as np
    from PIL import Image
    
    images_dir = os.path.join(output_dir, "images")
    
    for split, n_samples in [("train", 10), ("val", 2), ("test", 2)]:
        for subdir in ["A", "B", "label"]:
            os.makedirs(os.path.join(images_dir, split, subdir), exist_ok=True)
        
        for i in range(n_samples):
            filename = f"{i:05d}.png"
            
            # Create dummy images (256x256 RGB)
            img_A = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            img_B = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            
            # Create dummy mask (256x256 grayscale)
            mask = np.zeros((256, 256), dtype=np.uint8)
            # Add some random "changes"
            mask[50:100, 50:100] = 255
            mask_img = Image.fromarray(mask)
            
            img_A.save(os.path.join(images_dir, split, "A", filename))
            img_B.save(os.path.join(images_dir, split, "B", filename))
            mask_img.save(os.path.join(images_dir, split, "label", filename))
    
    print(f"Dummy dataset created at {output_dir}")
    print("Please replace with actual LEVIR-MCI data.")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LEVIR-MCI Dataset Download")
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
        archive_path = os.path.join(output_dir, "levir_mci.zip")
        
        if download_file(args.url, archive_path):
            print("Extracting archive...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
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
The LEVIR-MCI dataset requires manual download.

Steps:
1. Visit: https://github.com/Chen-Yang-Liu/LEVIR-MCI
2. Download the dataset following their instructions
3. Extract to: {output_dir}

Expected structure:
{output_dir}/
└── images/
    ├── train/
    │   ├── A/        # Before images
    │   ├── B/        # After images
    │   └── label/    # Change masks
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/

For testing purposes, we'll create a dummy dataset structure.
        """.format(output_dir=output_dir))
        
        try:
            create_dummy_dataset(output_dir)
        except ImportError:
            print("Note: Install PIL to create dummy images")
            print("Creating empty directories instead...")
            for split in ["train", "val", "test"]:
                for subdir in ["A", "B", "label"]:
                    os.makedirs(os.path.join(output_dir, "images", split, subdir), exist_ok=True)


if __name__ == "__main__":
    main()


