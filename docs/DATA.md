# Data Preparation Guide

This document provides detailed instructions for preparing datasets for DeltaVLM.

## Table of Contents

- [ChangeChat Dataset](#changechat-dataset)
- [LEVIR-MCI Dataset](#levir-mci-dataset)
- [Custom Dataset](#custom-dataset)
- [Annotation Format](#annotation-format)

---

## ChangeChat Dataset

### Overview

ChangeChat is a large-scale multi-turn change dialogue dataset containing:
- **11,099** bi-temporal image pairs
- **77,693** question-answer pairs
- Multi-turn conversation format for interactive change analysis

### Download

```bash
# Automatic download (if available)
python data/download_changechat.py --output_dir ./data/changechat

# Or download manually and extract to ./data/changechat/
```

### Directory Structure

```
data/changechat/
├── images/
│   ├── train/
│   │   ├── pair_00001_A.png
│   │   ├── pair_00001_B.png
│   │   └── ...
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### Annotation Format

```json
{
  "id": "unique_sample_id",
  "image": ["path/to/before.png", "path/to/after.png"],
  "conversations": [
    {"from": "human", "value": "<image>Describe the changes..."},
    {"from": "gpt", "value": "A new building has been constructed..."},
    {"from": "human", "value": "How many buildings changed?"},
    {"from": "gpt", "value": "Two buildings have changed..."}
  ]
}
```

---

## LEVIR-MCI Dataset

### Overview

LEVIR-MCI (LEVIR Multi-class Change Instance) provides:
- High-resolution bi-temporal image pairs
- Pixel-level change masks
- Multi-class annotations (roads, buildings)

### Download

```bash
# Download from official source
python data/download_levir_mci.py --output_dir ./data/levir_mci

# Or download from: https://github.com/Chen-Yang-Liu/LEVIR-MCI
```

### Directory Structure

```
data/levir_mci/
└── images/
    ├── train/
    │   ├── A/           # Before images (T1)
    │   │   ├── 00001.png
    │   │   └── ...
    │   ├── B/           # After images (T2)
    │   │   ├── 00001.png
    │   │   └── ...
    │   └── label/       # Change masks
    │       ├── 00001.png
    │       └── ...
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/
```

### Mask Format

| Value | Class |
|-------|-------|
| 0 | No change (background) |
| 128 | Road change |
| 255 | Building change |

**Note**: For binary segmentation, any non-zero value is treated as "change".

---

## Custom Dataset

### For Change Captioning

1. **Organize images**:
```
your_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

2. **Create annotations** following this format:
```json
[
  {
    "id": "sample_001",
    "image": ["train/before_001.png", "train/after_001.png"],
    "conversations": [
      {"from": "human", "value": "<image>Describe the changes."},
      {"from": "gpt", "value": "Your change description here."}
    ]
  }
]
```

3. **Update config** with your paths:
```yaml
datasets:
  your_dataset:
    build_info:
      annotations:
        train:
          storage: "path/to/annotations/train.json"
      images:
        storage: "path/to/images"
```

### For Mask Prediction

1. **Organize images**:
```
your_mask_dataset/
└── images/
    ├── train/
    │   ├── A/      # Before images
    │   ├── B/      # After images
    │   └── label/  # Binary masks
    ├── val/
    └── test/
```

2. **Requirements**:
- Matching filenames across A/, B/, label/ directories
- Masks: PNG format, 0 = no change, 255 = change
- Recommended resolution: 256×256

3. **Usage**:
```bash
python scripts/train_mask.py \
    --data_root ./your_mask_dataset/images \
    --pretrained ./checkpoint.pth
```

---

## Annotation Format

### Multi-turn Dialogue Format

The dialogue format supports multi-turn conversations:

```json
{
  "id": "sample_id",
  "image": ["before.png", "after.png"],
  "conversations": [
    {"from": "human", "value": "<image>Question 1?"},
    {"from": "gpt", "value": "Answer 1."},
    {"from": "human", "value": "Follow-up question?"},
    {"from": "gpt", "value": "Follow-up answer."}
  ]
}
```

**Key points**:
- First human message should contain `<image>` placeholder
- Alternating human/gpt turns
- Any number of turns supported

### Evaluation Format

For evaluation, use a simplified format:

```json
{
  "image_A": "path/to/before.png",
  "image_B": "path/to/after.png",
  "captions": ["Ground truth caption 1", "Ground truth caption 2"]
}
```

---

## Troubleshooting

### Common Issues

1. **File not found errors**
   - Verify paths in annotation files are relative to `vis_root`
   - Check file extensions match

2. **Image loading errors**
   - Ensure images are valid PNG/JPEG files
   - Check file permissions

3. **Mask dimension mismatch**
   - Verify mask dimensions match config (`mask_size`)
   - Check mask is grayscale or convertible

### Verification Script

```bash
# Verify ChangeChat dataset
python data/verify_dataset.py --dataset changechat --data_root ./data/changechat

# Verify LEVIR-MCI dataset
python data/verify_dataset.py --dataset levir_mci --data_root ./data/levir_mci
```


