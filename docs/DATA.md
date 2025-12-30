# Data Preparation Guide

This document provides detailed instructions for preparing datasets for DeltaVLM training and evaluation.

## Table of Contents

- [ChangeChat-105k Dataset](#changechat-105k-dataset)
- [LEVIR-CC Dataset](#levir-cc-dataset)
- [LEVIR-MCI Dataset](#levir-mci-dataset)
- [Custom Dataset](#custom-dataset)
- [Annotation Format](#annotation-format)

---

## ChangeChat-105k Dataset

### Overview

**ChangeChat-105k** is a large-scale instruction-following dataset specifically designed for Remote Sensing Image Change Analysis (RSICA). It was constructed through a hybrid pipeline combining rule-based methods and GPT-assisted generation from the LEVIR-CC and LEVIR-MCI source datasets.

### Key Statistics

- **Total Instruction-Response Pairs**: 105,107
- **Training Set**: 87,935 pairs
- **Test Set**: 17,172 pairs
- **Image Resolution**: 256 × 256 pixels at 0.5 m/pixel spatial resolution

### Instruction Types

The dataset covers six categories of instruction types:

| Type | Description | Generation Method |
|------|-------------|-------------------|
| **Change Captioning** | Generate natural language descriptions of observed changes | Rule-based conversion from LEVIR-CC captions |
| **Binary Change Classification** | Determine if changes occurred (yes/no response) | Rule-based from change maps |
| **Category-Specific Change Quantification** | Count specific object changes (buildings, roads) | Rule-based with OpenCV contour detection |
| **Change Localization** | Identify change locations in 3×3 grid format | Rule-based from change map spatial analysis |
| **Open-ended QA** | Answer diverse questions about changes | GPT-assisted with in-context learning |
| **Multi-turn Conversation** | Chain-of-thought style progressive analysis | Rule-based dialogue construction |

### Download

```bash
# Automatic download
python data/download_changechat.py --output_dir ./data/changechat

# Verify integrity
python data/verify_dataset.py --dataset changechat --data_root ./data/changechat
```

### Directory Structure

```
data/changechat/
├── images/
│   ├── train/
│   │   ├── pair_00001_A.png    # Before image (T1)
│   │   ├── pair_00001_B.png    # After image (T2)
│   │   └── ...
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### Annotation Format

Each annotation file contains a list of samples with the following structure:

```json
{
  "id": "unique_sample_id",
  "image": ["path/to/before.png", "path/to/after.png"],
  "conversations": [
    {"from": "human", "value": "<image>Please briefly describe the changes in these two images."},
    {"from": "gpt", "value": "A new building has been constructed in the center of the image, and a road has been extended in the bottom-left area."}
  ]
}
```

#### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the sample |
| `image` | list[str] | Paths to bi-temporal images [T1, T2] |
| `conversations` | list[dict] | Alternating human/gpt conversation turns |
| `conversations[i].from` | string | Speaker identifier ("human" or "gpt") |
| `conversations[i].value` | string | Message content; first human message includes `<image>` placeholder |

#### Multi-turn Dialogue Example

```json
{
  "id": "multiturn_00042",
  "image": ["train/pair_00042_A.png", "train/pair_00042_B.png"],
  "conversations": [
    {"from": "human", "value": "<image>Please judge whether these two images have changed or not. Answer yes or no."},
    {"from": "gpt", "value": "Yes."},
    {"from": "human", "value": "What types of objects have changed?"},
    {"from": "gpt", "value": "Buildings and roads have changed in the images."},
    {"from": "human", "value": "How many buildings have been added?"},
    {"from": "gpt", "value": "Three new buildings have been constructed."},
    {"from": "human", "value": "Please describe the changes in detail."},
    {"from": "gpt", "value": "Three new buildings have appeared in the top-right region. Additionally, a road has been extended from the center toward the bottom-left corner of the image."}
  ]
}
```

#### Change Localization Format

For localization tasks, changes are reported using a 3×3 grid notation:

| Code | Position |
|------|----------|
| TL | Top-Left |
| TC | Top-Center |
| TR | Top-Right |
| CL | Center-Left |
| CC | Center-Center |
| CR | Center-Right |
| BL | Bottom-Left |
| BC | Bottom-Center |
| BR | Bottom-Right |

Example response: `"Changes are located in TL, TC, and CC regions."`

---

## LEVIR-CC Dataset

### Overview

LEVIR-CC (LEVIR Change Captioning) is the primary source dataset for change captioning annotations. It provides bi-temporal image pairs with five human-written captions each.

### Statistics

- **Image Pairs**: 10,077
- **Captions per Pair**: 5
- **Total Captions**: 50,385
- **Image Size**: 256 × 256 pixels

### Download

LEVIR-CC can be obtained from the official repository:
- Repository: https://github.com/Chen-Yang-Liu/RSICC

---

## LEVIR-MCI Dataset

### Overview

LEVIR-MCI (LEVIR Multi-class Change Instance) extends LEVIR-CC with pixel-level change masks, supporting binary change detection and fine-grained change analysis.

### Statistics

- **Image Pairs**: 10,077 (same as LEVIR-CC)
- **Mask Classes**: 3 (background, road, building)

### Download

```bash
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
    └── test/
```

### Mask Format

| Pixel Value | Class |
|-------------|-------|
| 0 | No change (background) |
| 128 | Road change |
| 255 | Building change |

---

## Custom Dataset

### For Change Captioning Tasks

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

3. **Update configuration**:
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

### Requirements

- Images should be PNG or JPEG format
- Recommended resolution: 256×256 pixels
- Matching filenames for bi-temporal pairs
- First human message must contain `<image>` placeholder
- Alternating human/gpt conversation turns

---

## Troubleshooting

### Common Issues

1. **File not found errors**
   - Verify paths in annotation files are relative to the images directory
   - Check file extensions match exactly

2. **Image loading errors**
   - Ensure images are valid PNG/JPEG files
   - Verify file permissions

3. **Annotation parsing errors**
   - Validate JSON syntax
   - Ensure all required fields are present

### Verification Script

```bash
# Verify ChangeChat-105k dataset
python data/verify_dataset.py --dataset changechat --data_root ./data/changechat

# Verify LEVIR-MCI dataset  
python data/verify_dataset.py --dataset levir_mci --data_root ./data/levir_mci
```
