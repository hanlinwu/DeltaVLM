# DeltaVLM Datasets

This directory contains data preparation scripts and documentation for DeltaVLM datasets.

## Datasets Overview

### 1. ChangeChat Dataset

**ChangeChat** is a large-scale multi-turn change dialogue dataset for remote sensing image change analysis.

| Split | Image Pairs | Q-A Pairs |
|-------|-------------|-----------|
| Train | 8,879 | 62,153 |
| Val | 1,110 | 7,770 |
| Test | 1,110 | 7,770 |
| **Total** | **11,099** | **77,693** |

#### Download

```bash
# Download ChangeChat dataset
python data/download_changechat.py --output_dir ./data/changechat
```

#### Data Structure

```
data/changechat/
├── images/
│   ├── train/
│   │   ├── pair_00001_A.png   # Before image
│   │   ├── pair_00001_B.png   # After image
│   │   └── ...
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

#### Annotation Format

Each annotation file is a JSON list with the following structure:

```json
[
  {
    "id": "sample_001",
    "image": ["train/pair_00001_A.png", "train/pair_00001_B.png"],
    "conversations": [
      {"from": "human", "value": "<image>Please describe the changes between these two images."},
      {"from": "gpt", "value": "A new building has been constructed in the center of the image..."},
      {"from": "human", "value": "How many buildings have changed?"},
      {"from": "gpt", "value": "Two buildings have changed. One new building was added..."}
    ]
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `image` | list[str] | Paths to [before_image, after_image] |
| `conversations` | list[dict] | Multi-turn dialogue with "from" (human/gpt) and "value" |

---

### 2. LEVIR-MCI Dataset

**LEVIR-MCI** (LEVIR Multi-class Change Instance) is used for training the mask prediction branch.

#### Download

```bash
# Download LEVIR-MCI dataset
python data/download_levir_mci.py --output_dir ./data/levir_mci
```

#### Data Structure

```
data/levir_mci/
└── images/
    ├── train/
    │   ├── A/           # Before images
    │   │   ├── 00001.png
    │   │   └── ...
    │   ├── B/           # After images
    │   │   ├── 00001.png
    │   │   └── ...
    │   └── label/       # Change masks
    │       ├── 00001.png
    │       └── ...
    ├── val/
    └── test/
```

#### Mask Format

- **Format**: PNG images (grayscale or RGB)
- **Resolution**: 256×256 pixels
- **Classes**:
  - `0`: No change (background)
  - `128`: Road change
  - `255`: Building change
- **Binary conversion**: Any non-zero value → change

---

## Data Preparation

### Option 1: Automatic Download

```bash
# Download all datasets
./data/download_all.sh
```

### Option 2: Manual Download

1. **ChangeChat**: Contact the authors or download from official source
2. **LEVIR-MCI**: Download from [official repository](https://github.com/Chen-Yang-Liu/LEVIR-MCI)

### Verify Dataset

After downloading, verify the data structure:

```bash
python data/verify_dataset.py --dataset changechat --data_root ./data/changechat
python data/verify_dataset.py --dataset levir_mci --data_root ./data/levir_mci
```

---

## Custom Dataset

To use your own dataset, follow these formats:

### For Change Captioning

1. Organize images in `images/{split}/` directories
2. Create annotation JSON files with the format shown above
3. Update the config file paths

### For Mask Training

1. Create `A/`, `B/`, `label/` directories for each split
2. Ensure matching filenames across directories
3. Masks should be binary or multi-class PNG images

---

## Citation

If you use these datasets, please cite:

```bibtex
@article{changechat2024,
  title={ChangeChat: A Large-Scale Multi-Turn Change Dialogue Dataset},
  author={},
  year={2024}
}

@article{levir_mci2023,
  title={LEVIR-MCI: A Multi-class Change Instance Dataset},
  author={},
  year={2023}
}
```


