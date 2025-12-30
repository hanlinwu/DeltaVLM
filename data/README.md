# DeltaVLM Datasets

This directory contains data preparation scripts and documentation for DeltaVLM datasets.

## Datasets Overview

### ChangeChat-105k Dataset

**ChangeChat-105k** is a large-scale instruction-following dataset for Remote Sensing Image Change Analysis (RSICA). It was constructed through a hybrid pipeline combining rule-based methods and GPT-assisted generation.

#### Key Statistics

| Metric | Value |
|--------|-------|
| Total Instruction-Response Pairs | 105,107 |
| Training Set | 87,935 |
| Test Set | 17,172 |
| Image Resolution | 256 × 256 pixels |
| Spatial Resolution | 0.5 m/pixel |

#### Instruction Types

| Type | Training | Test | Generation Method |
|------|----------|------|-------------------|
| Change Captioning | 34,075 | 1,929 | Rule-based |
| Binary Change Classification | 6,815 | 1,929 | Rule-based |
| Category-specific Change Quantification | 6,815 | 1,929 | Rule-based |
| Change Localization | 6,815 | 1,929 | Rule-based |
| Open-ended QA | 26,600 | 7,527 | GPT-assisted |
| Multi-turn Conversation | 6,815 | 1,929 | Rule-based |

#### Download

```bash
# Download ChangeChat-105k dataset
python data/download_changechat.py --output_dir ./data/changechat
```

#### Data Structure

```
data/changechat/
├── images/
│   ├── train/
│   │   ├── pair_00001_A.png   # Before image (T1)
│   │   ├── pair_00001_B.png   # After image (T2)
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

### LEVIR-MCI Dataset

**LEVIR-MCI** (LEVIR Multi-class Change Instance) provides pixel-level change masks and serves as the source for generating quantification and localization annotations in ChangeChat-105k.

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

- **Format**: PNG images (grayscale)
- **Resolution**: 256×256 pixels
- **Classes**:
  - `0`: No change (background)
  - `128`: Road change
  - `255`: Building change

---

## Data Preparation

### Option 1: Automatic Download

```bash
# Download all datasets
./data/download_all.sh
```

### Option 2: Manual Download

1. **ChangeChat-105k**: Contact the authors or download from official source
2. **LEVIR-MCI**: Download from [official repository](https://github.com/Chen-Yang-Liu/LEVIR-MCI)
3. **LEVIR-CC**: Download from [official repository](https://github.com/Chen-Yang-Liu/RSICC)

### Verify Dataset

After downloading, verify the data structure:

```bash
python data/verify_dataset.py --dataset changechat --data_root ./data/changechat
python data/verify_dataset.py --dataset levir_mci --data_root ./data/levir_mci
```

---

## Custom Dataset

To use your own dataset for change captioning:

1. Organize images in `images/{split}/` directories
2. Create annotation JSON files with the format shown above
3. Update the config file paths

---

## Citation

If you use these datasets, please cite:

```bibtex
@article{deltavlm2024,
  title={DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception},
  author={Deng, Pei and Zhou, Wenqian and Wu, Hanlin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}

@article{liu2022remote,
  title={Remote sensing image change captioning with dual-branch transformers},
  author={Liu, Chenyang and others},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022}
}
```
