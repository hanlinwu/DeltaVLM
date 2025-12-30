# DeltaVLM

<div align="center">

**Interactive Remote Sensing Image Change Analysis with Vision-Language Models**

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://github.com/hanlinwu/DeltaVLM)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green)](LICENSE)

</div>

## 📋 Table of Contents
- [Introduction](#-introduction)
- [Method Overview](#-method-overview)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Model Zoo](#-model-zoo)
- [Citation](#-citation)
- [Acknowledgement](#-acknowledgement)

---

## 🌟 Introduction

**DeltaVLM** is a Vision-Language Model designed for **Interactive Remote Sensing Image Change Analysis (RSICA)**. Given a pair of bi-temporal remote sensing images (before and after), DeltaVLM can:

-  **Change Captioning**: Generate natural language descriptions of changes
-  **Interactive QA**: Answer questions about specific changes  
-  **Change Localization**: Predict binary change masks (with Mask Branch)

### Key Features

-  Built on **BLIP-2** architecture with **EVA-ViT-G** visual encoder and **Vicuna-7B** LLM
-  **CSRM (Change-aware Spatial Representation Module)** for difference-aware feature extraction
-  Trained on **ChangeChat-105k** - a large-scale multi-turn change dialogue dataset

---

## 🧠 Method Overview

<div align="center">
<img src="docs/assets/architecture.png" width="800">
</div>

DeltaVLM consists of three main components:

1. **Visual Encoder (EVA-ViT-G)**: Extracts features from bi-temporal images
2. **Change-aware Spatial Representation Module (CSRM)**: Models temporal differences with gated attention
3. **Q-Former + LLM (Vicuna-7B)**: Bridges vision and language for interactive dialogue

### Training Strategy

- **Stage 1**: Q-Former pre-training with image-text alignment
- **Stage 2**: Instruction tuning with multi-turn dialogue

---

## 🔧 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.3

### Setup

```bash
# Clone the repository
git clone https://github.com/hanlinwu/DeltaVLM.git
cd DeltaVLM

# Create conda environment
conda create -n deltavlm python=3.10 -y
conda activate deltavlm

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_pretrained.py
```

### Pre-trained Models Required

| Model | Size | Link |
|-------|------|------|
| EVA-ViT-G | 1.0B | [Download](https://huggingface.co/BAAI/EVA) |
| Vicuna-7B-v1.5 | 7B | [Download](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| BERT-base-uncased | 110M | [Download](https://huggingface.co/bert-base-uncased) |

Place models in the following structure:
```
pretrained/
├── vicuna-7b-v1.5/
├── bert-base-uncased/
└── eva_vit_g.pth
```

---

## 📦 Data Preparation

### ChangeChat-105k Dataset

ChangeChat is a large-scale multi-turn change dialogue dataset containing:
- 🖼️ **11,099** bi-temporal image pairs
- 💬 **77,693** question-answer pairs
- 🗣️ Multi-turn conversation format

```bash
# Download ChangeChat dataset
python data/download_changechat.py --output_dir ./data/changechat

# Verify dataset structure
python data/verify_dataset.py --dataset changechat
```

### LEVIR-MCI Dataset (for Mask Training)

For change mask prediction, we use the LEVIR-MCI dataset:

```bash
# Download LEVIR-MCI
python data/download_levir_mci.py --output_dir ./data/levir_mci
```

Expected data structure:
```
data/
├── changechat/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
└── levir_mci/
    └── images/
        ├── train/
        │   ├── A/
        │   ├── B/
        │   └── label/
        ├── val/
        └── test/
```

See [DATA.md](docs/DATA.md) for detailed data format documentation.

---

## 🚀 Training

### Stage 1: Q-Former Pre-training

```bash
python scripts/train.py --cfg_path configs/train_stage1.yaml
```

### Stage 2: Instruction Tuning

```bash
python scripts/train.py --cfg_path configs/train_stage2.yaml \
    model.pretrained=./output/stage1/checkpoint_best.pth
```

### Stage 3: Mask Branch Training (Optional)

Train the mask prediction branch while freezing other components:

```bash
python scripts/train_mask.py --cfg_path configs/train_mask.yaml \
    --pretrained ./output/stage2/checkpoint_best.pth
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 scripts/train.py --cfg_path configs/train_stage2.yaml
```

---

## 📊 Evaluation

### Change Captioning Evaluation

Evaluate on standard change captioning metrics (BLEU, CIDEr, METEOR, ROUGE):

```bash
python scripts/evaluate.py --cfg_path configs/evaluate.yaml \
    model.pretrained=./output/checkpoint_best.pth
```

### Mask Prediction Evaluation

```bash
python scripts/evaluate_mask.py \
    --checkpoint ./output/checkpoint_best.pth \
    --mask_branch ./output/mask_branch_best.pth \
    --data_root ./data/levir_mci/images
```

---

## 🎮 Inference

### Interactive Demo

```bash
python scripts/predict.py \
    --image_A path/to/before.png \
    --image_B path/to/after.png \
    --prompt "Describe the changes between these two images."
```

### Mask Prediction

```bash
python scripts/predict_mask.py \
    --image_A path/to/before.png \
    --image_B path/to/after.png \
    --output change_mask.png \
    --visualize
```

### Batch Inference

```bash
python scripts/batch_predict.py \
    --input_dir ./test_images \
    --output_dir ./results \
    --checkpoint ./output/checkpoint_best.pth
```

---

## 🏛️ Model Zoo

| Model | Backbone | BLEU-4 | CIDEr | Mask IoU | Download |
|-------|----------|--------|-------|----------|----------|
| DeltaVLM-7B | Vicuna-7B | 42.3 | 136.5 | - | [Link](#) |
| DeltaVLM-7B-Mask | Vicuna-7B | 42.3 | 136.5 | 68.2 | [Link](#) |

---

## 📁 Repository Structure

```
DeltaVLM/
├── configs/                # Configuration files
├── deltavlm/               # Main source code
│   ├── models/             # Model architectures
│   ├── datasets/           # Dataset loaders
│   ├── evaluation/         # Evaluation metrics
│   └── utils/              # Utility functions
├── scripts/                # Training & inference scripts
├── data/                   # Data preparation scripts
├── docs/                   # Documentation
└── pretrained/             # Pretrained model weights (download separately)
```

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{deltavlm2024,
  title={DeltaVLM: Interactive Remote Sensing Image Change Analysis with Vision-Language Models},
  author={},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```

---

## 🙏 Acknowledgement

This project builds upon the following excellent works:

- [BLIP-2](https://github.com/salesforce/LAVIS) - Vision-Language Pre-training
- [EVA](https://github.com/baaivision/EVA) - Vision Transformer
- [Vicuna](https://github.com/lm-sys/FastChat) - Large Language Model

---

## 📜 License

This project is released under the [BSD 3-Clause License](LICENSE).

---

<div align="center">

**⭐ If you find this project helpful, please give us a star! ⭐**

</div>
