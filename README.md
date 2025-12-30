# DeltaVLM

<div align="center">

**Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception**

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://github.com/hanlinwu/DeltaVLM)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green)](LICENSE)

</div>

## Table of Contents

- [Introduction](#introduction)
- [Method Overview](#method-overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

---

## Introduction

Accurate interpretation of land-cover changes in multi-temporal satellite imagery is critical for real-world applications such as disaster management, deforestation monitoring, and environmental surveillance. However, existing methods typically provide only one-shot change masks or static captions, limiting their ability to support interactive, query-driven analysis.

**DeltaVLM** introduces **Remote Sensing Image Change Analysis (RSICA)** as a new paradigm that combines the strengths of change detection and visual question answering to enable multi-turn, instruction-guided exploration of changes in bi-temporal remote sensing images.

### Key Capabilities

- **Change Captioning**: Generate natural language descriptions of changes between bi-temporal images
- **Binary Change Classification**: Determine whether changes occurred between image pairs
- **Category-Specific Change Quantification**: Count specific object changes (e.g., buildings, roads)
- **Change Localization**: Identify spatial locations of changes in a structured grid format
- **Open-ended Question Answering**: Answer diverse user queries about observed changes
- **Multi-turn Dialogue**: Support interactive, context-aware conversations for complex change analysis

### Core Innovations

1. **Bi-temporal Vision Encoder (Bi-VE)**: A fine-tuned vision encoder based on EVA-ViT-g/14 that captures temporal differences while preserving pre-trained knowledge through selective fine-tuning
2. **Instruction-guided Difference Perception Module (IDPM)**: Features a Cross-Semantic Relation Measuring (CSRM) mechanism to filter irrelevant variations (sensor noise, lighting changes) and retain semantically meaningful changes
3. **Instruction-guided Q-former**: Dynamically aligns visual difference features with user instructions for query-relevant response generation
4. **Frozen LLM Backbone**: Utilizes Vicuna-7B as the language decoder, keeping it frozen during training to preserve language capabilities while adapting only vision and alignment modules

---

## Method Overview

<div align="center">
<img src="docs/assets/architecture.png" width="800">
</div>

DeltaVLM is an end-to-end framework for interactive RSICA, comprising three main stages:

### 1. Bi-temporal Visual Feature Encoding

The Bi-temporal Vision Encoder (Bi-VE) processes each image independently using EVA-ViT-g/14. To adapt to remote sensing data while mitigating catastrophic forgetting, only the final two transformer blocks are fine-tuned while the first 37 layers remain frozen.

```
F_t1, F_t2 = Φ_BiVE(I_t1, I_t2)
```

### 2. Instruction-guided Difference Feature Extraction

The IDPM enhances bi-temporal features through the CSRM mechanism:

- **Contextualizing**: Fuses difference features with original temporal features to understand change-context relationships
- **Gating**: Generates relevance scores via sigmoid activation to weight each detected change
- **Filtering**: Selectively retains semantically relevant changes while suppressing noise

The Q-former then aligns these filtered features with user instructions through cross-attention.

### 3. Language Decoding

The frozen Vicuna-7B decoder generates context-aware natural language responses conditioned on the aligned difference features and user instructions.

---

## Installation

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

# Install package in development mode
pip install -e .
```

### Pre-trained Models

| Model | Size | Description | Link |
|-------|------|-------------|------|
| EVA-ViT-G | 1.0B | Vision encoder backbone | [Download](https://huggingface.co/BAAI/EVA) |
| Vicuna-7B-v1.5 | 7B | Language model (frozen) | [Download](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| BERT-base-uncased | 110M | Q-former initialization | [Download](https://huggingface.co/bert-base-uncased) |

Place models in the following structure:
```
pretrained/
├── vicuna-7b-v1.5/
├── bert-base-uncased/
└── eva_vit_g.pth
```

---

## Data Preparation

### ChangeChat-105k Dataset

**ChangeChat-105k** is a large-scale instruction-following dataset constructed to support the RSICA task. It contains **105,107 instruction-response pairs** derived from LEVIR-CC and LEVIR-MCI through a hybrid pipeline combining rule-based methods and GPT-assisted generation.

#### Dataset Statistics

| Instruction Type | Source Data | Generation Method | Training Set | Test Set |
|-----------------|-------------|-------------------|--------------|----------|
| Change Captioning | LEVIR-CC | Rule-based | 34,075 | 1,929 |
| Binary Change Classification | LEVIR-MCI | Rule-based | 6,815 | 1,929 |
| Category-specific Change Quantification | LEVIR-MCI | Rule-based | 6,815 | 1,929 |
| Change Localization | LEVIR-MCI | Rule-based | 6,815 | 1,929 |
| Open-ended QA | Derived | GPT-assisted | 26,600 | 7,527 |
| Multi-turn Conversation | Derived | Rule-based | 6,815 | 1,929 |
| **Total** | -- | -- | **87,935** | **17,172** |

#### Download

```bash
# Download ChangeChat-105k dataset
python data/download_changechat.py --output_dir ./data/changechat

# Verify dataset structure
python data/verify_dataset.py --dataset changechat
```

#### Expected Structure

```
data/
└── changechat/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json
```

See [DATA.md](docs/DATA.md) for detailed annotation format documentation.

---

## Training

DeltaVLM training consists of two stages:

### Stage 1: Q-Former Pre-training

Pre-train the Q-former for vision-language alignment:

```bash
python scripts/train.py --cfg_path configs/train_stage1.yaml
```

### Stage 2: Instruction Tuning

Fine-tune the model on ChangeChat-105k for instruction-following:

```bash
python scripts/train.py --cfg_path configs/train_stage2.yaml \
    model.pretrained=./output/stage1/checkpoint_best.pth
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 scripts/train.py --cfg_path configs/train_stage2.yaml
```

### Training Details

- **Optimizer**: AdamW with weight decay 0.05
- **Learning Rate**: 1e-5
- **Batch Size**: 24
- **Epochs**: 30
- **Data Augmentation**: Random cropping (0-5%), random rotation (±15°), resize to 224×224

---

## Evaluation

Evaluate DeltaVLM on different RSICA subtasks:

### Change Captioning

Evaluate using standard metrics (BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr):

```bash
python scripts/evaluate.py --cfg_path configs/evaluate.yaml \
    model.pretrained=./output/checkpoint_best.pth \
    --task captioning
```

### Binary Change Classification

```bash
python scripts/evaluate.py --cfg_path configs/evaluate.yaml \
    model.pretrained=./output/checkpoint_best.pth \
    --task classification
```

### Change Quantification

```bash
python scripts/evaluate.py --cfg_path configs/evaluate.yaml \
    model.pretrained=./output/checkpoint_best.pth \
    --task quantification
```

### Change Localization

```bash
python scripts/evaluate.py --cfg_path configs/evaluate.yaml \
    model.pretrained=./output/checkpoint_best.pth \
    --task localization
```

---

## Inference

### Single Image Pair

```bash
python scripts/predict.py \
    --image_A path/to/before.png \
    --image_B path/to/after.png \
    --prompt "Please briefly describe the changes in these two images."
```

### Interactive Mode

```bash
python scripts/predict.py \
    --image_A path/to/before.png \
    --image_B path/to/after.png \
    --interactive
```

### Batch Inference

```bash
python scripts/batch_predict.py \
    --input_dir ./test_images \
    --output_dir ./results \
    --checkpoint ./output/checkpoint_best.pth
```

---

## Model Zoo

| Model | Backbone | BLEU-4 | CIDEr | F1 (Classification) | Download |
|-------|----------|--------|-------|---------------------|----------|
| DeltaVLM-7B | Vicuna-7B | 62.51 | 136.72 | 93.83 | [Link](#) |

---

## Repository Structure

```
DeltaVLM/
├── configs/                    # Configuration files
│   ├── train_stage1.yaml
│   ├── train_stage2.yaml
│   └── evaluate.yaml
├── deltavlm/                   # Main source code
│   ├── models/
│   │   ├── base_model.py       # Base model class
│   │   ├── blip2_base.py       # BLIP-2 base architecture
│   │   ├── blip2_vicuna.py     # DeltaVLM main model
│   │   ├── eva_vit.py          # EVA-ViT vision encoder
│   │   ├── qformer.py          # Q-Former architecture
│   │   └── modeling_llama.py   # LLaMA model
│   ├── datasets/
│   │   ├── change_caption.py   # Dataset for change captioning
│   │   └── processors.py       # Image/text processors
│   └── utils/
│       ├── distributed.py      # Distributed training utilities
│       └── optims.py           # Optimizers and schedulers
├── scripts/                    # Training and inference scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/                       # Data preparation scripts
├── docs/                       # Documentation
├── pretrained/                 # Pre-trained weights (download separately)
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{deltavlm2024,
  title={DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception},
  author={Deng, Pei and Zhou, Wenqian and Wu, Hanlin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```

---

## Acknowledgement

This project builds upon the following works:

- [BLIP-2](https://github.com/salesforce/LAVIS) - Vision-Language Pre-training
- [EVA](https://github.com/baaivision/EVA) - Vision Transformer
- [Vicuna](https://github.com/lm-sys/FastChat) - Large Language Model
- [LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC) - Change Captioning Dataset
- [LEVIR-MCI](https://github.com/Chen-Yang-Liu/LEVIR-MCI) - Multi-class Change Instance Dataset

---

## License

This project is released under the [BSD 3-Clause License](LICENSE).
