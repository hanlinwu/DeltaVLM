# DeltaVLM

**Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception**

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2507.22346)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green)](LICENSE)

<p align="center">
  <img src="docs/assets/architecture.png" width="800">
</p>

## Introduction

DeltaVLM introduces **Remote Sensing Image Change Analysis (RSICA)** — a paradigm combining change detection and visual question answering for multi-turn, instruction-guided exploration of bi-temporal remote sensing images.

**Capabilities**: Change captioning, binary classification, quantification, localization, open-ended QA, multi-turn dialogue.

**Architecture**:
1. **Bi-temporal Vision Encoder**: Fine-tuned EVA-ViT-g/14 (last 2 blocks trainable)
2. **IDPM with CSRM**: Filters irrelevant variations, retains meaningful changes
3. **Instruction-guided Q-former**: Aligns visual differences with user instructions
4. **Frozen Vicuna-7B**: Language decoder

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/hanlinwu/DeltaVLM.git
cd DeltaVLM

conda create -n deltavlm python=3.10 -y
conda activate deltavlm

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

### 2. Download Pretrained Models

```bash
mkdir -p pretrained

# Vicuna-7B (requires HuggingFace login)
huggingface-cli login
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir pretrained/vicuna-7b-v1.5

# BERT (for Q-former)
huggingface-cli download bert-base-uncased --local-dir pretrained/bert-base-uncased

# EVA-ViT-G (auto-downloads during training, or manually)
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/EVA/eva_vit_g.pth -P pretrained/
```

### 3. Prepare Dataset

Download ChangeChat-105k (105,107 instruction-response pairs):

```bash
python data/download_changechat.py --output_dir ./data/changechat
```

Expected structure:
```
data/changechat/
├── images/{train,val,test}/
└── annotations/{train,val,test}.json
```

See [docs/DATA.md](docs/DATA.md) for annotation format.

### 4. Training

```bash
# Single GPU
python scripts/train.py --cfg_path configs/train_stage2.yaml

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train.py --cfg_path configs/train_stage2.yaml
```

### 5. Inference

```bash
python scripts/predict.py \
    --image_A path/to/before.png \
    --image_B path/to/after.png \
    --prompt "Describe the changes in these two images."
```

---

## Multi-turn Dialogue Example

<p align="center">
  <img src="docs/assets/multi-dialogue.png" width="800">
</p>

---

## ChangeChat-105k Dataset

| Instruction Type | Train | Test |
|-----------------|-------|------|
| Change Captioning | 34,075 | 1,929 |
| Binary Classification | 6,815 | 1,929 |
| Change Quantification | 6,815 | 1,929 |
| Change Localization | 6,815 | 1,929 |
| Open-ended QA | 26,600 | 7,527 |
| Multi-turn Dialogue | 6,815 | 1,929 |
| **Total** | **87,935** | **17,172** |

---

## Citation

```bibtex
@article{deltavlm2024,
  title={DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception},
  author={Deng, Pei and Zhou, Wenqian and Wu, Hanlin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```

## Acknowledgement

[BLIP-2](https://github.com/salesforce/LAVIS) | [EVA](https://github.com/baaivision/EVA) | [Vicuna](https://github.com/lm-sys/FastChat) | [LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC) | [LEVIR-MCI](https://github.com/Chen-Yang-Liu/LEVIR-MCI)

## License

[BSD 3-Clause License](LICENSE)
