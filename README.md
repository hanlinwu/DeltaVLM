# DeltaVLM

**Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception**

**Pei Deng, Wenqian Zhou, Hanlin Wu\***

[![arXiv](https://img.shields.io/badge/arXiv-2507.22346-red)](https://arxiv.org/abs/2507.22346)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/hanlinwu/DeltaVLM)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p align="center">
  <img src="docs/assets/architecture.png" width="800">
</p>

## Share us a ⭐ if this repo helps your research!

---

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [ChangeChat-105k Dataset](#changechat-105k-dataset)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

---

## Introduction

DeltaVLM introduces **Remote Sensing Image Change Analysis (RSICA)** — a paradigm combining change detection and visual question answering for multi-turn, instruction-guided exploration of bi-temporal remote sensing images.

**Capabilities**: Change captioning, binary classification, quantification, localization, open-ended QA, multi-turn dialogue.

<details>
<summary><b>Architecture Details</b></summary>

1. **Bi-temporal Vision Encoder (Bi-VE)**: EVA-ViT-g/14 with last 2 blocks fine-tuned
2. **IDPM with CSRM**: Cross-Semantic Relation Measuring to filter irrelevant variations
3. **Instruction-guided Q-Former**: Aligns visual differences with user instructions
4. **Frozen Vicuna-7B**: Language decoder for response generation

</details>

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

# Vicuna-7B (LLM backbone)
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir pretrained/vicuna-7b-v1.5

# BERT (Q-Former tokenizer)
huggingface-cli download bert-base-uncased --local-dir pretrained/bert-base-uncased

# DeltaVLM checkpoint
huggingface-cli download hanlinwu/DeltaVLM --local-dir pretrained/deltavlm
```

### 3. Run Inference

```bash
python scripts/predict.py \
    --image_A path/to/before.png \
    --image_B path/to/after.png \
    --checkpoint pretrained/deltavlm/checkpoint_best.pth \
    --llm_model pretrained/vicuna-7b-v1.5 \
    --bert_model pretrained/bert-base-uncased
```

<details>
<summary><b>Expected Output</b></summary>

```
Using device: cuda
Loading model from pretrained/deltavlm/checkpoint_best.pth...
Model loaded successfully!
Preprocessing images...
Prompt: Please briefly describe the changes in these two images.
Generating response...

==================================================
Generated Description:
A new building has appeared in the lower right area.
==================================================
```

</details>

---

## Training

### Prepare Dataset

Download ChangeChat-105k:

```bash
python data/download_changechat.py --output_dir ./data/changechat
```

<details>
<summary><b>Expected Data Structure</b></summary>

```
data/changechat/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

</details>

### Start Training

```bash
# Single GPU
python scripts/train.py --cfg_path configs/train_stage2.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py --cfg_path configs/train_stage2.yaml
```

---

## Evaluation

```bash
python scripts/evaluate.py --cfg_path configs/evaluate.yaml
```

| Metric | Description |
|--------|-------------|
| BLEU-1/2/3/4 | N-gram precision |
| METEOR | Semantic similarity |
| ROUGE-L | Longest common subsequence |
| CIDEr | Consensus-based image description |

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

If you find this work useful, please cite:

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

- [BLIP-2](https://github.com/salesforce/LAVIS) by Salesforce
- [EVA](https://github.com/baaivision/EVA) vision encoder
- [Vicuna](https://github.com/lm-sys/FastChat) language model
- [LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC) dataset
- [LEVIR-MCI](https://github.com/Chen-Yang-Liu/LEVIR-MCI) dataset

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
