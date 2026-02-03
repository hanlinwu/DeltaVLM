# DeltaVLM

**Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception**

[![Paper](https://img.shields.io/badge/TGRS-Paper-blue)](https://ieeexplore.ieee.org/document/xxxxx)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22346-red)](https://arxiv.org/abs/2507.22346)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green)](LICENSE)

<p align="center">
  <img src="docs/assets/architecture.png" width="800">
</p>

## Introduction

DeltaVLM introduces **Remote Sensing Image Change Analysis (RSICA)** — a paradigm combining change detection and visual question answering for multi-turn, instruction-guided exploration of bi-temporal remote sensing images.

**Capabilities**: Change captioning, binary classification, quantification, localization, open-ended QA, multi-turn dialogue.

**Architecture**:
1. **Bi-temporal Vision Encoder (Bi-VE)**: EVA-ViT-g/14 with last 2 blocks fine-tuned
2. **IDPM with CSRM**: Cross-Semantic Relation Measuring to filter irrelevant variations
3. **Instruction-guided Q-Former**: Aligns visual differences with user instructions
4. **Frozen Vicuna-7B**: Language decoder for response generation

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/hanlinwu/DeltaVLM.git
cd DeltaVLM

# Create conda environment
conda create -n deltavlm python=3.10 -y
conda activate deltavlm

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Download Pretrained Models

```bash
mkdir -p pretrained

# Vicuna-7B-v1.5 (requires HuggingFace login)
huggingface-cli login
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir pretrained/vicuna-7b-v1.5

# BERT Base (for Q-Former tokenizer)
huggingface-cli download bert-base-uncased --local-dir pretrained/bert-base-uncased

# EVA-ViT-G (auto-downloads on first run, or manually)
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth -P pretrained/
```

### 3. Download DeltaVLM Checkpoint

Download the pretrained DeltaVLM checkpoint:

```bash
# Option 1: From HuggingFace (recommended)
huggingface-cli download hanlinwu/DeltaVLM --local-dir pretrained/deltavlm

# Option 2: Direct link
wget https://huggingface.co/hanlinwu/DeltaVLM/resolve/main/checkpoint_best.pth -P pretrained/
```

---

## Minimal Reproducible Example (MRE)

### Single Image Pair Inference

Run inference on a pair of before/after images:

```bash
python scripts/predict.py \
    --image_A path/to/before.png \
    --image_B path/to/after.png \
    --checkpoint pretrained/checkpoint_best.pth \
    --llm_model pretrained/vicuna-7b-v1.5 \
    --prompt "Please briefly describe the changes in these two images."
```

**Expected Output:**
```
Using device: cuda
Loading model from pretrained/checkpoint_best.pth...
Model loaded successfully!
Preprocessing images...
Prompt: Please briefly describe the changes in these two images.
Generating response...

==================================================
Generated Description:
==================================================
A new building has appeared in the lower right area of the image.
==================================================
```

### Python API

```python
import torch
from PIL import Image
from deltavlm.models import Blip2VicunaInstruct
from deltavlm.datasets.processors import BlipImageEvalProcessor

# Load model
model = Blip2VicunaInstruct(
    vit_model="eva_clip_g",
    img_size=224,
    num_query_token=32,
    llm_model="pretrained/vicuna-7b-v1.5",
    qformer_text_input=True,
)

# Load checkpoint
checkpoint = torch.load("pretrained/checkpoint_best.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=False)
model = model.cuda().eval()

# Process images
processor = BlipImageEvalProcessor(image_size=224)
img_A = Image.open("before.png").convert("RGB")
img_B = Image.open("after.png").convert("RGB")
tensor_A, tensor_B = processor(img_A, img_B)

# Generate caption
samples = {
    "image_A": tensor_A.unsqueeze(0).cuda(),
    "image_B": tensor_B.unsqueeze(0).cuda(),
    "prompt": ["Please briefly describe the changes in these two images."],
}

with torch.no_grad():
    caption = model.generate(samples, num_beams=5)[0]

print(caption)
```

---

## Directory Structure

```
DeltaVLM/
├── configs/
│   ├── evaluate.yaml          # Evaluation configuration
│   └── train_stage2.yaml      # Training configuration
├── data/
│   └── download_changechat.py # Dataset download script
├── deltavlm/
│   ├── datasets/
│   │   ├── change_caption.py  # Dataset classes
│   │   └── processors.py      # Image/text processors
│   ├── models/
│   │   ├── blip2_vicuna.py    # Main DeltaVLM model
│   │   ├── blip2_base.py      # Base BLIP-2 class
│   │   ├── eva_vit.py         # Vision encoder
│   │   └── qformer.py         # Q-Former
│   └── utils/
│       └── distributed.py     # Distributed utilities
├── docs/
│   ├── assets/                # Images for README
│   ├── DATA.md               # Dataset documentation
│   └── INSTALL.md            # Installation guide
├── pretrained/                # Model weights (download)
│   ├── vicuna-7b-v1.5/
│   ├── bert-base-uncased/
│   ├── eva_vit_g.pth
│   └── checkpoint_best.pth
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── predict.py            # Inference script
├── requirements.txt
├── setup.py
└── README.md
```

---

## Training

### Prepare Dataset

Download ChangeChat-105k:

```bash
python data/download_changechat.py --output_dir ./data/changechat
```

Expected structure:
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

### Run Training

```bash
# Single GPU
python scripts/train.py --cfg_path configs/train_stage2.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py --cfg_path configs/train_stage2.yaml
```

---

## Evaluation

### Batch Evaluation on Dataset

```bash
python scripts/evaluate.py --cfg_path configs/evaluate.yaml
```

### Metrics

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

## Model Configuration

| Component | Specification |
|-----------|--------------|
| Vision Encoder | EVA-ViT-G (1.0B params) |
| Q-Former | 32 query tokens |
| LLM | Vicuna-7B-v1.5 (frozen) |
| Image Size | 224 × 224 |
| CSRM | Gating + Filtering mechanism |

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use CPU
   python scripts/predict.py --device cpu ...
   ```

2. **Missing Vicuna Weights**
   ```
   Error: Can't find model at pretrained/vicuna-7b-v1.5
   ```
   → Download Vicuna-7B-v1.5 from HuggingFace (see Step 2)

3. **EVA-ViT Download Fails**
   ```bash
   # Manual download
   wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
   mv eva_vit_g.pth pretrained/
   ```

4. **Transformers Version**
   ```
   BLIP-2 Vicuna requires transformers>=4.28
   ```
   → `pip install transformers==4.33.2`

---

## Citation

If you use DeltaVLM in your research, please cite:

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

## License

[BSD 3-Clause License](LICENSE)
