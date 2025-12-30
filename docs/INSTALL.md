# Installation Guide

This document provides detailed installation instructions for DeltaVLM.

## Table of Contents

- [Requirements](#requirements)
- [Quick Install](#quick-install)
- [Step-by-Step Installation](#step-by-step-installation)
- [Pretrained Models](#pretrained-models)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with at least 24GB VRAM (A100 recommended)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ for models and datasets

### Software
- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.3

---

## Quick Install

```bash
# Clone repository
git clone https://github.com/hanlinwu/DeltaVLM.git
cd DeltaVLM

# Create environment
conda create -n deltavlm python=3.10 -y
conda activate deltavlm

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_pretrained.py
```

---

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/hanlinwu/DeltaVLM.git
cd DeltaVLM
```

### 2. Create Conda Environment

```bash
# Create new environment
conda create -n deltavlm python=3.10 -y
conda activate deltavlm
```

### 3. Install PyTorch

Choose the appropriate command for your CUDA version:

```bash
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install DeltaVLM Package (Optional)

For development:
```bash
pip install -e .
```

---

## Pretrained Models

### Required Models

| Model | Size | Description | Download |
|-------|------|-------------|----------|
| EVA-ViT-G | 1.0B | Vision encoder | [HuggingFace](https://huggingface.co/BAAI/EVA) |
| Vicuna-7B-v1.5 | 7B | Language model | [HuggingFace](https://huggingface.co/lmsys/vicuna-7b-v1.5) |
| BERT-base-uncased | 110M | Tokenizer | [HuggingFace](https://huggingface.co/bert-base-uncased) |

### Download Models

```bash
# Create pretrained directory
mkdir -p pretrained

# Download Vicuna-7B (requires HuggingFace login)
huggingface-cli login
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir pretrained/vicuna-7b-v1.5

# Download BERT
huggingface-cli download bert-base-uncased --local-dir pretrained/bert-base-uncased

# EVA-ViT-G will be downloaded automatically during training
```

### Directory Structure

```
pretrained/
├── vicuna-7b-v1.5/
│   ├── config.json
│   ├── tokenizer.model
│   └── pytorch_model*.bin
├── bert-base-uncased/
│   ├── config.json
│   ├── tokenizer.json
│   └── pytorch_model.bin
└── eva_vit_g.pth  # (auto-downloaded)
```

### DeltaVLM Checkpoints

| Checkpoint | Description | Download |
|------------|-------------|----------|
| Stage 1 | Q-Former pre-training | [Link](#) |
| Stage 2 | Instruction tuning | [Link](#) |
| Mask Branch | Mask prediction | [Link](#) |

---

## Troubleshooting

### CUDA Out of Memory

1. **Reduce batch size**:
   ```yaml
   batch_size: 4  # or lower
   ```

2. **Use gradient checkpointing**:
   ```yaml
   use_grad_checkpoint: true
   ```

3. **Use mixed precision**:
   ```yaml
   vit_precision: "fp16"
   amp: true
   ```

### Module Not Found

Ensure DeltaVLM is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# or
pip install -e .
```

### Slow Data Loading

Increase number of workers:
```yaml
num_workers: 8
```

### Flash Attention (Optional)

For faster attention computation:
```bash
pip install flash-attn --no-build-isolation
```

### SpaCy Model

For lemmatization during evaluation:
```bash
python -m spacy download en_core_web_sm
```

---

## Verification

Run a quick test:

```bash
# Test imports
python -c "from deltavlm.models import Blip2VicunaInstruct; print('Import OK')"

# Test dataset loading
python -c "from deltavlm.datasets import ChangeMaskDataset; print('Dataset OK')"
```

---

## Next Steps

1. [Prepare datasets](DATA.md)
2. [Start training](../README.md#-training)
3. [Run inference](../README.md#-inference)


