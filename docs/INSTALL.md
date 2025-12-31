# Installation

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA >= 11.3
- GPU with 24GB+ VRAM (A100 recommended)

## Setup

```bash
git clone https://github.com/hanlinwu/DeltaVLM.git
cd DeltaVLM

conda create -n deltavlm python=3.10 -y
conda activate deltavlm

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Pretrained Models

Download and place in `pretrained/`:

```bash
mkdir -p pretrained

# Vicuna-7B (requires HuggingFace login)
huggingface-cli login
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir pretrained/vicuna-7b-v1.5

# BERT
huggingface-cli download bert-base-uncased --local-dir pretrained/bert-base-uncased

# EVA-ViT-G (optional, auto-downloads during training)
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/EVA/eva_vit_g.pth -P pretrained/
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import deltavlm; print('DeltaVLM installed successfully')"
```
