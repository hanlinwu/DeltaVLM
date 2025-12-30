# Pretrained Models

This directory should contain the pretrained model weights for DeltaVLM.

## Required Models

### 1. Vicuna-7B-v1.5

Large Language Model backbone.

```bash
# Download using HuggingFace CLI
huggingface-cli login
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ./pretrained/vicuna-7b-v1.5
```

Or manually:
- URL: https://huggingface.co/lmsys/vicuna-7b-v1.5

### 2. BERT-base-uncased

Used for Q-Former initialization.

```bash
huggingface-cli download bert-base-uncased --local-dir ./pretrained/bert-base-uncased
```

### 3. EVA-ViT-G

Vision encoder. Will be downloaded automatically during training, or:

```bash
# Manual download
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/EVA/eva_vit_g.pth -P ./pretrained/
```

## DeltaVLM Checkpoints

After training, place checkpoints here:

| File | Description |
|------|-------------|
| `deltavlm_stage1.pth` | Stage 1 pre-trained weights |
| `deltavlm_stage2.pth` | Stage 2 fine-tuned weights |
| `mask_branch_best.pth` | Mask branch weights |

## Directory Structure

```
pretrained/
├── README.md              # This file
├── vicuna-7b-v1.5/        # Vicuna LLM
│   ├── config.json
│   ├── tokenizer.model
│   └── *.bin
├── bert-base-uncased/     # BERT tokenizer
│   ├── config.json
│   └── *.bin
├── eva_vit_g.pth          # EVA-ViT vision encoder
├── deltavlm_stage2.pth    # Trained DeltaVLM weights
└── mask_branch_best.pth   # Mask branch weights
```

## Notes

- Total disk space required: ~30GB
- Vicuna-7B requires agreement to Meta's LLaMA license
- EVA-ViT-G is ~1GB


