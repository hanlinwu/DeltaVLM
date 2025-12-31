# Pretrained Models

## Required (User Downloads)

| Model | Description | Download |
|-------|-------------|----------|
| Vicuna-7B-v1.5 | LLM backbone (frozen) | `huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ./pretrained/vicuna-7b-v1.5` |
| BERT-base-uncased | Q-Former init | `huggingface-cli download bert-base-uncased --local-dir ./pretrained/bert-base-uncased` |
| EVA-ViT-G | Vision encoder | Auto-downloaded during training, or: `wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/EVA/eva_vit_g.pth` |

## DeltaVLM Checkpoints (Not Included)

Trained weights should be placed here after training:
- `deltavlm_stage2.pth` - Instruction-tuned model

## Expected Structure

```
pretrained/
├── vicuna-7b-v1.5/
├── bert-base-uncased/
└── eva_vit_g.pth
```
