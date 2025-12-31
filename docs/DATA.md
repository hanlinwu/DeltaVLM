# Data Preparation

## ChangeChat-105k Dataset

A large-scale instruction-following dataset with **105,107 instruction-response pairs** derived from LEVIR-CC and LEVIR-MCI through a hybrid rule-based and GPT-assisted pipeline.

### Instruction Types

| Type | Training | Test | Method |
|------|----------|------|--------|
| Change Captioning | 34,075 | 1,929 | Rule-based |
| Binary Change Classification | 6,815 | 1,929 | Rule-based |
| Category-specific Change Quantification | 6,815 | 1,929 | Rule-based |
| Change Localization | 6,815 | 1,929 | Rule-based |
| Open-ended QA | 26,600 | 7,527 | GPT-assisted |
| Multi-turn Conversation | 6,815 | 1,929 | Rule-based |
| **Total** | **87,935** | **17,172** | -- |

### Download

```bash
python data/download_changechat.py --output_dir ./data/changechat
```

### Expected Structure

```
data/changechat/
├── images/
│   ├── train/
│   │   ├── pair_00001_A.png
│   │   ├── pair_00001_B.png
│   │   └── ...
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

### Annotation Format

```json
{
  "id": "unique_sample_id",
  "image": ["path/to/before.png", "path/to/after.png"],
  "conversations": [
    {"from": "human", "value": "<image>Describe the changes..."},
    {"from": "gpt", "value": "A new building has been constructed..."}
  ]
}
```

---

## LEVIR-MCI Dataset

Source dataset providing pixel-level change masks. Used for deriving localization and quantification annotations.

- Download: https://github.com/Chen-Yang-Liu/LEVIR-MCI
- Resolution: 256×256 pixels at 0.5m/pixel

---

## Custom Dataset

To use your own bi-temporal images:

1. Organize images as `pair_XXXXX_A.png` (before) and `pair_XXXXX_B.png` (after)
2. Create annotation JSON following the format above
3. Update `build_info.annotations` and `build_info.images` paths in config files
