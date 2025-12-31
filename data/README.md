# Data

This directory contains download scripts for datasets. Actual data files are not included in the repository.

## Download Scripts

- `download_changechat.py` - Download ChangeChat-105k dataset
- `download_levir_mci.py` - Download LEVIR-MCI dataset

## Usage

```bash
python download_changechat.py --output_dir ./changechat
```

After downloading, the structure should be:

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
└── levir_mci/
    └── ...
```

See [docs/DATA.md](../docs/DATA.md) for detailed annotation format.
