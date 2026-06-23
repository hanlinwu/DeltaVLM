# DeltaVLM

This repository is the official implementation of [**DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception**](https://arxiv.org/abs/2507.22346) (Deng, Zhou & Wu, *Remote Sensing* 2026 · arXiv:2507.22346).

The paper introduces **remote-sensing image change analysis (RSICA)** — a new paradigm that combines change detection and visual question answering, enabling multi-turn, instruction-guided exploration of bi-temporal RS imagery — and ships two artifacts:

- **DeltaVLM** — an end-to-end architecture with three components: (1) a fine-tuned bi-temporal vision encoder for capturing temporal differences, (2) a visual difference perception module with a cross-semantic relation measuring (CSRM) mechanism, and (3) an instruction-guided Q-former that extracts query-relevant difference information and aligns it with textual instructions. The underlying LLM is kept **frozen**; only the vision and alignment modules are trained.
- **ChangeChat-105k** — a large-scale instruction-following dataset (built via a hybrid rule-based + GPT-assisted process) covering **six interaction types**: change captioning, classification, quantification, localization, open-ended QA, and multi-turn dialogues. Built on the bi-temporal imagery of [LEVIR-CC](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset).

This codebase is a flattened fork of [Salesforce/LAVIS](https://github.com/salesforce/LAVIS) — the registry / plug-in layers have been removed, and models, datasets, tasks, runners, and optimizers are wired together by explicit `arch` dispatch in [config.py](config.py) and [task.py](task.py).

---

## ChangeChat-105k Dataset

### What's in it

ChangeChat-105k is the instruction-following dataset introduced in the DeltaVLM paper. Each sample pairs a bi-temporal image pair `(image_A, image_B)` (pre / post views of the same tile) with a question and answer covering one of the **six interaction types** below — one test split per type:

| Interaction type | Example question | Test file |
| --- | --- | --- |
| Change captioning (5 reference captions / pair) | "Describe the change between these two scenes." | `changechat_105k_test.json` |
| Change classification (binary) | *"Have these two images changed? Answer yes or no."* | `changechat_105k_test_binary.json` |
| Change quantification | *"How many new roads / buildings have been constructed?"* | `changechat_105k_test_count.json` |
| Change localization (3×3 grid) | *"Indicate the locations where changes occurred using a 3×3 grid…"* | `changechat_105k_test_loc.json` |
| Open-ended QA | Free-form questions about the change | `changechat_105k_test_open.json` |
| Multi-turn dialogues | 6-turn dialogues about the change | `changechat_105k_test_dialog.json` |

Each instruction record (everything except the captioning split — see note below) looks like:

```json
{
  "id": 0,
  "image": ["train/A/train_000001.png", "train/B/train_000001.png"],
  "changeflag": 0,
  "conversations": [
    {"from": "human", "value": "<image> <image> Please briefly describe the changes in these two images."},
    {"from": "gpt",   "value": "There is no difference."}
  ]
}
```

`image` paths are **relative** to `dataset/images/`. `changeflag = 0` means the pair contains no change; non-zero values flag changed pairs. The captioning test split (`changechat_105k_test.json`) uses a different schema — `{image_A, image_B, captions: [5 refs], changeflag}` — to match the LEVIR-CC-style multi-reference caption evaluation used by [benchmark.py](benchmark.py).

### Splits and statistics

| Split | File | Samples |
| --- | --- | --- |
| Train (mixed tasks) | `changechat_105k_train.json` | 87,935 |
| Train (localization-augmented) | `changechat_105k_train_loc.json` | 87,935 |
| Test — captioning (5 refs / pair) | `changechat_105k_test.json` | 1,929 |
| Test — classification (binary) | `changechat_105k_test_binary.json` | 1,929 |
| Test — quantification | `changechat_105k_test_count.json` | 1,929 |
| Test — localization | `changechat_105k_test_loc.json` | 1,929 |
| Test — multi-turn dialogues (6-turn) | `changechat_105k_test_dialog.json` | 1,929 |
| Test — open QA | `changechat_105k_test_open.json` | 7,527 |

≈ **105k** instruction samples in total, built over the **10,077 bitemporal image pairs** in LEVIR-CC (train 7,590 / val 1,438 / test 2,135).

### Getting the data

ChangeChat-105k is split into two pieces with different licenses and hosting:

| Component | Source | License |
| --- | --- | --- |
| Annotations (8 JSON files) | Hugging Face: [`hlwu/changechat-105k`](https://huggingface.co/datasets/hlwu/changechat-105k) | CC-BY-4.0 |
| Bitemporal images | [LEVIR-CC](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) (download separately) | LEVIR-CC license |

The `dataset/` directory is `.gitignore`d, so cloning this repo gives you **only the code** — you must fetch both the annotations and the images yourself before training.

#### 1. Annotations

Pull from Hugging Face (~83 MB total):

```bash
# Whole repo
huggingface-cli download hlwu/changechat-105k --repo-type dataset --local-dir dataset/annotations

# Or just one file
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('hlwu/changechat-105k', 'changechat_105k_train.json', \
repo_type='dataset', local_dir='dataset/annotations')"
```

If you're behind the GFW, set `HF_ENDPOINT=https://hf-mirror.com` before either command (the project's [train.py](train.py) already does this).

#### 2. Images (LEVIR-CC)

The image pairs are **not** redistributed here. They are exactly the bitemporal tiles from the **LEVIR-CC** dataset and must be downloaded from the original authors:

1. Go to <https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset>.
2. Follow its README to obtain the image archive (the repo provides Google Drive and Baidu Pan links).
3. Unpack so the layout matches the paths referenced in our annotations:

   ```text
   dataset/
   ├── annotations/                    # ChangeChat-105k JSONs (from HF)
   │   ├── changechat_105k_train.json
   │   ├── changechat_105k_train_loc.json
   │   ├── changechat_105k_test.json
   │   ├── changechat_105k_test_dialog.json
   │   ├── changechat_105k_test_binary.json
   │   ├── changechat_105k_test_count.json
   │   ├── changechat_105k_test_loc.json
   │   └── changechat_105k_test_open.json
   └── images/                         # LEVIR-CC — download separately
       ├── train/
       │   ├── A/   train_000001.png … train_007590.png
       │   └── B/   train_000001.png … train_007590.png
       ├── val/
       │   ├── A/   val_000001.png   …
       │   └── B/
       └── test/
           ├── A/   test_000001.png  …
           └── B/
   ```

   `A/` is the pre-event view, `B/` is the post-event view of the same tile.

4. (Optional) If you want to compare against caption-only baselines, drop the original `LevirCCcaptions.json` next to `images/`.
5. (Optional) For mask-supervised training with [model/blip2_vicua_mask.py](model/blip2_vicua_mask.py), also download the LEVIR-CC change-detection masks from the same upstream repo.

If you use the images, please cite LEVIR-CC:

```bibtex
@article{liu2022remote,
  title   = {Remote Sensing Image Change Captioning With Dual-Branch Transformers:
             A New Method and a Large Scale Dataset},
  author  = {Liu, Chenyang and Zhao, Rui and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2022}, volume = {60}, pages = {1--20},
  doi     = {10.1109/TGRS.2022.3218921}
}
```

---

## Pretrained checkpoint

The released DeltaVLM checkpoint is hosted on Hugging Face:

- Model repo: [`hlwu/DeltaVLM`](https://huggingface.co/hlwu/DeltaVLM)
- Main weight file: `checkpoint_best.pth`

Download it into the same location expected by [configs/evaluate.yaml](configs/evaluate.yaml):

```bash
mkdir -p output/BLIP2/train_vicua/20250320054
huggingface-cli download hlwu/DeltaVLM checkpoint_best.pth \
  --local-dir output/BLIP2/train_vicua/20250320054
```

After download, the layout should be:

```text
output/
└── BLIP2/
    └── train_vicua/
        └── 20250320054/
            └── checkpoint_best.pth
```

The default evaluation config already points to this file through `model.pretrained:` in [configs/evaluate.yaml](configs/evaluate.yaml), so you do not need to edit the YAML if you keep the path above.

### Base LLM weights

This checkpoint does **not** contain the frozen Vicuna base model. The current code loads a Vicuna-7B-v1.5-compatible directory from `../vicuna-7b-v1.5`.

If you already have a compatible local model, make sure this path resolves correctly. One working setup is:

```bash
ln -s /path/to/your/vicuna-or-llava-v1.5-7b ../vicuna-7b-v1.5
```

## Environment

```bash
conda create -n deltavlm python=3.9 -y
conda activate deltavlm
pip install -r requirements.txt
```

> `transformers==4.33.2` is pinned — the LLaMA modeling code in [model/modeling_llama.py](model/modeling_llama.py) is from that era and newer releases will break it.

For HuggingFace access from inside China, [train.py](train.py) sets `HF_ENDPOINT=https://hf-mirror.com`; remove it if you don't need the mirror.

## Training

```bash
# Stage 1 — Q-Former pretraining
python train.py --cfg_path configs/train_stage1.yaml

# Stage 2 — Q-Former + LM finetuning
python train.py --cfg_path configs/train_stage2.yaml

# InstructBLIP / Vicuna-7B (the main config used in the paper)
python train.py --cfg_path configs/train_vicua.yaml
```

Background runs:

```bash
nohup python train.py --cfg_path configs/train_vicua.yaml > nohup.out 2>&1 &
```

Checkpoints land under `output/<run.output_dir>/<timestamp>/checkpoint_*.pth`.

## Evaluation

```bash
python evaluate.py --cfg_path configs/evaluate.yaml
```

The checkpoint to evaluate is set via `model.pretrained:` in [configs/evaluate.yaml](configs/evaluate.yaml). Caption metrics (BLEU, METEOR, ROUGE-L, CIDEr) are vendored under [eval_func/](eval_func/) and scored against `dataset/annotations/changechat_105k_{val,test}.json`.

## Quick smoke test

For a fast end-to-end inference check on all six sub-tasks, run:

```bash
conda activate deltavlm
python infer_subtasks.py --cfg_path configs/evaluate.yaml --n_samples 2
```

The script will:

- load `checkpoint_best.pth` from `output/BLIP2/train_vicua/20250320054/`
- read the six test annotation files under `dataset/annotations/`
- run caption / binary / count / loc / open / dialog inference
- save predictions to `output/subtask_smoke/`

To run the full test splits instead of a 2-sample smoke test:

```bash
python infer_subtasks.py --cfg_path configs/evaluate.yaml --n_samples 99999
```

---

## Repository layout

```text
DeltaVLM/
├── train.py / evaluate.py          # entry points
├── config.py / task.py / runner.py # config, task, training loop
├── dataset.py / dataset_mask.py    # ChangeChat-105k builders
├── processor.py                    # vis / text processors
├── model/                          # Blip2Qformer, Blip2OPT, Blip2VicunaInstruct, …
├── eval_func/                      # vendored COCO-caption metrics
├── configs/                        # YAML configs for each arch / stage
└── dataset/                        # gitignored — populate yourself (see § Getting the data)
    ├── annotations/                # ChangeChat-105k JSONs (HF: hlwu/changechat-105k)
    └── images/                     # LEVIR-CC images
```

See [CLAUDE.md](CLAUDE.md) for an architecture-level walkthrough (config layering, model dispatch, dataset pipeline, task layer, distributed setup).

## Citation

If you use DeltaVLM or ChangeChat-105k in your research, please cite:

```bibtex
@article{deng2026deltavlm,
  title   = {DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-Guided Difference Perception},
  author  = {Deng, Pei and Zhou, Wenqian and Wu, Hanlin},
  journal = {Remote Sensing},
  volume  = {18},
  number  = {4},
  pages   = {541},
  year    = {2026},
  publisher = {MDPI}
}
```

Please also cite **LEVIR-CC** (the source of the bitemporal images) — the bibtex is given above under [§ Images](#2-images-levir-cc).

## Acknowledgements

- [Salesforce/LAVIS](https://github.com/salesforce/LAVIS) — BLIP-2 / InstructBLIP backbone.
- [Chen-Yang-Liu/LEVIR-CC-Dataset](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) — bitemporal remote-sensing images used by ChangeChat-105k.
