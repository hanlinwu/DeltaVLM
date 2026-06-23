---
license: other
language:
  - en
tags:
  - remote-sensing
  - vision-language
  - image-change-captioning
  - pytorch
pipeline_tag: image-text-to-text
---

# DeltaVLM

This repository hosts the pretrained DeltaVLM checkpoint from the paper [DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception](https://arxiv.org/abs/2507.22346).

## Contents

- `checkpoint_best.pth`: pretrained DeltaVLM checkpoint
- `configs/evaluate.yaml`: example evaluation config used in the project codebase

## Important note

This checkpoint does **not** include the frozen base LLM weights. The original project loads a Vicuna-7B-v1.5-compatible model separately and then applies the DeltaVLM checkpoint on top.

In the local reproduction environment, the Vicuna path was substituted with a compatible `llavav1.5-7b` directory for loading.

## Code and dataset

- Code: https://github.com/hanlinwu/DeltaVLM
- Dataset annotations: https://huggingface.co/datasets/hlwu/changechat-105k
- Image source: https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset

## Example

After cloning the code repo and preparing the required base model plus dataset files, evaluation can be run with:

```bash
python infer_subtasks.py --cfg_path configs/evaluate.yaml --n_samples 2
```

## License

Please follow the license terms of this project repository, the referenced dataset, and the required upstream base model.
