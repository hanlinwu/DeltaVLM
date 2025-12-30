"""
ChangeChat Dataset for Multi-turn Change Captioning and QA

This module provides dataset classes for loading bi-temporal remote sensing
images and their corresponding change captions/dialogues.

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import copy
import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .processors import BlipImageEvalProcessor, Blip2ImageTrainProcessor, BlipCaptionProcessor


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""
    
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root: str = None,
        ann_paths: List[str] = []
    ):
        """
        Args:
            vis_processor: Visual processor for image augmentation
            text_processor: Text processor for caption processing
            vis_root: Root directory of images
            ann_paths: List of annotation file paths
        """
        self.vis_root = vis_root
        self.annotation = []
        
        for ann_path in ann_paths:
            if any(ext in ann_path for ext in ['csv', 'tsv']):
                df = pd.read_csv(ann_path)
                self.annotation.extend(df.to_dict(orient="records"))
            elif 'jsonl' in ann_path:
                with open(ann_path, "r") as f:
                    self.annotation.extend([json.loads(line) for line in f])
            else:
                with open(ann_path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.annotation.extend(loaded)
                    elif isinstance(loaded, dict):
                        self.annotation.extend([
                            {"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "data": v}
                            for k, v in loaded.items()
                        ])

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self._add_instance_ids()

    def __len__(self) -> int:
        return len(self.annotation)

    def collater(self, samples):
        """Custom collate function."""
        samples = [s for s in samples if s is not None]
        if not samples:
            return {}
        
        collated = {}
        for k in samples[0].keys():
            values = [sample[k] for sample in samples]
            if isinstance(values[0], torch.Tensor):
                collated[k] = torch.stack(values, dim=0)
            else:
                collated[k] = values
        return collated

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key: str = "instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ChangeCaptionDataset(BaseDataset):
    """
    Dataset for training change captioning with multi-turn dialogues.
    
    Loads bi-temporal image pairs and their corresponding conversation
    data from the ChangeChat dataset.
    """
    
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root: str = None,
        ann_paths: List[str] = []
    ):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        # Expand multi-turn conversations into individual samples
        self.annotation = self._expand_conversations(ann_paths)

    def _expand_conversations(self, ann_paths: List[str]) -> List[Dict]:
        """Expand multi-turn dialogues into individual Q-A pairs with context."""
        expanded = []
        
        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                data = json.load(f)
            
            for item in data:
                dialog = copy.deepcopy(item)
                all_turns = dialog.get('conversations', [])
                
                # Convert to Q-A pairs
                qa_pairs = [
                    {
                        "question": all_turns[i]['value'],
                        "answer": all_turns[i + 1]['value'],
                    }
                    for i in range(0, len(all_turns), 2)
                    if i + 1 < len(all_turns)
                ]
                
                # Create samples with dialogue history
                for turn_idx in range(len(qa_pairs)):
                    sample = copy.deepcopy(dialog)
                    
                    # Build context from previous turns
                    context = ' '.join([
                        f"q: {t['question']} a: {t['answer']}"
                        for t in qa_pairs[:turn_idx]
                    ]).strip()
                    
                    current = qa_pairs[turn_idx]
                    
                    if context:
                        sample["question"] = f"{context} q: {current['question']}"
                    else:
                        sample["question"] = f"q: {current['question']}"
                    
                    sample["answer"] = current["answer"]
                    sample.pop('conversations', None)
                    
                    expanded.append(sample)
        
        logging.info(f"Expanded {len(expanded)} Q-A samples from dialogues")
        return expanded

    def __getitem__(self, index: int) -> Optional[Dict]:
        ann = self.annotation[index]

        # Load bi-temporal images
        try:
            image_paths = ann.get("image", [])
            if isinstance(image_paths, list) and len(image_paths) >= 2:
                path_A = os.path.join(self.vis_root, image_paths[0])
                path_B = os.path.join(self.vis_root, image_paths[1])
            else:
                return None
            
            image_A = Image.open(path_A).convert("RGB")
            image_B = Image.open(path_B).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to load images at index {index}: {e}")
            return None

        # Apply visual processing
        image_A, image_B = self.vis_processor(image_A, image_B)

        # Extract text
        text_input = ann['question'].replace('<image>', '').strip()
        text_output = ann['answer']

        return {
            "image_A": image_A,
            "image_B": image_B,
            "text_input": self.text_processor(text_input),
            "text_output": self.text_processor(text_output),
            "image_id": ann.get("id", str(index)),
        }


class ChangeCaptionEvalDataset(BaseDataset):
    """
    Evaluation dataset for change captioning.
    
    Returns bi-temporal images with evaluation prompts for inference.
    """
    
    def __init__(
        self,
        vis_processor=None,
        text_processor=None,
        vis_root: str = None,
        ann_paths: List[str] = [],
        prompt: str = "Please briefly describe the changes in these two images."
    ):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.prompt = prompt

    def __getitem__(self, index: int) -> Dict:
        ann = self.annotation[index]

        # Load images
        path_A = os.path.join(self.vis_root, ann["image_A"])
        path_B = os.path.join(self.vis_root, ann["image_B"])

        image_A = Image.open(path_A).convert("RGB")
        image_B = Image.open(path_B).convert("RGB")

        # Apply processing
        image_A, image_B = self.vis_processor(image_A, image_B)

        # Extract image ID from filename
        img_id = ann["image_A"].split("/")[-1].replace(".png", "").split("_")[-1]

        return {
            "image_A": image_A,
            "image_B": image_B,
            "text_input": self.text_processor(self.prompt),
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class ChangeCaptionBuilder:
    """
    Builder class for creating ChangeCaptionDatasets.
    
    Handles processor initialization and dataset configuration.
    """
    
    train_dataset_cls = ChangeCaptionDataset
    eval_dataset_cls = ChangeCaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/changechat.yaml",
    }

    def __init__(self, cfg=None):
        self.config = cfg
        self.vis_processors = {"train": None, "eval": None}
        self.text_processors = {"train": None, "eval": None}

    def build_processors(self, cfg):
        """Build visual and text processors from config."""
        vis_cfg = cfg.get("vis_processor", {})
        txt_cfg = cfg.get("text_processor", {})

        # Visual processors
        if vis_cfg.get("train"):
            self.vis_processors["train"] = Blip2ImageTrainProcessor.from_config(
                vis_cfg["train"]
            )
        if vis_cfg.get("eval"):
            self.vis_processors["eval"] = BlipImageEvalProcessor.from_config(
                vis_cfg["eval"]
            )

        # Text processors
        if txt_cfg.get("train"):
            self.text_processors["train"] = BlipCaptionProcessor.from_config(
                txt_cfg["train"]
            )
        if txt_cfg.get("eval"):
            self.text_processors["eval"] = BlipCaptionProcessor.from_config(
                txt_cfg["eval"]
            )

    def build_datasets(self, cfg) -> Dict[str, Dataset]:
        """Build train/val/test datasets from config."""
        self.build_processors(cfg)

        build_info = cfg.build_info
        ann_info = build_info.annotations
        vis_path = build_info.images.storage

        datasets = {}
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = (split == "train")
            vis_processor = self.vis_processors["train" if is_train else "eval"]
            text_processor = self.text_processors["train" if is_train else "eval"]

            ann_paths = ann_info[split].storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                vis_root=vis_path,
                ann_paths=ann_paths,
            )

        return datasets


