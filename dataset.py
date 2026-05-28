import logging
import os
import shutil
import warnings
import copy 
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import *
import torch.distributed as dist
# from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
# from lavis.common.registry import registry
# from lavis.datasets.data_utils import extract_archive
# from lavis.processors.base_processor import BaseProcessor
# from omegaconf import OmegaConf
from torchvision.datasets.utils import download_url
from omegaconf import OmegaConf
from processor import *
import json
from typing import Iterable
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from collections import OrderedDict
from PIL import Image


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    return next(iter(cfg.values()))


class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None
    """
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset
    """

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

        # additional processors, each specified by a name in string.
        self.kw_processors = {}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)
        
        kw_proc_cfg = self.config.get("kw_processor")
        if kw_proc_cfg is not None:
            for name, cfg in kw_proc_cfg.items():
                self.kw_processors[name] = self._build_proc_from_cfg(cfg)
        
    @staticmethod
    def _build_proc_from_cfg(cfg):
        processor_class = {
                           'blip_image_eval': BlipImageEvalProcessor,
                           'blip_caption': BlipCaptionProcessor,
                           'blip2_image_train': Blip2ImageTrainProcessor
        }
        return (
            processor_class[cfg.name].from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        self._download_ann()
        self._download_vis()

    def _download_ann(self):
        """
        Download annotation files if necessary.
        All the vision-language datasets should have annotations of unified format.

        storage_path can be:
          (1) relative/absolute: will be prefixed with env.cache_root to make full path if relative.
          (2) basename/dirname: will be suffixed with base name of URL if dirname is provided.

        Local annotation paths should be relative.
        """
        anns = self.config.build_info.annotations

        splits = anns.keys()

        cache_root = get_cache_root()

        for split in splits:
            info = anns[split]

            urls, storage_paths = info.get("url", None), info.storage

            if isinstance(storage_paths, str):
                storage_paths = [storage_paths]

            # No url given (e.g. ChangeChat-105k, already on disk): nothing to download.
            if urls is None:
                continue

            if isinstance(urls, str):
                urls = [urls]

            assert len(urls) == len(storage_paths)

            for url_or_filename, storage_path in zip(urls, storage_paths):
                # if storage_path is relative, make it full by prefixing with cache_root.
                if not os.path.isabs(storage_path):
                    storage_path = os.path.join(cache_root, storage_path)

                dirname = os.path.dirname(storage_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                if os.path.isfile(url_or_filename):
                    src, dst = url_or_filename, storage_path
                    if not os.path.exists(dst):
                        shutil.copyfile(src=src, dst=dst)
                    else:
                        logging.info("Using existing file {}.".format(dst))
                else:
                    if os.path.isdir(storage_path):
                        # if only dirname is provided, suffix with basename of URL.
                        raise ValueError(
                            "Expecting storage_path to be a file path, got directory {}".format(
                                storage_path
                            )
                        )
                    else:
                        filename = os.path.basename(storage_path)

                    download_url(url=url_or_filename, root=dirname, filename=filename)

    def _download_vis(self):

        storage_path = self.config.build_info.get(self.data_type).storage
        storage_path = get_cache_path(storage_path)

        if not os.path.exists(storage_path):
            warnings.warn(
                f"""
                The specified path {storage_path} for visual inputs does not exist.
                Please provide a correct path to the visual inputs or
                refer to datasets/download_scripts/README.md for downloading instructions.
                """
            )

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
            )

        return datasets


class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
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
                       self.annotation.extend([{"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "data": v} for k, v in loaded.items()])


        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return {}
        collated_dict = {}
        keys = samples[0].keys() # Use the keys of the first sample as a reference
        for k in keys:
            values = [sample[k] for sample in samples]
            # If the value type for the key is torch.Tensor, stack them else return list
            collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
        return collated_dict
        # return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )
    

class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])

        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }


class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # image_path = os.path.join(self.vis_root, ann["image"])
        ##
        #增加
        image_path_A = os.path.join(self.vis_root, ann["image_A"])
        image_path_B = os.path.join(self.vis_root, ann["image_B"])
        ######

        # image = Image.open(image_path).convert("RGB")
        ##
        image_A = Image.open(image_path_A).convert("RGB")
        image_B = Image.open(image_path_B).convert("RGB")

        # image = self.vis_processor(image)
        # print("#"*30)
        # print(self.vis_processor)
        # image_A = self.vis_processor(image_A)
        # image_B = self.vis_processor(image_B)
        image_A, image_B = self.vis_processor(image_A, image_B)
        # caption = self.text_processor(ann["captions"])

        img_id = ann["image_A"].split("/")[-1].strip(".png").split("_")[-1]
        text_input = "Please briefly describe the changes in these two images."
        # text_input = ann['question'].replace('<image>', '').strip()
        # Please briefly describe the changes in these two images.
        # Please judge whether these two images have changed. Please answer yes or no.
        # Please determine how many roads and buildings have changed?
        # Please indicate the locations where changes have occurred in the buildings and roads, using a 3x3 grid (namely, 'top', 'center', and 'bottom' vertically and 'left', 'center', 'right' horizontally).

        ## test open questions
        # image_path_A = os.path.join(self.vis_root, ann["image"][0])
        # image_path_B = os.path.join(self.vis_root, ann["image"][1])
        # image_A = Image.open(image_path_A).convert("RGB")
        # image_B = Image.open(image_path_B).convert("RGB")
        # image_A, image_B = self.vis_processor(image_A, image_B)
        # img_id = ann['id']
        # text_input = ann['conversations'][0]['value'].replace('<image>', '').strip()

        return {
            "image_A": image_A,
            "image_B":image_B,
            "text_input": self.text_processor(text_input),
            "image_id": img_id,
            "instance_id": ann["instance_id"],
            # "changeflag": ann["changeflag"],
            
        }


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # self.img_ids = {}
        # n = 0
        # for ann in self.annotation:
        #     img_id = ann["image_id"]
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1
        ## 重写
        self.annotation = []
        for ann_path in ann_paths:
            data = json.load(open(ann_path, "r"))

            for d in data:
                dialog = copy.deepcopy(d)
                all_turns = dialog['conversations']
                all_turns = [
                            {
                                "answer": all_turns[d+1]['value'],
                                "question": all_turns[d]['value'],
                            }
                            for d in range(0, len(all_turns),2)
                        ]
                for i in range(len(all_turns)):
                    dialog_instance = copy.deepcopy(dialog)
                    dialogue_context = ' '.join([f" q: {t['question']} a: {t['answer']}" for t in all_turns[:i]]).strip()
                    last_turn = all_turns[i]

                    question = last_turn["question"]
                    answer = last_turn["answer"]
                    if dialogue_context=='':
                        dialog_instance["question"]= 'q: '+ question
                    else:
                        dialog_instance["question"] = dialogue_context + '  q: ' + question
                    dialog_instance["answer"]=answer
                    dialog_instance['conversations'] = ''

                    self.annotation.append(dialog_instance)

    # 重写
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        # image_path = os.path.join(self.vis_root, ann["image"])

        ##图像处理部分
        image_path_A = os.path.join(self.vis_root, ann["image"][0])
        image_path_B = os.path.join(self.vis_root, ann["image"][1])
        
        try:
            image_A = Image.open(image_path_A).convert("RGB")
            image_B = Image.open(image_path_B).convert("RGB")
        except:
            return None # image does not exist
        # print(self.vis_processor)
        image_A, image_B = self.vis_processor(image_A, image_B)
        # caption = self.text_processor(ann["captions"])
        ##文本部分
        text_input = ann['question'].replace('<image>', '').strip()
        text_output = ann['answer']
        return {
            "image_A": image_A,
            "image_B":image_B,
            # "text_input": caption,
            # "changeflag": ann["changeflag"],
            "text_input": self.text_processor(text_input),
            "text_output": self.text_processor(text_output),
            "image_id": ann["id"]
        }


class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "defaults_cap.yaml",
    }


## llava150k_dialogue_instruct
class LLaVA150kInstructDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor,ann_paths, vis_root):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, ann_paths=ann_paths, vis_root=vis_root)
        self.inner_dataset = self.annotation
        self.location = vis_root

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, index):
        example = self.inner_dataset[index]
        text_input = example['conversations'][0]['value'].replace('<image>', '').strip()
        text_output = example['conversations'][1]['value']
        image_id = example['image']
        image_path = os.path.join(self.location, image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        return {
            "image": image,
            "instance_id":image_id,
            "text_input": self.text_processor(text_input),
            "text_output": self.text_processor(text_output),
            "image_path": image_path
        }


class LLaVA150kDialInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = LLaVA150kInstructDataset
    eval_dataset_cls = LLaVA150kInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "defaults_cap.yaml",
    }

