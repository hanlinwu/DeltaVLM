import re
from randaugment import RandomAugment
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from omegaconf import OmegaConf
import random
from torchvision.transforms import functional as F



class PairTransforms:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2):
        # 生成相同的变换参数
        i, j, h, w = transforms.RandomResizedCrop.get_params(img1, scale=(0.95, 1.0), ratio=(1. / 1., 1. / 1.))
        angle = random.uniform(-15, 15)


        # 应用相同的变换
        img1 = F.resized_crop(img1, i, j, h, w, self.size)
        img2 = F.resized_crop(img2, i, j, h, w, self.size)

        img1 = F.rotate(img1, angle)
        img2 = F.rotate(img2, angle)


        return img1, img2

class BaseProcessor:
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        # self.pair_transforms = PairTransforms(size=(image_size, image_size))
        self.transform = transforms.Compose(
            [
               
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item1, itme2):
        # img_A, img_B =self.pair_transforms(item1, itme2)
        img_A = self.transform(item1)
        img_B = self.transform(itme2)
        return img_A, img_B

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)

class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption

class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.pair_transforms = PairTransforms(size=(image_size, image_size))
        self.transform = transforms.Compose(
            [
               
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item1, itme2):
        # img_A, img_B =self.pair_transforms(item1, itme2)
        img_A = self.transform(item1)
        img_B = self.transform(itme2)
        return img_A, img_B

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )