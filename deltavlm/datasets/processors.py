"""
Image and Text Processors for DeltaVLM

Provides preprocessing for bi-temporal remote sensing images
and text captions with synchronized augmentation support.

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import (
    resize, center_crop, to_tensor, normalize,
    hflip, vflip, rotate, affine
)


# ImageNet normalization
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


class BlipImageEvalProcessor:
    """
    Image processor for evaluation.
    
    Applies deterministic transforms: resize and normalize.
    Handles bi-temporal image pairs with identical processing.
    """
    
    def __init__(self, image_size: int = 224, mean=None, std=None):
        self.image_size = image_size
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def __call__(self, image_A: Image.Image, image_B: Image.Image = None):
        """
        Process image pair.
        
        Args:
            image_A: Before image
            image_B: After image (optional)
        
        Returns:
            Processed tensor(s)
        """
        tensor_A = self.transform(image_A)
        
        if image_B is not None:
            tensor_B = self.transform(image_B)
            return tensor_A, tensor_B
        
        return tensor_A
    
    @classmethod
    def from_config(cls, cfg):
        image_size = cfg.get("image_size", 224)
        return cls(image_size=image_size)


class Blip2ImageTrainProcessor:
    """
    Image processor for training with data augmentation.
    
    Applies synchronized random transforms to bi-temporal pairs:
    - Random crop
    - Random horizontal/vertical flip
    - Color jittering (different for each image)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean=None,
        std=None,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
    ):
        self.image_size = image_size
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.normalize = transforms.Normalize(self.mean, self.std)
    
    def __call__(self, image_A: Image.Image, image_B: Image.Image = None):
        """
        Process image pair with synchronized augmentation.
        """
        # Random parameters (same for both images)
        scale = random.uniform(self.min_scale, self.max_scale)
        do_hflip = random.random() < 0.5
        
        # Process image A
        tensor_A = self._process_single(image_A, scale, do_hflip)
        
        if image_B is not None:
            tensor_B = self._process_single(image_B, scale, do_hflip)
            return tensor_A, tensor_B
        
        return tensor_A
    
    def _process_single(
        self, image: Image.Image, scale: float, do_hflip: bool
    ) -> torch.Tensor:
        # Resize with scale
        w, h = image.size
        new_size = int(self.image_size / scale)
        image = resize(image, new_size, interpolation=transforms.InterpolationMode.BICUBIC)
        
        # Center crop to target size
        image = center_crop(image, self.image_size)
        
        # Horizontal flip
        if do_hflip:
            image = hflip(image)
        
        # To tensor and normalize
        tensor = to_tensor(image)
        tensor = self.normalize(tensor)
        
        return tensor
    
    @classmethod
    def from_config(cls, cfg):
        image_size = cfg.get("image_size", 224)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)
        return cls(image_size=image_size, min_scale=min_scale, max_scale=max_scale)


class BlipCaptionProcessor:
    """
    Text processor for captions.
    
    Applies basic text cleaning and formatting.
    """
    
    def __init__(self, prompt: str = "", max_words: int = 50):
        self.prompt = prompt
        self.max_words = max_words
    
    def __call__(self, caption: str) -> str:
        # Clean and format
        caption = self._clean_text(caption)
        
        # Truncate if too long
        words = caption.split()
        if len(words) > self.max_words:
            caption = ' '.join(words[:self.max_words])
        
        # Add prompt if specified
        if self.prompt:
            caption = self.prompt + caption
        
        return caption
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text
    
    @classmethod
    def from_config(cls, cfg):
        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)
        return cls(prompt=prompt, max_words=max_words)


class MaskAwarePairTransforms:
    """
    Synchronized transforms for bi-temporal images and mask.
    
    Applies identical geometric transforms to all inputs:
    - Random crop
    - Random rotation
    - Random flip
    
    Used for mask branch training where spatial consistency is crucial.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mask_size: int = 256,
        crop_scale: Tuple[float, float] = (0.95, 1.0),
        rotation_range: Tuple[int, int] = (-15, 15),
        flip_prob: float = 0.5,
        mean=None,
        std=None,
    ):
        self.image_size = image_size
        self.mask_size = mask_size
        self.crop_scale = crop_scale
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        
        self.normalize = transforms.Normalize(self.mean, self.std)
    
    def __call__(
        self,
        image_A: Image.Image,
        image_B: Image.Image,
        mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply synchronized transforms to image pair and mask.
        
        Returns:
            (tensor_A, tensor_B, mask_tensor)
        """
        # Random parameters
        scale = random.uniform(*self.crop_scale)
        angle = random.uniform(*self.rotation_range)
        do_hflip = random.random() < self.flip_prob
        do_vflip = random.random() < self.flip_prob
        
        # Process images
        tensor_A = self._process_image(image_A, scale, angle, do_hflip, do_vflip)
        tensor_B = self._process_image(image_B, scale, angle, do_hflip, do_vflip)
        
        # Process mask (no normalization, nearest interpolation)
        mask_tensor = self._process_mask(mask, scale, angle, do_hflip, do_vflip)
        
        return tensor_A, tensor_B, mask_tensor
    
    def _process_image(
        self,
        image: Image.Image,
        scale: float,
        angle: float,
        do_hflip: bool,
        do_vflip: bool,
    ) -> torch.Tensor:
        # Resize
        size = int(self.image_size / scale)
        image = resize(image, size, interpolation=transforms.InterpolationMode.BICUBIC)
        
        # Center crop
        image = center_crop(image, self.image_size)
        
        # Rotation
        if angle != 0:
            image = rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        
        # Flips
        if do_hflip:
            image = hflip(image)
        if do_vflip:
            image = vflip(image)
        
        # To tensor and normalize
        tensor = to_tensor(image)
        tensor = self.normalize(tensor)
        
        return tensor
    
    def _process_mask(
        self,
        mask: Image.Image,
        scale: float,
        angle: float,
        do_hflip: bool,
        do_vflip: bool,
    ) -> torch.Tensor:
        # Resize
        size = int(self.mask_size / scale)
        mask = resize(mask, size, interpolation=transforms.InterpolationMode.NEAREST)
        
        # Center crop
        mask = center_crop(mask, self.mask_size)
        
        # Rotation
        if angle != 0:
            mask = rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        
        # Flips
        if do_hflip:
            mask = hflip(mask)
        if do_vflip:
            mask = vflip(mask)
        
        # Convert to binary tensor
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            mask_arr = (mask_arr.sum(axis=2) > 0).astype(np.float32)
        else:
            mask_arr = (mask_arr > 0).astype(np.float32)
        
        return torch.from_numpy(mask_arr).unsqueeze(0)


class MaskEvalTransforms:
    """
    Deterministic transforms for mask evaluation.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mask_size: int = 256,
        mean=None,
        std=None,
    ):
        self.image_size = image_size
        self.mask_size = mask_size
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def __call__(
        self,
        image_A: Image.Image,
        image_B: Image.Image,
        mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Process images
        tensor_A = self.image_transform(image_A)
        tensor_B = self.image_transform(image_B)
        
        # Process mask
        mask = resize(mask, self.mask_size, interpolation=transforms.InterpolationMode.NEAREST)
        mask_arr = np.array(mask)
        if mask_arr.ndim == 3:
            mask_arr = (mask_arr.sum(axis=2) > 0).astype(np.float32)
        else:
            mask_arr = (mask_arr > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        
        return tensor_A, tensor_B, mask_tensor


