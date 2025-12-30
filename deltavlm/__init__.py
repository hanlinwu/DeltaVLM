"""
DeltaVLM: Interactive Remote Sensing Image Change Analysis with Vision-Language Models

This package provides the core components for:
- Change captioning and interactive QA on bi-temporal remote sensing images
- Change mask prediction with AnyUp-style upsampling
"""

__version__ = "1.0.0"

from .models import Blip2VicunaInstruct, Blip2VicunaMask

__all__ = [
    "Blip2VicunaInstruct",
    "Blip2VicunaMask",
]


