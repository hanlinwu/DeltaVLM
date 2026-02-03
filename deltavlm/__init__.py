"""
DeltaVLM: Interactive Remote Sensing Image Change Analysis with Vision-Language Models

This package provides the core components for:
- Change captioning and interactive QA on bi-temporal remote sensing images
"""

__version__ = "1.0.0"

from .models import Blip2VicunaInstruct

__all__ = [
    "Blip2VicunaInstruct",
]


