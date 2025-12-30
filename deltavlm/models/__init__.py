"""
DeltaVLM Models

This module contains the core model implementations for DeltaVLM.

Available Models:
- Blip2VicunaInstruct: Main DeltaVLM model with CSRM for change captioning
- Blip2Base: Base class for BLIP-2 based models

Available Components:
- VisionTransformer: EVA-ViT-G visual encoder
- BertModel: Q-Former encoder
- LlamaForCausalLM: LLaMA language model

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

from .base_model import BaseModel, BaseEncoder
from .blip_outputs import BlipOutput, BlipOutputFeatures, BlipIntermediateOutput
from .blip2_base import Blip2Base
from .blip2_vicuna import Blip2VicunaInstruct
from .eva_vit import VisionTransformer, create_eva_vit_g
from .qformer import BertModel, BertLMHeadModel


__all__ = [
    # Base classes
    "BaseModel",
    "BaseEncoder",
    
    # BLIP-2 models
    "Blip2Base",
    "Blip2VicunaInstruct",
    
    # Output classes
    "BlipOutput",
    "BlipOutputFeatures",
    "BlipIntermediateOutput",
    
    # Components
    "VisionTransformer",
    "create_eva_vit_g",
    "BertModel",
    "BertLMHeadModel",
]
