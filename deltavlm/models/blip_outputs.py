"""
BLIP Output Data Classes for DeltaVLM

Data classes for intermediate and final outputs of BLIP-based models.

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


@dataclass
class BlipSimilarity(ModelOutput):
    """Similarity scores between image and text embeddings."""
    sim_i2t: torch.FloatTensor = None
    sim_t2i: torch.FloatTensor = None

    sim_i2t_m: Optional[torch.FloatTensor] = None
    sim_t2i_m: Optional[torch.FloatTensor] = None

    sim_i2t_targets: Optional[torch.FloatTensor] = None
    sim_t2i_targets: Optional[torch.FloatTensor] = None


@dataclass
class BlipIntermediateOutput(ModelOutput):
    """
    Data class for intermediate outputs of BLIP models.

    Attributes:
        image_embeds (torch.FloatTensor): Image embeddings, shape (batch_size, num_patches, embed_dim).
        text_embeds (torch.FloatTensor): Text embeddings, shape (batch_size, seq_len, embed_dim).
        image_embeds_m (torch.FloatTensor): Image embeddings from momentum visual encoder.
        text_embeds_m (torch.FloatTensor): Text embeddings from momentum text encoder.
        encoder_output (BaseModelOutputWithPoolingAndCrossAttentions): Output from image-grounded text encoder.
        encoder_output_neg (BaseModelOutputWithPoolingAndCrossAttentions): Output for negative pairs.
        decoder_output (CausalLMOutputWithCrossAttentions): Output from image-grounded text decoder.
        decoder_labels (torch.LongTensor): Labels for the captioning loss.
        itm_logits (torch.FloatTensor): Logits for image-text matching loss, shape (batch_size * 3, 2).
        itm_labels (torch.LongTensor): Labels for image-text matching loss, shape (batch_size * 3,).
    """

    # uni-modal features
    image_embeds: torch.FloatTensor = None
    text_embeds: Optional[torch.FloatTensor] = None

    image_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None

    # intermediate outputs of multimodal encoder
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None

    itm_logits: Optional[torch.FloatTensor] = None
    itm_labels: Optional[torch.LongTensor] = None

    # intermediate outputs of multimodal decoder
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None


@dataclass
class BlipOutput(ModelOutput):
    """
    Output class for BLIP models.
    
    Attributes:
        sims (BlipSimilarity): Image-text similarity scores (optional for some finetuned models).
        intermediate_output (BlipIntermediateOutput): Intermediate outputs.
        loss (torch.FloatTensor): Total loss.
        loss_itc (torch.FloatTensor): Image-text contrastive loss.
        loss_itm (torch.FloatTensor): Image-text matching loss.
        loss_lm (torch.FloatTensor): Language modeling loss.
    """
    sims: Optional[BlipSimilarity] = None

    intermediate_output: BlipIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_itc: Optional[torch.FloatTensor] = None

    loss_itm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None


@dataclass
class BlipOutputFeatures(ModelOutput):
    """
    Data class of features from BlipFeatureExtractor.

    Attributes:
        image_embeds (torch.FloatTensor): Shape (batch_size, num_patches+1, embed_dim), optional.
        image_embeds_proj (torch.FloatTensor): Shape (batch_size, num_patches+1, feature_dim), optional.
        text_embeds (torch.FloatTensor): Shape (batch_size, sequence_length+1, embed_dim), optional.
        text_embeds_proj (torch.FloatTensor): Shape (batch_size, sequence_length+1, feature_dim), optional.
        multimodal_embeds (torch.FloatTensor): Multimodal embeddings, optional.

    Note:
        The first embedding or feature is for the [CLS] token.
        Features are obtained by projecting the corresponding embedding into a normalized low-dimensional space.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    image_embeds_proj: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None

    multimodal_embeds: Optional[torch.FloatTensor] = None


