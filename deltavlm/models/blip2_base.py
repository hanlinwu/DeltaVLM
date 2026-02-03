"""
BLIP-2 Base Model for DeltaVLM

Based on Salesforce BLIP-2 implementation with modifications for remote sensing change analysis.

Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
"""

import contextlib
import logging
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from .base_model import BaseModel, disabled_train
from .qformer import BertConfig, BertLMHeadModel
from .eva_vit import create_eva_vit_g
from deltavlm.utils.distributed import download_cached_file, is_url

from transformers import BertTokenizer

# Support local model paths via environment variable
BERT_MODEL_PATH = os.environ.get("BERT_MODEL_PATH", "bert-base-uncased")


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Blip2Base(BaseModel):
    """
    Base class for BLIP-2 models.
    
    Provides:
    - Vision encoder initialization (EVA-ViT-G)
    - Q-Former initialization
    - Tokenizer initialization
    - Mixed precision support
    """
    
    def __init__(self):
        super().__init__()

    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        """Initialize BERT tokenizer."""
        tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL_PATH, 
            truncation_side=truncation_side,
            local_files_only=os.path.isdir(BERT_MODEL_PATH)
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        """Context manager for mixed precision."""
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        """
        Initialize Q-Former model.
        
        Args:
            num_query_token: Number of query tokens
            vision_width: Width of vision features
            cross_attention_freq: Frequency of cross-attention layers
            
        Returns:
            Qformer: Q-Former model
            query_tokens: Learnable query tokens
        """
        local_only = os.path.isdir(BERT_MODEL_PATH)
        encoder_config = BertConfig.from_pretrained(BERT_MODEL_PATH, local_files_only=local_only)
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel.from_pretrained(
            BERT_MODEL_PATH, config=encoder_config, local_files_only=local_only
        )
        
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        return Qformer, query_tokens

    def init_vision_encoder(
        self, 
        model_name: str,
        img_size: int,
        drop_path_rate: float,
        use_grad_checkpoint: bool,
        precision: str
    ):
        """
        Initialize vision encoder.
        
        Args:
            model_name: Name of vision encoder ("eva_clip_g")
            img_size: Input image size
            drop_path_rate: Drop path rate for regularization
            use_grad_checkpoint: Whether to use gradient checkpointing
            precision: Precision ("fp16" or "fp32")
            
        Returns:
            visual_encoder: Vision encoder model
            ln_vision: Layer normalization for vision features
        """
        assert model_name in ["eva_clip_g"], f"Unsupported model: {model_name}"
        
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename: str):
        """Load pretrained weights."""
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError(f"Checkpoint not found: {url_or_filename}")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded checkpoint from {url_or_filename}")

        return msg

    def get_optimizer_params(self, weight_decay: float, lr_scale: float = 1.0):
        """
        Get optimizer parameters with layer-wise learning rate decay.
        
        Args:
            weight_decay: Weight decay value
            lr_scale: Learning rate scale factor
            
        Returns:
            List of parameter groups for optimizer
        """
        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(
            lr_scale ** (vit_num_layers + 1 - i) 
            for i in range(vit_num_layers + 2)
        )

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # No weight decay for bias and LayerNorm
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.0
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
                
            # Layer-wise learning rate for ViT
            if "visual_encoder" in name:
                layer_id = self.visual_encoder.get_num_layer(
                    name.replace("visual_encoder.", "")
                )
                group_name = f"vit_layer_{layer_id}_{group_name}"
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                scale = lr_scales[layer_id] if layer_id is not None else 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        return list(parameter_group_vars.values())

    @property
    def lemmatizer(self):
        """Lazy-loaded spaCy lemmatizer."""
        if self._lemmatizer is None:
            try:
                import spacy
                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    "Please install spacy and en_core_web_sm:\n"
                    "pip install spacy\n"
                    "python -m spacy download en_core_web_sm"
                )
                exit(1)
        return self._lemmatizer

    def _lemmatize(self, answers):
        """Apply lemmatization to answers."""
        def apply(answer):
            doc = self.lemmatizer(answer)
            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            return " ".join(words)
        return [apply(answer) for answer in answers]


