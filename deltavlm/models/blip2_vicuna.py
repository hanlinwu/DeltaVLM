"""
DeltaVLM: BLIP-2 with Vicuna for Remote Sensing Change Analysis

This module implements the core DeltaVLM model that combines:
- EVA-ViT-G visual encoder for bi-temporal image encoding
- Change-aware Spatial Representation Module (CSRM) for difference modeling
- Q-Former for vision-language alignment
- Vicuna-7B LLM for natural language generation

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import logging
import string
from packaging import version

import torch
import torch.nn as nn
from torch.nn import functional as F

import transformers
from .blip2_base import Blip2Base, disabled_train
from .base_model import all_gather_with_grad, concat_all_gather


class Blip2VicunaInstruct(Blip2Base):
    """
    DeltaVLM model based on BLIP-2 with Vicuna LLM.
    
    This model processes bi-temporal remote sensing images and generates
    natural language descriptions of changes or answers questions about them.
    
    Architecture:
        1. EVA-ViT-G encodes both images independently
        2. CSRM (Change-aware Spatial Representation Module) models temporal differences
        3. Q-Former aligns visual features with text
        4. Vicuna-7B generates natural language output
    
    Args:
        vit_model: Vision encoder type ("eva_clip_g")
        img_size: Input image size (default: 224)
        drop_path_rate: Drop path rate for ViT
        use_grad_checkpoint: Enable gradient checkpointing
        vit_precision: ViT precision ("fp16" or "fp32")
        freeze_vit: Whether to freeze the vision encoder
        num_query_token: Number of Q-Former query tokens
        llm_model: Path to LLM model
        prompt: Default prompt template
        max_txt_len: Maximum input text length
        max_output_txt_len: Maximum output text length
        apply_lemmatizer: Whether to apply lemmatization
        qformer_text_input: Whether Q-Former receives text input
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model: str = "eva_clip_g",
        img_size: int = 224,
        drop_path_rate: float = 0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        num_query_token: int = 32,
        llm_model: str = "",
        prompt: str = "",
        max_txt_len: int = 128,
        max_output_txt_len: int = 256,
        apply_lemmatizer: bool = False,
        qformer_text_input: bool = True,
    ):
        super().__init__()
        
        # Validate transformers version
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.28"), \
            "BLIP-2 Vicuna requires transformers>=4.28"
        
        from transformers import LlamaTokenizer
        from .modeling_llama import LlamaForCausalLM
        
        # Initialize tokenizer
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        # Initialize vision encoder
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.visual_encoder = self.visual_encoder.float()
        self.ln_vision = self.ln_vision.float()
        
        # Freeze vision encoder (except last 2 blocks for fine-tuning)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                # Keep last 2 blocks trainable
                if "blocks.37" in name or "blocks.38" in name:
                    continue
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("Frozen vision encoder (except last 2 blocks)")

        # Initialize Q-Former
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        
        if qformer_text_input:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        else:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        self.Qformer.cls = None

        # Skip LLM loading for mask-only training mode
        self.skip_llm = getattr(self, '_skip_llm', False)
        
        if not self.skip_llm:
            # Initialize LLM
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(
                llm_model or "lmsys/vicuna-7b-v1.5",
                use_fast=False,
                truncation_side="left"
            )
            self.llm_model = LlamaForCausalLM.from_pretrained(
                llm_model or "lmsys/vicuna-7b-v1.5",
                torch_dtype=torch.float16
            )
            
            # Add special tokens
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
            self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
            self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

            # Freeze LLM
            for param in self.llm_model.parameters():
                param.requires_grad = False

            # LLM projection
            self.llm_proj = nn.Linear(
                self.Qformer.config.hidden_size,
                self.llm_model.config.hidden_size
            )
            
            prompt_tokens = self.llm_tokenizer(prompt, return_tensors="pt")
            self.prompt_length = prompt_tokens.attention_mask.sum(1)
        else:
            self.llm_tokenizer = None
            self.llm_model = None
            self.llm_proj = None
            self.prompt_length = 0
            logging.info("Skipping LLM loading for mask-only training")

        # Configuration
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        self.qformer_text_input = qformer_text_input
        self._lemmatizer = None

        # ============ Change-aware Spatial Representation Module (CSRM) ============
        # These layers model the temporal difference between bi-temporal images
        feat_dim = self.visual_encoder.num_features  # 1408 for EVA-ViT-G
        
        # Context modeling
        self.context1 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.context2 = nn.Linear(feat_dim, feat_dim)
        
        # Gating mechanism
        self.gate1 = nn.Linear(feat_dim, feat_dim, bias=False)
        self.gate2 = nn.Linear(feat_dim, feat_dim)
        
        self.dropout = nn.Dropout(0.5)
        
        # Feature fusion
        self.context3 = nn.Linear(3 * feat_dim, feat_dim)
        
        # Contrastive learning projections (optional)
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, 256)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, 256)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        """Concatenate input and output tokens for training."""
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
            
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        
        return llm_tokens, input_part_targets_len

    def forward(self, samples):
        """
        Forward pass for training.
        
        Args:
            samples: Dict containing:
                - image_A: (B, 3, H, W) before image
                - image_B: (B, 3, H, W) after image
                - text_input: List of input questions
                - text_output: List of target answers
                
        Returns:
            Dict with 'loss' key
        """
        image_A = samples["image_A"]
        image_B = samples["image_B"]

        # Encode bi-temporal images
        with self.maybe_autocast():
            input_bef = self.ln_vision(self.visual_encoder(image_A))
            input_aft = self.ln_vision(self.visual_encoder(image_B))

        # Concatenate features (simple approach without CSRM for training)
        image_embeds = torch.cat((input_bef, input_aft), dim=1)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long
        ).to(image_A.device)

        # Q-Former encoding
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_A.device)
            
            query_atts = torch.ones(
                query_tokens.size()[:-1], dtype=torch.long
            ).to(image_A.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        # Project to LLM space
        inputs_llm = self.llm_proj(
            query_output.last_hidden_state[:, :query_tokens.size(1), :]
        )
        atts_llm = torch.ones(
            inputs_llm.size()[:-1], dtype=torch.long
        ).to(image_A.device)

        # Tokenize input and output
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image_A.device)

        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image_A.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        # Mask padding tokens
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # Mask input tokens (only compute loss on output)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # Mask query tokens
        empty_targets = torch.ones(
            atts_llm.size(), dtype=torch.long
        ).to(image_A.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        # Prepare LLM inputs
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        # Forward through LLM
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling: bool = False,
        num_beams: int = 1,
        max_length: int = 256,
        min_length: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.5,
        length_penalty: float = 1,
        num_captions: int = 1,
        temperature: float = 1,
    ):
        """
        Generate text from bi-temporal images.
        
        Args:
            samples: Dict with 'image_A', 'image_B', and optionally 'prompt'
            use_nucleus_sampling: Use nucleus sampling
            num_beams: Number of beams for beam search
            max_length: Maximum output length
            min_length: Minimum output length
            top_p: Top-p for nucleus sampling
            repetition_penalty: Repetition penalty
            length_penalty: Length penalty
            num_captions: Number of captions to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated text strings
        """
        self.llm_tokenizer.padding_side = "left"

        prompt = samples.get("prompt", self.prompt)
        image_A = samples["image_A"]
        image_B = samples["image_B"]
        bs = image_A.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs

        # Encode images with CSRM
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_A.device)
            query_atts = torch.ones(
                query_tokens.size()[:-1], dtype=torch.long
            ).to(image_A.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        with self.maybe_autocast():
            # Extract features
            input_bef = self.ln_vision(self.visual_encoder(image_A))
            input_aft = self.ln_vision(self.visual_encoder(image_B))

            # Apply CSRM for difference modeling
            input_diff = input_aft - input_bef

            # Context and gating for before image
            input_bef_context = torch.tanh(
                self.context1(input_diff) + self.context2(input_bef)
            )
            input_bef_context = self.dropout(input_bef_context)
            input_bef_gate = torch.sigmoid(
                self.gate1(input_diff) + self.gate2(input_bef)
            )
            input_bef_gate = self.dropout(input_bef_gate)
            input_befs = input_bef_gate * input_bef_context

            # Context and gating for after image
            input_aft_context = torch.tanh(
                self.context1(input_diff) + self.context2(input_aft)
            )
            input_aft_context = self.dropout(input_aft_context)
            input_aft_gate = torch.sigmoid(
                self.gate1(input_diff) + self.gate2(input_aft)
            )
            input_aft_gate = self.dropout(input_aft_gate)
            input_afts = input_aft_gate * input_aft_context

            # Fuse features
            input_bef = input_bef.permute(0, 2, 1)
            input_aft = input_aft.permute(0, 2, 1)
            input_befs = input_befs.permute(0, 2, 1)
            input_afts = input_afts.permute(0, 2, 1)
            input_diff = input_diff.permute(0, 2, 1)

            input_before = torch.cat([input_bef, input_diff, input_befs], 1)
            input_after = torch.cat([input_aft, input_diff, input_afts], 1)

            input_before = input_before.permute(0, 2, 1)
            input_after = input_after.permute(0, 2, 1)
            
            image_embeds_A = self.context3(input_before)
            image_embeds_B = self.context3(input_after)
            
            image_embeds = torch.cat((image_embeds_A, image_embeds_B), dim=1)
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
            ).to(image_A.device)

            # Q-Former
            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(
                query_output.last_hidden_state[:, :query_tokens.size(1), :]
            )
            atts_llm = torch.ones(
                inputs_llm.size()[:-1], dtype=torch.long
            ).to(image_A.device)

        # Prepare LLM inputs
        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image_A.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
            
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        # Decode outputs
        outputs[outputs == 0] = 2  # Convert 0 to EOS
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams: int = 5,
        inference_method: str = "generate",
        max_len: int = 256,
        min_len: int = 1,
        num_ans_candidates: int = 128,
        answer_list=None,
        prompt: str = "",
        length_penalty: float = 1,
        **kwargs
    ):
        """
        Predict answers for question answering.
        
        Args:
            samples: Dict with images and text_input
            num_beams: Number of beams
            max_len: Maximum answer length
            min_len: Minimum answer length
            prompt: Optional prompt template
            
        Returns:
            List of answer strings
        """
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            text_input = [prompt.format(q) for q in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if samples.get("apply_lemmatizer", False):
            output_text = self._lemmatize(output_text)

        return output_text

    @classmethod
    def from_config(cls, cfg):
        """Create model from config."""
        model = cls(
            vit_model=cfg.get("vit_model", "eva_clip_g"),
            img_size=cfg.get("image_size", 224),
            drop_path_rate=cfg.get("drop_path_rate", 0),
            use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
            vit_precision=cfg.get("vit_precision", "fp16"),
            freeze_vit=cfg.get("freeze_vit", True),
            num_query_token=cfg.get("num_query_token", 32),
            llm_model=cfg.get("llm_model", ""),
            prompt=cfg.get("prompt", ""),
            max_txt_len=cfg.get("max_txt_len", 128),
            max_output_txt_len=cfg.get("max_output_txt_len", 256),
            apply_lemmatizer=cfg.get("apply_lemmatizer", False),
            qformer_text_input=cfg.get("qformer_text_input", True),
        )

        model.load_checkpoint_from_config(cfg)
        return model


