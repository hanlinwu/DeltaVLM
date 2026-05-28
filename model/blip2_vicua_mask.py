"""
DeltaVLM with Mask Branch
=========================
Extends Blip2VicunaInstruct with an AnyUp-style mask prediction branch.

The mask branch is a side-path that:
1. Takes features from CSRM (before Q-Former)
2. Produces high-resolution binary change masks
3. Does NOT affect the original caption/QA output

Training modes:
- 'mask_only': Only train mask decoder (freeze everything else)
- 'joint': Train both branches (future work)
"""
import logging
import torch
import torch.nn as nn
from .blip2_vicua import Blip2VicunaInstruct
from .mask_decoder import ChangeMaskDecoder, build_mask_decoder


class Blip2VicunaMask(Blip2VicunaInstruct):
    """
    DeltaVLM with mask prediction branch.
    
    Additional args:
        mask_hidden_dim: Hidden dimension for mask decoder (default: 128)
        mask_output_size: Output mask resolution (default: 256)
        use_mask_attention: Use attention in mask decoder (default: True)
    """
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        # Mask branch args
        mask_hidden_dim=128,
        mask_output_size=256,
        use_mask_attention=True,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
        )
        
        # Build mask decoder
        self.mask_decoder = build_mask_decoder(
            in_ch=self.visual_encoder.num_features,  # 1408 for EVA-ViT-G
            out_sz=mask_output_size,
            hid=mask_hidden_dim,
            light=not use_mask_attention
        )
        
        self.mask_output_size = mask_output_size
        logging.info(f"Mask decoder initialized with {sum(p.numel() for p in self.mask_decoder.parameters()):,} parameters")
    
    def freeze_for_mask_training(self):
        """Freeze all modules except mask decoder for mask-only training."""
        # Freeze visual encoder
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.ln_vision.parameters():
            param.requires_grad = False
        
        # Freeze CSRM components
        for param in self.context1.parameters():
            param.requires_grad = False
        for param in self.context2.parameters():
            param.requires_grad = False
        for param in self.gate1.parameters():
            param.requires_grad = False
        for param in self.gate2.parameters():
            param.requires_grad = False
        for param in self.context3.parameters():
            param.requires_grad = False
        
        # Freeze Q-Former
        for param in self.Qformer.parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = False
        
        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False
        for param in self.llm_proj.parameters():
            param.requires_grad = False
        
        # Freeze other projections
        if hasattr(self, 'vision_proj'):
            for param in self.vision_proj.parameters():
                param.requires_grad = False
        if hasattr(self, 'text_proj'):
            for param in self.text_proj.parameters():
                param.requires_grad = False
        
        # Ensure mask decoder is trainable
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logging.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def extract_csrm_features(self, imageA, imageB):
        """
        Extract CSRM (Change-aware Spatial Relation Module) features.
        
        Returns:
            image_embeds: [B, 514, 1408] concatenated bi-temporal features
            diff_features: [B, 257, 1408] difference features for mask prediction
        """
        with self.maybe_autocast():
            input_bef = self.ln_vision(self.visual_encoder(imageA))
            input_aft = self.ln_vision(self.visual_encoder(imageB))
        
        # Compute difference
        input_diff = input_aft - input_bef
        
        # CSRM gating mechanism
        input_bef_context = torch.tanh(self.context1(input_diff) + self.context2(input_bef))
        input_bef_context = self.dropout(input_bef_context)
        input_bef_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_bef))
        input_bef_gate = self.dropout(input_bef_gate)
        input_befs = input_bef_gate * input_bef_context

        input_aft_context = torch.tanh(self.context1(input_diff) + self.context2(input_aft))
        input_aft_context = self.dropout(input_aft_context)
        input_aft_gate = torch.sigmoid(self.gate1(input_diff) + self.gate2(input_aft))
        input_aft_gate = self.dropout(input_aft_gate)
        input_afts = input_aft_gate * input_aft_context

        # Concatenate features
        input_bef_perm = input_bef.permute(0, 2, 1)
        input_aft_perm = input_aft.permute(0, 2, 1)
        input_befs_perm = input_befs.permute(0, 2, 1)
        input_afts_perm = input_afts.permute(0, 2, 1)
        input_diff_perm = input_diff.permute(0, 2, 1)

        input_before = torch.cat([input_bef_perm, input_diff_perm, input_befs_perm], 1)
        input_after = torch.cat([input_aft_perm, input_diff_perm, input_afts_perm], 1)

        input_before = input_before.permute(0, 2, 1)
        input_after = input_after.permute(0, 2, 1)
        image_embedsA = self.context3(input_before)
        image_embedsB = self.context3(input_after)

        # Concatenated embeddings for Q-Former
        image_embeds = torch.cat((image_embedsA, image_embedsB), dim=1)
        
        # Return both for different purposes
        return image_embeds, input_diff
    
    def forward_mask(self, imageA, imageB, gt_mask=None, return_loss=True):
        """
        Forward pass for mask prediction only.
        
        Args:
            imageA: [B, 3, H, W] pre-change image
            imageB: [B, 3, H, W] post-change image
            gt_mask: [B, 1, H, W] ground truth mask (optional)
            return_loss: Whether to compute and return loss
        
        Returns:
            dict with 'mask_pred' and optionally 'mask_loss'
        """
        # Extract features
        _, diff_features = self.extract_csrm_features(imageA, imageB)
        
        # Predict mask
        mask_pred = self.mask_decoder(diff_features, output_size=(self.mask_output_size, self.mask_output_size))
        
        result = {"mask_pred": mask_pred}
        
        if return_loss and gt_mask is not None:
            result["mask_loss"] = self.mask_decoder.get_loss(mask_pred, gt_mask)
        
        return result
    
    def forward(self, samples, mode='caption'):
        """
        Forward pass with mode selection.
        
        Args:
            samples: dict with 'image_A', 'image_B', and optionally 'gt_mask'
            mode: 'caption' (original), 'mask' (mask only), or 'both'
        
        Returns:
            dict with outputs based on mode
        """
        if mode == 'mask':
            return self.forward_mask(
                samples["image_A"],
                samples["image_B"],
                samples.get("gt_mask"),
                return_loss=True
            )
        elif mode == 'caption':
            # Original forward
            return super().forward(samples)
        elif mode == 'both':
            # Joint forward (for future joint training)
            caption_out = super().forward(samples)
            mask_out = self.forward_mask(
                samples["image_A"],
                samples["image_B"],
                samples.get("gt_mask"),
                return_loss=True
            )
            return {**caption_out, **mask_out}
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @torch.no_grad()
    def predict_mask(self, imageA, imageB, threshold=0.5):
        """
        Inference: predict binary mask.
        
        Args:
            imageA, imageB: [B, 3, H, W] input images
            threshold: Binarization threshold
        
        Returns:
            mask: [B, 1, H, W] binary mask (0 or 1)
            mask_prob: [B, 1, H, W] probability mask (0 to 1)
        """
        self.eval()
        _, diff_features = self.extract_csrm_features(imageA, imageB)
        mask_prob = self.mask_decoder(diff_features)
        mask_binary = (mask_prob > threshold).float()
        return mask_binary, mask_prob
    
    @classmethod
    def from_pretrained_base(cls, base_ckpt_path, **kwargs):
        """
        Create mask model from pretrained base model checkpoint.
        
        Args:
            base_ckpt_path: Path to base model checkpoint
            **kwargs: Additional args for mask decoder
        """
        # Load checkpoint
        ckpt = torch.load(base_ckpt_path, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        # Create model
        model = cls(**kwargs)
        
        # Load base weights (ignore mask decoder)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        # Filter expected missing keys (mask decoder)
        missing = [k for k in missing if not k.startswith('mask_decoder')]
        
        if missing:
            logging.warning(f"Missing keys: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys: {unexpected}")
        
        logging.info(f"Loaded base model from {base_ckpt_path}")
        return model


if __name__ == "__main__":
    print("Blip2VicunaMask module loaded successfully")
