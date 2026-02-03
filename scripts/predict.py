#!/usr/bin/env python3
"""
Inference Script for DeltaVLM Change Captioning

Generates natural language descriptions of changes between bi-temporal images.

Usage:
    python scripts/predict.py \
        --image_A path/to/before.png \
        --image_B path/to/after.png \
        --prompt "Describe the changes between these two images."

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import os
import sys

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate change descriptions")
    
    parser.add_argument("--image_A", type=str, required=True,
                       help="Path to before image")
    parser.add_argument("--image_B", type=str, required=True,
                       help="Path to after image")
    parser.add_argument("--checkpoint", type=str, default="pretrained/checkpoint_best.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str,
                       default="Please briefly describe the changes in these two images.",
                       help="Prompt for generation")
    parser.add_argument("--llm_model", type=str, default="lmsys/vicuna-7b-v1.5",
                       help="Path to LLM model")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum output length")
    parser.add_argument("--min_length", type=int, default=1,
                       help="Minimum output length")
    parser.add_argument("--num_beams", type=int, default=5,
                       help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p for nucleus sampling")
    
    # Other
    parser.add_argument("--image_size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    return parser.parse_args()


def load_model(checkpoint_path, llm_model, device='cuda'):
    """Load DeltaVLM model."""
    from deltavlm.models import Blip2VicunaInstruct
    
    print(f"Loading model from {checkpoint_path}...")
    
    model = Blip2VicunaInstruct(
        vit_model='eva_clip_g',
        img_size=224,
        freeze_vit=True,
        num_query_token=32,
        llm_model=llm_model,
        qformer_text_input=True,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def preprocess_images(image_A_path, image_B_path, image_size=224):
    """Load and preprocess images."""
    from deltavlm.datasets.processors import BlipImageEvalProcessor
    
    processor = BlipImageEvalProcessor(image_size=image_size)
    
    img_A = Image.open(image_A_path).convert("RGB")
    img_B = Image.open(image_B_path).convert("RGB")
    
    img_A_tensor, img_B_tensor = processor(img_A, img_B)
    
    return img_A_tensor.unsqueeze(0), img_B_tensor.unsqueeze(0)


def generate(model, image_A, image_B, prompt, args, device='cuda'):
    """Generate change description."""
    samples = {
        'image_A': image_A.to(device),
        'image_B': image_B.to(device),
        'prompt': [prompt],
    }
    
    with torch.no_grad():
        output = model.generate(
            samples,
            num_beams=args.num_beams,
            max_length=args.max_length,
            min_length=args.min_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    
    return output[0]


def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image_A):
        print(f"Error: Image A not found: {args.image_A}")
        return
    if not os.path.exists(args.image_B):
        print(f"Error: Image B not found: {args.image_B}")
        return
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.llm_model, device)
    
    # Preprocess images
    print("Preprocessing images...")
    img_A, img_B = preprocess_images(args.image_A, args.image_B, args.image_size)
    
    # Generate
    print(f"Prompt: {args.prompt}")
    print("Generating response...")
    
    response = generate(model, img_A, img_B, args.prompt, args, device)
    
    print("\n" + "=" * 50)
    print("Generated Description:")
    print("=" * 50)
    print(response)
    print("=" * 50)


if __name__ == "__main__":
    main()


