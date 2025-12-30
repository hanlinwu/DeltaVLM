#!/usr/bin/env python3
"""
Evaluation Script for DeltaVLM

Evaluates DeltaVLM on change captioning benchmarks.

Usage:
    python scripts/evaluate.py \
        --checkpoint path/to/checkpoint.pth \
        --data_root path/to/data \
        --ann_path path/to/test.json

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deltavlm.models import Blip2VicunaInstruct
from deltavlm.datasets.change_caption import ChangeCapDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeltaVLM")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--llm_model", type=str, default="lmsys/vicuna-7b-v1.5",
                       help="Path to LLM model")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to dataset root")
    parser.add_argument("--ann_path", type=str, required=True,
                       help="Path to annotation JSON file")
    
    # Generation
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum generation length")
    parser.add_argument("--beam_size", type=int, default=5,
                       help="Beam search size")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    
    # Misc
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--output_path", type=str, default="./results.json",
                       help="Output path for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    return parser.parse_args()


def load_model(checkpoint_path, llm_model, device):
    """Load trained model."""
    model = Blip2VicunaInstruct(
        vit_model="eva_clip_g",
        img_size=224,
        llm_model=llm_model,
        max_txt_len=128,
        max_output_txt_len=256,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def evaluate(model, dataloader, args, device):
    """Run evaluation."""
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            image_A = batch['image_A'].to(device)
            image_B = batch['image_B'].to(device)
            
            # Generate captions
            batch_samples = {
                'image_A': image_A,
                'image_B': image_B,
                'prompt': batch.get('prompt', ['Describe the changes between the two images.'] * len(image_A)),
            }
            
            outputs = model.generate(
                batch_samples,
                max_length=args.max_length,
                num_beams=args.beam_size,
            )
            
            # Collect results
            for i, caption in enumerate(outputs):
                result = {
                    'image_id': batch['image_id'][i] if 'image_id' in batch else i,
                    'generated': caption,
                    'reference': batch['caption'][i] if 'caption' in batch else None,
                }
                results.append(result)
    
    return results


def compute_metrics(results):
    """Compute evaluation metrics (placeholder)."""
    # In practice, you would use pycocoevalcap or similar
    metrics = {
        'num_samples': len(results),
        'avg_length': sum(len(r['generated'].split()) for r in results) / len(results),
    }
    
    # Add note about external evaluation
    logging.info("Note: For full metrics (BLEU, CIDEr, METEOR), use pycocoevalcap")
    
    return metrics


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.llm_model, device)
    
    # Build dataset
    logging.info("Building dataset...")
    dataset = ChangeCapDataset(
        vis_root=args.data_root,
        ann_path=args.ann_path,
        split='test',
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    logging.info(f"Test samples: {len(dataset)}")
    
    # Run evaluation
    logging.info("Running evaluation...")
    results = evaluate(model, dataloader, args, device)
    
    # Compute metrics
    metrics = compute_metrics(results)
    logging.info(f"Metrics: {metrics}")
    
    # Save results
    output_data = {
        'metrics': metrics,
        'results': results,
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()


