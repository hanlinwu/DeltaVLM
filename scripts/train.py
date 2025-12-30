#!/usr/bin/env python3
"""
Training Script for DeltaVLM

Trains the DeltaVLM model for change captioning on bi-temporal remote sensing images.

Usage:
    python scripts/train.py --cfg_path configs/train_stage2.yaml

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import datetime
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deltavlm.models import Blip2VicunaInstruct
from deltavlm.datasets.change_caption import ChangeCapDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeltaVLM")
    
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to dataset root directory")
    parser.add_argument("--ann_path", type=str, required=True,
                       help="Path to annotation JSON file")
    
    # Model
    parser.add_argument("--pretrained", type=str, default=None,
                       help="Path to pretrained checkpoint")
    parser.add_argument("--llm_model", type=str, default="lmsys/vicuna-7b-v1.5",
                       help="Path to LLM model")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Input image size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    # Misc
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--log_freq", type=int, default=50,
                       help="Logging frequency (iterations)")
    parser.add_argument("--save_freq", type=int, default=1,
                       help="Checkpoint saving frequency (epochs)")
    
    # Config file
    parser.add_argument("--cfg_path", type=str, default=None,
                       help="Path to YAML config file")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.cfg_path and os.path.exists(args.cfg_path):
        import yaml
        with open(args.cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        for key, value in cfg.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return args


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    
    return rank, world_size, local_rank


def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    # Create output directory
    if is_main_process:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.output_dir, timestamp)
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')) if is_main_process else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )
    
    if is_main_process:
        logging.info(f"Arguments: {args}")
        logging.info(f"Output directory: {args.output_dir}")
    
    set_seed(args.seed + rank)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Rank {rank}: Using device {device}")
    
    # TensorBoard
    writer = None
    if is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    
    # Build dataset
    logging.info("Building dataset...")
    train_dataset = ChangeCapDataset(
        vis_root=args.data_root,
        ann_path=args.ann_path,
        split='train',
    )
    
    if world_size > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    if is_main_process:
        logging.info(f"Train samples: {len(train_dataset)}")
    
    # Build model
    logging.info("Building model...")
    model = Blip2VicunaInstruct(
        vit_model="eva_clip_g",
        img_size=args.image_size,
        llm_model=args.llm_model,
        max_txt_len=128,
        max_output_txt_len=256,
    )
    
    # Load pretrained if specified
    if args.pretrained:
        logging.info(f"Loading pretrained from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    
    # Distributed
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Training loop
    logging.info("Starting training...")
    global_step = 0
    
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward
            outputs = model(batch)
            loss = outputs['loss']
            
            # Backward
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            total_loss += loss.item() * args.grad_accum_steps
            
            # Logging
            if is_main_process and (batch_idx + 1) % args.log_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Iter [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} LR: {lr:.2e}"
                )
                if writer:
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    writer.add_scalar('train/lr', lr, global_step)
        
        # Epoch end
        avg_loss = total_loss / len(train_loader)
        if is_main_process:
            logging.info(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if is_main_process and (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args),
            }
            save_path = os.path.join(args.output_dir, f'checkpoint_{epoch+1}.pth')
            torch.save(checkpoint, save_path)
            logging.info(f"Checkpoint saved to {save_path}")
    
    # Final save
    if is_main_process:
        checkpoint = {
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': args.epochs - 1,
            'args': vars(args),
        }
        save_path = os.path.join(args.output_dir, 'checkpoint_final.pth')
        torch.save(checkpoint, save_path)
        logging.info(f"Final checkpoint saved to {save_path}")
        
        if writer:
            writer.close()
    
    if world_size > 1:
        dist.destroy_process_group()
    
    logging.info("Training complete!")


if __name__ == "__main__":
    main()


