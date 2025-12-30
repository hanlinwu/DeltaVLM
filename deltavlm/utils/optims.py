"""
Learning Rate Schedulers for DeltaVLM

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import math


class LinearWarmupCosineLRScheduler:
    """
    Linear warmup followed by cosine annealing learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        max_epoch: Maximum training epochs
        min_lr: Minimum learning rate
        init_lr: Initial learning rate (after warmup)
        warmup_steps: Number of warmup steps
        warmup_start_lr: Starting learning rate for warmup
        decay_rate: Optional decay rate (not used in cosine)
    """
    
    def __init__(
        self,
        optimizer,
        max_epoch: int,
        min_lr: float,
        init_lr: float,
        warmup_steps: int = 0,
        warmup_start_lr: float = -1,
        decay_rate: float = None,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        
        self._step_count = 0
        self._cur_epoch = 0
    
    def step(self, cur_epoch: int, cur_step: int):
        """
        Update learning rate.
        
        Args:
            cur_epoch: Current epoch (0-indexed)
            cur_step: Current step within epoch
        """
        total_step = cur_epoch * self._steps_per_epoch + cur_step
        self._step_count = total_step
        self._cur_epoch = cur_epoch
        
        if total_step < self.warmup_steps:
            # Linear warmup
            warmup_ratio = total_step / self.warmup_steps
            lr = self.warmup_start_lr + warmup_ratio * (self.init_lr - self.warmup_start_lr)
        else:
            # Cosine annealing
            progress = (total_step - self.warmup_steps) / max(
                1, self._total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.init_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
        
        return lr
    
    def set_steps_per_epoch(self, steps_per_epoch: int):
        """Set number of steps per epoch."""
        self._steps_per_epoch = steps_per_epoch
        self._total_steps = self.max_epoch * steps_per_epoch
    
    @property
    def current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class LinearWarmupStepLRScheduler:
    """
    Linear warmup followed by step decay learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        max_epoch: Maximum training epochs
        min_lr: Minimum learning rate
        init_lr: Initial learning rate
        decay_rate: Decay rate per step
        warmup_steps: Number of warmup steps
        warmup_start_lr: Starting learning rate for warmup
    """
    
    def __init__(
        self,
        optimizer,
        max_epoch: int,
        min_lr: float,
        init_lr: float,
        decay_rate: float = 1.0,
        warmup_steps: int = 0,
        warmup_start_lr: float = -1,
        **kwargs
    ):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        
        self._step_count = 0
    
    def step(self, cur_epoch: int, cur_step: int):
        """Update learning rate."""
        total_step = cur_epoch * self._steps_per_epoch + cur_step
        self._step_count = total_step
        
        if total_step < self.warmup_steps:
            # Linear warmup
            warmup_ratio = total_step / self.warmup_steps
            lr = self.warmup_start_lr + warmup_ratio * (self.init_lr - self.warmup_start_lr)
        else:
            # Step decay
            decay_steps = (total_step - self.warmup_steps) // self._steps_per_epoch
            lr = max(self.init_lr * (self.decay_rate ** decay_steps), self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * param_group.get("lr_scale", 1.0)
        
        return lr
    
    def set_steps_per_epoch(self, steps_per_epoch: int):
        """Set number of steps per epoch."""
        self._steps_per_epoch = steps_per_epoch


# Scheduler name mapping
lr_scheduler_name_mapping = {
    "linear_warmup_cosine_lr": LinearWarmupCosineLRScheduler,
    "linear_warmup_step_lr": LinearWarmupStepLRScheduler,
}


