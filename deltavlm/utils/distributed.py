"""
Distributed Training Utilities

Copyright (c) 2024
SPDX-License-Identifier: BSD-3-Clause
"""

import datetime
import functools
import logging
import os
import subprocess
import tempfile
from urllib.parse import urlparse

import torch
import torch.distributed as dist


def is_url(url_or_filename: str) -> bool:
    """Check if a string is a URL."""
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def download_cached_file(url: str, check_hash: bool = True, progress: bool = False):
    """
    Download a file from URL and cache it locally.
    
    Uses torch.hub.download_url_to_file internally.
    """
    from torch.hub import download_url_to_file, urlparse
    
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    
    # Create cache directory
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "deltavlm")
    os.makedirs(cache_dir, exist_ok=True)
    
    cached_file = os.path.join(cache_dir, filename)
    
    if not os.path.exists(cached_file):
        logging.info(f"Downloading {url} to {cached_file}")
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    else:
        logging.info(f"Using cached file {cached_file}")
    
    return cached_file


def get_rank() -> int:
    """Get current process rank."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def is_dist_avail_and_initialized() -> bool:
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def main_process(func):
    """Decorator to run function only on main process."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


def init_distributed_mode(cfg):
    """
    Initialize distributed training mode.
    
    Supports:
    - SLURM
    - torch.distributed.launch
    - Single GPU
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.gpu = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
    else:
        logging.info("Not using distributed mode")
        cfg.distributed = False
        cfg.rank = 0
        cfg.world_size = 1
        cfg.gpu = 0
        return

    cfg.distributed = True

    torch.cuda.set_device(cfg.gpu)
    
    dist_backend = "nccl"
    logging.info(
        f"Distributed init (rank {cfg.rank}): "
        f"world_size={cfg.world_size}, gpu={cfg.gpu}"
    )
    
    dist.init_process_group(
        backend=dist_backend,
        init_method=cfg.get("dist_url", "env://"),
        world_size=cfg.world_size,
        rank=cfg.rank,
        timeout=datetime.timedelta(minutes=60),
    )
    
    dist.barrier()


def setup_logger():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def now() -> str:
    """Get current timestamp string."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_cache_root() -> str:
    """Get cache root directory."""
    return os.path.join(os.path.expanduser("~"), ".cache", "deltavlm")


def get_cache_path(path: str) -> str:
    """Get absolute cache path."""
    if os.path.isabs(path):
        return path
    return os.path.join(get_cache_root(), path)


def get_abs_path(path: str) -> str:
    """Get absolute path relative to this file's directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", path)


