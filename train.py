"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config import Config
from utils import get_rank

# import lavis.tasks as tasks
# from lavis.common.logger import setup_logger
# from lavis.common.optims import (
#     LinearWarmupCosineLRScheduler,
#     LinearWarmupStepLRScheduler,
# )
# from lavis.common.registry import registry
from utils import now, init_distributed_mode, setup_logger

# # imports modules for registration
# from lavis.datasets.builders import *
# from lavis.models import *
# from lavis.processors import *
# from lavis.runners import *
# from lavis.tasks import *
from task import ImageTextPretrainTask
from runner import RunnerBase
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg_path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # 增加为了克服多卡问题
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(-1)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = RunnerBase #runner_base

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = ImageTextPretrainTask.setup_task(cfg=cfg)
    datasets = task.build_datasets(cfg)   # image_text_pretrain->BaseTask->
    model = task.build_model(cfg)    #

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
