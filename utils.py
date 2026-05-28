import datetime
import functools
import os

import torch
import torch.distributed as dist
import timm.models.hub as timm_hub
from omegaconf import OmegaConf
import webdataset as wds
import logging
from typing import Iterable
import time
import random
import re
import shutil
from typing import Optional

from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    extract_archive,
)
from urllib.parse import urlparse
import urllib
from torch.utils.model_zoo import tqdm


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return
    

def setup_logger():
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_abs_path(rel_path):
    root_dir = os.path.dirname(os.path.abspath(__file__))  #library_root就是root_dir
    return os.path.join(root_dir, rel_path)


def is_main_process():
    return get_rank() == 0


def get_cache_path(rel_path):
    # Copyright (c) 2022, salesforce.com, inc.
    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_cfg = OmegaConf.load(os.path.join(root_dir, "configs", "default.yaml"))

    # registry.register_path("library_root", root_dir)
    repo_root = os.path.join(root_dir, "..")
    # registry.register_path("repo_root", repo_root)
    cache_root = os.path.join(root_dir, default_cfg.env.cache_root)
    # registry.register_path("cache_root", cache_root)
    return os.path.expanduser(os.path.join(cache_root, rel_path))


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def apply_to_sample(f, sample):
    ## add check for datasets that return none samples for missing items
    if sample == None or len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def reorg_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()

    # reorganize by split
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()


def get_cache_root():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_cfg = OmegaConf.load(os.path.join(root_dir, "configs", "default.yaml"))
    # repo_root = os.path.join(root_dir, "..")
    cache_root = os.path.join(root_dir, default_cfg.env.cache_root)
    return cache_root


def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        shutil.rmtree(dir)
    logging.info(f"Deleted contents of directory: {dir}")


def is_convertible_to_int(value):
    return bool(re.match(r'^-?\d+$', str(value)))

#####################################################
def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> None:
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
                                  If None, use the basename of the URL.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir(root)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # expand redirect chain if needed
    url = get_redirected_url(url)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead."
                " Downloading " + url + " to " + fpath
            )
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "vissl"})
        ) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


def to_google_drive_download_url(view_url: str) -> str:
    """
    Utility function to transform a view URL of google drive
    to a download URL for google drive
    Example input:
        https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view
    Example output:
        https://drive.google.com/uc?export=download&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp
    """
    splits = view_url.split("/")
    assert splits[-1] == "view"
    file_id = splits[-2]
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        print(f"Error creating directory: {dir_path}")
    return is_success


def get_redirected_url(url: str):
    """
    Given a URL, returns the URL it redirects to or the
    original URL in case of no indirection
    """
    import requests

    with requests.Session() as session:
        with session.get(url, stream=True, allow_redirects=True) as response:
            if response.history:
                return response.url
            else:
                return url
