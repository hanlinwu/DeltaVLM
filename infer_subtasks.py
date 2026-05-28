"""Smoke-test inference for every ChangeChat-105k sub-task.

For each annotation file under dataset/annotations/changechat_105k_test*.json,
load N samples, run model.predict_answers, and dump per-sub-task JSON results.

Usage:
    python infer_subtasks.py --cfg_path configs/evaluate.yaml [--n_samples 4]
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from processor import BlipImageEvalProcessor
from task import ImageTextPretrainTask
from utils import init_distributed_mode, setup_logger


ANNO_ROOT = "dataset/annotations"
IMG_ROOT = "dataset/images"

# (filename, label, prompt-source). For test.json the prompt is fixed; for the
# other files we pull the prompt from conversations[0]['value'].
SUBTASKS = [
    ("changechat_105k_test.json", "caption", "fixed"),
    ("changechat_105k_test_binary.json", "binary", "from_conv"),
    ("changechat_105k_test_count.json", "count", "from_conv"),
    ("changechat_105k_test_loc.json", "loc", "from_conv"),
    ("changechat_105k_test_open.json", "open", "from_conv"),
    ("changechat_105k_test_dialog.json", "dialog", "from_conv"),
]

DEFAULT_CAPTION_PROMPT = "Please briefly describe the changes in these two images."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default="configs/evaluate.yaml")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="How many examples to run per sub-task.")
    parser.add_argument("--out_dir", default="output/subtask_smoke")
    parser.add_argument("--options", nargs="+", default=None)
    return parser.parse_args()


def load_samples(path, n):
    with open(path, "r") as f:
        data = json.load(f)
    return data[: n]


def build_inputs(record, kind, processor, device):
    """Return a samples dict suitable for model.predict_answers."""
    if kind == "caption":
        path_a = os.path.join(IMG_ROOT, record["image_A"])
        path_b = os.path.join(IMG_ROOT, record["image_B"])
        prompt = DEFAULT_CAPTION_PROMPT
        gt = record.get("captions", [])
        img_id = record["image_A"].split("/")[-1].rstrip(".png").split("_")[-1]
    else:
        path_a = os.path.join(IMG_ROOT, record["image"][0])
        path_b = os.path.join(IMG_ROOT, record["image"][1])
        # For dialog: first turn only — multi-turn conditioning would require
        # custom plumbing; we just probe whether the model produces an answer.
        prompt = record["conversations"][0]["value"].replace("<image>", "").strip()
        gt = [t["value"] for t in record["conversations"] if t["from"] == "gpt"]
        img_id = record.get("id", "")

    img_a = Image.open(path_a).convert("RGB")
    img_b = Image.open(path_b).convert("RGB")
    img_a, img_b = processor(img_a, img_b)

    samples = {
        "image_A": img_a.unsqueeze(0).to(device),
        "image_B": img_b.unsqueeze(0).to(device),
        "text_input": [prompt],
        "prompt": [prompt],
    }
    return samples, prompt, gt, img_id


def main():
    args = parse_args()

    cfg_args = SimpleNamespace(cfg_path=args.cfg_path, options=args.options)
    cfg = Config(cfg_args)
    # Force single-process / no DDP for a smoke test.
    cfg.run_cfg.distributed = False
    cfg.run_cfg.world_size = 1

    init_distributed_mode(cfg.run_cfg)
    setup_logger()
    cfg.pretty_print()

    task = ImageTextPretrainTask.setup_task(cfg=cfg)
    model = task.build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"[infer] model on {device}; checkpoint = {cfg.model_cfg.pretrained}")

    processor = BlipImageEvalProcessor(image_size=224)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for fname, label, _ in SUBTASKS:
        path = os.path.join(ANNO_ROOT, fname)
        if not os.path.isfile(path):
            print(f"[infer] SKIP {label}: {path} missing")
            continue
        records = load_samples(path, args.n_samples)
        print(f"\n===== sub-task: {label} ({len(records)} samples from {fname}) =====")
        results = []
        for rec in records:
            try:
                samples, prompt, gt, img_id = build_inputs(rec, label, processor, device)
            except FileNotFoundError as e:
                print(f"[infer] missing image, skip: {e}")
                continue
            with torch.no_grad():
                pred = model.predict_answers(samples, num_beams=1, max_len=128, min_len=1)
            pred_text = pred[0] if isinstance(pred, list) else pred
            print(f"  id={img_id}")
            print(f"  prompt = {prompt}")
            print(f"  pred   = {pred_text}")
            if gt:
                print(f"  gt     = {gt[0] if len(gt) == 1 else gt}")
            results.append({
                "id": img_id,
                "prompt": prompt,
                "prediction": pred_text,
                "ground_truth": gt,
            })
        out_path = os.path.join(args.out_dir, f"{label}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        summary[label] = {"n": len(results), "out": out_path}
        print(f"[infer] wrote {out_path}")

    print("\n===== summary =====")
    for k, v in summary.items():
        print(f"  {k}: n={v['n']:>3d}  ->  {v['out']}")


if __name__ == "__main__":
    main()
