import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.registry import build_dataset
from src.models.registry import build_model
from src.utils.config import load_config
from src.utils.metrics import accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    inputs, targets = zip(*batch)
    return torch.stack(inputs, dim=0), torch.tensor(targets)


def evaluate(model, loader, device):
    model.eval()
    running_acc = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            if batch is None:
                continue
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            running_acc += accuracy(logits, targets)

    return running_acc / max(len(loader), 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.device:
        cfg["eval"]["device"] = args.device

    device = torch.device(cfg["eval"]["device"])
    pin_memory = device.type == "cuda"

    test_ds = build_dataset(cfg["data"], split="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin_memory,
        collate_fn=_collate_skip_none,
    )

    model = build_model(cfg["model"]).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"])

    acc = evaluate(model, test_loader, device)
    print(f"test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
