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
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    inputs, targets = zip(*batch)
    return torch.stack(inputs, dim=0), torch.tensor(targets)


def train_one_epoch(model, loader, criterion, optimizer, device, log_interval):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if batch is None:
            continue
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(logits, targets)

        if log_interval and (step + 1) % log_interval == 0:
            avg_loss = running_loss / (step + 1)
            avg_acc = running_acc / (step + 1)
            tqdm.write(f"step {step + 1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    return running_loss / max(len(loader), 1), running_acc / max(len(loader), 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            if batch is None:
                continue
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item()
            running_acc += accuracy(logits, targets)

    return running_loss / max(len(loader), 1), running_acc / max(len(loader), 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.device:
        cfg["train"]["device"] = args.device
        cfg["eval"]["device"] = args.device
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size

    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg["train"]["device"])
    pin_memory = device.type == "cuda"

    train_ds = build_dataset(cfg["data"], split="train")
    test_ds = build_dataset(cfg["data"], split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin_memory,
        collate_fn=_collate_skip_none,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=pin_memory,
        collate_fn=_collate_skip_none,
    )

    model = build_model(cfg["model"]).to(device)

    class_weights = None
    if hasattr(train_ds, "samples") and hasattr(train_ds, "class_to_idx"):
        counts = [0] * len(train_ds.class_to_idx)
        for _, label in train_ds.samples:
            counts[label] += 1
        weights = [0.0 if c == 0 else 1.0 / c for c in counts]
        class_weights = torch.tensor(weights, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg["train"]["log_interval"],
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"epoch {epoch + 1}/{cfg['train']['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict(), "acc": best_acc}, "checkpoints/best.pt")


if __name__ == "__main__":
    main()
