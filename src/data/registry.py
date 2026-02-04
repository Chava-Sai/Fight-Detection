from typing import Dict

from .rwf2000 import RWF2000Dataset
from .xd_violence import XDViolenceDataset

DATASET_REGISTRY: Dict[str, object] = {
    "rwf2000": RWF2000Dataset,
    "xd_violence": XDViolenceDataset,
}


def build_dataset(cfg: dict, split: str):
    name = cfg["dataset"]
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")

    dataset_cls = DATASET_REGISTRY[name]
    return dataset_cls(
        root=cfg["data_root"],
        split=split,
        num_frames=cfg["num_frames"],
        img_size=cfg["img_size"],
        mean=cfg["mean"],
        std=cfg["std"],
        flip_prob=cfg["flip_prob"],
    )
