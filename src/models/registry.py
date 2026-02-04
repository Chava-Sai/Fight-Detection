from typing import Dict

from .baseline import build_r3d18, build_swin3d_t

MODEL_REGISTRY: Dict[str, object] = {
    "r3d18": build_r3d18,
    "swin3d_t": build_swin3d_t,
}


def build_model(cfg: dict):
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    return MODEL_REGISTRY[name](
        num_classes=cfg.get("num_classes", 2),
        pretrained=cfg.get("pretrained", False),
    )
