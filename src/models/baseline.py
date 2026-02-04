import torch.nn as nn
from torchvision.models.video import r3d_18

try:
    from torchvision.models.video import swin3d_t
    HAS_SWIN3D = True
except Exception:
    HAS_SWIN3D = False


def build_r3d18(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    if pretrained:
        try:
            model = r3d_18(weights="DEFAULT")
        except TypeError:
            model = r3d_18(pretrained=True)
    else:
        try:
            model = r3d_18(weights=None)
        except TypeError:
            model = r3d_18(pretrained=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_swin3d_t(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    if not HAS_SWIN3D:
        raise ImportError("swin3d_t is not available in this torchvision version.")

    if pretrained:
        try:
            model = swin3d_t(weights="DEFAULT")
        except TypeError:
            model = swin3d_t(pretrained=True)
    else:
        try:
            model = swin3d_t(weights=None)
        except TypeError:
            model = swin3d_t(pretrained=False)

    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
