import random
from typing import List

import torch
import torch.nn.functional as F


def apply_video_transforms(
    frames: torch.Tensor,
    train: bool,
    img_size: int,
    mean: List[float],
    std: List[float],
    flip_prob: float,
) -> torch.Tensor:
    # frames: (T, H, W, C), uint8
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

    frames = F.interpolate(
        frames, size=(img_size, img_size), mode="bilinear", align_corners=False
    )

    if train and flip_prob > 0 and random.random() < flip_prob:
        frames = torch.flip(frames, dims=[3])

    mean_t = torch.tensor(mean, device=frames.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=frames.device).view(1, 3, 1, 1)
    frames = (frames - mean_t) / std_t

    frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
    return frames
