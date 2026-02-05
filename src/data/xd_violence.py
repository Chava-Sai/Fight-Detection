import os
from typing import List, Tuple

from torch.utils.data import Dataset
import random

from .transforms import apply_video_transforms
from .video import VIDEO_EXTS, read_video_frames


class XDViolenceDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        num_frames: int,
        img_size: int,
        mean: List[float],
        std: List[float],
        flip_prob: float,
    ) -> None:
        self.root = root
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.flip_prob = flip_prob

        # Based on observed Kaggle structure:
        # root/{train,test}/{Fighting,Normal,Shooting}/
        self.class_names = ["Normal", "Fighting", "Shooting"]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        split_dir = os.path.join(self.root, self.split)
        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(VIDEO_EXTS):
                    path = os.path.join(class_dir, fname)
                    label = self.class_to_idx[class_name]
                    samples.append((path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        for _ in range(3):
            path, label = self.samples[index]
            try:
                frames = read_video_frames(path, self.num_frames, train=self.split == "train")
                break
            except Exception:
                index = random.randint(0, len(self.samples) - 1)
        else:
            return None
        frames = apply_video_transforms(
            frames,
            train=self.split == "train",
            img_size=self.img_size,
            mean=self.mean,
            std=self.std,
            flip_prob=self.flip_prob,
        )
        return frames, label
