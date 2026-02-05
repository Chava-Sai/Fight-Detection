import random
from typing import List

import numpy as np
import torch

try:
    import decord

    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    import cv2

    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def _sample_indices(num_total: int, num_frames: int, train: bool) -> List[int]:
    if num_total <= 0:
        return [0] * num_frames

    if num_total >= num_frames:
        if train:
            start = random.randint(0, num_total - num_frames)
            return list(range(start, start + num_frames))
        return np.linspace(0, num_total - 1, num_frames).astype(int).tolist()

    return np.linspace(0, num_total - 1, num_frames).astype(int).tolist()


def read_video_frames(path: str, num_frames: int, train: bool) -> torch.Tensor:
    if HAS_DECORD:
        try:
            vr = decord.VideoReader(path)
            idx = _sample_indices(len(vr), num_frames, train)
            frames = vr.get_batch(idx)
            return frames
        except Exception as exc:
            raise RuntimeError(f"Failed to read video with decord: {path}") from exc

    if not HAS_CV2:
        raise RuntimeError("Neither decord nor opencv-python is available for video loading.")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = _sample_indices(total, num_frames, train)
    frames = []
    got_any = False
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        got_any = True
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not got_any or not frames:
        raise RuntimeError(f"Failed to read frames from video: {path}")
    return torch.from_numpy(np.stack(frames, axis=0))
