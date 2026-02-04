# Real-Time Abnormal Activity Detection

This repo is a starter baseline for real-time abnormal activity detection from CCTV or drone video. We begin with a strong, simple baseline for fight detection (RWF-2000) and a clean data/model pipeline that we can extend to multi-stream (RGB + pose) and temporal attention models later.

## What is included
- Clean project layout with configs, data loaders, and training scripts
- RWF-2000 dataset loader
- Baseline 3D CNN model (`r3d_18` from torchvision)
- Train/eval loop with accuracy reporting

## Quick start

1. Create a Python environment and install deps.

```bash
pip install -r requirements.txt
```

2. Download and extract the RWF-2000 dataset.

Expected folder structure:

```
RWF-2000/
  train/
    Fight/
    NonFight/
  test/
    Fight/
    NonFight/
```

3. Point the config to the dataset path and run training.

```bash
python scripts/train.py --config configs/rwf2000_baseline.yaml --data-root /path/to/RWF-2000
```

4. Optional: Verify dataset layout.

```bash
python scripts/verify_dataset.py --data-root /path/to/RWF-2000
```

## Next steps (we will do together)
- Add pose stream (OpenPose or HRNet) and fusion
- Add temporal transformer block (TimeSformer-like)
- Add anomaly datasets (XD-Violence / UCF-Crime)
- Real-time inference and deployment profiling

## Repo layout
```
configs/
  rwf2000_baseline.yaml
scripts/
  train.py
  eval.py
  verify_dataset.py
src/
  data/
  models/
  utils/
```
