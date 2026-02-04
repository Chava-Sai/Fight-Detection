import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.video import VIDEO_EXTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def count_videos(path):
    if not os.path.isdir(path):
        return 0
    return sum(
        1
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(VIDEO_EXTS)
    )


def main():
    args = parse_args()
    root = args.data_root
    splits = ["train", "test"]
    classes = ["NonFight", "Fight"]

    print(f"Checking RWF-2000 at: {root}")
    for split in splits:
        for cls in classes:
            path = os.path.join(root, split, cls)
            count = count_videos(path)
            status = "OK" if count > 0 else "MISSING"
            print(f"{split}/{cls}: {count} videos ({status})")


if __name__ == "__main__":
    main()
