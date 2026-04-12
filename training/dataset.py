"""
SurfaceIdiot - Grasp Dataset

Loads all episodes under data/<object>/<episode>/ and produces
(image, joint_history, target_joints) tuples for behavior cloning.

Input feature at each step t:
    image         : wrist-view RGB  (3×224×224)  — what the robot sees
    joint_history : finger angles at [t-H, …, t-1]  (H×5)

Target:
    joint angles at t  (5,)  — the glove reading we want to reproduce

The dataset uses a sliding window over each episode so every frame
(after the first H) becomes an independent training sample.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_DIM = len(FINGERS)  # 5


def _load_metadata(ep_dir: Path) -> Optional[dict]:
    meta_path = ep_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def _extract_joints(frame_meta: dict) -> Optional[np.ndarray]:
    """Pull the 5-dim finger vector out of one metadata frame entry."""
    try:
        f = frame_meta["glove"]["fingers"]
        return np.array([f[k] for k in FINGERS], dtype=np.float32)
    except (KeyError, TypeError):
        return None


def build_image_transform(train: bool = True, img_size: int = 224) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize((img_size + 16, img_size + 16)),
            T.RandomCrop(img_size),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomHorizontalFlip(p=0.1),   # slight augmentation; hand is mostly symmetric
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


class GraspSample:
    """Lightweight index entry — no images loaded at construction time."""
    __slots__ = ("ep_dir", "frame_idx", "frames_meta")

    def __init__(self, ep_dir: Path, frame_idx: int, frames_meta: list):
        self.ep_dir      = ep_dir
        self.frame_idx   = frame_idx
        self.frames_meta = frames_meta


class GraspDataset(Dataset):
    """
    Args:
        data_root   : path to the top-level data/ directory
        objects     : list of object names to include, or None for all
        history_len : number of past frames (H) fed as joint history
        train       : if True, applies data augmentation
        img_size    : spatial size of resized image (both sides)
        camera      : "wrist" or "side" — which camera view to use as input
    """

    def __init__(
        self,
        data_root: str = "data",
        objects: Optional[List[str]] = None,
        history_len: int = 6,
        train: bool = True,
        img_size: int = 224,
        camera: str = "wrist",
    ):
        self.data_root    = Path(data_root)
        self.history_len  = history_len
        self.transform    = build_image_transform(train, img_size)
        self.camera       = camera
        self.samples: List[GraspSample] = []

        self._scan(objects)

    def _scan(self, objects: Optional[List[str]]):
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        obj_dirs = sorted(self.data_root.iterdir())
        if objects:
            obj_dirs = [d for d in obj_dirs if d.name in objects]

        for obj_dir in obj_dirs:
            if not obj_dir.is_dir():
                continue
            for ep_dir in sorted(obj_dir.iterdir()):
                meta = _load_metadata(ep_dir)
                if meta is None:
                    continue
                frames = meta.get("frames", [])
                if len(frames) < self.history_len + 1:
                    continue
                # Validate that all frames have joint data and images
                valid_frames = []
                for fm in frames:
                    joints = _extract_joints(fm)
                    if joints is None:
                        continue
                    valid_frames.append(fm)

                # Sliding window: every frame index i ≥ history_len is a sample
                for i in range(self.history_len, len(valid_frames)):
                    self.samples.append(GraspSample(ep_dir, i, valid_frames))

        print(f"GraspDataset: {len(self.samples)} samples "
              f"from {self.data_root} (camera='{self.camera}')")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        i      = sample.frame_idx
        frames = sample.frames_meta
        ep_dir = sample.ep_dir

        # ── Wrist-view image at frame i ──
        img_path = ep_dir / f"{self.camera}_{i:06d}.jpg"
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Return a black image if the file is missing (shouldn't happen after scan)
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        image = self.transform(img)

        # ── Joint history [t-H … t-1] ──
        history = np.stack([
            _extract_joints(frames[j])
            for j in range(i - self.history_len, i)
        ])  # shape: (H, 5)
        joint_history = torch.from_numpy(history)  # float32

        # ── Target: joint angles at frame i ──
        target = torch.from_numpy(_extract_joints(frames[i]))  # (5,)

        return image, joint_history, target


# ─── Quick dataset inspection ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../data")
    parser.add_argument("--history_len", type=int, default=6)
    args = parser.parse_args()

    ds = GraspDataset(args.data_root, history_len=args.history_len, train=False)
    if len(ds) == 0:
        print("No samples found. Collect data first.")
    else:
        img, hist, tgt = ds[0]
        print(f"Image shape    : {tuple(img.shape)}")
        print(f"History shape  : {tuple(hist.shape)}")
        print(f"Target shape   : {tuple(tgt.shape)}")
        print(f"Target (sample): {tgt.numpy()}")
