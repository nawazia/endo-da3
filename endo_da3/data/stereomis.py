"""
StereoMIS dataset loader — Stage 2a depth training with RAFT-Stereo pseudo-GT.

Reference: Hayoz et al., "Learning how to robustly estimate camera pose in
           endoscopic videos." IJCARS 2023. https://zenodo.org/records/8154924

Data layout (after extract_stereomis.py):
    <root>/P1/left/000000.jpg    ...  (left frames, rectified, 640×512)
    <root>/P1/right/000000.jpg   ...  (right frames, rectified, 640×512)
    <root>/P1/depth_pseudogt/000000.npy  ...  (float16, metres, from RAFT-Stereo)
    <root>/P1/rectified_K.json   (fx, fy, cx, cy after rectification)
    <root>/P1/baseline.txt       (B in metres)

Stereo geometry:
    Left camera  : world frame origin  (c2w = I)
    Right camera : pure x-translation  (c2w[:3,3] = [B, 0, 0])
    Both cameras share K after rectification (per sequence, not fixed).

Training split: P1 only (P2_*/P3 are test sequences).
Image size: 640×512 (W×H). Loader resizes short side to img_size and
            center-crops to img_size×img_size, adjusting K accordingly.

Each sample is a stereo pair (S=2):
    images : (2, 3, H, W)  — [left, right]
    depths : (2, H, W)     — [RAFT pseudo-GT for left, zeros for right]
    K      : (3, 3)        — shared intrinsics (adjusted for resize+crop)
    c2w    : (2, 4, 4)     — [I, [I|B·x̂]]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from endo_da3.data.base import EndoDepthDataset

# Training sequence — P2_*/P3 are test only
TRAIN_SEQ = "P1"
_ORIG_W, _ORIG_H = 640, 512


def _adjust_K(K: np.ndarray, orig_w: int, orig_h: int, img_size: int) -> np.ndarray:
    """Adjust K for Resize(short_side→img_size) + CenterCrop(img_size×img_size)."""
    scale = img_size / min(orig_h, orig_w)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    K = K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale

    crop_x = (new_w - img_size) / 2.0
    crop_y = (new_h - img_size) / 2.0
    K[0, 2] -= crop_x
    K[1, 2] -= crop_y

    return K


def _img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class StereoMISDataset(EndoDepthDataset):
    """
    Single StereoMIS sequence as stereo pairs for Stage 2a training.

    Args:
        root     : Path to extracted StereoMIS root (contains P1/, P2_0/, etc.)
        seq      : Sequence name (default "P1" — the only training sequence)
        img_size : Square crop size (default 336)
    """

    name = "stereomis"

    def __init__(
        self,
        root: str | Path,
        seq: str = TRAIN_SEQ,
        img_size: int = 336,
    ):
        self.root     = Path(root)
        self.seq_dir  = self.root / seq
        self.img_size = img_size
        self._tf      = _img_transform(img_size)

        # Per-sequence calibration
        with open(self.seq_dir / "rectified_K.json") as f:
            kdata = json.load(f)
        K_raw = np.array([
            [kdata["fx"],  0.,          kdata["cx"]],
            [0.,           kdata["fy"], kdata["cy"]],
            [0.,           0.,          1.         ],
        ], dtype=np.float64)
        self.K = torch.from_numpy(
            _adjust_K(K_raw, _ORIG_W, _ORIG_H, img_size)
        ).float()

        with open(self.seq_dir / "baseline.txt") as f:
            B = float(f.read().strip())

        # Fixed stereo c2w for this sequence
        c2w_left  = np.eye(4, dtype=np.float32)
        c2w_right = np.eye(4, dtype=np.float32)
        c2w_right[0, 3] = B
        self._c2w = torch.from_numpy(np.stack([c2w_left, c2w_right]))  # (2,4,4)

        # Frame list
        self._left_dir  = self.seq_dir / "left"
        self._right_dir = self.seq_dir / "right"
        self._depth_dir = self.seq_dir / "depth_pseudogt"

        self._stems = sorted(p.stem for p in self._left_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self._stems[idx]

        left_img  = self._tf(Image.open(self._left_dir  / f"{stem}.jpg").convert("RGB"))
        right_img = self._tf(Image.open(self._right_dir / f"{stem}.jpg").convert("RGB"))

        # Depth: pseudo-GT for left only, apply same resize+centercrop as image
        depth_np = np.load(self._depth_dir / f"{stem}.npy").astype(np.float32)
        depth_t  = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        scale    = self.img_size / min(_ORIG_H, _ORIG_W)
        new_h    = round(_ORIG_H * scale)
        new_w    = round(_ORIG_W * scale)
        depth_t  = F.interpolate(depth_t, size=(new_h, new_w),
                                 mode="bilinear", align_corners=False)
        crop_left = (new_w - self.img_size) // 2
        crop_top  = (new_h - self.img_size) // 2
        depth_t   = depth_t[:, :, crop_top:crop_top + self.img_size,
                                   crop_left:crop_left + self.img_size]
        left_depth = depth_t.squeeze()  # (H, W)

        return {
            "images": torch.stack([left_img, right_img]),           # (2, 3, H, W)
            "depths": torch.stack([left_depth,
                                   torch.zeros_like(left_depth)]),  # (2, H, W)
            "K":      self.K,                                        # (3, 3)
            "c2w":    self._c2w,                                     # (2, 4, 4)
        }
