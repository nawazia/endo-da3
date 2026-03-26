"""
Hamlyn daVinci stereo dataset — Stage 2a depth training with RAFT-Stereo pseudo-GT.

Reference: Ye et al., "Self-Supervised Siamese Learning on Stereo Image Pairs for
           Depth Estimation in Robotic Surgery." Hamlyn Symposium 2017.
           https://hamlyn.doc.ic.ac.uk/vision/

Data layout:
    <root>/train/image_0/000001.png  ...  (left frames, rectified)
    <root>/train/image_1/000001.png  ...  (right frames, rectified)
    <root>/train/depth_pseudogt/000001.npy  ...  (float16, metres, from RAFT-Stereo)
    <root>/test/  (same structure)

Camera params (both cameras post-rectification, fixed for all sequences):
    fx = fy = 373.47833  px = 182.91805  py = 113.72999
    Baseline B = 5.63117 mm → depth = fx * B / disparity = 2.103 / disparity

Stereo geometry:
    Left camera  : world frame origin  (c2w = I)
    Right camera : pure x-translation  (c2w[:3,3] = [B, 0, 0])
    Both cameras share the same K after rectification.

Image size: 384×192 (W×H). Loader resizes short side to img_size and center-crops
            to img_size×img_size, adjusting K accordingly.

Each sample is a stereo pair (S=2):
    images : (2, 3, H, W)  — [left, right]
    depths : (2, H, W)     — [RAFT pseudo-GT for left, zeros for right]
    K      : (3, 3)        — shared intrinsics
    c2w    : (2, 4, 4)     — [I, [I|B·x̂]]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from endo_da3.data.base import EndoDepthDataset

# ---------------------------------------------------------------------------
# Camera intrinsics (fixed for all Hamlyn daVinci sequences)
# ---------------------------------------------------------------------------
_FX = _FY = 373.47833252
_CX = 182.91804504
_CY = 113.72999573
_W  = 384
_H  = 192
_B  = 5.63117313e-3   # stereo baseline in metres (|tx| from Hamlyn extrinsics)

# Stereo c2w matrices (fixed, built once)
_C2W_LEFT  = np.eye(4, dtype=np.float32)
_C2W_RIGHT = np.eye(4, dtype=np.float32)
_C2W_RIGHT[0, 3] = _B   # right camera centre at x=+B in left-camera world frame


def _build_K() -> np.ndarray:
    return np.array([
        [_FX,  0., _CX],
        [ 0., _FY, _CY],
        [ 0.,  0.,  1.],
    ], dtype=np.float64)


def _adjust_K(K: np.ndarray, orig_w: int, orig_h: int, img_size: int) -> np.ndarray:
    """
    Adjust K for:
      1. Resize: short side → img_size  (preserves aspect ratio)
      2. CenterCrop: img_size × img_size (crops the longer dimension)
    """
    scale = img_size / min(orig_h, orig_w)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    K = K.copy()
    K[0, 0] *= scale          # fx
    K[1, 1] *= scale          # fy
    K[0, 2] *= scale          # cx
    K[1, 2] *= scale          # cy

    # CenterCrop: offset applied to the larger dimension
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HamlynDataset(EndoDepthDataset):
    """
    Stereo pairs from the Hamlyn daVinci dataset.

    Each sample is a left+right stereo pair at the same timestamp (S=2).
    Left frame has RAFT-Stereo pseudo-GT depth; right frame depth is zeros
    (masked out in all loss terms that require valid depth).

    Splits:
        train : <root>/train/  (34,240 frames)
        val   : last 10% of test split (~720 frames)
        test  : first 90% of <root>/test/ (~6,471 frames)

    Args:
        root     : Path to Hamlyn daVinci root (contains train/ and test/).
        split    : "train" | "val" | "test"
        img_size : Resize/crop to this square size (default 336).
    """

    name = "hamlyn"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 336,
    ):
        self.root     = Path(root)
        self.img_size = img_size
        self._tf      = _img_transform(img_size)

        self.K = torch.from_numpy(
            _adjust_K(_build_K(), _W, _H, img_size)
        ).float()

        # Fixed stereo c2w (same for every sample)
        self._c2w = torch.from_numpy(
            np.stack([_C2W_LEFT, _C2W_RIGHT])   # (2, 4, 4)
        )

        # Resolve split → data directory and frame list
        if split == "train":
            data_dir = self.root / "train"
        else:
            data_dir = self.root / "test"

        img0_dir = data_dir / "image_0"
        all_stems = sorted(p.stem for p in img0_dir.glob("*.png"))

        if split == "val":
            cut = int(len(all_stems) * 0.9)
            stems = all_stems[cut:]
        elif split == "test":
            cut = int(len(all_stems) * 0.9)
            stems = all_stems[:cut]
        else:
            stems = all_stems

        self._samples: list[tuple[str, Path]] = [
            (stem, data_dir) for stem in stems
        ]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        stem, data_dir = self._samples[idx]

        img0_dir  = data_dir / "image_0"
        img1_dir  = data_dir / "image_1"
        depth_dir = data_dir / "depth_pseudogt"

        # Left image
        left_img  = self._tf(Image.open(img0_dir / f"{stem}.png").convert("RGB"))
        # Right image
        right_img = self._tf(Image.open(img1_dir / f"{stem}.png").convert("RGB"))

        # Left depth — apply same resize+center-crop as the image transform
        depth_np = np.load(depth_dir / f"{stem}.npy").astype(np.float32)  # (H, W)
        depth_t  = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        scale    = self.img_size / min(_H, _W)
        new_h    = round(_H * scale)
        new_w    = round(_W * scale)
        depth_t  = F.interpolate(
            depth_t, size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        )
        crop_left = (new_w - self.img_size) // 2
        depth_t   = depth_t[:, :, :, crop_left: crop_left + self.img_size]
        left_depth = depth_t.squeeze()   # (H, W)

        return {
            "images": torch.stack([left_img, right_img]),          # (2, 3, H, W)
            "depths": torch.stack([left_depth,
                                   torch.zeros_like(left_depth)]), # (2, H, W)
            "K":      self.K,                                       # (3, 3)
            "c2w":    self._c2w,                                    # (2, 4, 4)
        }
