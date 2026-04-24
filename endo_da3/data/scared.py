"""
SCARED dataset loader — Stage 3 training with structured-light GT depth.

Reference: Allan et al., "Stereo correspondence and reconstruction of
           endoscopic data challenge." MICCAI 2021.
           https://endovissub2019-scared.grand-challenge.org/

Keyframe data layout (after tools/extract_scared.py):
    <root>/extracted/dataset_N/keyframe_K/
        K_left.json       # left intrinsics
        K_right.json      # right intrinsics
        stereo.json       # R, T (metres), baseline_m
        left.png          # left keyframe image  (1024×1280)
        right.png         # right keyframe image
        depth_left.npy    # float32 (H,W), metres, 0=invalid
        depth_right.npy

Video frame data layout (after tools/extract_scared_video.py):
    <root>/extracted/dataset_N/keyframe_K/video/
        K_rect.json       # rectified left intrinsics (fx,fy,cx,cy,width,height)
        stereo_rect.json  # R=I, T=[baseline,0,0], baseline_m
        XXXXXX/
            left.png      # rectified left frame  (1024×1280)
            right.png     # rectified right frame
            depth_left.npy  # float16 RAFT pseudo-GT, metres, 0=invalid

Stereo geometry:
    keyframe : c2w_right = [R^T | -R^T @ T]  where R,T from endoscope_calibration.yaml
    video    : c2w_right = [I | T]            where T=[baseline,0,0] (rectified, R=I)

Training split:  dataset_1–3, 6–7  (train=True; 4&5 excluded: calibration errors per Allan2021)
Validation split: dataset_8–9  (train=False)

Each sample is a stereo pair (S=2):
    images : (2, 3, H, W)
    depths : (2, H, W)    — left: GT/pseudo-GT depth; right: zeros for video, GT for keyframes
    K      : (3, 3)       — left-camera intrinsics, adjusted for resize+crop
    c2w    : (2, 4, 4)    — [I, right-camera pose in left frame]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from endo_da3.data.base import EndoDepthDataset

_ORIG_H, _ORIG_W = 1024, 1280

TRAIN_DATASETS = [1, 2, 3, 6, 7]     # dataset_1–3, 6–7 (4&5 excluded: calibration errors)
VAL_DATASETS   = [8, 9]              # dataset_8, dataset_9


def _adjust_K(K: np.ndarray, orig_h: int, orig_w: int, img_size: int) -> np.ndarray:
    scale  = img_size / min(orig_h, orig_w)
    new_h  = round(orig_h * scale)
    new_w  = round(orig_w * scale)
    K = K.copy()
    K[0, 0] *= scale;  K[1, 1] *= scale
    K[0, 2] *= scale;  K[1, 2] *= scale
    K[0, 2] -= (new_w - img_size) / 2.0
    K[1, 2] -= (new_h - img_size) / 2.0
    return K


def _img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _resize_crop_depth(depth_np: np.ndarray, img_size: int) -> torch.Tensor:
    depth_np = np.nan_to_num(depth_np.astype(np.float32), nan=0.0)
    t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    scale  = img_size / min(_ORIG_H, _ORIG_W)
    new_h  = round(_ORIG_H * scale)
    new_w  = round(_ORIG_W * scale)
    t = F.interpolate(t, size=(new_h, new_w), mode="nearest")
    cy = (new_h - img_size) // 2
    cx = (new_w - img_size) // 2
    return t[0, 0, cy:cy + img_size, cx:cx + img_size]   # (H,W)


class SCAREDDataset(EndoDepthDataset):
    """
    SCARED stereo pairs with depth supervision.

    Args:
        root     : Path to SCARED root (contains extracted/ subdirectory)
        train    : True → dataset_1–7, False → dataset_8–9
        video    : False → keyframes (structured-light GT, both depths)
                   True  → video frames (RAFT pseudo-GT, left depth only)
        img_size : Square output size (default 336)
    """

    name = "scared"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        video: bool = False,
        img_size: int = 336,
    ):
        self.root     = Path(root)
        self.img_size = img_size
        self.video    = video
        self._tf      = _img_transform(img_size)

        ds_ids    = TRAIN_DATASETS if train else VAL_DATASETS
        extracted = self.root / "extracted"

        self._samples: list[Path] = []
        for ds_id in ds_ids:
            ds_dir = extracted / f"dataset_{ds_id}"
            if not ds_dir.exists():
                continue
            for kf_dir in sorted(ds_dir.glob("keyframe_*"),
                                  key=lambda p: int(p.name.split("_")[1])):
                if video:
                    vid_dir = kf_dir / "video"
                    if not (vid_dir / "K_rect.json").exists():
                        continue
                    for frame_dir in sorted(vid_dir.glob("[0-9]*")):
                        if (frame_dir / "depth_left.npy").exists():
                            self._samples.append(frame_dir)
                else:
                    if (kf_dir / "left.png").exists():
                        self._samples.append(kf_dir)

        if not self._samples:
            kind = "video frames" if video else "keyframes"
            raise FileNotFoundError(
                f"No SCARED {kind} found under {extracted}. "
                f"Run tools/extract_scared{'_video' if video else ''}.py first."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]

        if self.video:
            return self._getitem_video(sample)
        else:
            return self._getitem_keyframe(sample)

    def _getitem_keyframe(self, kf: Path) -> dict:
        left_img  = self._tf(Image.open(kf / "left.png" ).convert("RGB"))
        right_img = self._tf(Image.open(kf / "right.png").convert("RGB"))

        with open(kf / "K_left.json") as f:
            kd = json.load(f)
        K_raw = np.array([
            [kd["fx"], 0.,       kd["cx"]],
            [0.,       kd["fy"], kd["cy"]],
            [0.,       0.,       1.      ],
        ], dtype=np.float64)
        K = torch.from_numpy(
            _adjust_K(K_raw, _ORIG_H, _ORIG_W, self.img_size)
        ).float()

        d_left  = _resize_crop_depth(np.load(kf / "depth_left.npy"),  self.img_size)
        d_right = _resize_crop_depth(np.load(kf / "depth_right.npy"), self.img_size)

        with open(kf / "stereo.json") as f:
            st = json.load(f)
        R = np.array(st["R"], dtype=np.float64)
        T = np.array(st["T"], dtype=np.float64)

        c2w_left  = np.eye(4, dtype=np.float32)
        c2w_right = np.eye(4, dtype=np.float32)
        c2w_right[:3, :3] = R.T
        c2w_right[:3,  3] = -R.T @ T

        return {
            "images": torch.stack([left_img, right_img]),
            "depths": torch.stack([d_left, d_right]),
            "K":      K,
            "c2w":    torch.from_numpy(np.stack([c2w_left, c2w_right])).float(),
        }

    def _getitem_video(self, frame_dir: Path) -> dict:
        vid_dir = frame_dir.parent   # video/

        left_img  = self._tf(Image.open(frame_dir / "left.png" ).convert("RGB"))
        right_img = self._tf(Image.open(frame_dir / "right.png").convert("RGB"))

        with open(vid_dir / "K_rect.json") as f:
            kd = json.load(f)
        K_raw = np.array([
            [kd["fx"], 0.,       kd["cx"]],
            [0.,       kd["fy"], kd["cy"]],
            [0.,       0.,       1.      ],
        ], dtype=np.float64)
        K = torch.from_numpy(
            _adjust_K(K_raw, _ORIG_H, _ORIG_W, self.img_size)
        ).float()

        d_left = _resize_crop_depth(np.load(frame_dir / "depth_left.npy"), self.img_size)
        d_right = torch.zeros_like(d_left)

        with open(vid_dir / "stereo_rect.json") as f:
            st = json.load(f)
        baseline_m = st["baseline_m"]

        c2w_left  = np.eye(4, dtype=np.float32)
        c2w_right = np.eye(4, dtype=np.float32)
        c2w_right[0, 3] = baseline_m   # R=I, T=[baseline,0,0]

        return {
            "images": torch.stack([left_img, right_img]),
            "depths": torch.stack([d_left, d_right]),
            "K":      K,
            "c2w":    torch.from_numpy(np.stack([c2w_left, c2w_right])).float(),
        }
