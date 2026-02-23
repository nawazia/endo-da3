"""
PolypSense3D — Virtual subset for Stage 1 depth training.

Reference: Zhang et al., "PolypSense3D: A Multi-Source Benchmark Dataset for
           Depth-Aware Polyp Size Measurement in Endoscopy"
           NeurIPS 2025  https://doi.org/10.7910/DVN/LKDIEK

Data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LKDIEK
      Download Virtual-Dataset-For-PolypSense3D.7z and extract to <dest>/.

Layout after extraction (root = <dest>):
    root/Virtual Dataset For PolypSense3D/depth_estimation/
        camera.txt                          9 space-separated floats (3×3 K, row-major)
        position_rotation.csv               (tX,tY,tZ,rX,rY,rZ,rW,time)
        images/image_NNNN.jpg               320×320 RGB (8241 frames)
        depths/aov_image_NNNN.png           320×320 RGBA uint8 (R = depth in mm)

Depth encoding:
    depth_metres = R_channel / 1000.0
    R = 0 → invalid pixel (background); masked to 0 in output.
    Range: ~0 – 25.5 cm  |  mean ≈ 6–7 cm

Pose convention:
    CSV columns: tX, tY, tZ (camera position, Unity metres), rX, rY, rZ, rW (quat x,y,z,w)
    Unity left-handed (Y-up) → right-handed OpenCV with Y-flip (same as EndoSLAM).
    Absolute position ~7–8 m from world origin (arbitrary Unity placement).
    → Poses are made RELATIVE to the first frame of each window.

Splits (temporal, single sequence):
    train : first 80 % of frames (~6 591 windows at seq_len=2, stride=1)
    val   : last  20 % of frames (~1 647 windows)

Camera intrinsics (K, for 320×320):
    fx = fy = 200.0,  cx = 161.93,  cy = 163.17
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision import transforms

from endo_da3.data.base import EndoDepthDataset


# ── constants ─────────────────────────────────────────────────────────────────

_ORIG_SIZE   = 320
_DEPTH_SCALE = 1.0 / 1000.0        # R channel (mm) → metres
_DATA_SUBDIR = "Virtual Dataset For PolypSense3D/depth_estimation"

# Unity left-handed → OpenCV right-handed: flip Y axis (same as EndoSLAM)
_TM = np.array([[1,  0, 0, 0],
                [0, -1, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1]], dtype=np.float64)

_TRAIN_FRAC = 0.80


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_K(cam_txt: Path, orig: int, target: int) -> np.ndarray:
    """Parse cam.txt (9 space-separated floats, row-major 3×3) and scale to target."""
    vals = list(map(float, cam_txt.read_text().split()))
    K = np.array(vals, dtype=np.float64).reshape(3, 3)
    s = target / orig
    K[0, 0] *= s; K[1, 1] *= s   # fx, fy
    K[0, 2] *= s; K[1, 2] *= s   # cx, cy
    return K


def _load_poses(csv_path: Path) -> np.ndarray:
    """
    Return (N, 4, 4) float64 c2w matrices in right-handed coordinates.

    CSV columns: tX, tY, tZ, rX, rY, rZ, rW, time(s)
    The published CSV has a truncated last line; invalid rows are silently dropped.
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                         invalid_raise=False)
    data = data[~np.isnan(data).any(axis=1)]

    tx, ty, tz = data[:, 0], data[:, 1], data[:, 2]
    qx, qy, qz, qw = data[:, 3], data[:, 4], data[:, 5], data[:, 6]

    quats  = np.stack([qx, qy, qz, qw], axis=1)
    R_mats = Rotation.from_quat(quats).as_matrix()

    c2w = np.zeros((len(data), 4, 4), dtype=np.float64)
    c2w[:, :3, :3] = R_mats
    c2w[:, 0, 3] = tx
    c2w[:, 1, 3] = ty
    c2w[:, 2, 3] = tz
    c2w[:, 3, 3] = 1.0

    # Unity LH → OpenCV RH
    return np.einsum("ij,njk,kl->nil", _TM, c2w, _TM)


def _relative_c2w(c2w_abs: np.ndarray) -> np.ndarray:
    """(S, 4, 4) absolute → relative; c2w_rel[0] = I."""
    return np.einsum("ij,sjk->sik", np.linalg.inv(c2w_abs[0]), c2w_abs)


def _load_depth(path: Path) -> np.ndarray:
    """Return (H, W) float32 depth in metres (R / 1000); invalid pixels = 0."""
    arr = np.array(Image.open(path))          # RGBA uint8
    return arr[..., 0].astype(np.float32) * _DEPTH_SCALE


def _img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── dataset ───────────────────────────────────────────────────────────────────

class PolypSense3DVirtualDataset(EndoDepthDataset):
    """
    PolypSense3D virtual (Unity) subset — single colon sequence.

    Temporal split: first 80 % → train, last 20 % → val.

    Args:
        root      : Path to the extracted PolypSense3D root
                    (must contain "Virtual Dataset For PolypSense3D/").
        split     : "train" | "val"
        img_size  : Resize to this square resolution (default 336).
        seq_len   : Consecutive frames per sample (default 2).
        stride    : Step between consecutive frames (default 1).
        with_pose : Return c2w matrices (relative, right-handed).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 336,
        seq_len: int = 2,
        stride: int = 1,
        with_pose: bool = False,
    ):
        self.root      = Path(root)
        self.img_size  = img_size
        self.seq_len   = seq_len
        self.stride    = stride
        self.with_pose = with_pose
        self._tf = _img_transform(img_size)

        data_dir   = self.root / _DATA_SUBDIR
        frames_dir = data_dir / "images"
        depths_dir = data_dir / "depths"

        K = _load_K(data_dir / "camera.txt", _ORIG_SIZE, img_size)

        # Sorted frame indices
        frame_nums = sorted(
            int(p.stem.split("_")[-1])
            for p in frames_dir.glob("image_*.jpg")
        )

        poses = None
        if with_pose:
            poses  = _load_poses(data_dir / "position_rotation.csv")
            n_valid = min(len(frame_nums), len(poses))
        else:
            n_valid = len(frame_nums)

        # Temporal 80/20 split
        split_idx = int(n_valid * _TRAIN_FRAC)
        valid_range = range(0, split_idx) if split == "train" else range(split_idx, n_valid)

        # Sliding windows
        window = (seq_len - 1) * stride + 1
        self._samples: list[tuple] = []
        for start in range(len(valid_range) - window + 1):
            idxs = [valid_range.start + start + j * stride for j in range(seq_len)]
            self._samples.append((frames_dir, depths_dir, K, poses, idxs))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        frames_dir, depths_dir, K, poses, frame_idxs = self._samples[idx]

        images, depths = [], []
        for fi in frame_idxs:
            img_path   = frames_dir / f"image_{fi:04d}.jpg"
            depth_path = depths_dir / f"aov_image_{fi:04d}.png"

            images.append(self._tf(Image.open(img_path).convert("RGB")))

            d_raw = torch.from_numpy(_load_depth(depth_path)).unsqueeze(0).unsqueeze(0)
            d_res = F.interpolate(d_raw, size=(self.img_size, self.img_size),
                                  mode="bilinear", align_corners=False).squeeze()
            depths.append(d_res)

        out = {
            "images": torch.stack(images),          # (S, 3, H, W)
            "depths": torch.stack(depths),           # (S, H, W)
            "K":      torch.from_numpy(K).float(),   # (3, 3)
        }

        if self.with_pose and poses is not None:
            c2w_abs = np.stack([poses[fi] for fi in frame_idxs])
            out["c2w"] = torch.from_numpy(_relative_c2w(c2w_abs)).float()

        return out
