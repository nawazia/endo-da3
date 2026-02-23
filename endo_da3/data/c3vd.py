"""
C3VD v2 dataset — consecutive-frame pairs for Stage 1 depth training.

Reference: Bobrow et al., "Colonoscopy 3D Video Dataset with Paired Depth from
           2D-3D Registration" (C3VD) — https://arxiv.org/abs/2206.01990

Directory structure (after download_c3vd.sh):
    {root}/{split}/{phantom}_{segment}_{texture}_{view}/
        rgb/NNNN.png              1350×1080 colour frames
        depth/NNNN_depth.tiff    16-bit [0,65535] → [0,100] mm
        pose.txt                 N×16 comma-sep row-major 4×4 (translation mm)

Camera: Scaramuzza omnidirectional (camera_intrinsics.txt at dataset root).
Pose  : stored as [R | 0 ; t^T | 1] (row-vector convention, mm).
        Converted to standard c2w:  c2w[:3,:3] = R^T,  c2w[:3,3] = t / 1000.

Split (segment-level, frozen before any training):
    train : c1 + c2 — ascending, transverse1, descending, sigmoid1, sigmoid2
    val   : c1 + c2 — cecum, rectum, transverse2   (held-out shapes)
    test  : c0 (v1 format — different layout, no pose.txt; handled separately)

Camera-model note:
    The shared camera_intrinsics.txt describes a Scaramuzza omnidirectional
    model.  For Stage 1 we use an effective pinhole approximation built from
    the first polynomial coefficient a0 (≈ focal length at image centre).
    This introduces a ray-direction error that grows from 0° at the centre to
    roughly 10° at the corners of the 336×336 crop.  This is acceptable for
    depth-feature learning; replace with the full Scaramuzza unprojection for
    Stage 2+ metric tasks.
    TODO: implement _scaramuzza_ray_grid() and pass as 'ray_grid' to da3_loss.

Coordinate-system note:
    Poses are stored in a right-handed coordinate system (det(R) ≈ +1) with
    translations in mm.  The Scaramuzza model uses image-plane coords (y-down),
    consistent with OpenCV convention.  No additional axis flip is applied.
    TODO: verify against a ground-truth point-cloud reconstruction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from endo_da3.data.base import EndoDepthDataset


# ── constants ────────────────────────────────────────────────────────────────

_ORIG_W, _ORIG_H = 1350, 1080      # native C3VD v2 resolution
_CROP = 1080                        # centre-crop to 1080×1080 (square)
_CROP_X0 = (_ORIG_W - _CROP) // 2  # 135 pixels removed from left
_DEPTH_SCALE = 0.1 / 65535.0       # raw uint16 → metres  (0–100 mm range)


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_intrinsics(txt: Path) -> dict:
    """Return dict of Scaramuzza omnidirectional camera parameters."""
    params: dict = {}
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            params[k.strip()] = float(v.strip())
    return params


def _approx_K(params: dict, target_size: int) -> np.ndarray:
    """
    Build an effective pinhole 3×3 intrinsic matrix from Scaramuzza params.

    Steps:
      1. Centre-crop 1350×1080 → 1080×1080: shift cx by _CROP_X0.
      2. Resize 1080 → target_size: scale cx, cy, and a0 by target_size / 1080.

    Approximation error ≤ ~10° at image corners; acceptable for Stage 1.
    """
    scale = target_size / _CROP
    cx = (params["cx"] - _CROP_X0) * scale
    cy = params["cy"] * scale
    f  = params["a0"] * scale
    return np.array([[f, 0.0, cx],
                     [0.0, f, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _load_poses(pose_txt: Path) -> np.ndarray:
    """
    Parse pose.txt: N rows of 16 comma-separated floats.

    Stored format:  [R | 0 ; t^T | 1]  (row-vector convention, mm).
    Returned:       (N, 4, 4) c2w in metres, column-vector OpenCV convention.
        c2w[:3,:3] = R^T   (rotation)
        c2w[:3, 3] = t / 1000.0   (translation in metres)
    """
    raw = np.loadtxt(str(pose_txt), delimiter=",")
    if raw.ndim == 1:
        raw = raw[np.newaxis]
    M = raw.reshape(-1, 4, 4)          # (N, 4, 4) stored matrices

    c2w = np.zeros_like(M)
    c2w[:, :3, :3] = M[:, :3, :3].transpose(0, 2, 1)   # R^T
    c2w[:, :3,  3] = M[:,  3, :3] / 1000.0             # mm → m
    c2w[:,  3,  3] = 1.0
    return c2w


def _img_transform(img_size: int):
    """ImageNet-normalised transform: resize shorter side → crop to square."""
    return transforms.Compose([
        transforms.Resize(img_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _load_depth(path: Path, target_size: int) -> torch.Tensor:
    """
    Load a 16-bit depth TIFF, centre-crop 1350×1080 → 1080×1080,
    resize to target_size×target_size, scale to metres.

    Returns (target_size, target_size) float32 depth in metres.
    """
    raw = np.array(Image.open(path), dtype=np.int32)      # (H, W)
    # Centre crop horizontally
    raw = raw[:, _CROP_X0 : _CROP_X0 + _CROP]             # (1080, 1080)
    depth_m = (raw * _DEPTH_SCALE).astype(np.float32)

    t = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    t = torch.nn.functional.interpolate(
        t, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    ).squeeze()                                                  # (H, W)
    return t


# ── dataset ──────────────────────────────────────────────────────────────────

class C3VDDataset(EndoDepthDataset):
    """
    C3VD v2 dataset for Stage 1 depth training.

    Split (segment-level, frozen):
        train : ascending, transverse1, descending, sigmoid1, sigmoid2
                (c1 + c2, v1 + v2)
        val   : cecum, rectum, transverse2  (held-out shapes)
                (c1 + c2, v1 only)
        test  : c0 sequences (v1 format — separate class; not used here)

    Args:
        root      : Path to the C3VD root (contains train/, val/, test/,
                    camera_intrinsics.txt).
        split     : "train" | "val"
        img_size  : Resize images to this square size (default 336).
        seq_len   : Consecutive frames per sample (default 2).
        stride    : Step between frames (default 1).
    """

    name = "C3VD"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 336,
        seq_len: int = 2,
        stride: int = 1,
        **_kwargs,   # absorb unused kwargs (e.g. with_pose from factory)
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.seq_len = seq_len
        self.stride = stride
        self._tf = _img_transform(img_size)

        # Shared intrinsics
        cam_params = _parse_intrinsics(self.root / "camera_intrinsics.txt")
        K = _approx_K(cam_params, img_size)
        self._K = torch.from_numpy(K).float()               # (3, 3)

        # Collect sequence directories for this split
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"C3VD split directory not found: {split_dir}")

        seq_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())

        # Build sliding-window sample index
        self._samples: list[tuple] = []

        for seq_dir in seq_dirs:
            rgb_dir   = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            pose_file = seq_dir / "pose.txt"

            if not (rgb_dir.exists() and depth_dir.exists() and pose_file.exists()):
                # c0 (test) sequences have a different layout; skip silently
                continue

            poses   = _load_poses(pose_file)         # (N_pose, 4, 4)
            n_pose  = len(poses)
            n_rgb   = len(sorted(rgb_dir.glob("*.png")))
            n_frames = min(n_rgb, n_pose)            # guard against mismatch

            if n_rgb != n_pose:
                import warnings
                warnings.warn(
                    f"C3VD: {seq_dir.name} has {n_rgb} rgb frames but "
                    f"{n_pose} poses — using first {n_frames}.",
                    stacklevel=2,
                )

            window = (seq_len - 1) * stride + 1
            for start in range(n_frames - window + 1):
                idxs = [start + j * stride for j in range(seq_len)]
                self._samples.append((seq_dir, poses, idxs))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        seq_dir, poses, frame_idxs = self._samples[idx]

        images, depths = [], []
        for fi in frame_idxs:
            rgb_path   = seq_dir / "rgb"   / f"{fi:04d}.png"
            depth_path = seq_dir / "depth" / f"{fi:04d}_depth.tiff"

            images.append(self._tf(Image.open(rgb_path).convert("RGB")))
            depths.append(_load_depth(depth_path, self.img_size))

        c2w = np.stack([poses[fi] for fi in frame_idxs])   # (S, 4, 4)

        return {
            "images": torch.stack(images),                      # (S, 3, H, W)
            "depths": torch.stack(depths),                      # (S, H, W)
            "K":      self._K,                                  # (3, 3)
            "c2w":    torch.from_numpy(c2w).float(),            # (S, 4, 4)
        }
