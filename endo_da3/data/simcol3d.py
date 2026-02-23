"""
SimCol3D dataset — consecutive-frame pairs for Stage 1 depth training.

Reference: Rau et al., "SimCol3D -- 3D Reconstruction during Colonoscopy Challenge"
           https://arxiv.org/abs/2307.11261

Depth encoding  : 16-bit greyscale PNG, [0, 65535] → [0, 20 cm] → divide by 65535 * 0.20 = metres
Camera convention: original poses are in Unity's left-handed system; converted to right-handed
                   by the Y-flip TM = diag(1, -1, 1, 1):  c2w_rh = TM @ c2w_lh @ TM
Intrinsics       : cam.txt stores K as 9 space-separated floats (row-major 3×3)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision import transforms
from endo_da3.data.base import EndoDepthDataset


# ── constants ────────────────────────────────────────────────────────────────
_ORIG_SIZE = 475          # native SimCol3D image resolution (square)
_DEPTH_SCALE = 0.20 / 65535.0   # raw uint16 → metres  (20 cm max)
_TM = np.array([[1, 0, 0, 0],   # Y-flip: Unity left-handed → OpenCV right-handed
                [0,-1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float64)


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_K(cam_txt: Path) -> np.ndarray:
    """Return 3×3 float64 intrinsic matrix from cam.txt."""
    vals = list(map(float, cam_txt.read_text().split()))
    return np.array(vals, dtype=np.float64).reshape(3, 3)


def _scale_K(K: np.ndarray, orig: int, target: int) -> np.ndarray:
    """Scale intrinsics from origxorig to targetxtarget."""
    s = target / orig
    K = K.copy()
    K[0, 0] *= s   # fx
    K[1, 1] *= s   # fy
    K[0, 2] *= s   # cx
    K[1, 2] *= s   # cy
    return K


def _load_poses(scene_dir: Path, seq_prefix: str, seq_idx: int) -> np.ndarray:
    """
    Return (N, 4, 4) float64 c2w matrices in right-handed coordinates, metres.

    SimCol3D world space is in centimetres; translations are divided by 100
    so they are consistent with depths (loaded in metres via _DEPTH_SCALE).

    seq_prefix : 'S', 'B', or 'O'
    seq_idx    : integer index (e.g. 1, 2, …)
    """
    tag = f"{seq_prefix}{seq_idx}"
    translations = np.loadtxt(scene_dir / f"SavedPosition_{tag}.txt") / 100.0  # cm → m
    quats        = np.loadtxt(scene_dir / f"SavedRotationQuaternion_{tag}.txt")  # (N, 4) xyzw

    R_mats = Rotation.from_quat(quats).as_matrix()   # (N, 3, 3)

    c2w_list = []
    for i in range(len(translations)):
        P = np.eye(4, dtype=np.float64)
        P[:3, :3] = R_mats[i]
        P[:3,  3] = translations[i]
        P_rh = _TM @ P @ _TM   # convert left-handed → right-handed
        c2w_list.append(P_rh)

    return np.stack(c2w_list)   # (N, 4, 4)


def _load_depth(path: Path) -> np.ndarray:
    """Return (H, W) float32 depth in metres."""
    raw = np.array(Image.open(path), dtype=np.int32)   # PIL I;16 → int32
    return (raw * _DEPTH_SCALE).astype(np.float32)


def _img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── dataset ──────────────────────────────────────────────────────────────────

class SimCol3DDataset(EndoDepthDataset):
    """
    Splits (defined once and frozen):
        train : Colon_I  S1-S3, S6-S8, S11-S13   (9 seqs)
                Colon_II B1-B3, B6-B8, B11-B13   (9 seqs) — 18 seqs total
        val   : Colon_I  S4, S9, S14
                Colon_II B4, B9, B14              —  6 seqs  (held out from train)
        test  : Colon_I  S5, S10, S15
                Colon_II B5, B10, B15
                Colon_III O1-O3                   —  9 seqs (official MICCAI, untouched)

    Val is carved from the official train sequences (scene-level, not frame-level).
    The official test set is never used during training or checkpoint selection.

    Yields consecutive S-frame windows from SimCol3D sequences.

    Args:
        root      : Path to the SimCol3D root (contains SyntheticColon_I, …).
        split     : "train" | "val" | "test" — see split table above.
        img_size  : Resize images to this square size (default 336).
        seq_len   : Number of consecutive frames per sample S (default 2).
        stride    : Step between consecutive frames (default 1).
        with_pose : If True, also return c2w matrices (S, 4, 4).
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
        self.root = Path(root)
        self.img_size = img_size
        self.seq_len = seq_len
        self.stride = stride
        self.with_pose = with_pose
        self._tf = _img_transform(img_size)

        # val sequences are carved from the official train file (last seq in each triplet)
        _VAL_SEQS = {
            "SyntheticColon_I":  {"Frames_S4",  "Frames_S9",  "Frames_S14"},
            "SyntheticColon_II": {"Frames_B4",  "Frames_B9",  "Frames_B14"},
        }

        if split == "test":
            split_file = self.root / "misc" / "test_file.txt"
            all_paths = [l.strip().rstrip("/") for l in split_file.read_text().splitlines() if l.strip()]
            seq_rel_paths = all_paths
        else:
            split_file = self.root / "misc" / "train_file.txt"
            all_paths = [l.strip().rstrip("/") for l in split_file.read_text().splitlines() if l.strip()]
            if split == "val":
                seq_rel_paths = [p for p in all_paths
                                 if Path(p).name in _VAL_SEQS.get(Path(p).parent.name, set())]
            else:  # "train"
                seq_rel_paths = [p for p in all_paths
                                 if Path(p).name not in _VAL_SEQS.get(Path(p).parent.name, set())]

        # Build index: list of (seq_dir, K, poses_or_None, [frame_indices])
        self._samples: list[tuple] = []

        for rel in seq_rel_paths:
            parts = rel.split("/")                # e.g. ["SyntheticColon_I", "Frames_S1"]
            scene_name, frames_name = parts[0], parts[1]
            scene_dir = self.root / scene_name
            frames_dir = scene_dir / frames_name

            if not frames_dir.exists():
                continue

            K = _scale_K(_load_K(scene_dir / "cam.txt"), _ORIG_SIZE, img_size)

            # Infer prefix and index from folder name e.g. "Frames_S1" → prefix='S', idx=1
            m = re.match(r"Frames_([A-Z]+)(\d+)$", frames_name)
            seq_prefix, seq_idx = m.group(1), int(m.group(2))

            poses = _load_poses(scene_dir, seq_prefix, seq_idx) if with_pose else None

            # Sorted frame indices
            frame_paths = sorted(frames_dir.glob("FrameBuffer_*.png"))
            n_frames = len(frame_paths)

            # Sliding window of seq_len frames with given stride
            window = (seq_len - 1) * stride + 1
            for start in range(n_frames - window + 1):
                idxs = [start + j * stride for j in range(seq_len)]
                self._samples.append((frames_dir, K, poses, idxs))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        frames_dir, K, poses, frame_idxs = self._samples[idx]

        images, depths = [], []
        for fi in frame_idxs:
            rgb_path   = frames_dir / f"FrameBuffer_{fi:04d}.png"
            depth_path = frames_dir / f"Depth_{fi:04d}.png"

            images.append(self._tf(Image.open(rgb_path).convert("RGB")))
            depth_raw = torch.from_numpy(_load_depth(depth_path)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            depth_resized = torch.nn.functional.interpolate(
                depth_raw, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            ).squeeze()  # (H,W)
            depths.append(depth_resized)

        out = {
            "images": torch.stack(images),   # (S, 3, H, W)
            "depths": torch.stack(depths),   # (S, H, W)
            "K":      torch.from_numpy(K).float(),  # (3, 3)
        }

        if self.with_pose and poses is not None:
            c2w = np.stack([poses[fi] for fi in frame_idxs])  # (S, 4, 4)
            out["c2w"] = torch.from_numpy(c2w).float()

        return out
