"""
EndoSLAM synthetic dataset — UnityCam subset for Stage 1 depth training.

Reference: Ozyoruk et al., "EndoSLAM Dataset and An Unsupervised Monocular
           Visual Odometry and Depth Estimation Approach for Endoscopic Videos"
           Medical Image Analysis 2021, https://arxiv.org/abs/2105.01929

Data: https://data.mendeley.com/datasets/cd2rtzm23r/1
      Extract with:  bash scripts/extract_endoslam.sh --zip EndoSLAM.zip --dest /path/to/EndoSLAM

Layout after extraction (root = /path/to/EndoSLAM):
    root/UnityCam/
        Calibration/cam.txt                 9 comma-separated floats (3×3 K, row-major)
        Colon/
            Poses/colon_position_rotation.csv   (tX,tY,tZ,rX,rY,rZ,rW,time)
            Frames/image_NNNN.png               320×320 RGB
            Pixelwise Depths/aov_image_NNNN.png 320×320 RGBA (R channel = depth in cm)
        Small Intestine/   (same layout, intestine_position_rotation.csv)
        Stomach/           (same layout, stomach_position_rotation.csv)

Depth encoding:
    depth_metres = R_channel / 100.0
    Range: ~1–63 cm (Colon), ~1–70 cm (Stomach), ~1–50 cm (Small Intestine)

Pose convention:
    CSV columns: tX, tY, tZ (camera position, Unity metres), rX, rY, rZ, rW (quaternion x,y,z,w)
    Unity is left-handed (Y-up); converted to right-handed OpenCV with Y-flip.
    Absolute camera position is ~9 m from the world origin (arbitrary Unity scene placement).
    → Poses are MADE RELATIVE to the first frame of each window to keep DA3 scale
      normalisation comparable across datasets (scale ≈ mean depth ≈ 0.1 m).

Splits:
    Scene-level temporal split (80 / 20 by frame index):
    train : first 80 % of each scene (Colon 17 509 | Intestine 10 041 | Stomach 1 235)
    val   : last  20 % of each scene (Colon  4 377 | Intestine  2 510 | Stomach   308)
    → train ≈ 28 782 windows  |  val ≈ 7 192 windows  (seq_len=2, stride=1)

Camera intrinsics (K, for 320×320):
    fx = 156.04,  fy = 155.75,  cx = 178.56,  cy = 181.80
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

_ORIG_SIZE    = 320                  # native image resolution (square)
_DEPTH_SCALE  = 1.0 / 100.0         # R channel (cm) → metres

# Unity left-handed → OpenCV right-handed: flip Y axis
_TM = np.array([[1,  0, 0, 0],
                [0, -1, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1]], dtype=np.float64)

_SCENES = {
    "Colon":           "colon_position_rotation.csv",
    "Small Intestine": "intestine_position_rotation.csv",
    "Stomach":         "stomach_position_rotation.csv",
}

_TRAIN_FRAC = 0.80


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_K(cam_txt: Path, orig: int, target: int) -> np.ndarray:
    """Parse cam.txt (9 comma-separated floats, row-major 3×3) and scale to target."""
    vals = list(map(float, cam_txt.read_text().split(",")))
    K = np.array(vals, dtype=np.float64).reshape(3, 3)
    s = target / orig
    K[0, 0] *= s; K[1, 1] *= s   # fx, fy
    K[0, 2] *= s; K[1, 2] *= s   # cx, cy
    return K


def _load_poses(csv_path: Path) -> np.ndarray:
    """
    Return (N, 4, 4) float64 c2w matrices in right-handed coordinates.

    CSV columns: tX, tY, tZ, rX, rY, rZ, rW, time(s)
    Conversion:
      1. Build Unity c2w_lh from quaternion (x,y,z,w) and translation (metres).
      2. Apply Y-flip: c2w_rh = TM @ c2w_lh @ TM   (Unity LH → OpenCV RH).

    Note: the published CSVs have a truncated last line — use genfromtxt with
    invalid_raise=False to silently drop any rows that cannot be parsed.
    """
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                         invalid_raise=False)          # (N, 8)
    # Drop any rows where genfromtxt emitted NaN due to a short/bad line
    data = data[~np.isnan(data).any(axis=1)]
    tx, ty, tz = data[:, 0], data[:, 1], data[:, 2]
    qx, qy, qz, qw = data[:, 3], data[:, 4], data[:, 5], data[:, 6]

    quats = np.stack([qx, qy, qz, qw], axis=1)      # (N, 4) xyzw
    R_mats = Rotation.from_quat(quats).as_matrix()   # (N, 3, 3)

    c2w = np.zeros((len(data), 4, 4), dtype=np.float64)
    c2w[:, :3, :3] = R_mats
    c2w[:, 0, 3] = tx
    c2w[:, 1, 3] = ty
    c2w[:, 2, 3] = tz
    c2w[:, 3, 3] = 1.0

    # Convert each pose: c2w_rh = TM @ c2w_lh @ TM
    c2w_rh = np.einsum("ij,njk,kl->nil", _TM, c2w, _TM)
    return c2w_rh


def _relative_c2w(c2w_abs: np.ndarray) -> np.ndarray:
    """
    Make poses relative to the first frame.

    c2w_abs : (S, 4, 4)
    Returns  : (S, 4, 4)  where c2w_rel[0] = I
    """
    c2w_ref_inv = np.linalg.inv(c2w_abs[0])
    return np.einsum("ij,sjk->sik", c2w_ref_inv, c2w_abs)


def _load_depth(path: Path) -> np.ndarray:
    """Return (H, W) float32 depth in metres (R channel / 100)."""
    arr = np.array(Image.open(path))   # RGBA uint8
    return (arr[..., 0].astype(np.float32)) * _DEPTH_SCALE


def _img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── dataset ───────────────────────────────────────────────────────────────────

class EndoSLAMSynthDataset(EndoDepthDataset):
    """
    EndoSLAM synthetic (UnityCam) subset — Colon, Small Intestine, Stomach.

    Scene-level temporal split: first 80 % → train, last 20 % → val.

    Args:
        root      : Path to the extracted EndoSLAM root
                    (must contain UnityCam/ with the layout above).
        split     : "train" | "val"
        img_size  : Resize to this square resolution (default 336).
        seq_len   : Consecutive frames per sample S (default 2).
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
        self.root     = Path(root)
        self.img_size = img_size
        self.seq_len  = seq_len
        self.stride   = stride
        self.with_pose = with_pose
        self._tf = _img_transform(img_size)

        unity_dir = self.root / "UnityCam"
        K = _load_K(unity_dir / "Calibration" / "cam.txt", _ORIG_SIZE, img_size)

        self._samples: list[tuple] = []   # (frames_dir, depths_dir, K, poses_or_None, [idxs])

        for scene_name, pose_csv_name in _SCENES.items():
            scene_dir  = unity_dir / scene_name
            frames_dir = scene_dir / "Frames"
            depths_dir = scene_dir / "Pixelwise Depths"

            if not frames_dir.exists() or not depths_dir.exists():
                import warnings
                warnings.warn(f"EndoSLAM: {scene_name} not found at {scene_dir} — skipping.")
                continue

            # Sorted frame numbers (numeric sort, not alphabetical)
            frame_nums = sorted(
                int(p.stem.split("_")[-1])
                for p in frames_dir.glob("image_*.png")
            )

            # Load poses and determine valid range
            poses = None
            if with_pose:
                poses = _load_poses(scene_dir / "Poses" / pose_csv_name)
                n_valid = min(len(frame_nums), len(poses))
            else:
                n_valid = len(frame_nums)

            # Temporal split (80/20 by frame index in the valid range)
            split_idx = int(n_valid * _TRAIN_FRAC)
            if split == "train":
                valid_range = range(0, split_idx)
            else:
                valid_range = range(split_idx, n_valid)

            # Sliding windows within the split
            window = (seq_len - 1) * stride + 1
            for start in range(len(valid_range) - window + 1):
                idxs = [valid_range.start + start + j * stride for j in range(seq_len)]
                self._samples.append((frames_dir, depths_dir, K, poses, idxs))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        frames_dir, depths_dir, K, poses, frame_idxs = self._samples[idx]

        images, depths = [], []
        for fi in frame_idxs:
            rgb_path   = frames_dir / f"image_{fi:04d}.png"
            # Handle 5-digit frame numbers (fi >= 10000)
            if not rgb_path.exists():
                rgb_path = frames_dir / f"image_{fi:05d}.png"
            depth_path = depths_dir / f"aov_image_{fi:04d}.png"
            if not depth_path.exists():
                depth_path = depths_dir / f"aov_image_{fi:05d}.png"

            images.append(self._tf(Image.open(rgb_path).convert("RGB")))

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
            c2w_abs = np.stack([poses[fi] for fi in frame_idxs])   # (S, 4, 4)
            c2w_rel = _relative_c2w(c2w_abs)                        # (S, 4, 4), [0]=I
            out["c2w"] = torch.from_numpy(c2w_rel).float()

        return out
