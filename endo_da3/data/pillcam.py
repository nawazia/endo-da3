"""
PillCam SB3 capsule endoscopy dataset — EndoSLAM ex-vivo subset for Stage 2b.

Reference: Ozyoruk et al., "EndoSLAM Dataset and An Unsupervised Monocular
           Visual Odometry and Depth Estimation Approach for Endoscopic Videos."
           Medical Image Analysis 2021.

Data layout (after extracting Cameras/PillCam.zip):
    <root>/PillCam/Calibration/cam1.txt   (front camera C1)
    <root>/PillCam/Calibration/cam2.txt   (rear  camera C2)
    <root>/PillCam/Colon-I(L-shaped)/TumorfreeTrajectory_1/Frames/C1/01111OutRGB.bmp
    <root>/PillCam/Colon-I(L-shaped)/TumorfreeTrajectory_1/Frames/C2/01112OutRGB.bmp
    <root>/PillCam/Colon-I(L-shaped)/TumorfreeTrajectory_1/Poses/test1
    ...

Cameras: PillCam SB3 dual-capsule endoscope.
  C1 (front): fx≈74.2, fy≈74.2, cx≈130.0, cy≈130.0, k1≈0.199, k2≈-0.128
  C2 (rear) : fx≈76.1, fy≈76.1, cx≈130.9, cy≈130.9, k1≈0.199, k2≈-0.132
  Image size : 256×256 (square BMP)
  C1 frames have odd global indices; C2 frames have even global indices.
  C1 and C2 face opposite directions → treated as independent sequences.
  Distortion: pincushion (k1>0). Undistortion breaks tissue structure at this
  magnitude — images are loaded raw and K_raw is used directly.

Poses: EM tracker at ~1 kHz, space-delimited file with derivative columns
  interleaved.  Relevant columns (0-indexed):
    t(0)  px(1)  py(3)  pz(5)  Qx(37)  Qy(39)  Qz(41)  Qw(43)
  Evenly subsampled to one pose per frame.

Each sample is a sliding window of `seq_len` consecutive frames from the same
(trajectory, camera) pair, returned with relative poses (c2w[0] = I).
No GT depth — used with Stage 2b photometric loss only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from endo_da3.data.base import EndoDepthDataset


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_cam_txt(path: Path) -> dict:
    cal = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            cal[k.strip()] = float(v.strip())
    return cal


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion → 3×3 rotation matrix (float32)."""
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float32)


def _load_poses(pose_file: Path, n_frames: int) -> np.ndarray:
    """
    Parse space-delimited PillCam pose file and subsample to n_frames poses.

    The EM tracker outputs ~1 kHz data with derivative columns interleaved.
    Relevant column indices (0-based):
        t=0  px=1  py=3  pz=5  Qx=37  Qy=39  Qz=41  Qw=43

    Returns (n_frames, 4, 4) float32 c2w matrices.
    """
    _COLS = [0, 1, 3, 5, 37, 39, 41, 43]   # t, px, py, pz, Qx, Qy, Qz, Qw

    rows = []
    for line in pose_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) <= 43:
            continue
        try:
            rows.append([float(parts[c]) for c in _COLS])
        except ValueError:
            continue  # header / non-numeric

    rows = np.array(rows, dtype=np.float64)   # (N_pose, 8)

    # evenly subsample EM-tracker poses to match frame count
    indices = np.round(np.linspace(0, len(rows) - 1, n_frames)).astype(int)
    rows = rows[indices]

    c2w_list = []
    for row in rows:
        R = _quat_to_rot(*row[4:8])           # Qx Qy Qz Qw
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3,  3] = row[1:4].astype(np.float32)   # px py pz
        c2w_list.append(c2w)

    return np.stack(c2w_list)   # (n_frames, 4, 4)


def _adjust_K(K: np.ndarray, orig_w: int, orig_h: int, img_size: int) -> np.ndarray:
    """Adjust K for Resize(img_size) + CenterCrop(img_size)."""
    scale = img_size / min(orig_h, orig_w)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    K = K.copy()
    K[0, 0] *= scale; K[1, 1] *= scale
    K[0, 2] *= scale; K[1, 2] *= scale
    K[0, 2] -= (new_w - img_size) / 2.0
    K[1, 2] -= (new_h - img_size) / 2.0
    return K


def _img_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _build_K(cal: dict) -> np.ndarray:
    return np.array([
        [cal["fx"],       0., cal["cx"]],
        [      0., cal["fy"], cal["cy"]],
        [      0.,       0.,        1.],
    ], dtype=np.float64)


# ── dataset ───────────────────────────────────────────────────────────────────

class PillCamDataset(EndoDepthDataset):
    """
    PillCam SB3 capsule endoscopy trajectories from the EndoSLAM ex-vivo dataset.

    C1 (front) and C2 (rear) cameras face opposite directions and have separate
    calibrations — they are collected as independent sequences.  Each sample is
    a sliding window of `seq_len` consecutive frames from the same
    (trajectory, camera) pair, with relative poses from the EM tracker
    (c2w[0] = I).  No GT depth — returns zeros in the 'depths' field.

    Images are loaded without undistortion: the pincushion distortion (k1≈+0.2)
    correction warps tissue structure too aggressively at this magnitude.
    K_raw is used directly and adjusted for Resize + CenterCrop.

    Used for Stage 2b self-supervised training with photometric reprojection loss.

    Args:
        root     : EndoSLAM root directory (contains PillCam/ subdirectory).
        img_size : Square output size after Resize + CenterCrop (default 336).
        seq_len  : Number of consecutive frames per sample (default 2).
        stride   : Step between windows (default 1).
        cameras  : Which cameras to include: "both" | "C1" | "C2" (default "both").
    """

    name = "pillcam"

    _ORIG_W = _ORIG_H = 256   # PillCam native resolution

    def __init__(
        self,
        root: str | Path,
        img_size: int = 336,
        seq_len: int = 2,
        stride: int = 1,
        cameras: str = "both",
    ):
        self.root     = Path(root)
        self.img_size = img_size
        self.seq_len  = seq_len
        self._tf      = _img_transform(img_size)

        pill_root = self.root / "PillCam"

        # ── per-camera intrinsics (no undistortion) ───────────────────────────
        self._K: dict[str, torch.Tensor] = {}
        for cam_id in ("C1", "C2"):
            cal_file = f"cam{'1' if cam_id == 'C1' else '2'}.txt"
            cal = _parse_cam_txt(pill_root / "Calibration" / cal_file)
            K_out = torch.from_numpy(
                _adjust_K(_build_K(cal), self._ORIG_W, self._ORIG_H, img_size)
            ).float()
            self._K[cam_id] = K_out

        # ── collect (trajectory, camera) windows ─────────────────────────────
        self._windows: list[tuple[list[Path], np.ndarray, str]] = []
        # each entry: (frame_paths, poses[i:i+seq_len], cam_id)

        active_cams = ["C1", "C2"] if cameras == "both" else [cameras]

        for organ_dir in sorted(pill_root.iterdir()):
            if not organ_dir.is_dir() or organ_dir.name == "Calibration":
                continue
            for traj_dir in sorted(organ_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                frames_dir = traj_dir / "Frames"
                poses_dir  = traj_dir / "Poses"
                if not frames_dir.exists() or not poses_dir.exists():
                    continue

                pose_files = sorted(poses_dir.iterdir())
                if not pose_files:
                    continue
                pose_file = pose_files[0]   # one file per trajectory

                for cam_id in active_cams:
                    cam_dir = frames_dir / cam_id
                    if not cam_dir.exists():
                        continue
                    frame_paths = sorted(cam_dir.glob("*.bmp"))
                    if len(frame_paths) < seq_len:
                        continue

                    poses = _load_poses(pose_file, len(frame_paths))  # (N,4,4)

                    for i in range(0, len(frame_paths) - seq_len + 1, stride):
                        self._windows.append((
                            frame_paths[i : i + seq_len],
                            poses[i : i + seq_len],
                            cam_id,
                        ))

    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        frame_paths, c2w_abs, cam_id = self._windows[idx]

        # relative poses: c2w[0] = I
        c2w_0_inv = np.linalg.inv(c2w_abs[0])
        c2w_rel   = (c2w_0_inv @ c2w_abs).astype(np.float32)

        imgs = [self._tf(Image.open(fp).convert("RGB")) for fp in frame_paths]

        S = len(imgs)
        return {
            "images": torch.stack(imgs),                                    # (S, 3, H, W)
            "depths": torch.zeros(S, self.img_size, self.img_size),         # (S, H, W) — no GT
            "K":      self._K[cam_id],                                      # (3, 3)
            "c2w":    torch.from_numpy(c2w_rel),                            # (S, 4, 4)
        }
