"""
MiroCam capsule endoscopy dataset — EndoSLAM ex-vivo subset for Stage 2b.

Reference: Ozyoruk et al., "EndoSLAM Dataset and An Unsupervised Monocular
           Visual Odometry and Depth Estimation Approach for Endoscopic Videos."
           Medical Image Analysis 2021.

Data layout (after extracting Cameras/MiroCam.zip):
    <root>/MiroCam/Calibration/cam.txt
    <root>/MiroCam/Colon-III(L-shaped)/TumorfreeTrajectory_1/Frames/0001.png
    <root>/MiroCam/Colon-III(L-shaped)/TumorfreeTrajectory_1/Poses/teste1.txt
    ...

Camera: MiroCam® MC1000-W capsule endoscope.
  Image size   : 320×320 (square)
  Focal length : fx≈156, fy≈156
  Distortion   : k1≈-0.249, k2≈0.061 (barrel — corrected in loader)

Poses: EM tracker at ~1 kHz, evenly subsampled to one pose per frame.
  Format: semicolon-delimited text, columns t;px;py;pz;qx;qy;qz;qw
  Units : seconds; metres; metres; metres; unit quaternion

Each sample is a sliding window of `seq_len` consecutive frames from the same
trajectory, returned with relative poses (c2w[0] = I).  No GT depth is
available — this dataset is used with the Stage 2b photometric loss only.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
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
    Parse semicolon-delimited pose file and subsample to n_frames poses.

    Returns (n_frames, 4, 4) float32 c2w matrices.
    """
    rows = []
    for line in pose_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 8:
            continue
        try:
            rows.append([float(p) for p in parts[:8]])
        except ValueError:
            continue  # header / unit rows

    rows = np.array(rows, dtype=np.float64)   # (N_pose, 8)

    # evenly subsample EM-tracker poses to match frame count
    indices = np.round(np.linspace(0, len(rows) - 1, n_frames)).astype(int)
    rows = rows[indices]

    c2w_list = []
    for row in rows:
        R = _quat_to_rot(*row[4:8])
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3,  3] = row[1:4].astype(np.float32)
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


# ── dataset ───────────────────────────────────────────────────────────────────

class MiroCamDataset(EndoDepthDataset):
    """
    MiroCam capsule endoscopy trajectories from the EndoSLAM ex-vivo dataset.

    Used for Stage 2b self-supervised training with photometric reprojection
    loss.  No GT depth — returns zeros in the 'depths' field.

    Each sample is a sliding window of `seq_len` consecutive undistorted frames
    from the same trajectory, with relative poses from the EM tracker
    (c2w[0] = I).

    Args:
        root     : EndoSLAM root directory (contains MiroCam/ subdirectory).
        img_size : Square output size after Resize + CenterCrop (default 336).
        seq_len  : Number of consecutive frames per sample (default 2).
        stride   : Step between windows (default 1).
    """

    name = "mirocam"

    def __init__(
        self,
        root: str | Path,
        img_size: int = 336,
        seq_len: int = 2,
        stride: int = 1,
    ):
        self.root     = Path(root)
        self.img_size = img_size
        self.seq_len  = seq_len
        self._tf      = _img_transform(img_size)

        # ── calibration ──────────────────────────────────────────────────────
        cam_txt = self.root / "MiroCam" / "Calibration" / "cam.txt"
        cal     = _parse_cam_txt(cam_txt)

        orig_w = orig_h = 320   # MiroCam native resolution

        K_raw = np.array([
            [cal["fx"],       0., cal["cx"]],
            [      0., cal["fy"], cal["cy"]],
            [      0.,       0.,        1.],
        ], dtype=np.float64)
        dist = np.array([cal["k1"], cal["k2"], 0., 0.], dtype=np.float64)

        # Compute undistorted (rectified) K; alpha=0 removes black border pixels
        K_undist, _ = cv2.getOptimalNewCameraMatrix(
            K_raw, dist, (orig_w, orig_h), alpha=0
        )
        self._K_raw    = K_raw
        self._dist     = dist
        self._K_undist = K_undist

        # Final K adjusted for Resize + CenterCrop
        self.K = torch.from_numpy(
            _adjust_K(K_undist, orig_w, orig_h, img_size)
        ).float()

        # ── collect trajectories ──────────────────────────────────────────────
        self._windows: list[tuple[list[Path], np.ndarray]] = []

        cam_root = self.root / "MiroCam"
        for organ_dir in sorted(cam_root.iterdir()):
            if not organ_dir.is_dir() or organ_dir.name == "Calibration":
                continue
            for traj_dir in sorted(organ_dir.iterdir()):
                if not traj_dir.is_dir():
                    continue
                frames_dir = traj_dir / "Frames"
                poses_dir  = traj_dir / "Poses"
                if not frames_dir.exists() or not poses_dir.exists():
                    continue

                frame_paths = sorted(frames_dir.glob("*.png"))
                pose_files  = sorted(poses_dir.glob("*.txt"))
                if not frame_paths or not pose_files:
                    continue

                poses = _load_poses(pose_files[0], len(frame_paths))  # (N,4,4)

                for i in range(0, len(frame_paths) - seq_len + 1, stride):
                    self._windows.append((
                        frame_paths[i : i + seq_len],
                        poses[i : i + seq_len],
                    ))

    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        frame_paths, c2w_abs = self._windows[idx]   # (S,4,4)

        # relative poses: c2w[0] = I
        c2w_0_inv = np.linalg.inv(c2w_abs[0])
        c2w_rel   = (c2w_0_inv @ c2w_abs).astype(np.float32)

        # load, undistort, and transform each frame
        imgs = []
        for fp in frame_paths:
            img_np    = np.array(Image.open(fp).convert("RGB"))
            img_undist = cv2.undistort(img_np, self._K_raw, self._dist,
                                       newCameraMatrix=self._K_undist)
            imgs.append(self._tf(Image.fromarray(img_undist)))

        S = len(imgs)
        return {
            "images": torch.stack(imgs),                                    # (S, 3, H, W)
            "depths": torch.zeros(S, self.img_size, self.img_size),         # (S, H, W) — no GT
            "K":      self.K,                                               # (3, 3)
            "c2w":    torch.from_numpy(c2w_rel),                            # (S, 4, 4)
        }
