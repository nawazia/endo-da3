"""
Compare kinematic GT (scene_points) vs RAFT-Stereo on SCARED intermediate frames.

For a chosen dataset/keyframe sequence, at frame indices [0, 1, 5, 10, 20]:
  - Extracts left+right images from rgb.mp4
  - Extracts kinematic depth from scene_points (top half of XYZ tiff, Z channel)
  - Runs RAFT-Stereo (with rectification) to get RAFT depth
  - Computes left→right photometric reprojection error for both depth sources
  - Reports how reprojection error evolves with frame index

Reprojection error is computed in the original (unrectified) image space using
per-frame calibration from frame_data.tar.gz.

Usage:
    python tools/compare_kinematic_raft.py \
        --scared-root ~/code/data/SCARED \
        --dataset 1 --keyframe 1 \
        --frames 0 1 5 10 20
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import tifffile
from tqdm import tqdm

RAFT_ROOT = Path("/home/in4218/code/RAFT-Stereo")
sys.path.insert(0, str(RAFT_ROOT))
sys.path.insert(0, str(RAFT_ROOT / "core"))
from raft_stereo import RAFTStereo      # noqa: E402
from utils.utils import InputPadder     # noqa: E402

RAFT_CKPT = RAFT_ROOT / "models" / "iraftstereo_rvc.pth"
IMG_H, IMG_W = 1024, 1280


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(device):
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dims",         nargs="+", type=int, default=[128]*3)
    p.add_argument("--corr_implementation", default="reg")
    p.add_argument("--shared_backbone",     action="store_true")
    p.add_argument("--corr_levels",         type=int, default=4)
    p.add_argument("--corr_radius",         type=int, default=4)
    p.add_argument("--n_downsample",        type=int, default=2)
    p.add_argument("--context_norm",        default="instance")
    p.add_argument("--slow_fast_gru",       action="store_true")
    p.add_argument("--n_gru_layers",        type=int, default=3)
    p.add_argument("--mixed_precision",     action="store_true")
    args, _ = p.parse_known_args()
    model = torch.nn.DataParallel(RAFTStereo(args))
    model.load_state_dict(torch.load(str(RAFT_CKPT), map_location="cpu"))
    return model.module.to(device).eval()


# ── calibration / rectification ───────────────────────────────────────────────

def load_yaml_cal(zip_path, ds, kf):
    """Load endoscope_calibration.yaml from zip → K1,D1,K2,D2,R,T(m)."""
    with zipfile.ZipFile(zip_path) as zf:
        yaml_bytes = zf.read(f"dataset_{ds}/keyframe_{kf}/endoscope_calibration.yaml")
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(yaml_bytes); tmp = f.name
    fs = cv2.FileStorage(tmp, cv2.FILE_STORAGE_READ)
    K1 = fs.getNode("M1").mat().astype(np.float64)
    K2 = fs.getNode("M2").mat().astype(np.float64)
    D1 = fs.getNode("D1").mat().astype(np.float64).flatten()
    D2 = fs.getNode("D2").mat().astype(np.float64).flatten()
    R  = fs.getNode("R").mat().astype(np.float64)
    T  = fs.getNode("T").mat().astype(np.float64).reshape(3) / 1000.0
    fs.release()
    import os; os.unlink(tmp)
    return K1, D1, K2, D2, R, T


def build_rectification(K1, D1, K2, D2, R, T):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (IMG_W, IMG_H), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (IMG_W, IMG_H), cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (IMG_W, IMG_H), cv2.CV_32FC1)
    fx_rect    = float(P1[0, 0])
    baseline_m = float(-P2[0, 3] / P1[0, 0])
    return map1L, map2L, map1R, map2R, fx_rect, baseline_m


# ── RAFT inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def raft_depth(model, left_bgr, right_bgr, map1L, map2L, map1R, map2R,
               fx_rect, baseline_m, device, iters=32):
    """Rectify → RAFT disparity → depth in rectified space."""
    lR = cv2.remap(left_bgr,  map1L, map2L, cv2.INTER_LINEAR)
    rR = cv2.remap(right_bgr, map1R, map2R, cv2.INTER_LINEAR)

    def to_t(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    lt, rt = to_t(lR), to_t(rR)
    padder = InputPadder(lt.shape, divis_by=32)
    lt, rt = padder.pad(lt, rt)
    _, disp = model(lt, rt, iters=iters, test_mode=True)
    disp = padder.unpad(disp)
    disp_np = -disp.squeeze().cpu().numpy()   # positive px
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp_np > 0, fx_rect * baseline_m / disp_np, 0.0)
    return depth.astype(np.float32), lR, rR   # depth in rectified space


# ── reprojection error ────────────────────────────────────────────────────────

def reproj_error_rectified(depth, left_rect, right_rect, fx, baseline_m):
    """
    Left→right reprojection error in rectified space.
    depth: (H,W) metres in rectified left camera.
    Disparity d = fx*B/depth → right_x = left_x - d.
    """
    H, W = depth.shape
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    valid = depth > 0
    disp  = np.where(valid, fx * baseline_m / depth, 0.0)
    rx    = xx - disp   # projected x in right image

    # Build sampling grid for cv2.remap
    map_x = rx.astype(np.float32)
    map_y = yy.astype(np.float32)

    right_warped = cv2.remap(right_rect.astype(np.float32),
                             map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    left_f  = left_rect.astype(np.float32)
    diff    = np.abs(left_f - right_warped)

    # Only count pixels that project inside the right image and have valid depth
    in_bounds = (rx >= 0) & (rx < W) & valid
    if in_bounds.sum() == 0:
        return np.nan, 0
    mae = float(diff[in_bounds].mean())
    return mae, int(in_bounds.sum())


def reproj_error_original(xyz_left_mm, left_bgr, right_bgr, frame_cal):
    """
    Left→right reprojection in original (unrectified) space using frame_data cal.
    xyz_left_mm: (H,W,3) XYZ in left camera coords (mm).
    """
    KL = np.array(frame_cal["KL"], dtype=np.float64)
    KR = np.array(frame_cal["KR"], dtype=np.float64)
    DL = np.array(frame_cal["DL"], dtype=np.float64).flatten()
    DR = np.array(frame_cal["DR"], dtype=np.float64).flatten()
    R  = np.array(frame_cal["R"],  dtype=np.float64)
    T  = np.array(frame_cal["T"],  dtype=np.float64).reshape(3) / 1000.0  # mm→m

    H, W = xyz_left_mm.shape[:2]
    valid = ~(np.all(xyz_left_mm == 0, axis=-1))

    pts_left = xyz_left_mm[valid].astype(np.float64) / 1000.0   # mm→m

    # Transform left camera 3D points → right camera coords
    pts_right = (R @ pts_left.T).T + T[None, :]                 # (N,3) metres

    # Project into right image
    pts_right_cv = pts_right[:, :2] / pts_right[:, 2:3]         # normalised
    pts2d_r, _ = cv2.projectPoints(
        pts_right.reshape(-1, 1, 3),
        np.zeros(3), np.zeros(3), KR, DR)
    pts2d_r = pts2d_r.reshape(-1, 2)

    # Valid projections inside right image
    in_bounds = ((pts2d_r[:, 0] >= 0) & (pts2d_r[:, 0] < W) &
                 (pts2d_r[:, 1] >= 0) & (pts2d_r[:, 1] < H))

    if in_bounds.sum() == 0:
        return np.nan, 0

    # Get pixel coordinates in left image for the same valid points
    ys_l, xs_l = np.where(valid)
    xs_l = xs_l[in_bounds]
    ys_l = ys_l[in_bounds]
    pts2d_r_valid = pts2d_r[in_bounds]

    # Sample right image at projected coords (nearest neighbour)
    rx = np.clip(np.round(pts2d_r_valid[:, 0]).astype(int), 0, W-1)
    ry = np.clip(np.round(pts2d_r_valid[:, 1]).astype(int), 0, H-1)

    left_px  = left_bgr[ys_l, xs_l].astype(np.float32)
    right_px = right_bgr[ry, rx].astype(np.float32)

    mae = float(np.abs(left_px - right_px).mean())
    return mae, int(in_bounds.sum())


# ── data loading from zip ─────────────────────────────────────────────────────

def load_mp4_frames(zip_path, ds, kf, frame_indices):
    """Extract rgb.mp4 from zip and read specific frames."""
    with zipfile.ZipFile(zip_path) as zf:
        mp4_bytes = zf.read(f"dataset_{ds}/keyframe_{kf}/data/rgb.mp4")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(mp4_bytes); tmp = f.name

    cap = cv2.VideoCapture(tmp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    target = set(frame_indices)
    for i in range(max(frame_indices) + 1):
        ret, frame = cap.read()
        if not ret: break
        if i in target:
            left  = frame[:IMG_H, :, :]
            right = frame[IMG_H:, :, :]
            frames[i] = (left, right)
    cap.release()
    import os; os.unlink(tmp)
    return frames, total


def load_scene_points_frames(zip_path, ds, kf, frame_indices):
    """Extract scene_points.tar.gz from zip and read specific TIFF frames."""
    with zipfile.ZipFile(zip_path) as zf:
        sp_bytes = zf.read(f"dataset_{ds}/keyframe_{kf}/data/scene_points.tar.gz")
    tf = tarfile.open(fileobj=io.BytesIO(sp_bytes))
    members = {int(m.name.replace("scene_points", "").replace(".tiff", "")): m
               for m in tf.getmembers()}
    result = {}
    for idx in frame_indices:
        if idx not in members:
            continue
        raw = tifffile.imread(io.BytesIO(tf.extractfile(members[idx]).read()))
        xyz_left  = raw[:IMG_H]    # top half = left camera XYZ (mm)
        result[idx] = xyz_left
    return result


def load_frame_data(zip_path, ds, kf, frame_indices):
    """Load per-frame calibration from frame_data.tar.gz."""
    with zipfile.ZipFile(zip_path) as zf:
        fd_bytes = zf.read(f"dataset_{ds}/keyframe_{kf}/data/frame_data.tar.gz")
    tf = tarfile.open(fileobj=io.BytesIO(fd_bytes))
    members = {int(m.name.replace("frame_data", "").replace(".json", "")): m
               for m in tf.getmembers()}
    result = {}
    for idx in frame_indices:
        if idx not in members:
            continue
        data = json.loads(tf.extractfile(members[idx]).read())
        result[idx] = data["camera-calibration"]
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scared-root", default="/home/in4218/code/data/SCARED")
    ap.add_argument("--dataset",  type=int, default=1)
    ap.add_argument("--keyframe", type=int, default=1)
    ap.add_argument("--frames",   type=int, nargs="+", default=[0, 1, 5, 10, 20])
    ap.add_argument("--iters",    type=int, default=32)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device   = torch.device(args.device)
    zip_path = Path(args.scared_root) / f"dataset_{args.dataset}.zip"
    ds, kf   = args.dataset, args.keyframe

    print(f"Dataset {ds} / Keyframe {kf}  —  frames {args.frames}")

    # Keyframe calibration for RAFT rectification
    K1, D1, K2, D2, R_stereo, T_stereo = load_yaml_cal(zip_path, ds, kf)
    map1L, map2L, map1R, map2R, fx_rect, baseline_m = build_rectification(
        K1, D1, K2, D2, R_stereo, T_stereo)
    print(f"Rectified: fx={fx_rect:.1f}  B={baseline_m*1000:.2f}mm")

    print("Loading RAFT-Stereo …")
    model = build_model(device)

    print("Loading data from zip …")
    rgb_frames, n_total = load_mp4_frames(zip_path, ds, kf, args.frames)
    print(f"mp4 total frames: {n_total}")
    sp_frames   = load_scene_points_frames(zip_path, ds, kf, args.frames)
    fd_frames   = load_frame_data(zip_path, ds, kf, args.frames)

    print(f"\n{'Frame':>6}  {'RAFT reproj (rect)':>20}  {'Kinematic reproj (orig)':>24}  {'n_raft':>8}  {'n_kine':>8}")
    print("-" * 75)

    results = {}   # idx → (left_bgr, depth_raft, depth_kine, err_raft, err_kine)

    for idx in sorted(args.frames):
        if idx not in rgb_frames:
            print(f"{idx:>6}  (frame not available in mp4)")
            continue
        if idx not in sp_frames:
            print(f"{idx:>6}  (frame not available in scene_points)")
            continue

        left_bgr, right_bgr = rgb_frames[idx]

        # RAFT depth + reprojection error in rectified space
        depth_raft, left_rect, right_rect = raft_depth(
            model, left_bgr, right_bgr,
            map1L, map2L, map1R, map2R,
            fx_rect, baseline_m, device, iters=args.iters)
        err_raft, n_raft = reproj_error_rectified(
            depth_raft, left_rect, right_rect, fx_rect, baseline_m)

        # Kinematic depth + reprojection error in original space
        xyz_left = sp_frames[idx]
        z_kine = xyz_left[..., 2].astype(np.float32) / 1000.0
        z_kine[np.all(xyz_left == 0, axis=-1)] = 0.0
        frame_cal = fd_frames.get(idx)
        if frame_cal is not None:
            err_kine, n_kine = reproj_error_original(
                xyz_left, left_bgr, right_bgr, frame_cal)
        else:
            err_kine, n_kine = np.nan, 0

        print(f"{idx:>6}  {err_raft:>18.3f}px  {err_kine:>22.3f}px  {n_raft:>8,}  {n_kine:>8,}")
        results[idx] = (left_bgr, depth_raft, z_kine, err_raft, err_kine)

    print("\nNote: reprojection error in mean absolute pixel intensity difference (0-255 scale)")

    # ── visualisation ─────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def colorize(depth: np.ndarray) -> np.ndarray:
        """depth (H,W) metres → uint8 RGB magma_r, bright=close."""
        valid = depth > 0
        if valid.sum() == 0:
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        vmin, vmax = np.percentile(depth[valid], [2, 98])
        norm = (depth.clip(vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
        norm[~valid] = 0
        rgba = plt.get_cmap("magma_r")(norm)
        rgb  = (rgba[..., :3] * 255).astype(np.uint8)
        rgb[~valid] = 0
        return rgb

    # Load structured-light GT from extracted dir
    ext_root     = Path(args.scared_root) / "extracted"
    gt_path      = ext_root / f"dataset_{ds}" / f"keyframe_{kf}" / "depth_left.npy"
    kf_left_path = ext_root / f"dataset_{ds}" / f"keyframe_{kf}" / "left.png"
    gt_depth = np.load(gt_path)      if gt_path.exists()      else None
    kf_left  = cv2.cvtColor(cv2.imread(str(kf_left_path)), cv2.COLOR_BGR2RGB) \
               if kf_left_path.exists() else None

    frame_list = sorted(results.keys())
    # Rows: keyframe GT reference + one per frame index
    # Cols: RGB | RAFT depth | Kinematic depth
    n_rows = 1 + len(frame_list)
    n_cols = 3
    cell_h, cell_w = 256, 320

    canvas = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)

    def place(row, col, img):
        img_r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = img_r

    # Row 0 — keyframe structured-light GT reference
    if kf_left is not None:
        place(0, 0, kf_left)
    if gt_depth is not None:
        place(0, 1, colorize(gt_depth))
        place(0, 2, colorize(gt_depth))

    # Subsequent rows — intermediate frames
    for row, idx in enumerate(frame_list, start=1):
        left_bgr, depth_raft, depth_kine, err_raft, err_kine = results[idx]
        place(row, 0, cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB))
        place(row, 1, colorize(depth_raft))
        place(row, 2, colorize(depth_kine))

    # Labels
    col_labels = ["Left RGB", "RAFT depth", "Kinematic depth"]
    row_labels  = [f"Keyframe {kf}\n(struct-light GT)"] + \
                  [f"Frame {idx}\nRAFT={results[idx][3]:.1f}px  kine={results[idx][4]:.1f}px"
                   for idx in frame_list]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_w / 80, n_rows * cell_h / 80))
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].imshow(canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w])
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_labels[c], fontsize=9, fontweight="bold")
        axes[r, 0].set_ylabel(row_labels[r], fontsize=7, rotation=0,
                              labelpad=60, va="center")

    plt.tight_layout()

    out_path = Path("/home/in4218/code/endo-da3") / \
        f"kinematic_vs_raft_ds{ds}_kf{kf}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"\nVisualization saved to {out_path}")


if __name__ == "__main__":
    main()
