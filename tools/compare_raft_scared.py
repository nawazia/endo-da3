"""
Compare RAFT-Stereo pseudo-GT vs structured-light GT depth on SCARED keyframes.

For each extracted keyframe (datasets 1-7, keyframes 1-5):
  - Rectifies the stereo pair using the full calibration from the zip
  - Runs RAFT-Stereo on the rectified pair
  - Compares against structured-light GT depth (depth_left.npy, unrectified space)
  - Reports AbsRel / δ<1.25 / δ<1.05 / MAE / RMSE (median-scale aligned)

Usage:
    python tools/compare_raft_scared.py \
        --scared-root    ~/code/data/SCARED \
        --scared-extracted ~/code/data/SCARED/extracted \
        --datasets 1 2 3 4 5 6 7
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

RAFT_ROOT = Path("/home/in4218/code/RAFT-Stereo")
sys.path.insert(0, str(RAFT_ROOT))
sys.path.insert(0, str(RAFT_ROOT / "core"))

from raft_stereo import RAFTStereo      # noqa: E402
from utils.utils import InputPadder     # noqa: E402

RAFT_CKPT = RAFT_ROOT / "models" / "iraftstereo_rvc.pth"
IMG_H, IMG_W = 1024, 1280


# ── calibration ───────────────────────────────────────────────────────────────

def load_calibration_from_zip(zip_path: Path, ds: int, kf: int) -> dict:
    """
    Extract and parse endoscope_calibration.yaml from inside the dataset zip.
    Returns K1, D1, K2, D2 (float64), R (3x3), T (3,) in metres.
    """
    yaml_name = f"dataset_{ds}/keyframe_{kf}/endoscope_calibration.yaml"
    with zipfile.ZipFile(zip_path) as zf:
        yaml_bytes = zf.read(yaml_name)

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(yaml_bytes)
        tmp = f.name

    fs = cv2.FileStorage(tmp, cv2.FILE_STORAGE_READ)
    K1 = fs.getNode("M1").mat().astype(np.float64)
    K2 = fs.getNode("M2").mat().astype(np.float64)
    D1 = fs.getNode("D1").mat().astype(np.float64).flatten()
    D2 = fs.getNode("D2").mat().astype(np.float64).flatten()
    R  = fs.getNode("R").mat().astype(np.float64)
    T  = fs.getNode("T").mat().astype(np.float64).reshape(3) / 1000.0  # mm → m
    fs.release()

    import os; os.unlink(tmp)
    return {"K1": K1, "D1": D1, "K2": K2, "D2": D2, "R": R, "T": T}


def compute_rectification(cal: dict):
    """
    Run cv2.stereoRectify → rectification maps + rectified fx and baseline.
    Returns (map1_left, map2_left, map1_right, map2_right, fx_rect, baseline_m).
    """
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cal["K1"], cal["D1"],
        cal["K2"], cal["D2"],
        (IMG_W, IMG_H),
        cal["R"], cal["T"],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map1L, map2L = cv2.initUndistortRectifyMap(
        cal["K1"], cal["D1"], R1, P1, (IMG_W, IMG_H), cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(
        cal["K2"], cal["D2"], R2, P2, (IMG_W, IMG_H), cv2.CV_32FC1)

    fx_rect    = float(P1[0, 0])
    baseline_m = float(-P2[0, 3] / P1[0, 0])   # P2[0,3] = -fx * baseline
    return map1L, map2L, map1R, map2R, fx_rect, baseline_m


def rectify_image(img: np.ndarray, map1, map2) -> np.ndarray:
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)


# ── RAFT ──────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> torch.nn.Module:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dims",         nargs="+", type=int, default=[128] * 3)
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


def img_to_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def run_raft(model, left_t: torch.Tensor, right_t: torch.Tensor,
             iters: int = 32) -> np.ndarray:
    """Returns disparity (H, W) in pixels, positive."""
    padder = InputPadder(left_t.shape, divis_by=32)
    l, r   = padder.pad(left_t, right_t)
    _, disp = model(l, r, iters=iters, test_mode=True)
    disp   = padder.unpad(disp)
    return -disp.squeeze().cpu().numpy()


def disp_to_depth(disp: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(disp > 0, fx * baseline / disp, 0.0).astype(np.float32)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Median-scale aligned. gt=0 means invalid."""
    mask = (gt > 0) & np.isfinite(pred) & (pred > 0)
    if mask.sum() < 10:
        return {"AbsRel": np.nan, "d1": np.nan, "d3": np.nan,
                "MAE": np.nan, "RMSE": np.nan, "n": 0}
    p = pred[mask] * (np.median(gt[mask]) / np.median(pred[mask]))
    g = gt[mask]
    ratio   = np.maximum(p / g, g / p)
    return {
        "AbsRel": float(np.mean(np.abs(p - g) / g)),
        "d1":     float(np.mean(ratio < 1.25)),
        "d3":     float(np.mean(ratio < 1.05)),
        "d4":     float(np.mean(ratio < 1.02)),
        "d5":     float(np.mean(ratio < 1.01)),
        "MAE":    float(np.mean(np.abs(p - g))),
        "RMSE":   float(np.sqrt(np.mean((p - g) ** 2))),
        "n":      int(mask.sum()),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scared-root",      default="/home/in4218/code/data/SCARED")
    ap.add_argument("--scared-extracted", default="/home/in4218/code/data/SCARED/extracted")
    ap.add_argument("--datasets", nargs="+", type=int, default=list(range(1, 8)))
    ap.add_argument("--iters",  type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device   = torch.device(args.device)
    ext_root = Path(args.scared_extracted)
    zip_root = Path(args.scared_root)

    # Collect keyframes
    keyframes = []
    for ds in sorted(args.datasets):
        ds_dir = ext_root / f"dataset_{ds}"
        if not ds_dir.exists():
            print(f"  dataset_{ds} not found in extracted, skipping")
            continue
        for kf_dir in sorted(ds_dir.glob("keyframe_*"),
                              key=lambda p: int(p.name.split("_")[1])):
            if not (kf_dir / "left.png").exists():
                continue
            kf_num = int(kf_dir.name.split("_")[1])
            keyframes.append((ds, kf_num, kf_dir))

    print(f"Found {len(keyframes)} keyframes across datasets {args.datasets}")
    print(f"Loading RAFT-Stereo …")
    model = build_model(device)

    rows = []
    for ds, kf, kf_dir in tqdm(keyframes, desc="keyframes"):
        zip_path = zip_root / f"dataset_{ds}.zip"
        if not zip_path.exists():
            tqdm.write(f"  dataset_{ds}.zip not found, skipping kf{kf}")
            continue

        # Full calibration from zip → rectification
        try:
            cal = load_calibration_from_zip(zip_path, ds, kf)
        except Exception as e:
            tqdm.write(f"  dataset_{ds}/kf{kf}: calibration error: {e}")
            continue

        map1L, map2L, map1R, map2R, fx_rect, baseline_m = compute_rectification(cal)

        # Load and rectify images
        left_bgr  = cv2.imread(str(kf_dir / "left.png"))
        right_bgr = cv2.imread(str(kf_dir / "right.png"))
        left_rect  = rectify_image(left_bgr,  map1L, map2L)
        right_rect = rectify_image(right_bgr, map1R, map2R)

        # RAFT on rectified pair
        left_t  = img_to_tensor(left_rect,  device)
        right_t = img_to_tensor(right_rect, device)
        disp    = run_raft(model, left_t, right_t, iters=args.iters)

        # Depth in rectified space
        depth_rect = disp_to_depth(disp, fx_rect, baseline_m)

        # GT depth is in original (unrectified) left camera space.
        # Warp RAFT depth back to original space for a fair comparison.
        # We use the inverse rectification map (remap with INTER_NEAREST).
        # Simpler: warp GT into rectified space instead.
        gt_orig = np.load(kf_dir / "depth_left.npy")   # (H, W) metres
        gt_rect = cv2.remap(gt_orig, map1L, map2L, interpolation=cv2.INTER_NEAREST)

        m = compute_metrics(depth_rect, gt_rect)
        rows.append((ds, kf, m))

        tqdm.write(
            f"  dataset_{ds}/keyframe_{kf}  "
            f"AbsRel={m['AbsRel']:.4f}  δ<1.25={m['d1']:.4f}  δ<1.05={m['d3']:.4f}  "
            f"δ<1.02={m['d4']:.4f}  δ<1.01={m['d5']:.4f}  "
            f"MAE={m['MAE']*1000:.1f}mm  RMSE={m['RMSE']*1000:.1f}mm"
        )

    # Summary
    valid = [m for _, _, m in rows if not np.isnan(m["AbsRel"])]
    if valid:
        print("\n" + "="*72)
        print(f"{'Checkpoint':<32} {'AbsRel':>8} {'δ<1.25':>8} {'δ<1.05':>8} "
              f"{'δ<1.02':>8} {'δ<1.01':>8} {'MAE':>10} {'RMSE':>10}")
        print("-"*72)
        print(f"{'RAFT-Stereo (mean)':<32} "
              f"{np.mean([m['AbsRel'] for m in valid]):>8.4f} "
              f"{np.mean([m['d1']     for m in valid]):>8.4f} "
              f"{np.mean([m['d3']     for m in valid]):>8.4f} "
              f"{np.mean([m['d4']     for m in valid]):>8.4f} "
              f"{np.mean([m['d5']     for m in valid]):>8.4f} "
              f"{np.mean([m['MAE']    for m in valid])*1000:>9.1f}mm "
              f"{np.mean([m['RMSE']   for m in valid])*1000:>9.1f}mm")
        print("="*72)
        print(f"({len(valid)}/{len(rows)} keyframes evaluated)")


if __name__ == "__main__":
    main()
