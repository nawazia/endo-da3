"""
Extract SCARED keyframes from downloaded zip files into a training-ready structure.

For each dataset_N / keyframe_K:
  1. Parse endoscope_calibration.yaml → K.json + stereo extrinsics
  2. Copy Left_Image.png, Right_Image.png
  3. Convert left_depth_map.tiff / right_depth_map.tiff (XYZ in mm) → depth_left.npy / depth_right.npy
     (Z in metres, float32, 0 = invalid)

Output structure:
    <out_root>/dataset_N/keyframe_K/
        K_left.json       # {"fx","fy","cx","cy","width","height"}
        K_right.json
        stereo.json       # {"R": [[3x3]], "T": [tx,ty,tz] in metres, "baseline_m": B}
        left.png
        right.png
        depth_left.npy    # float32 (H,W), metres, 0=invalid
        depth_right.npy

SCARED depth is stored in mm → divided by 1000 to give metres.
Image resolution: 1024 x 1280 (H x W).

Usage:
    python tools/extract_scared.py [--scared-root /path/to/SCARED]
                                   [--out-root /path/to/output]
                                   [--datasets 1 2 3 ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

IMG_H, IMG_W = 1024, 1280


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calibration(yaml_path: Path) -> dict:
    """
    Load endoscope_calibration.yaml (OpenCV FileStorage).
    Returns {"K_left", "K_right", "R", "T_mm"} all as float64 numpy arrays.
    T_mm is the stereo translation in mm (T[0] ≈ -4mm baseline).
    """
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    M1 = fs.getNode("M1").mat().astype(np.float64)   # left K
    M2 = fs.getNode("M2").mat().astype(np.float64)   # right K
    R  = fs.getNode("R").mat().astype(np.float64)    # stereo rotation (3x3)
    T  = fs.getNode("T").mat().astype(np.float64).reshape(3)  # stereo translation (mm)
    fs.release()
    return {"K_left": M1, "K_right": M2, "R": R, "T_mm": T}


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------

def xyz_tiff_to_depth_m(tiff_path: Path) -> np.ndarray:
    """
    Load (H, W, 3) XYZ float32 tiff in mm → Z in metres float32.
    Pixels where all XYZ == 0 are set to 0 (invalid).
    """
    xyz = tifffile.imread(str(tiff_path)).astype(np.float32)
    z = xyz[..., 2].copy()
    # Invalid pixels: all-zero XYZ or NaN (SCARED uses both conventions)
    invalid = ((xyz[..., 0] == 0) & (xyz[..., 1] == 0) & (xyz[..., 2] == 0)) \
              | np.isnan(xyz).any(axis=-1)
    z[invalid] = 0.0
    return np.nan_to_num(z / 1000.0, nan=0.0)


# ---------------------------------------------------------------------------
# Per-keyframe extraction
# ---------------------------------------------------------------------------

def extract_keyframe(kf_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calibration
    cal = load_calibration(kf_dir / "endoscope_calibration.yaml")

    def _k_dict(K):
        return {"fx": float(K[0,0]), "fy": float(K[1,1]),
                "cx": float(K[0,2]), "cy": float(K[1,2]),
                "width": IMG_W, "height": IMG_H}

    with open(out_dir / "K_left.json",  "w") as f:
        json.dump(_k_dict(cal["K_left"]),  f, indent=2)
    with open(out_dir / "K_right.json", "w") as f:
        json.dump(_k_dict(cal["K_right"]), f, indent=2)

    # Stereo extrinsics — T in mm → metres
    T_m = cal["T_mm"] / 1000.0
    baseline_m = float(np.linalg.norm(T_m))
    with open(out_dir / "stereo.json", "w") as f:
        json.dump({
            "R":          cal["R"].tolist(),
            "T":          T_m.tolist(),      # metres
            "baseline_m": baseline_m,
        }, f, indent=2)

    # Images
    shutil.copy2(kf_dir / "Left_Image.png",  out_dir / "left.png")
    shutil.copy2(kf_dir / "Right_Image.png", out_dir / "right.png")

    # Depths
    np.save(out_dir / "depth_left.npy",
            xyz_tiff_to_depth_m(kf_dir / "left_depth_map.tiff"))
    np.save(out_dir / "depth_right.npy",
            xyz_tiff_to_depth_m(kf_dir / "right_depth_map.tiff"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scared-root", default="/home/in4218/code/data/SCARED")
    ap.add_argument("--out-root",    default=None,
                    help="Output root (default: <scared-root>/extracted)")
    ap.add_argument("--datasets", nargs="+", type=int, default=None,
                    help="Dataset numbers to process (default: all 1–9)")
    args = ap.parse_args()

    scared_root = Path(args.scared_root)
    out_root    = Path(args.out_root) if args.out_root \
                  else scared_root / "extracted"

    zip_files = sorted(
        z for z in scared_root.glob("dataset_*.zip")
        if not z.stem.startswith("test_")
    )
    if args.datasets:
        zip_files = [z for z in zip_files
                     if int(z.stem.split("_")[1]) in args.datasets]

    if not zip_files:
        raise FileNotFoundError(f"No dataset_*.zip files in {scared_root}")

    print(f"Processing {len(zip_files)} datasets → {out_root}")

    for zip_path in tqdm(zip_files, desc="datasets"):
        ds_num = zip_path.stem.split("_")[1]

        with tempfile.TemporaryDirectory(prefix=f"scared_ds{ds_num}_") as tmpdir:
            tmpdir = Path(tmpdir)
            with zipfile.ZipFile(zip_path) as zf:
                # Only extract the files we need (skip scene_points / rgb.mp4)
                needed = [
                    n for n in zf.namelist()
                    if any(n.endswith(x) for x in (
                        "endoscope_calibration.yaml",
                        "Left_Image.png",
                        "Right_Image.png",
                        "left_depth_map.tiff",
                        "right_depth_map.tiff",
                    ))
                ]
                for name in tqdm(needed, desc=f"  ds{ds_num} extract", leave=False):
                    zf.extract(name, tmpdir)

            # Find the dataset directory
            ds_dirs = list(tmpdir.glob(f"dataset_{ds_num}"))
            ds_dir  = ds_dirs[0] if ds_dirs else tmpdir

            kf_dirs = sorted(
                ds_dir.glob("keyframe_*"),
                key=lambda p: int(p.name.split("_")[1])
            )

            for kf_dir in tqdm(kf_dirs, desc=f"  ds{ds_num} kf", leave=False):
                kf_num = kf_dir.name.split("_")[1]
                out_dir = out_root / f"dataset_{ds_num}" / f"keyframe_{kf_num}"
                extract_keyframe(kf_dir, out_dir)

    print(f"\nDone. Extracted to {out_root}")
    print(f"Total keyframes: "
          f"{len(list(out_root.glob('dataset_*/keyframe_*')))}")


if __name__ == "__main__":
    main()
