"""
Extract and rectify StereoMIS stereo frames from vertically-stacked MP4 videos.

For each sequence in the StereoMIS zip:
  1. Read video frames (1280×2048, top=left, bottom=right)
  2. Undistort each camera using kc_* coefficients from StereoCalibration.ini
  3. Stereo-rectify using R, T between cameras (OpenCV stereoRectify)
  4. Save rectified left/right frames as JPEG at 640×512

Output structure:
    <out_root>/<seq>/left/000000.jpg ...
    <out_root>/<seq>/right/000000.jpg ...
    <out_root>/<seq>/rectified_K.json        (shared K after rectification)
    <out_root>/<seq>/baseline.txt            (B in metres)
    <out_root>/<seq>/StereoCalibration.ini   (copy of original)
    <out_root>/<seq>/groundtruth.txt         (copy of original)
    <out_root>/<seq>/train_split.csv / test_split.csv

Usage:
    python tools/extract_stereomis.py [--seq P1 P2_0 ...] [--stride 1]
"""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ZIP_PATH  = Path("/home/in4218/code/data/StereoMIS.zip")
OUT_ROOT  = Path("/home/in4218/code/data/StereoMIS")
ZIP_ROOT  = "StereoMIS_0_0_1"

# All sequences in the dataset
ALL_SEQS  = ["P1", "P2_0", "P2_1", "P2_2", "P2_3",
             "P2_4", "P2_5", "P2_6", "P2_7", "P2_8", "P3"]

# Output resolution per camera after rectification
OUT_W, OUT_H = 640, 512


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def parse_calibration(ini_text: str) -> dict:
    """Parse StereoCalibration.ini → dict with left/right camera params."""
    cfg = configparser.ConfigParser()
    cfg.read_string(ini_text)

    def _cam(section):
        s = cfg[section]
        K = np.array([
            [float(s["fc_x"]), 0.,             float(s["cc_x"])],
            [0.,               float(s["fc_y"]), float(s["cc_y"])],
            [0.,               0.,              1.              ],
        ])
        # kc_0..4 → OpenCV (k1,k2,p1,p2,k3)
        dist = np.array([float(s[f"kc_{i}"]) for i in range(5)])
        R = np.array([float(s[f"R_{i}"]) for i in range(9)]).reshape(3, 3)
        T = np.array([float(s[f"T_{i}"]) for i in range(3)])
        w, h = int(s["res_x"]), int(s["res_y"])
        return K, dist, R, T, w, h

    Kl, dl, Rl, Tl, w, h = _cam("StereoLeft")
    Kr, dr, Rr, Tr, _, _ = _cam("StereoRight")

    # Relative rotation/translation: right camera in left camera frame
    # Left is at origin (Rl=I, Tl=0), right has Rr, Tr
    R_rel = Rr @ Rl.T
    T_rel = Tr - R_rel @ Tl

    return dict(Kl=Kl, dl=dl, Kr=Kr, dr=dr, R_rel=R_rel, T_rel=T_rel, w=w, h=h)


def compute_rectification(cal: dict) -> tuple:
    """Run OpenCV stereoRectify → returns (map1l, map2l, map1r, map2r, K_rect, B)."""
    w, h = cal["w"], cal["h"]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cal["Kl"], cal["dl"],
        cal["Kr"], cal["dr"],
        (w, h),
        cal["R_rel"], cal["T_rel"].reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=(OUT_W, OUT_H),
    )

    map1l, map2l = cv2.initUndistortRectifyMap(
        cal["Kl"], cal["dl"], R1, P1, (OUT_W, OUT_H), cv2.CV_32FC1)
    map1r, map2r = cv2.initUndistortRectifyMap(
        cal["Kr"], cal["dr"], R2, P2, (OUT_W, OUT_H), cv2.CV_32FC1)

    # Shared intrinsics after rectification (from P1)
    K_rect = P1[:3, :3].copy()

    # Baseline in metres: B = -P2[0,3] / P2[0,0]  (P2[0,3] = -fx*B)
    B = float(-P2[0, 3] / P2[0, 0]) / 1000.0   # T is in mm → convert to m

    return map1l, map2l, map1r, map2r, K_rect, B


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def find_video(zf: zipfile.ZipFile, seq: str) -> str:
    """Return zip path of the video file for a sequence."""
    for name in zf.namelist():
        if f"/{seq}/" in name and name.endswith(".mp4"):
            return name
    raise FileNotFoundError(f"No video found for sequence {seq}")


def zip_text(zf: zipfile.ZipFile, path: str) -> str:
    with zf.open(path) as f:
        return f.read().decode("utf-8")


def load_split_ranges(zf: zipfile.ZipFile, seq: str) -> list[tuple[int, int]] | None:
    """
    Return list of (start, end) frame index ranges from the split CSV, or None
    if no split file exists (extract all frames).
    P1 has train_split.csv (multiple ranges), P2/P3 have test_split.csv (one range).
    """
    for fname in ["train_split.csv", "test_split.csv"]:
        zpath = f"{ZIP_ROOT}/{seq}/{fname}"
        if zpath in zf.namelist():
            ranges = []
            for line in zip_text(zf, zpath).strip().splitlines()[1:]:  # skip header
                start, end = map(int, line.strip().split(","))
                ranges.append((start, end))
            return ranges
    return None


def in_split(frame_idx: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= frame_idx <= end for start, end in ranges)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_sequence(zf: zipfile.ZipFile, seq: str, stride: int):
    seq_dir = OUT_ROOT / seq
    left_dir  = seq_dir / "left"
    right_dir = seq_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    # Calibration
    cal_path = f"{ZIP_ROOT}/{seq}/StereoCalibration.ini"
    cal      = parse_calibration(zip_text(zf, cal_path))
    map1l, map2l, map1r, map2r, K_rect, B = compute_rectification(cal)

    # Save metadata
    with open(seq_dir / "rectified_K.json", "w") as f:
        json.dump({"fx": K_rect[0, 0], "fy": K_rect[1, 1],
                   "cx": K_rect[0, 2], "cy": K_rect[1, 2],
                   "width": OUT_W, "height": OUT_H}, f, indent=2)
    with open(seq_dir / "baseline.txt", "w") as f:
        f.write(f"{B}\n")

    # Copy auxiliary files
    for fname in ["StereoCalibration.ini", "groundtruth.txt",
                  "train_split.csv", "test_split.csv"]:
        zpath = f"{ZIP_ROOT}/{seq}/{fname}"
        if zpath in zf.namelist():
            with zf.open(zpath) as src, open(seq_dir / fname, "wb") as dst:
                shutil.copyfileobj(src, dst)

    # Extract video to a temp file (OpenCV needs seekable file)
    video_zip_path = find_video(zf, seq)
    tmp_video = Path(f"/tmp/stereomis_{seq}.mp4")
    print(f"  Extracting video to {tmp_video} …")
    with zf.open(video_zip_path) as src, open(tmp_video, "wb") as dst:
        shutil.copyfileobj(src, dst)

    split_ranges = load_split_ranges(zf, seq)
    if split_ranges:
        total_valid = sum(e - s + 1 for s, e in split_ranges)
        print(f"  Split ranges: {split_ranges}  ({total_valid} valid frames)")
    else:
        total_valid = None
        print("  No split file found — extracting all frames")

    cap = cv2.VideoCapture(str(tmp_video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved = 0

    pbar = tqdm(total=(total_valid or n_frames) // stride, desc=f"{seq}", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames outside the valid split range
        if split_ranges and not in_split(frame_idx, split_ranges):
            frame_idx += 1
            continue

        if frame_idx % stride == 0:
            # Split vertically-stacked frame
            left_raw  = frame[:1024, :, :]
            right_raw = frame[1024:, :, :]

            # Undistort + rectify
            left_rect  = cv2.remap(left_raw,  map1l, map2l, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, map1r, map2r, cv2.INTER_LINEAR)

            stem = f"{saved:06d}"
            cv2.imwrite(str(left_dir  / f"{stem}.jpg"), left_rect,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(right_dir / f"{stem}.jpg"), right_rect,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()
    tmp_video.unlink(missing_ok=True)
    print(f"  {seq}: saved {saved} stereo pairs → {seq_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",    nargs="+", default=ALL_SEQS,
                    help="Sequences to extract (default: all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Save every Nth frame (default: 1 = all frames)")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for seq in args.seq:
            print(f"\n=== {seq} ===")
            extract_sequence(zf, seq, stride=args.stride)

    print("\nDone.")


if __name__ == "__main__":
    main()
