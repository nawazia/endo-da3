"""
Extract SCARED video frames and generate RAFT-Stereo pseudo-GT depth.

For each dataset/keyframe sequence (train: 1,2,3,6,7 / val: 8,9):
  - Reads rgb.mp4 (left=top 1024px, right=bottom 1024px)
  - Rectifies stereo pair using endoscope_calibration.yaml from zip
  - Runs RAFT-Stereo → disparity → depth in rectified space
  - Saves every --stride-th frame to extracted/dataset_N/keyframe_K/video/XXXXXX/
      left.png          (rectified, uint8)
      right.png         (rectified, uint8)
      depth_left.npy    (float16, metres, 0=invalid)
  - Saves per-keyframe calibration (rectified K + baseline):
      extracted/dataset_N/keyframe_K/video/K_rect.json
      extracted/dataset_N/keyframe_K/video/stereo_rect.json

Skips datasets 4 & 5 (bad intrinsics per Allan et al. 2021).

Usage:
    python tools/extract_scared_video.py \\
        --scared-root ~/code/data/SCARED \\
        --datasets 1 2 3 6 7 8 9 \\
        --stride 1 \\
        --iters 32 \\
        --batch-size 4
"""

from __future__ import annotations

import argparse
import io
import json
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

RAFT_CKPT  = RAFT_ROOT / "models" / "iraftstereo_rvc.pth"
IMG_H, IMG_W = 1024, 1280
BAD_DATASETS = {4, 5}


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> torch.nn.Module:
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


# ── calibration ───────────────────────────────────────────────────────────────

def load_and_rectify(zip_path: Path, ds: int, kf: int):
    """
    Load calibration from zip and compute rectification.
    Returns (map1L, map2L, map1R, map2R, K_rect_dict, stereo_rect_dict).
    """
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

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (IMG_W, IMG_H), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (IMG_W, IMG_H), cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (IMG_W, IMG_H), cv2.CV_32FC1)

    fx_rect    = float(P1[0, 0])
    fy_rect    = float(P1[1, 1])
    cx_rect    = float(P1[0, 2])
    cy_rect    = float(P1[1, 2])
    baseline_m = float(-P2[0, 3] / P1[0, 0])

    K_rect = {"fx": fx_rect, "fy": fy_rect, "cx": cx_rect, "cy": cy_rect,
              "width": IMG_W, "height": IMG_H}
    # After rectification: R=I, T=[baseline, 0, 0]
    stereo_rect = {"R": np.eye(3).tolist(), "T": [baseline_m, 0.0, 0.0],
                   "baseline_m": baseline_m}

    return map1L, map2L, map1R, map2R, fx_rect, baseline_m, K_rect, stereo_rect


# ── RAFT batch inference ───────────────────────────────────────────────────────

@torch.no_grad()
def raft_batch(model: torch.nn.Module,
               lefts: list[np.ndarray],
               rights: list[np.ndarray],
               fx: float, baseline: float,
               device: torch.device,
               iters: int = 32) -> list[np.ndarray]:
    """
    Run RAFT on a batch of rectified BGR images.
    Returns list of float16 depth maps (metres).
    """
    def to_t(bgr_list):
        tensors = []
        for bgr in bgr_list:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            tensors.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return torch.stack(tensors).to(device)   # (B, 3, H, W)

    lt = to_t(lefts)
    rt = to_t(rights)
    padder = InputPadder(lt.shape, divis_by=32)
    lt, rt = padder.pad(lt, rt)
    _, disp = model(lt, rt, iters=iters, test_mode=True)
    disp = padder.unpad(disp)
    disp_np = -disp.squeeze(1).cpu().numpy()   # (B, H, W) positive

    depths = []
    for d in disp_np:
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(d > 0, fx * baseline / d, 0.0).astype(np.float16)
        depths.append(depth)
    return depths


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scared-root",  default="/home/in4218/code/data/SCARED")
    ap.add_argument("--datasets",     nargs="+", type=int,
                    default=[1, 2, 3, 6, 7, 8, 9])
    ap.add_argument("--stride",       type=int, default=1,
                    help="Save every N-th frame (default: 1 = all frames)")
    ap.add_argument("--iters",        type=int, default=32)
    ap.add_argument("--batch-size",   type=int, default=4)
    ap.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device    = torch.device(args.device)
    scared_root = Path(args.scared_root)
    ext_root    = scared_root / "extracted"

    datasets = [d for d in args.datasets if d not in BAD_DATASETS]
    skipped  = [d for d in args.datasets if d in BAD_DATASETS]
    if skipped:
        print(f"Skipping datasets {skipped} (bad intrinsics per Allan et al. 2021)")

    print(f"Loading RAFT-Stereo …")
    model = build_model(device)

    total_saved = 0

    for ds in datasets:
        zip_path = scared_root / f"dataset_{ds}.zip"
        if not zip_path.exists():
            print(f"  dataset_{ds}.zip not found, skipping")
            continue

        with zipfile.ZipFile(zip_path) as zf:
            mp4_names = sorted(n for n in zf.namelist()
                               if n.endswith("data/rgb.mp4"))

        for mp4_name in mp4_names:
            kf_num = int(mp4_name.split("/")[1].split("_")[1])
            video_dir = ext_root / f"dataset_{ds}" / f"keyframe_{kf_num}" / "video"
            video_dir.mkdir(parents=True, exist_ok=True)

            # Calibration
            try:
                map1L, map2L, map1R, map2R, fx_rect, baseline_m, K_rect, stereo_rect = \
                    load_and_rectify(zip_path, ds, kf_num)
            except Exception as e:
                print(f"  dataset_{ds}/kf{kf_num}: calibration error: {e}, skipping")
                continue

            # Save calibration files (once per keyframe)
            with open(video_dir / "K_rect.json",     "w") as f: json.dump(K_rect,     f, indent=2)
            with open(video_dir / "stereo_rect.json","w") as f: json.dump(stereo_rect, f, indent=2)

            # Extract mp4
            with zipfile.ZipFile(zip_path) as zf:
                mp4_bytes = zf.read(mp4_name)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(mp4_bytes); tmp_mp4 = f.name

            cap       = cv2.VideoCapture(tmp_mp4)
            n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            n_to_save = len(range(0, n_frames, args.stride))

            print(f"  dataset_{ds}/kf{kf_num}: {n_frames} frames → "
                  f"{n_to_save} to save (stride={args.stride})")

            # Batch processing
            batch_lefts, batch_rights, batch_indices = [], [], []
            frame_idx = 0

            pbar = tqdm(total=n_to_save, desc=f"  ds{ds}/kf{kf_num}", leave=False)

            def flush_batch():
                if not batch_lefts:
                    return
                depths = raft_batch(model, batch_lefts, batch_rights,
                                    fx_rect, baseline_m, device, iters=args.iters)
                for i, (lR, rR, depth) in enumerate(zip(batch_lefts, batch_rights, depths)):
                    fidx = batch_indices[i]
                    frame_dir = video_dir / f"{fidx:06d}"
                    frame_dir.mkdir(exist_ok=True)
                    cv2.imwrite(str(frame_dir / "left.png"),  lR)
                    cv2.imwrite(str(frame_dir / "right.png"), rR)
                    np.save(frame_dir / "depth_left.npy", depth)
                batch_lefts.clear(); batch_rights.clear(); batch_indices.clear()
                pbar.update(len(depths))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % args.stride == 0:
                    # Check if already done
                    frame_dir = video_dir / f"{frame_idx:06d}"
                    if not (frame_dir / "depth_left.npy").exists():
                        left_bgr  = frame[:IMG_H]
                        right_bgr = frame[IMG_H:]
                        lR = cv2.remap(left_bgr,  map1L, map2L, cv2.INTER_LINEAR)
                        rR = cv2.remap(right_bgr, map1R, map2R, cv2.INTER_LINEAR)
                        batch_lefts.append(lR)
                        batch_rights.append(rR)
                        batch_indices.append(frame_idx)

                        if len(batch_lefts) >= args.batch_size:
                            flush_batch()
                    else:
                        pbar.update(1)

                frame_idx += 1

            flush_batch()
            pbar.close()
            cap.release()

            import os; os.unlink(tmp_mp4)
            total_saved += n_to_save

    print(f"\nDone. Total frames saved: {total_saved}")


if __name__ == "__main__":
    main()
