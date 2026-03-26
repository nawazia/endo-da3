"""
Generate pseudo-GT depth maps for StereoMIS using RAFT-Stereo.

Reads rectified left/right JPEG frames extracted by extract_stereomis.py and
saves one float16 .npy depth map (metres) per frame into:
    <data_root>/<seq>/depth_pseudogt/000000.npy  ...

Depth = fx * B / disparity, where fx and B are read from per-sequence
rectified_K.json and baseline.txt.

Usage:
    python tools/generate_stereomis_pseudogt.py [--seq P1] [--batch_size 16]
"""

import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

RAFT_ROOT  = Path("/home/in4218/code/RAFT-Stereo")
DATA_ROOT  = Path("/home/in4218/code/data/StereoMIS")

sys.path.insert(0, str(RAFT_ROOT))
sys.path.insert(0, str(RAFT_ROOT / "core"))

from raft_stereo import RAFTStereo          # noqa: E402
from utils.utils import InputPadder         # noqa: E402

RAFT_CKPT = RAFT_ROOT / "models" / "iraftstereo_rvc.pth"

# All sequences (P1=train, P2_*/P3=test)
DEFAULT_SEQS = ["P1", "P2_0", "P2_1", "P2_2", "P2_3",
                "P2_4", "P2_5", "P2_6", "P2_7", "P2_8", "P3"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB")).astype(np.uint8)
    t   = torch.from_numpy(img).permute(2, 0, 1).float()
    return t.unsqueeze(0).to(device)


def disparity_to_depth(disp_px: np.ndarray, fx: float, B: float) -> np.ndarray:
    """disp_px: (H, W) float pixels → depth in metres, float16."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp_px > 0, fx * B / disp_px, 0.0)
    return depth.astype(np.float16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_sequence(seq: str, model: torch.nn.Module,
                     device: torch.device, batch_size: int, iters: int):
    seq_dir   = DATA_ROOT / seq
    left_dir  = seq_dir / "left"
    right_dir = seq_dir / "right"
    out_dir   = seq_dir / "depth_pseudogt"
    out_dir.mkdir(exist_ok=True)

    # Per-sequence camera params
    with open(seq_dir / "rectified_K.json") as f:
        K = json.load(f)
    fx = K["fx"]
    with open(seq_dir / "baseline.txt") as f:
        B = float(f.read().strip())

    print(f"  fx={fx:.2f}  B={B*1000:.3f}mm  "
          f"depth_const={fx*B:.4f} m·px")

    left_files  = sorted(left_dir.glob("*.jpg"))
    right_files = sorted(right_dir.glob("*.jpg"))
    assert len(left_files) == len(right_files), "left/right count mismatch"

    # Skip already generated
    todo = [(l, r) for l, r in zip(left_files, right_files)
            if not (out_dir / (l.stem + ".npy")).exists()]
    print(f"  Total: {len(left_files)}  Todo: {len(todo)}")

    if not todo:
        print("  All depth maps already generated.")
        return

    for i in tqdm(range(0, len(todo), batch_size), desc=seq, unit="batch"):
        batch = todo[i: i + batch_size]

        lefts  = torch.cat([load_image(l, device) for l, _ in batch])
        rights = torch.cat([load_image(r, device) for _, r in batch])

        padder = InputPadder(lefts.shape, divis_by=32)
        lefts, rights = padder.pad(lefts, rights)

        with torch.no_grad():
            _, disp_up = model(lefts, rights, iters=iters, test_mode=True)

        disp_up = padder.unpad(disp_up)
        disp_np = -disp_up.squeeze(1).cpu().numpy()   # (B, H, W), positive

        for (l_path, _), disp in zip(batch, disp_np):
            depth = disparity_to_depth(disp, fx, B)
            np.save(out_dir / (l_path.stem + ".npy"), depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",        nargs="+", default=DEFAULT_SEQS,
                    help="Sequences to process (default: P1)")
    ap.add_argument("--iters",      type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading RAFT-Stereo from {RAFT_CKPT} …")
    model = build_model(device)

    for seq in args.seq:
        print(f"\n=== {seq} ===")
        process_sequence(seq, model, device, args.batch_size, args.iters)

    print("\nDone.")


if __name__ == "__main__":
    main()
