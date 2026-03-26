"""
Generate pseudo-GT depth maps for Hamlyn daVinci stereo dataset using RAFT-Stereo.

Outputs one float16 .npy depth map (metres) per stereo pair into:
    <data_root>/depth_pseudogt/000001.npy  ...

Usage:
    python tools/generate_hamlyn_pseudogt.py [--split train|test] [--batch_size 8]
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

RAFT_ROOT = Path("/home/in4218/code/RAFT-Stereo")
# RAFT-Stereo needs both the repo root (for `from core.xxx` inside raft_stereo.py)
# and the core dir (so `from raft_stereo import RAFTStereo` resolves)
sys.path.insert(0, str(RAFT_ROOT))
sys.path.insert(0, str(RAFT_ROOT / "core"))

from raft_stereo import RAFTStereo          # noqa: E402
from utils.utils import InputPadder         # noqa: E402

# ---------------------------------------------------------------------------
# Hamlyn camera params
# ---------------------------------------------------------------------------
FX        = 373.47833252
BASELINE  = 5.63117313e-3   # metres

DATA_ROOT  = Path("/home/in4218/code/data/Hamlyn/daVinci")
RAFT_CKPT  = RAFT_ROOT / "models" / "iraftstereo_rvc.pth"


# ---------------------------------------------------------------------------
# RAFT-Stereo model
# ---------------------------------------------------------------------------

def build_model(device: torch.device) -> torch.nn.Module:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dims",        nargs="+", type=int, default=[128] * 3)
    parser.add_argument("--corr_implementation", default="reg")
    parser.add_argument("--shared_backbone",    action="store_true")
    parser.add_argument("--corr_levels",        type=int, default=4)
    parser.add_argument("--corr_radius",        type=int, default=4)
    parser.add_argument("--n_downsample",       type=int, default=2)
    parser.add_argument("--context_norm",       default="instance")
    parser.add_argument("--slow_fast_gru",      action="store_true")
    parser.add_argument("--n_gru_layers",       type=int, default=3)
    parser.add_argument("--mixed_precision",    action="store_true")
    args, _ = parser.parse_known_args()

    model = torch.nn.DataParallel(RAFTStereo(args))
    model.load_state_dict(torch.load(str(RAFT_CKPT), map_location="cpu"))
    model = model.module.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = np.array(Image.open(path)).astype(np.uint8)          # (H, W, 3)
    t   = torch.from_numpy(img).permute(2, 0, 1).float()       # (3, H, W)
    return t.unsqueeze(0).to(device)                            # (1, 3, H, W)


@torch.no_grad()
def disparity_to_depth(disp_px: np.ndarray) -> np.ndarray:
    """disp_px: (H, W) float, pixels.  Returns depth in metres, float16."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp_px > 0, FX * BASELINE / disp_px, 0.0)
    return depth.astype(np.float16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",      default="train", choices=["train", "test"])
    ap.add_argument("--iters",      type=int, default=32, help="RAFT update iters")
    ap.add_argument("--batch_size", type=int, default=8,  help="images per GPU batch")
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device   = torch.device(args.device)
    split_dir = DATA_ROOT / args.split

    left_dir  = split_dir / "image_0"
    right_dir = split_dir / "image_1"
    out_dir   = split_dir / "depth_pseudogt"
    out_dir.mkdir(exist_ok=True)

    left_files  = sorted(left_dir.glob("*.png"))
    right_files = sorted(right_dir.glob("*.png"))
    assert len(left_files) == len(right_files), "left/right count mismatch"

    # Skip already-generated files
    todo = [(l, r) for l, r in zip(left_files, right_files)
            if not (out_dir / (l.stem + ".npy")).exists()]
    print(f"Split: {args.split}  |  Total: {len(left_files)}  |  Todo: {len(todo)}")

    if not todo:
        print("All depth maps already generated.")
        return

    print(f"Loading RAFT-Stereo from {RAFT_CKPT} …")
    model = build_model(device)

    # Process in batches
    batch_size = args.batch_size
    for i in tqdm(range(0, len(todo), batch_size), unit="batch"):
        batch = todo[i : i + batch_size]

        lefts  = torch.cat([load_image(l, device) for l, _ in batch])   # (B, 3, H, W)
        rights = torch.cat([load_image(r, device) for _, r in batch])

        padder = InputPadder(lefts.shape, divis_by=32)
        lefts, rights = padder.pad(lefts, rights)

        with torch.no_grad():
            _, disp_up = model(lefts, rights, iters=args.iters, test_mode=True)

        disp_up = padder.unpad(disp_up)                 # (B, 1, H, W)
        disp_np = -disp_up.squeeze(1).cpu().numpy()    # (B, H, W) — RAFT returns negative flow

        for (l_path, _), disp in zip(batch, disp_np):
            depth = disparity_to_depth(disp)
            np.save(out_dir / (l_path.stem + ".npy"), depth)


if __name__ == "__main__":
    main()
