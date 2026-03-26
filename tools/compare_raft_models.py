"""
Compare RAFT-Stereo pseudo-GT depth from sceneflow vs iraftstereo_rvc on a
small sample of Hamlyn frames.

Outputs: tools/raft_comparison.png
         - rows = frames, cols = [Left RGB | SceneFlow depth | RVC depth | diff]
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image
from tqdm import tqdm

RAFT_ROOT  = Path("/home/in4218/code/RAFT-Stereo")
DATA_ROOT  = Path("/home/in4218/code/data/Hamlyn/daVinci")
OUT_PATH   = Path(__file__).parent / "raft_comparison.png"
N_FRAMES   = 10

sys.path.insert(0, str(RAFT_ROOT))
sys.path.insert(0, str(RAFT_ROOT / "core"))

from raft_stereo import RAFTStereo        # noqa: E402
from utils.utils import InputPadder       # noqa: E402

FX       = 373.47833252
BASELINE = 5.63117313e-3


def build_model(context_norm: str, device: torch.device):
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dims",         nargs="+", type=int, default=[128] * 3)
    p.add_argument("--corr_implementation", default="reg")
    p.add_argument("--shared_backbone",     action="store_true")
    p.add_argument("--corr_levels",         type=int, default=4)
    p.add_argument("--corr_radius",         type=int, default=4)
    p.add_argument("--n_downsample",        type=int, default=2)
    p.add_argument("--context_norm",        default=context_norm)
    p.add_argument("--slow_fast_gru",       action="store_true")
    p.add_argument("--n_gru_layers",        type=int, default=3)
    p.add_argument("--mixed_precision",     action="store_true")
    args, _ = p.parse_known_args()
    return args


def load_model(ckpt_path: Path, context_norm: str, device: torch.device):
    args = build_model(context_norm, device)
    model = torch.nn.DataParallel(RAFTStereo(args))
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    return model.module.to(device).eval()


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    img = np.array(Image.open(path)).astype(np.uint8)
    t   = torch.from_numpy(img).permute(2, 0, 1).float()
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def run(model, left: torch.Tensor, right: torch.Tensor, iters: int = 32) -> np.ndarray:
    padder = InputPadder(left.shape, divis_by=32)
    l, r   = padder.pad(left, right)
    _, disp_up = model(l, r, iters=iters, test_mode=True)
    disp_up    = padder.unpad(disp_up)
    disp_np    = -disp_up.squeeze().cpu().numpy()       # (H, W), positive
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp_np > 0, FX * BASELINE / disp_np, 0.0)
    return depth.astype(np.float32)


def depth_to_rgb(d: np.ndarray, vmin=None, vmax=None, cmap="magma_r") -> np.ndarray:
    valid = d > 0
    if vmin is None:
        vmin, vmax = (np.percentile(d[valid], [2, 98]) if valid.any() else (0, 1))
    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    return (plt.get_cmap(cmap)(norm(d))[..., :3] * 255).astype(np.uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir   = DATA_ROOT / "test"
    all_left    = sorted((split_dir / "image_0").glob("*.png"))
    all_right   = sorted((split_dir / "image_1").glob("*.png"))
    rng         = np.random.default_rng(42)
    indices     = rng.choice(len(all_left), size=N_FRAMES, replace=False)
    left_files  = [all_left[i]  for i in sorted(indices)]
    right_files = [all_right[i] for i in sorted(indices)]

    print("Loading SceneFlow model…")
    sf_model  = load_model(RAFT_ROOT / "models" / "raftstereo-sceneflow.pth",
                           context_norm="batch", device=device)
    print("Loading RVC model…")
    rvc_model = load_model(RAFT_ROOT / "models" / "iraftstereo_rvc.pth",
                           context_norm="instance", device=device)

    rows = []
    for lp, rp in tqdm(zip(left_files, right_files), total=N_FRAMES):
        left  = load_image(lp, device)
        right = load_image(rp, device)

        d_sf  = run(sf_model,  left, right)
        d_rvc = run(rvc_model, left, right)

        # shared scale for fair comparison
        valid = d_sf > 0
        vmin, vmax = np.percentile(d_sf[valid], [2, 98]) if valid.any() else (0, 1)

        rgb   = np.array(Image.open(lp).convert("RGB"))
        diff  = np.abs(d_sf - d_rvc)

        rows.append((rgb, d_sf, d_rvc, diff, vmin, vmax))

    # ── plot ──────────────────────────────────────────────────────────────────
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 4, n * 2.5))
    for c, title in enumerate(["Left RGB", "SceneFlow depth", "RVC depth", "|SF − RVC|"]):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")

    for r, (rgb, d_sf, d_rvc, diff, vmin, vmax) in enumerate(rows):
        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(depth_to_rgb(d_sf,  vmin, vmax))
        axes[r, 2].imshow(depth_to_rgb(d_rvc, vmin, vmax))
        # diff: use its own scale
        dmax = np.percentile(diff[diff > 0], 98) if (diff > 0).any() else 1.0
        axes[r, 3].imshow(depth_to_rgb(diff, 0, dmax, cmap="hot"))
        for ax in axes[r]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
