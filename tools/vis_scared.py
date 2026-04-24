"""
Visualise SCARED val predictions (datasets 8 & 9, 10 keyframes total).

Layout: 10 rows × N cols
    [Input | GT depth | DA3-BASE | Stage 2a | Stage 3 | ...]

Depth maps use shared per-row scale (GT 2nd–98th percentile over valid pixels).
Predictions are median-aligned to GT before display.
Invalid pixels shown in dark grey.

Usage:
    python tools/vis_scared.py \
        --scared-root ~/code/data/SCARED \
        --gastronet ~/code/GastroNet/gastronet/dinov2.pth \
        --out scared_vis.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endo_da3 import EndoDA3
from endo_da3.lora import inject_lora
from tools.eval_scared import load_scared_val_samples

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 336


def depth_to_rgb(depth, valid, vmin, vmax, cmap="magma_r"):
    d    = np.array(depth, dtype=np.float32)
    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    rgba = plt.get_cmap(cmap)(norm(d))
    rgba[~valid] = [0.15, 0.15, 0.15, 1.0]
    return (rgba[..., :3] * 255).astype(np.uint8)


def load_gastronet_sd(path):
    ckpt = torch.load(path, map_location="cpu")
    return {k.replace("backbone.", ""): v
            for k, v in ckpt["teacher"].items()
            if k.startswith("backbone.")}


def build_model(gastro_sd, ckpt_path, use_lora, device):
    m = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device)
    m.replace_backbone(gastro_sd)
    if use_lora:
        inject_lora(m, rank=4, lora_alpha=4.0)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        m.load_state_dict(sd, strict=False)
    return m.eval()


@torch.no_grad()
def run_model(model, pil_img, orig_h, orig_w, device):
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    x    = tf(pil_img).unsqueeze(0).unsqueeze(0).to(device)
    pred = model(x)["depth"].squeeze().cpu().float()
    pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0),
                         size=(orig_h, orig_w),
                         mode="bilinear", align_corners=False).squeeze().numpy()
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scared-root", required=True)
    p.add_argument("--gastronet",   required=True)
    p.add_argument("--stage2a",     default="runs/stage2a/best.pt")
    p.add_argument("--stage3",      default="runs/stage3/best.pt")
    p.add_argument("--out",         default="scared_vis.png")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device    = args.device
    gastro_sd = load_gastronet_sd(args.gastronet)

    model_specs = [("DA3-BASE", None, False)]
    for label, path, lora in [
        ("Stage 2a", args.stage2a, True),
        ("Stage 3",  args.stage3,  True),
    ]:
        if Path(path).exists():
            model_specs.append((label, path, lora))
        else:
            print(f"  {label} not found ({path}) — skipping")

    samples = load_scared_val_samples(args.scared_root)
    print(f"Loaded {len(samples)} SCARED val keyframes, {len(model_specs)} models")

    n_rows    = len(samples)
    n_cols    = 2 + len(model_specs)
    col_titles = ["Input", "GT depth"] + [label for label, _, _ in model_specs]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.0, n_rows * 2.5))

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")

    # Pre-fill input + GT columns, cache scale
    for r, s in enumerate(samples):
        gt    = s["depth_m"]
        valid = s["valid"]
        gt_valid = gt[valid]
        vmin  = float(np.percentile(gt_valid, 2))
        vmax  = float(np.percentile(gt_valid, 98))
        s["vmin"] = vmin
        s["vmax"] = vmax
        orig_h, orig_w = gt.shape
        s["orig_h"] = orig_h
        s["orig_w"] = orig_w

        axes[r, 0].imshow(s["image"])
        axes[r, 0].axis("off")
        axes[r, 1].imshow(depth_to_rgb(gt, valid, vmin, vmax))
        axes[r, 1].axis("off")
        axes[r, 0].set_ylabel(s["id"], fontsize=7, rotation=0,
                               labelpad=80, va="center")

    # Run each model, then free
    for col_idx, (label, path, lora) in enumerate(model_specs, start=2):
        print(f"  Running {label}…")
        model = build_model(gastro_sd, path, lora, device)
        for r, s in enumerate(samples):
            pred  = run_model(model, s["image"], s["orig_h"], s["orig_w"], device)
            valid = s["valid"]
            med_p = np.median(pred[valid])
            med_g = np.median(s["depth_m"][valid])
            if med_p > 1e-6:
                pred = pred * (med_g / med_p)
            axes[r, col_idx].imshow(
                depth_to_rgb(pred, valid, s["vmin"], s["vmax"])
            )
            axes[r, col_idx].axis("off")
        del model
        torch.cuda.empty_cache()

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
