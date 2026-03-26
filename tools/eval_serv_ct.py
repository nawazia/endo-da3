"""
SERV-CT evaluation — AbsRel, δ<1.25, RMSE on 16 CT/RGB-scan-registered frames.

Reference: Edwards et al., "SERV-CT: A disparity dataset from CT for validation
           of endoscopic 3D reconstruction." Medical Image Analysis 2021.

Data layout (after extracting with tools or partial unzip):
    <root>/SERV-CT-ALL/Experiment_1/Left_rectified/001.png
    <root>/SERV-CT-ALL/Experiment_1/Reference_CT/DepthL/001.png
    <root>/SERV-CT-ALL/Experiment_1/Reference_CT/OcclusionL/001.png
    <root>/SERV-CT-ALL/Experiment_1/Rectified_calibration/001.json
    <root>/SERV-CT-ALL/Experiment_2/Left_rectified/009.png
    <root>/SERV-CT-ALL/Experiment_2/Reference_RGB/DepthL/009.png
    <root>/SERV-CT-ALL/Experiment_2/Reference_RGB/OcclusionL/009.png
    <root>/SERV-CT-ALL/Experiment_2/Rectified_calibration/009.json

Depth format: 16-bit PNG, value = depth_mm × 256 → depth_m = uint16 / 256 / 1000
Valid pixels: OcclusionL is NOT pure yellow (255,255,0) — that colour marks
              pixels outside the reference scan coverage.

Metrics (scale-invariant via median alignment):
    AbsRel : mean |pred - gt| / gt
    δ<1.25 : fraction where max(pred/gt, gt/pred) < 1.25
    RMSE   : sqrt(mean (pred - gt)²)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── helpers ───────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_EXPERIMENTS = [
    ("Experiment_1", "Reference_CT",  [f"{i:03d}" for i in range(1, 9)]),
    ("Experiment_2", "Reference_RGB", [f"{i:03d}" for i in range(9, 17)]),
]


def _load_K(json_path: Path) -> np.ndarray:
    """Extract 3×3 K from P1 projection matrix in calibration JSON."""
    data = json.loads(json_path.read_text())["P1"]["data"]
    # P1 is 3×4; K = first 3×3 block
    P1 = np.array(data).reshape(3, 4)
    return P1[:, :3].astype(np.float32)


def _load_depth_m(depth_png: Path) -> np.ndarray:
    """Load 16-bit depth PNG → float32 metres."""
    d = np.array(Image.open(depth_png)).astype(np.float32)
    return d / 256.0 / 1000.0   # mm×256 → metres


def _valid_mask(occlusion_png: Path) -> np.ndarray:
    """Boolean mask: True where pixel has valid reference depth (not yellow)."""
    occ = np.array(Image.open(occlusion_png))
    yellow = (occ[:, :, 0] == 255) & (occ[:, :, 1] == 255) & (occ[:, :, 2] == 0)
    return ~yellow


def load_servct_samples(root: str | Path) -> list[dict]:
    """
    Load all 16 SERV-CT samples.

    Returns list of dicts with keys:
        id       : str  (e.g. "001")
        exp      : str  (e.g. "Experiment_1")
        image    : PIL Image (720×576 RGB)
        depth_m  : (576, 720) float32 metres
        valid    : (576, 720) bool
        K        : (3, 3) float32
    """
    root = Path(root) / "SERV-CT-ALL"
    samples = []
    for exp, ref, ids in _EXPERIMENTS:
        exp_dir = root / exp
        for sid in ids:
            samples.append({
                "id":      sid,
                "exp":     exp,
                "image":   Image.open(exp_dir / "Left_rectified" / f"{sid}.png").convert("RGB"),
                "depth_m": _load_depth_m(exp_dir / ref / "DepthL" / f"{sid}.png"),
                "valid":   _valid_mask(exp_dir / ref / "OcclusionL" / f"{sid}.png"),
                "K":       _load_K(exp_dir / "Rectified_calibration" / f"{sid}.json"),
            })
    return samples


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    pred: np.ndarray,    # (H, W) predicted depth (any scale)
    gt:   np.ndarray,    # (H, W) GT depth in metres
    mask: np.ndarray,    # (H, W) bool — valid pixels
) -> dict:
    """
    Scale-invariant depth metrics via median alignment.
    pred is scaled by  s = median(gt[mask]) / median(pred[mask])  before eval.
    """
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    # median scale alignment
    med_p = np.median(p)
    med_g = np.median(g)
    if med_p < 1e-6:
        return {"AbsRel": float("nan"), "d1": float("nan"), "RMSE": float("nan")}
    s = med_g / med_p
    p = p * s

    abs_rel = float(np.mean(np.abs(p - g) / g))
    ratio   = np.maximum(p / g, g / p)
    d1      = float(np.mean(ratio < 1.25))
    rmse    = float(np.sqrt(np.mean((p - g) ** 2)))

    return {"AbsRel": abs_rel, "d1": d1, "RMSE": rmse}


# ── inference + eval ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, root: str | Path, img_size: int = 336,
             device: str = "cuda") -> dict:
    """
    Run model on all 16 SERV-CT frames and return averaged metrics.

    Args:
        model    : EndoDA3 model (eval mode)
        root     : SERV-CT root (contains SERV-CT-ALL/)
        img_size : inference resolution (must match model)
        device   : torch device string

    Returns:
        dict with AbsRel, d1, RMSE (averaged over all 16 frames)
    """
    tf = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

    samples = load_servct_samples(root)
    orig_h, orig_w = 576, 720

    all_metrics = []
    for s in samples:
        # run model
        x = tf(s["image"]).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,3,H,W)
        pred = model(x)["depth"].squeeze().cpu().float().numpy()  # (img_size, img_size)

        # resize pred back to original resolution for fair comparison with GT
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        pred_orig = F.interpolate(pred_t, size=(orig_h, orig_w),
                                  mode="bilinear", align_corners=False)
        pred_orig = pred_orig.squeeze().numpy()

        m = compute_metrics(pred_orig, s["depth_m"], s["valid"])
        all_metrics.append(m)

    avg = {k: float(np.nanmean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    return avg


# ── standalone script ─────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser("SERV-CT evaluation")
    p.add_argument("--ckpt",        required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--gastronet",   required=True, help="Path to GastroNet dinov2.pth")
    p.add_argument("--serv-ct-root", required=True, help="SERV-CT root (contains SERV-CT-ALL/)")
    p.add_argument("--lora-rank",   type=int, default=0,
                   help="LoRA rank if checkpoint uses LoRA (0 = no LoRA)")
    p.add_argument("--lora-alpha",  type=float, default=4.0)
    p.add_argument("--img-size",    type=int, default=336)
    args = p.parse_args()

    from endo_da3 import EndoDA3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EndoDA3.from_pretrained(img_size=args.img_size, with_camera=False, device=device)

    ckpt = torch.load(args.gastronet, map_location="cpu")
    gastro_sd = {k.replace("backbone.", ""): v
                 for k, v in ckpt["teacher"].items()
                 if k.startswith("backbone.")}
    model.replace_backbone(gastro_sd)

    if args.lora_rank > 0:
        from endo_da3.lora import inject_lora
        inject_lora(model, rank=args.lora_rank, lora_alpha=args.lora_alpha)

    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    print(f"Evaluating on SERV-CT (16 frames)…")
    metrics = evaluate(model, args.serv_ct_root, img_size=args.img_size, device=device)
    print(f"  AbsRel : {metrics['AbsRel']:.4f}")
    print(f"  δ<1.25 : {metrics['d1']:.4f}")
    print(f"  RMSE   : {metrics['RMSE']:.4f} m")


if __name__ == "__main__":
    main()
