"""
SCARED evaluation — AbsRel, δ<1.25, RMSE on datasets 8 & 9 (val split).

Each dataset has 5 keyframes with structured-light GT depth (left camera).
Invalid pixels (depth == 0) are excluded from all metrics.
Metrics are scale-invariant via median alignment (same as SERV-CT eval).

Data layout (after extract_scared.py):
    <root>/extracted/dataset_8/keyframe_0/left.png
    <root>/extracted/dataset_8/keyframe_0/depth_left.npy
    <root>/extracted/dataset_8/keyframe_0/K_left.json
    ...
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

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

VAL_DATASETS = [8, 9]
N_KEYFRAMES  = 5


def load_scared_val_samples(root: str | Path) -> list[dict]:
    """
    Load all val keyframes (datasets 8 & 9, 5 keyframes each = 10 samples).

    Returns list of dicts:
        id       : str  (e.g. "dataset_8/keyframe_0")
        image    : PIL Image (1280×1024 RGB)
        depth_m  : (1024, 1280) float32 metres
        valid    : (1024, 1280) bool
    """
    root    = Path(root) / "extracted"
    samples = []
    for ds in VAL_DATASETS:
        ds_dir = root / f"dataset_{ds}"
        for kf in range(N_KEYFRAMES):
            kf_dir  = ds_dir / f"keyframe_{kf}"
            if not kf_dir.exists():
                continue
            depth = np.load(kf_dir / "depth_left.npy").astype(np.float32)
            samples.append({
                "id":      f"dataset_{ds}/keyframe_{kf}",
                "image":   Image.open(kf_dir / "left.png").convert("RGB"),
                "depth_m": depth,
                "valid":   depth > 0,
            })
    return samples


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """Scale-invariant metrics via median alignment."""
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    med_p = np.median(p)
    if med_p < 1e-6:
        return {"AbsRel": float("nan"), "d1": float("nan"), "d2": float("nan"), "d3": float("nan"), "RMSE": float("nan")}
    p = p * (np.median(g) / med_p)
    abs_rel = float(np.mean(np.abs(p - g) / g))
    ratio   = np.maximum(p / g, g / p)
    d1      = float(np.mean(ratio < 1.25))
    d3      = float(np.mean(ratio < 1.05))
    mae     = float(np.mean(np.abs(p - g)))
    rmse    = float(np.sqrt(np.mean((p - g) ** 2)))
    return {"AbsRel": abs_rel, "d1": d1, "d3": d3, "MAE": mae, "RMSE": rmse}


@torch.no_grad()
def evaluate(model, root: str | Path, img_size: int = 336,
             device: str = "cuda") -> dict:
    """Run model on all SCARED val keyframes and return averaged metrics."""
    tf = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

    samples  = load_scared_val_samples(root)
    all_metrics = []

    for s in samples:
        orig_h, orig_w = s["depth_m"].shape
        x    = tf(s["image"]).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x)["depth"].squeeze().cpu().float()
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0),
                             size=(orig_h, orig_w),
                             mode="bilinear", align_corners=False).squeeze().numpy()
        all_metrics.append(compute_metrics(pred, s["depth_m"], s["valid"]))

    return {k: float(np.nanmean([m[k] for m in all_metrics])) for k in all_metrics[0]}
