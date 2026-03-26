"""
Depth comparison across training stages.

Image sources:
  SOH          — Sydney Opera House  (natural,   OOD)
  Kvasir       — endoscopy z-line    (endoscopy, OOD)
  Hamlyn       — real robotic surgery (Stage 2a in-dist)
  StereoMIS    — in-vivo porcine surgery (Stage 2a in-dist)
  SimCol3D     — synthetic colon     (Stage 1 in-dist)
  C3VD         — ex-vivo colon       (Stage 1 in-dist)
  EndoSLAM     — synthetic colon     (Stage 1 in-dist)
  PolypSense3D — synthetic polyp     (Stage 1 in-dist)

Layout: 8 rows × 6 cols
  [Input | GT/Pseudo-GT | DA3-BASE | Endo-DA3+GastroNet | Stage 1 | Stage 2a]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SOH_PATH       = '/home/in4218/code/data/SOH/000.png'
KVASIR_PATH    = '/home/in4218/code/data/kvasir-dataset-v2/normal-z-line/0be91a4e-be3d-4c06-92d7-6e0ee417f55a.jpg'
HAMLYN_ROOT      = Path('/home/in4218/code/data/Hamlyn/daVinci')
STEREOMIS_ROOT   = Path('/home/in4218/code/data/StereoMIS')
GASTRONET_CKPT = '/home/in4218/code/GastroNet/gastronet/dinov2.pth'
STAGE1_CKPT    = '/home/in4218/code/endo-da3/runs/stage1/last.pt'
STAGE2A_CKPT         = '/home/in4218/code/endo-da3/runs/stage2a/best.pt'
STAGE2B_DISTILL_CKPT = '/home/in4218/code/endo-da3/runs/stage2b_distill/last.pt'
STAGE3_CKPT          = '/home/in4218/code/endo-da3/runs/stage3/best.pt'
DATA           = Path('/home/in4218/code/data')
OUT_PATH       = '/home/in4218/code/endo-da3/depth_comparison.png'

IMG_SIZE = 336
LORA_RANK  = 4
LORA_ALPHA = 4.0
_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
    ])


def load_gastronet_sd():
    ckpt = torch.load(GASTRONET_CKPT, map_location='cpu')
    return {k.replace('backbone.', ''): v
            for k, v in ckpt['teacher'].items()
            if k.startswith('backbone.')}


def denorm(t):
    """(3,H,W) normalised tensor → (H,W,3) uint8 for display."""
    img = t.permute(1, 2, 0).cpu().numpy()
    return ((img * _STD + _MEAN).clip(0, 1) * 255).astype(np.uint8)


def depth_to_rgb(depth, vmin=None, vmax=None, cmap='magma_r'):
    """Depth tensor/array (H,W) → (H,W,3) uint8 colourmap.

    If vmin/vmax provided (per-row shared scale), depths are comparable across
    columns. Otherwise falls back to per-image percentile stretch.
    """
    d = depth.detach().cpu().float().numpy() if hasattr(depth, 'detach') else np.asarray(depth, dtype=np.float32)
    if vmin is None or vmax is None:
        valid = d > 0
        if valid.any():
            vmin, vmax = np.percentile(d[valid], [2, 98])
        else:
            vmin, vmax = float(d.min()), float(d.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    return (plt.get_cmap(cmap)(norm(d))[..., :3] * 255).astype(np.uint8)


def row_scale(*depths):
    """Compute shared vmin/vmax across a row of depth maps (ignores None and zeros)."""
    vals = []
    for d in depths:
        if d is None:
            continue
        arr = d.detach().cpu().float().numpy() if hasattr(d, 'detach') else np.asarray(d, dtype=np.float32)
        valid = arr[arr > 0]
        if valid.size:
            vals.append(valid)
    if not vals:
        return 0.0, 1.0
    all_vals = np.concatenate(vals)
    return float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))


@torch.no_grad()
def run_single(model, pil_img, device):
    x = get_transform()(pil_img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,3,H,W)
    return model(x)['depth'].squeeze().cpu().float()


@torch.no_grad()
def run_pair(model, pil_img, device):
    x = get_transform()(pil_img).unsqueeze(0).unsqueeze(1)
    x = x.expand(-1, 2, -1, -1, -1).to(device)                        # (1,2,3,H,W)
    return model(x)['depth'][0, 0].detach().cpu().float()


def pil_from_dataset(ds_cls, root):
    """Returns (pil_image, gt_depth_tensor_or_None)."""
    ds = ds_cls(root, split='val', img_size=IMG_SIZE, seq_len=1, with_pose=False)
    sample = ds[0]
    pil = Image.fromarray(denorm(sample['images'][0]))
    gt  = sample['depths'][0]   # (H, W) float32, metres
    return pil, gt


def pil_from_hamlyn():
    """Returns (pil_image, pseudo_gt_depth_tensor)."""
    from endo_da3.data.hamlyn import HamlynDataset
    ds = HamlynDataset(HAMLYN_ROOT, split='val', img_size=IMG_SIZE)
    sample = ds[0]
    pil = Image.fromarray(denorm(sample['images'][0]))
    gt  = sample['depths'][0]   # (H, W) RAFT pseudo-GT, metres
    return pil, gt


def pil_from_stereomis():
    """Returns (pil_image, pseudo_gt_depth_tensor) from StereoMIS P1."""
    from endo_da3.data.stereomis import StereoMISDataset
    ds = StereoMISDataset(STEREOMIS_ROOT, seq='P1', img_size=IMG_SIZE)
    sample = ds[len(ds) // 2]   # mid-sequence for variety
    pil = Image.fromarray(denorm(sample['images'][0]))
    gt  = sample['depths'][0]   # (H, W) RAFT pseudo-GT, metres
    return pil, gt


def main():
    from endo_da3 import EndoDA3
    from endo_da3.lora import inject_lora
    from endo_da3.data import (
        SimCol3DDataset, C3VDDataset,
        EndoSLAMSynthDataset, PolypSense3DVirtualDataset,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gastro_sd = load_gastronet_sd()

    print("Loading DA3-BASE...")
    da3 = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device)

    print("Loading Endo-DA3 + GastroNet (no fine-tuning)...")
    endo = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device)
    endo.replace_backbone(gastro_sd)

    print("Loading Stage 1...")
    stage1 = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device)
    stage1.replace_backbone(gastro_sd)
    sd = torch.load(STAGE1_CKPT, map_location=device)
    stage1.load_state_dict(sd['model'] if 'model' in sd else sd)

    stage2a = None
    if Path(STAGE2A_CKPT).exists():
        print("Loading Stage 2a...")
        stage2a = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device)
        stage2a.replace_backbone(gastro_sd)
        inject_lora(stage2a, rank=LORA_RANK, lora_alpha=LORA_ALPHA)
        sd = torch.load(STAGE2A_CKPT, map_location=device)
        stage2a.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
    else:
        print(f"Stage 2a checkpoint not found ({STAGE2A_CKPT}) — skipping column.")

    stage2b_distill = None
    if Path(STAGE2B_DISTILL_CKPT).exists():
        print("Loading Stage 2b-distill...")
        stage2b_distill = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device)
        stage2b_distill.replace_backbone(gastro_sd)
        inject_lora(stage2b_distill, rank=LORA_RANK, lora_alpha=LORA_ALPHA)
        sd = torch.load(STAGE2B_DISTILL_CKPT, map_location=device)
        stage2b_distill.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
    else:
        print(f"Stage 2b-distill checkpoint not found ({STAGE2B_DISTILL_CKPT}) — skipping column.")

    stage3 = None
    if Path(STAGE3_CKPT).exists():
        print("Loading Stage 3...")
        stage3 = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device)
        stage3.replace_backbone(gastro_sd)
        inject_lora(stage3, rank=LORA_RANK, lora_alpha=LORA_ALPHA)
        sd = torch.load(STAGE3_CKPT, map_location=device)
        stage3.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
    else:
        print(f"Stage 3 checkpoint not found ({STAGE3_CKPT}) — skipping column.")

    # ── Image sources — (label, pil, gt_or_None) ─────────────────────────────
    print("Loading images...")
    hamlyn_pil, hamlyn_gt       = pil_from_hamlyn()
    stereomis_pil, stereomis_gt = pil_from_stereomis()
    sources = [
        ('SOH (natural)',           Image.open(SOH_PATH).convert('RGB'),    None),
        ('Kvasir (endo)',           Image.open(KVASIR_PATH).convert('RGB'), None),
        ('Hamlyn (real surgery)',   hamlyn_pil,                             hamlyn_gt),
        ('StereoMIS (porcine)',     stereomis_pil,                          stereomis_gt),
        ('SimCol3D',                *pil_from_dataset(SimCol3DDataset,            DATA / 'SimCol3D')),
        ('C3VD',                    *pil_from_dataset(C3VDDataset,                DATA / 'C3VD')),
        ('EndoSLAM',                *pil_from_dataset(EndoSLAMSynthDataset,       DATA / 'EndoSLAM')),
        ('PolypSense3D',            *pil_from_dataset(PolypSense3DVirtualDataset, DATA / 'PolypSense3D')),
    ]

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running inference...")
    rows = []
    for label, pil_img, gt in sources:
        print(f"  {label}...")
        d_da3           = run_single(da3,            pil_img, device)
        d_endo          = run_single(endo,           pil_img, device)
        d_stage1        = run_pair(stage1,           pil_img, device)
        d_stage2a       = run_pair(stage2a,          pil_img, device) if stage2a else None
        d_stage2b_dist  = run_pair(stage2b_distill,  pil_img, device) if stage2b_distill else None
        d_stage3        = run_pair(stage3,            pil_img, device) if stage3 else None
        pil_sq          = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        msg = f"    DA3: {d_da3.mean():.3f}m  Stage1: {d_stage1.mean():.4f}m"
        if d_stage2a is not None:
            msg += f"  Stage2a: {d_stage2a.mean():.4f}m"
        if d_stage2b_dist is not None:
            msg += f"  2b-distill: {d_stage2b_dist.mean():.4f}m"
        if d_stage3 is not None:
            msg += f"  Stage3: {d_stage3.mean():.4f}m"
        print(msg)
        rows.append((label, pil_sq, gt, d_da3, d_endo, d_stage1, d_stage2a, d_stage2b_dist, d_stage3))

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_cols = 5 + (1 if stage2a else 0) + (1 if stage2b_distill else 0) + (1 if stage3 else 0)
    col_titles = ['Input', 'GT / Pseudo-GT', 'DA3-BASE', 'Endo-DA3\n(GastroNet init)', 'Stage 1']
    if stage2a:
        col_titles.append('Stage 2a')
    if stage2b_distill:
        col_titles.append('Stage 2b\n(distill)')
    if stage3:
        col_titles.append('Stage 3\n(SCARED)')

    n = len(rows)
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 3.5, n * 3.2))

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11, fontweight='bold')

    blank = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    for r, (label, pil_img, gt, d_da3, d_endo, d_stage1, d_stage2a, d_stage2b_dist, d_stage3) in enumerate(rows):
        # Each depth map normalised independently — models have different output
        # scales so per-image percentile stretch gives the fairest visual comparison.
        axes[r, 0].imshow(pil_img)
        axes[r, 1].imshow(depth_to_rgb(gt) if gt is not None else blank)
        axes[r, 2].imshow(depth_to_rgb(d_da3))
        axes[r, 3].imshow(depth_to_rgb(d_endo))
        axes[r, 4].imshow(depth_to_rgb(d_stage1))
        col = 5
        if stage2a:
            axes[r, col].imshow(depth_to_rgb(d_stage2a) if d_stage2a is not None else blank)
            col += 1
        if stage2b_distill:
            axes[r, col].imshow(depth_to_rgb(d_stage2b_dist) if d_stage2b_dist is not None else blank)
            col += 1
        if stage3:
            axes[r, col].imshow(depth_to_rgb(d_stage3) if d_stage3 is not None else blank)
        for ax in axes[r]:
            ax.axis('off')
        y = 1.0 - (r + 0.5) / n
        fig.text(0.01, y, label, fontsize=10, va='center', ha='left', fontweight='bold')

    plt.tight_layout(rect=[0.08, 0, 1, 1])
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {OUT_PATH}")


if __name__ == '__main__':
    main()
