"""
Stage 1 — Align: train DualDPT head + DA3-specific params on SimCol3D.

Frozen   : GastroNet backbone weights (patch_embed, pos_embed, blocks.*.qkv/proj/mlp/norm)
Trainable: DualDPT head, camera_token, q_norm/k_norm (blocks 4-11)

Loss (DA3-style from arXiv:2511.10647):
    L = LD + LM + LP + Lgrad
    LD    — confidence-weighted L1 depth + log-barrier on confidence
    LM    — L1 ray-map loss (uses GT c2w + K from SimCol3D)
    LP    — point-map loss: D̂⊙d + t vs GT point map
    Lgrad — depth gradient loss

Run:
    python train/stage1.py \\
        --data-path ~/code/data \\
        --gastronet /path/to/dinov2.pth \\
        --out-dir runs/stage1
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endo_da3 import EndoDA3
from endo_da3.data import make_stage1_loaders
from endo_da3.loss import da3_loss


# ── freeze helpers ────────────────────────────────────────────────────────────

# GastroNet backbone weights — frozen in Stage 1
_GASTRO_PREFIXES = (
    "backbone.patch_embed",
    "backbone.pos_embed",
    "backbone.norm",
    "backbone.cls_token",
    "backbone.mask_token",
)
_GASTRO_BLOCK_KEYS = ("attn.qkv", "attn.proj", "mlp.", "norm1", "norm2")


def freeze_gastronet(model: EndoDA3):
    """Freeze GastroNet backbone weights; keep DA3-specific tokens/norms trainable."""
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in _GASTRO_PREFIXES):
            param.requires_grad_(False)
            continue
        if name.startswith("backbone.blocks."):
            # DA3-specific per-block params: q_norm, k_norm → trainable
            if any(k in name for k in ("q_norm", "k_norm")):
                param.requires_grad_(True)
            elif any(k in name for k in _GASTRO_BLOCK_KEYS):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        else:
            # head, camera_token, etc. → trainable
            param.requires_grad_(True)


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── visualisation ────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _to_colormap(depth: np.ndarray) -> np.ndarray:
    """Normalise a (H,W) depth map → (H,W,3) uint8 magma_r."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(depth[depth > 0], [2, 98]) if (depth > 0).any() else (0, 1)
    norm = (depth.clip(vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
    return (plt.get_cmap("magma_r")(norm)[..., :3] * 255).astype(np.uint8)


def _save_depth_vis(vis_batch: tuple, path):
    """Save pred depth vs GT depth for the first sample of the batch."""
    images, gt_depths, pred_depths = vis_batch
    # first item, first view: (3,H,W), (H,W), (H,W)
    img   = images[0, 0].permute(1, 2, 0).numpy()
    img   = ((img * _IMAGENET_STD + _IMAGENET_MEAN).clip(0, 1) * 255).astype(np.uint8)
    gt    = _to_colormap(gt_depths[0, 0].numpy())
    pred  = _to_colormap(pred_depths[0, 0].numpy())

    canvas = np.concatenate([img, gt, pred], axis=1)   # side by side
    PILImage.fromarray(canvas).save(path)


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── model ────────────────────────────────────────────────────────────────
    print("Loading Endo-DA3 (DA3-BASE weights)…")
    model = EndoDA3.from_pretrained(img_size=args.img_size, with_camera=False, device=device)

    print("Swapping backbone with GastroNet…")
    ckpt = torch.load(args.gastronet, map_location="cpu")
    gastro_sd = {k.replace("backbone.", ""): v
                 for k, v in ckpt["teacher"].items()
                 if k.startswith("backbone.")}
    model.replace_backbone(gastro_sd)

    freeze_gastronet(model)

    total, trainable = count_params(model)
    print(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # ── data ─────────────────────────────────────────────────────────────────
    data = Path(args.data_path)
    train_loader, val_loader, ds_names = make_stage1_loaders(
        simcol_root=str(data / "SimCol3D"),
        c3vd_root=str(data / "C3VD"),
        endoslam_root=str(data / "EndoSLAM"),
        polypsense3d_root=str(data / "PolypSense3D"),
        img_size=args.img_size, seq_len=args.seq_len, stride=args.stride,
        batch_size=args.batch_size, num_workers=args.workers,
    )
    print(f"Datasets : {ds_names}")
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # ── optimiser ────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.05,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))

    best_val = math.inf
    global_step = 0
    start_epoch = 1

    if args.resume:
        print(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt["global_step"]
        best_val     = ckpt["val_loss"]
        print(f"  Resumed at epoch {start_epoch}  best_val={best_val:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        running = {"total": 0.0, "LD": 0.0, "LM": 0.0, "LP": 0.0, "Lgrad": 0.0}
        n_log = 0

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device)   # (B, S, 3, H, W)
            depths = batch["depths"].to(device)   # (B, S, H, W)
            c2w    = batch["c2w"].to(device)      # (B, S, 4, 4)
            K      = batch["K"].to(device)        # (B, 3, 3)

            opt.zero_grad(set_to_none=True)
            out = model(images)

            loss, terms = da3_loss(out, depths, c2w, K, alpha=args.alpha)
            loss.backward()

            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            scheduler.step()

            running["total"] += loss.item()
            for k in ("LD", "LM", "LP", "Lgrad"):
                running[k] += terms[k]
            n_log += 1
            global_step += 1

            if (step + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch:3d}  step {step+1:5d}/{len(train_loader)}"
                    f"  loss {running['total']/n_log:.4f}"
                    f"  LD {running['LD']/n_log:.3f}"
                    f"  LM {running['LM']/n_log:.3f}"
                    f"  LP {running['LP']/n_log:.3f}"
                    f"  Lg {running['Lgrad']/n_log:.3f}"
                    f"  lr {lr:.2e}  t {time.time()-t0:.0f}s"
                )
                writer.add_scalars("train", {k: running[k]/n_log for k in running}, global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                running = {k: 0.0 for k in running}
                n_log = 0

        # ── val ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        vis_batch = None
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"val epoch {epoch}", leave=False)):
                images = batch["images"].to(device)
                depths = batch["depths"].to(device)
                c2w    = batch["c2w"].to(device)
                K      = batch["K"].to(device)
                out    = model(images)
                loss, _ = da3_loss(out, depths, c2w, K, alpha=args.alpha)
                val_loss += loss.item()
                if i == 0:
                    vis_batch = (images.cpu(), depths.cpu(), out["depth"].cpu())

        val_loss /= len(val_loader)
        writer.add_scalar("val/loss", val_loss, epoch)
        print(f"Epoch {epoch:3d} — val_loss {val_loss:.4f}  [{time.time()-t0:.0f}s]")

        # ── depth visualisation ───────────────────────────────────────────
        if vis_batch is not None:
            _save_depth_vis(vis_batch, out_dir / f"depth_epoch{epoch:03d}.png")

        # ── checkpoint ────────────────────────────────────────────────────
        torch.save({
            "epoch":       epoch,
            "global_step": global_step,
            "model":       model.state_dict(),
            "opt":         opt.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "val_loss":    val_loss,
        }, out_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  new best: {best_val:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Stage 1: Align decoder on SimCol3D")
    p.add_argument("--data-path",  required=True,
                   help="Root data directory containing SimCol3D/, C3VD/, EndoSLAM/, PolypSense3D/")
    p.add_argument("--gastronet",  required=True, help="Path to GastroNet dinov2.pth")
    p.add_argument("--out-dir",    default="runs/stage1")
    p.add_argument("--resume",     default=None, help="Path to last.pt to resume from")
    p.add_argument("--img-size",   type=int,   default=336)
    p.add_argument("--seq-len",    type=int,   default=2)
    p.add_argument("--stride",     type=int,   default=1)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--alpha",      type=float, default=1.0,
                   help="Lgrad weight (paper: α=1)")
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--log-every",  type=int,   default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
