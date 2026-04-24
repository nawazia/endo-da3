"""
Stage 3 — SCARED surgical fine-tuning with keyframes + video pseudo-GT.

Frozen   : GastroNet backbone base weights
Trainable: LoRA adapters (rank-4), DualDPT head, camera_token, q_norm/k_norm

Loss: full DA3 loss (LD + LM + LP + α·Lgrad)
      Keyframes: structured-light GT for both cameras
      Video frames: RAFT pseudo-GT for left camera only (right depth = 0)

Data:
    Train : SCARED keyframes ds1-3,6-7 (25 pairs) +
            SCARED video frames ds1-3,6-7 (~17,206 frames)
    Val   : SCARED dataset_8–9 keyframes (10 pairs, structured-light GT)
    Eval  : SERV-CT per epoch (AbsRel / δ<1.25 / RMSE) — primary metric for best.pt

Initialisation: Stage 2a best.pt (GastroNet + LoRA already loaded)

Run:
    python train/stage3.py \\
        --stage2a-ckpt runs/stage2a/best.pt \\
        --gastronet ~/code/GastroNet/gastronet/dinov2.pth \\
        --scared-root ~/code/data/SCARED \\
        --serv-ct-root ~/code/data/SERV-CT \\
        --out-dir runs/stage3
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endo_da3 import EndoDA3
from endo_da3.data.loaders import make_stage3_loaders
from endo_da3.lora import inject_lora, count_lora_params
from endo_da3.loss import da3_loss

from train.stage1 import freeze_gastronet, count_params
from tools.eval_serv_ct import evaluate as serv_ct_evaluate


# ── visualisation ─────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _to_colormap(depth: np.ndarray) -> np.ndarray:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = depth > 0
    vmin, vmax = np.percentile(depth[valid], [2, 98]) if valid.any() else (0, 1)
    norm = (depth.clip(vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
    return (plt.get_cmap("magma_r")(norm)[..., :3] * 255).astype(np.uint8)


def _make_row(images, gt_depths, pred_depths) -> np.ndarray:
    """[RGB | GT depth | Pred depth] for first sample, left camera."""
    img  = images[0, 0].permute(1, 2, 0).numpy()
    img  = ((img * _IMAGENET_STD + _IMAGENET_MEAN).clip(0, 1) * 255).astype(np.uint8)
    gt   = _to_colormap(gt_depths[0, 0].numpy())
    pred = _to_colormap(pred_depths[0, 0].numpy())
    return np.concatenate([img, gt, pred], axis=1)


def _save_vis(row: np.ndarray, path):
    PILImage.fromarray(row).save(path)


# ── model loading ──────────────────────────────────────────────────────────────

def load_model(args, device):
    print("Loading Endo-DA3 (DA3-BASE weights)…")
    model = EndoDA3.from_pretrained(
        img_size=args.img_size, with_camera=False, device=device
    )

    print("Swapping backbone with GastroNet…")
    ckpt = torch.load(args.gastronet, map_location="cpu")
    gastro_sd = {k.replace("backbone.", ""): v
                 for k, v in ckpt["teacher"].items()
                 if k.startswith("backbone.")}
    model.replace_backbone(gastro_sd)

    # Freeze backbone base weights first, then inject LoRA on top (same order as
    # Stage 2a — freeze_gastronet must run before inject_lora so LoRA adapters
    # are added after the freeze and remain trainable)
    freeze_gastronet(model)
    inject_lora(model, rank=args.lora_rank, lora_alpha=args.lora_alpha)

    print(f"Loading Stage 2a checkpoint: {args.stage2a_ckpt}")
    sd = torch.load(args.stage2a_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)

    lora_p, trainable = count_lora_params(model)
    total, _ = count_params(model)
    print(f"Parameters: {total/1e6:.1f}M total | "
          f"{trainable/1e6:.1f}M trainable | "
          f"{lora_p/1e3:.1f}K LoRA")

    return model.to(device)


# ── training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args, device)

    train_loader, val_kf_loader, val_vid_loader, ds_names = make_stage3_loaders(
        scared_root=args.scared_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f"Train: {len(train_loader)} batches/epoch  |  "
          f"Val keyframes: {len(val_kf_loader)}  |  Val video: {len(val_vid_loader)} batches")

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

    best_abs_rel = math.inf
    global_step  = 0

    for epoch in range(1, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        running = {"total": 0.0, "LD": 0.0, "LM": 0.0, "LP": 0.0, "Lgrad": 0.0}
        n_log   = 0

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            depths = batch["depths"].to(device)
            c2w    = batch["c2w"].to(device)
            K      = batch["K"].to(device)

            opt.zero_grad(set_to_none=True)
            out  = model(images)

            bad_keys = [k for k, v in out.items()
                        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()]
            if bad_keys:
                print(f"  Inf/NaN in {bad_keys} at step {step} — skipping batch")
                del out; torch.cuda.empty_cache()
                scheduler.step()
                continue

            loss, terms = da3_loss(out, depths, c2w, K, alpha=args.alpha)

            if not torch.isfinite(loss):
                print(f"  NaN/Inf loss at step {step}: {terms} — skipping batch")
                del out, loss; torch.cuda.empty_cache()
                scheduler.step()
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            scheduler.step()

            running["total"] += loss.item()
            for k in ("LD", "LM", "LP", "Lgrad"):
                running[k] += terms[k]
            n_log       += 1
            global_step += 1

            if (step + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch:3d}  step {step+1:5d}/{len(train_loader)}"
                    f"  loss {running['total']/max(n_log,1):.4f}"
                    f"  LD {running['LD']/max(n_log,1):.3f}"
                    f"  LM {running['LM']/max(n_log,1):.3f}"
                    f"  LP {running['LP']/max(n_log,1):.3f}"
                    f"  Lg {running['Lgrad']/max(n_log,1):.3f}"
                    f"  lr {lr:.2e}  t {time.time()-t0:.0f}s"
                )
                writer.add_scalars("train", {k: running[k]/max(n_log,1) for k in running}, global_step)
                writer.add_scalar("lr", lr, global_step)
                running = {k: 0.0 for k in running}
                n_log   = 0

        # ── val ───────────────────────────────────────────────────────────
        model.eval()
        vis_batch = None

        def _run_val(loader):
            total_loss, total_l1, n = 0.0, 0.0, 0
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    images = batch["images"].to(device)
                    depths = batch["depths"].to(device)
                    c2w    = batch["c2w"].to(device)
                    K      = batch["K"].to(device)
                    out    = model(images)
                    loss, _ = da3_loss(out, depths, c2w, K, alpha=args.alpha)
                    total_loss += loss.item()
                    mask = (depths > 0).float()
                    l1   = (out["depth"] - depths).abs()
                    total_l1 += (mask * l1).sum().item() / mask.sum().clamp(min=1).item()
                    n += 1
                    if i == 0:
                        nonlocal vis_batch
                        vis_batch = (images.cpu(), depths.cpu(), out["depth"].cpu())
            return total_loss / max(n, 1), total_l1 / max(n, 1)

        kf_loss,  kf_l1  = _run_val(val_kf_loader)
        vid_loss, vid_l1 = _run_val(val_vid_loader)

        writer.add_scalar("val_kf/loss",     kf_loss,  epoch)
        writer.add_scalar("val_kf/depth_l1", kf_l1,    epoch)
        writer.add_scalar("val_vid/loss",    vid_loss,  epoch)
        writer.add_scalar("val_vid/depth_l1", vid_l1,  epoch)

        # ── SERV-CT eval ───────────────────────────────────────────────────
        serv_ct = serv_ct_evaluate(model, args.serv_ct_root, device=device)
        writer.add_scalars("serv_ct", serv_ct, epoch)
        print(
            f"Epoch {epoch:3d} — "
            f"kf_l1 {kf_l1*1000:.1f}mm  vid_l1 {vid_l1*1000:.1f}mm  "
            f"SERV-CT AbsRel {serv_ct['AbsRel']:.4f}  "
            f"δ<1.25 {serv_ct['d1']:.4f}  "
            f"RMSE {serv_ct['RMSE']*1000:.1f}mm  "
            f"[{time.time()-t0:.0f}s]"
        )

        # ── visualisation ──────────────────────────────────────────────────
        if vis_batch:
            row = _make_row(*vis_batch)
            _save_vis(row, out_dir / f"depth_epoch{epoch:03d}.png")

        # ── checkpoints ────────────────────────────────────────────────────
        torch.save({
            "epoch":        epoch,
            "global_step":  global_step,
            "model":        model.state_dict(),
            "opt":          opt.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "serv_ct":      serv_ct,
            "val_kf_l1":    kf_l1,
            "val_vid_l1":   vid_l1,
        }, out_dir / "last.pt")

        if serv_ct["AbsRel"] < best_abs_rel:
            best_abs_rel = serv_ct["AbsRel"]
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  ★ new best SERV-CT AbsRel: {best_abs_rel:.4f}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Stage 3: SCARED structured-light fine-tuning")
    p.add_argument("--stage2a-ckpt",  required=True)
    p.add_argument("--gastronet",     required=True)
    p.add_argument("--scared-root",   required=True)
    p.add_argument("--serv-ct-root",  required=True)
    p.add_argument("--out-dir",       default="runs/stage3")
    p.add_argument("--img-size",      type=int,   default=336)
    p.add_argument("--batch-size",    type=int,   default=2)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--alpha",         type=float, default=1.0)
    p.add_argument("--lora-rank",     type=int,   default=4)
    p.add_argument("--lora-alpha",    type=float, default=4.0)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--log-every",     type=int,   default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
