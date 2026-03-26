"""
Stage 2b — Self-supervised capsule endoscopy adaptation: fine-tune LoRA adapters
           + DualDPT head on MiroCam + PillCam ex-vivo videos using photometric
           reprojection loss (no GT depth required).

Frozen   : GastroNet backbone base weights
Trainable: LoRA adapters, DualDPT head, camera_token, q_norm/k_norm

Loss: γ·Lphoto + α·Lsmooth  (stage2b_loss — Monodepth2-style with auto-mask)
      Lphoto: min-pooled SSIM+L1 reprojection error across source frames
      Lsmooth: edge-aware depth smoothness

Initialisation: Stage 2a last.pt (already contains LoRA weights)

Run:
    python train/stage2b.py \\
        --stage2a-ckpt runs/stage2a/best.pt \\
        --gastronet ~/code/GastroNet/gastronet/dinov2.pth \\
        --endoslam-root ~/code/data/EndoSLAM \\
        --out-dir runs/stage2b
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
from endo_da3.data.loaders import make_stage2b_loaders
from endo_da3.lora import inject_lora, count_lora_params
from endo_da3.loss import stage2b_loss

from train.stage1 import freeze_gastronet, count_params
from tools.eval_serv_ct import evaluate as serv_ct_evaluate


# ── visualisation ─────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _to_colormap(depth: np.ndarray) -> np.ndarray:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = depth > 0
    vmin, vmax = np.percentile(depth[valid], [2, 98]) if valid.any() else (float(depth.min()), float(depth.max()))
    norm = (depth.clip(vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
    return (plt.get_cmap("magma_r")(norm)[..., :3] * 255).astype(np.uint8)


def _denorm(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor → (H,W,3) uint8."""
    img = t.permute(1, 2, 0).numpy()
    return ((img * _IMAGENET_STD + _IMAGENET_MEAN).clip(0, 1) * 255).astype(np.uint8)


def _make_row(images, pred_depths) -> np.ndarray:
    """[RGB | Pred depth] strip for first sample in batch. (H, 2W, 3)"""
    img  = _denorm(images[0, 0])
    pred = _to_colormap(pred_depths[0, 0].numpy())
    return np.concatenate([img, pred], axis=1)


def _save_vis_grid(rows: list[tuple[str, np.ndarray]], path):
    label_h = 16
    strips  = []
    for _, strip in rows:
        label = np.zeros((label_h, strip.shape[1], 3), dtype=np.uint8)
        strips.append(label)
        strips.append(strip)
    canvas = np.concatenate(strips, axis=0)
    PILImage.fromarray(canvas).save(path)


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── model ─────────────────────────────────────────────────────────────────
    print("Loading Endo-DA3 (DA3-BASE weights)…")
    model = EndoDA3.from_pretrained(img_size=args.img_size, with_camera=False, device=device)

    print("Swapping backbone with GastroNet…")
    ckpt = torch.load(args.gastronet, map_location="cpu")
    gastro_sd = {k.replace("backbone.", ""): v
                 for k, v in ckpt["teacher"].items()
                 if k.startswith("backbone.")}
    model.replace_backbone(gastro_sd)

    # Freeze backbone base (same policy as Stage 1 / 2a)
    freeze_gastronet(model)

    # Inject LoRA with same config as Stage 2a (must match before loading sd)
    n_lora = inject_lora(model, rank=args.lora_rank, lora_alpha=args.lora_alpha)
    print(f"Injected {n_lora} LoRA adapters (rank={args.lora_rank})")

    print(f"Loading Stage 2a checkpoint: {args.stage2a_ckpt}")
    s2a_sd = torch.load(args.stage2a_ckpt, map_location="cpu")
    if isinstance(s2a_sd, dict) and "model" in s2a_sd:
        s2a_sd = s2a_sd["model"]
    model.load_state_dict(s2a_sd, strict=False)

    lora_params, trainable = count_lora_params(model)
    total, _ = count_params(model)
    print(f"Parameters: {total/1e6:.1f}M total | "
          f"{trainable/1e6:.1f}M trainable | "
          f"{lora_params/1e3:.1f}K LoRA")

    model.to(device)

    # ── fixed OOD images for visual check ─────────────────────────────────────
    from torchvision import transforms
    _tf = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ood_batches: list[tuple[str, torch.Tensor]] = []
    _OOD = [
        ("SOH",    "/home/in4218/code/data/SOH/000.png"),
        ("Kvasir", "/home/in4218/code/data/kvasir-dataset-v2/normal-z-line/0be91a4e-be3d-4c06-92d7-6e0ee417f55a.jpg"),
    ]
    for name, path in _OOD:
        try:
            img = PILImage.open(path).convert("RGB")
            ood_batches.append((name, _tf(img).unsqueeze(0).unsqueeze(0)))  # (1,1,3,H,W)
        except FileNotFoundError:
            pass

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, ds_names = make_stage2b_loaders(
        endoslam_root=args.endoslam_root,
        img_size=args.img_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f"Train datasets: {ds_names}  |  {len(train_loader)} batches/epoch")

    # ── optimiser ─────────────────────────────────────────────────────────────
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

    best_loss   = math.inf
    global_step = 0
    start_epoch = 1

    if args.resume:
        print(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_loss   = ckpt.get("train_loss", math.inf)
        print(f"  Resumed at epoch {start_epoch}  best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {"total": 0.0, "Lphoto": 0.0, "Lsmooth": 0.0}
        n_log   = 0
        vis_batch = None

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device)   # (B, S, 3, H, W)
            c2w    = batch["c2w"].to(device)      # (B, S, 4, 4)
            K      = batch["K"].to(device)        # (B, 3, 3)

            opt.zero_grad(set_to_none=True)
            out = model(images)

            loss, terms = stage2b_loss(
                out, images, c2w, K,
                alpha=args.alpha, gamma=args.gamma,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            scheduler.step()

            running["total"]   += loss.item()
            running["Lphoto"]  += terms["Lphoto"]
            running["Lsmooth"] += terms["Lsmooth"]
            n_log      += 1
            global_step += 1

            if step == 0 and vis_batch is None:
                vis_batch = (images.cpu(), out["depth"].detach().cpu())

            if (step + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch:3d}  step {step+1:5d}/{len(train_loader)}"
                    f"  loss {running['total']/n_log:.4f}"
                    f"  Lphoto {running['Lphoto']/n_log:.4f}"
                    f"  Lsmooth {running['Lsmooth']/n_log:.4f}"
                    f"  lr {lr:.2e}  t {time.time()-t0:.0f}s"
                )
                writer.add_scalars("train", {k: running[k]/n_log for k in running}, global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                running = {k: 0.0 for k in running}
                n_log   = 0

        epoch_loss = running["total"] / max(n_log, 1)
        print(f"Epoch {epoch:3d} — train_loss {epoch_loss:.4f}  [{time.time()-t0:.0f}s]")
        writer.add_scalar("epoch/train_loss", epoch_loss, epoch)

        # ── SERV-CT eval ──────────────────────────────────────────────────────
        serv_metrics = {}
        if args.serv_ct_root:
            model.eval()
            with torch.no_grad():
                serv_metrics = serv_ct_evaluate(
                    model, args.serv_ct_root, img_size=args.img_size, device=str(device)
                )
            model.train()
            print(f"  SERV-CT — AbsRel {serv_metrics['AbsRel']:.4f}"
                  f"  δ<1.25 {serv_metrics['d1']:.4f}"
                  f"  RMSE {serv_metrics['RMSE']:.4f}m")
            writer.add_scalar("serv_ct/AbsRel", serv_metrics["AbsRel"], epoch)
            writer.add_scalar("serv_ct/d1",     serv_metrics["d1"],     epoch)
            writer.add_scalar("serv_ct/RMSE",   serv_metrics["RMSE"],   epoch)

        # ── visualisation ─────────────────────────────────────────────────────
        if vis_batch is not None:
            rows = [("CapsuleEndo", _make_row(*vis_batch))]
            model.eval()
            with torch.no_grad():
                for name, ood_images in ood_batches:
                    ood_out = model(ood_images.to(device))
                    rows.append((name, _make_row(ood_images, ood_out["depth"].cpu())))
            model.train()
            _save_vis_grid(rows, out_dir / f"depth_epoch{epoch:03d}.png")

        # ── checkpoint ────────────────────────────────────────────────────────
        torch.save({
            "epoch":        epoch,
            "global_step":  global_step,
            "model":        model.state_dict(),
            "opt":          opt.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "train_loss":   epoch_loss,
            "serv_ct":      serv_metrics,
        }, out_dir / "last.pt")

        val_metric = serv_metrics.get("AbsRel", epoch_loss)
        if val_metric < best_loss:
            best_loss = val_metric
            torch.save(model.state_dict(), out_dir / "best.pt")
            key = "AbsRel" if serv_metrics else "train_loss"
            print(f"  new best {key}: {best_loss:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Stage 2b: Self-supervised capsule endoscopy adaptation")
    p.add_argument("--stage2a-ckpt",   required=True, help="Path to Stage 2a last.pt")
    p.add_argument("--gastronet",      required=True, help="Path to GastroNet dinov2.pth")
    p.add_argument("--endoslam-root",  required=True,
                   help="EndoSLAM root (must contain extracted MiroCam/ and PillCam/)")
    p.add_argument("--serv-ct-root",   default=None,
                   help="SERV-CT root (contains SERV-CT-ALL/) — used for per-epoch eval")
    p.add_argument("--out-dir",        default="runs/stage2b")
    p.add_argument("--resume",         default=None, help="Path to stage2b last.pt to resume")
    p.add_argument("--img-size",       type=int,   default=336)
    p.add_argument("--seq-len",        type=int,   default=2)
    p.add_argument("--batch-size",     type=int,   default=8)
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--lr",             type=float, default=2e-5,
                   help="Lower than Stage 2a — fine-tuning with weaker signal")
    p.add_argument("--alpha",          type=float, default=0.1, help="Lsmooth weight")
    p.add_argument("--gamma",          type=float, default=1.0, help="Lphoto weight")
    p.add_argument("--lora-rank",      type=int,   default=4)
    p.add_argument("--lora-alpha",     type=float, default=4.0)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--log-every",      type=int,   default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
