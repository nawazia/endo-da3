"""
Stage 2b-distill — Teacher-student distillation on PolypSense3D clinical.

Teacher : DA3-BASE (frozen, original DINOv2 backbone, no GastroNet)
Student : Endo-DA3 (GastroNet backbone + LoRA, initialised from Stage 2a best.pt)

Loss: β·Ldistill + α·Lsmooth
    Ldistill: scale-invariant L1 between per-frame median-normalised depth maps
    Lsmooth : edge-aware depth smoothness regularisation

Trainable: LoRA adapters, DualDPT head, camera_token, q_norm/k_norm
Frozen   : GastroNet backbone base weights

Initialisation: Stage 2a best.pt

Per-epoch SERV-CT evaluation (AbsRel / δ<1.25 / RMSE); best.pt saved on AbsRel.

Run:
    python train/stage2b_distill.py \\
        --stage2a-ckpt runs/stage2a/best.pt \\
        --gastronet ~/code/GastroNet/gastronet/dinov2.pth \\
        --polypsense3d-root ~/code/data/PolypSense3D \\
        --serv-ct-root ~/code/data/SERV-CT \\
        --out-dir runs/stage2b_distill
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
from endo_da3.data.loaders import make_stage2b_distill_loaders
from endo_da3.lora import inject_lora, count_lora_params
from endo_da3.loss import distillation_loss

from train.stage1 import freeze_gastronet, count_params
from tools.eval_serv_ct import evaluate as serv_ct_evaluate


# ── visualisation ─────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _to_colormap(depth: np.ndarray) -> np.ndarray:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = depth > 0
    if valid.any():
        vmin, vmax = np.percentile(depth[valid], [2, 98])
    else:
        vmin, vmax = float(depth.min()), float(depth.max())
    norm = (depth.clip(vmin, vmax) - vmin) / max(vmax - vmin, 1e-6)
    return (plt.get_cmap("magma_r")(norm)[..., :3] * 255).astype(np.uint8)


def _denorm(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).numpy()
    return ((img * _IMAGENET_STD + _IMAGENET_MEAN).clip(0, 1) * 255).astype(np.uint8)


def _make_quad(images, stage2a_depths, student_depths, teacher_depths) -> np.ndarray:
    """[RGB | Stage 2a | Student (current) | Teacher (DA3-BASE)] strip for first sample."""
    img     = _denorm(images[0, 0])
    s2a     = _to_colormap(stage2a_depths[0, 0].numpy())
    student = _to_colormap(student_depths[0, 0].numpy())
    teacher = _to_colormap(teacher_depths[0, 0].numpy())
    return np.concatenate([img, s2a, student, teacher], axis=1)


def _save_vis_grid(rows: list[tuple[str, np.ndarray]], path):
    label_h = 16
    strips = []
    for _, strip in rows:
        strips.append(np.zeros((label_h, strip.shape[1], 3), dtype=np.uint8))
        strips.append(strip)
    PILImage.fromarray(np.concatenate(strips, axis=0)).save(path)


# ── training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── teacher (DA3-BASE, frozen) ────────────────────────────────────────────
    print("Loading DA3-BASE teacher (frozen)...")
    teacher = EndoDA3.from_pretrained(img_size=args.img_size, with_camera=False, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── student (GastroNet + Stage 2a LoRA) ───────────────────────────────────
    print("Loading Endo-DA3 student (DA3-BASE weights)...")
    student = EndoDA3.from_pretrained(img_size=args.img_size, with_camera=False, device=device)

    print("Swapping backbone with GastroNet...")
    ckpt = torch.load(args.gastronet, map_location="cpu")
    gastro_sd = {k.replace("backbone.", ""): v
                 for k, v in ckpt["teacher"].items()
                 if k.startswith("backbone.")}
    student.replace_backbone(gastro_sd)

    freeze_gastronet(student)

    n_lora = inject_lora(student, rank=args.lora_rank, lora_alpha=args.lora_alpha)
    print(f"Injected {n_lora} LoRA adapters (rank={args.lora_rank})")

    print(f"Loading Stage 2a checkpoint: {args.stage2a_ckpt}")
    s2a_sd = torch.load(args.stage2a_ckpt, map_location="cpu")
    if isinstance(s2a_sd, dict) and "model" in s2a_sd:
        s2a_sd = s2a_sd["model"]
    student.load_state_dict(s2a_sd, strict=False)

    lora_params, trainable = count_lora_params(student)
    total, _ = count_params(student)
    print(f"Parameters: {total/1e6:.1f}M total | "
          f"{trainable/1e6:.1f}M trainable | "
          f"{lora_params/1e3:.1f}K LoRA")

    student.to(device)

    # ── Stage 2a reference model (frozen, for visualisation only) ─────────────
    print("Loading Stage 2a reference (frozen, for vis)...")
    stage2a_ref = EndoDA3.from_pretrained(img_size=args.img_size, with_camera=False, device=device)
    stage2a_ref.replace_backbone(gastro_sd)
    inject_lora(stage2a_ref, rank=args.lora_rank, lora_alpha=args.lora_alpha)
    stage2a_ref.load_state_dict(s2a_sd, strict=False)
    stage2a_ref.eval()
    for p in stage2a_ref.parameters():
        p.requires_grad_(False)
    stage2a_ref.to(device)

    # ── fixed OOD images for visual check ─────────────────────────────────────
    from torchvision import transforms
    _tf = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ood_batches: list[tuple[str, torch.Tensor]] = []
    for name, path in [
        ("SOH",    "/home/in4218/code/data/SOH/000.png"),
        ("Kvasir", "/home/in4218/code/data/kvasir-dataset-v2/normal-z-line/"
                   "0be91a4e-be3d-4c06-92d7-6e0ee417f55a.jpg"),
    ]:
        try:
            img = PILImage.open(path).convert("RGB")
            ood_batches.append((name, _tf(img).unsqueeze(0).unsqueeze(0)))
        except FileNotFoundError:
            pass

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, ds_names = make_stage2b_distill_loaders(
        polypsense3d_root=args.polypsense3d_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        undistort=args.undistort,
    )
    print(f"Train dataset: {ds_names}  |  {len(train_loader)} batches/epoch")

    # ── optimiser ─────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
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
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_loss   = ckpt.get("train_loss", math.inf)
        print(f"  Resumed at epoch {start_epoch}  best_loss={best_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        t0 = time.time()
        running = {"total": 0.0, "Ldistill": 0.0, "Lsmooth": 0.0}
        n_log    = 0
        vis_batch = None

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device)   # (B, 1, 3, H, W)

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_out   = teacher(images)
                teacher_depth = teacher_out["depth"].detach()

            student_out = student(images)

            loss, terms = distillation_loss(
                student_out, teacher_depth, images,
                beta=args.beta, alpha=args.alpha,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            scheduler.step()

            running["total"]    += loss.item()
            running["Ldistill"] += terms["Ldistill"]
            running["Lsmooth"]  += terms["Lsmooth"]
            n_log      += 1
            global_step += 1

            if step == 0 and vis_batch is None:
                with torch.no_grad():
                    s2a_depth = stage2a_ref(images)["depth"].cpu()
                vis_batch = (
                    images.cpu(),
                    s2a_depth,
                    student_out["depth"].detach().cpu(),
                    teacher_depth.cpu(),
                )

            if (step + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch:3d}  step {step+1:5d}/{len(train_loader)}"
                    f"  loss {running['total']/n_log:.4f}"
                    f"  Ldistill {running['Ldistill']/n_log:.4f}"
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
            student.eval()
            with torch.no_grad():
                serv_metrics = serv_ct_evaluate(
                    student, args.serv_ct_root, img_size=args.img_size, device=str(device)
                )
            student.train()
            print(f"  SERV-CT — AbsRel {serv_metrics['AbsRel']:.4f}"
                  f"  δ<1.25 {serv_metrics['d1']:.4f}"
                  f"  RMSE {serv_metrics['RMSE']:.4f}m")
            writer.add_scalar("serv_ct/AbsRel", serv_metrics["AbsRel"], epoch)
            writer.add_scalar("serv_ct/d1",     serv_metrics["d1"],     epoch)
            writer.add_scalar("serv_ct/RMSE",   serv_metrics["RMSE"],   epoch)

        # ── visualisation ─────────────────────────────────────────────────────
        if vis_batch is not None:
            rows = [("PolypSense3D", _make_quad(*vis_batch))]
            student.eval()
            with torch.no_grad():
                for name, ood_images in ood_batches:
                    ood_dev      = ood_images.to(device)
                    ood_s2a      = stage2a_ref(ood_dev)
                    ood_student  = student(ood_dev)
                    ood_teacher  = teacher(ood_dev)
                    rows.append((name, _make_quad(
                        ood_images,
                        ood_s2a["depth"].cpu(),
                        ood_student["depth"].cpu(),
                        ood_teacher["depth"].cpu(),
                    )))
            student.train()
            _save_vis_grid(rows, out_dir / f"depth_epoch{epoch:03d}.png")

        # ── checkpoint ────────────────────────────────────────────────────────
        torch.save({
            "epoch":       epoch,
            "global_step": global_step,
            "model":       student.state_dict(),
            "opt":         opt.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "train_loss":  epoch_loss,
            "serv_ct":     serv_metrics,
        }, out_dir / "last.pt")

        val_metric = serv_metrics.get("AbsRel", epoch_loss)
        if val_metric < best_loss:
            best_loss = val_metric
            torch.save(student.state_dict(), out_dir / "best.pt")
            key = "AbsRel" if serv_metrics else "train_loss"
            print(f"  new best {key}: {best_loss:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Stage 2b-distill: DA3-BASE → Endo-DA3 distillation")
    p.add_argument("--stage2a-ckpt",        required=True,
                   help="Path to Stage 2a best.pt")
    p.add_argument("--gastronet",           required=True,
                   help="Path to GastroNet dinov2.pth")
    p.add_argument("--polypsense3d-root",   required=True,
                   help="PolypSense3D root (contains Clinical-Dataset-For-PolypSense3D/)")
    p.add_argument("--serv-ct-root",        default=None,
                   help="SERV-CT root for per-epoch evaluation")
    p.add_argument("--out-dir",             default="runs/stage2b_distill")
    p.add_argument("--resume",              default=None,
                   help="Path to stage2b_distill last.pt to resume")
    p.add_argument("--img-size",            type=int,   default=336)
    p.add_argument("--batch-size",          type=int,   default=8)
    p.add_argument("--epochs",              type=int,   default=20)
    p.add_argument("--lr",                  type=float, default=2e-5)
    p.add_argument("--beta",                type=float, default=1.0,
                   help="Ldistill weight")
    p.add_argument("--alpha",               type=float, default=0.1,
                   help="Lsmooth weight")
    p.add_argument("--lora-rank",           type=int,   default=4)
    p.add_argument("--lora-alpha",          type=float, default=4.0)
    p.add_argument("--workers",             type=int,   default=4)
    p.add_argument("--log-every",           type=int,   default=50)
    p.add_argument("--no-undistort",        action="store_true",
                   help="Skip barrel-distortion correction")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.undistort = not args.no_undistort
    train(args)
