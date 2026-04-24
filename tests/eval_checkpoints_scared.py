"""
Evaluate all stage checkpoints on SCARED val (datasets 8 & 9).

Usage:
    python tests/eval_checkpoints_scared.py
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endo_da3 import EndoDA3
from endo_da3.lora import inject_lora
from tools.eval_scared import evaluate

GASTRONET = "/home/in4218/code/GastroNet/gastronet/dinov2.pth"
SCARED    = "/home/in4218/code/data/SCARED"
IMG_SIZE  = 336

CHECKPOINTS = [
    # (label, ckpt_path, use_lora)
    ("Stage 1  (best)", "runs/stage1/best.pt",   False),
    ("Stage 1  (last)", "runs/stage1/last.pt",   False),
    ("Stage 2a (best)", "runs/stage2a/best.pt",  True),
    ("Stage 2a (last)", "runs/stage2a/last.pt",  True),
    ("Stage 2b (best)", "runs/stage2b/best.pt",  True),
    ("Stage 2b (last)", "runs/stage2b/last.pt",  True),
    ("Stage 2b-distill (best)", "runs/stage2b_distill/best.pt", True),
    ("Stage 2b-distill (last)", "runs/stage2b_distill/last.pt", True),
    ("Stage 3  (best)", "runs/stage3/best.pt",   True),
    ("Stage 3  (last)", "runs/stage3/last.pt",   True),
]


def load_model(ckpt_path: str, use_lora: bool, device: str) -> EndoDA3:
    model = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device)
    ckpt = torch.load(GASTRONET, map_location="cpu")
    gastro_sd = {k.replace("backbone.", ""): v
                 for k, v in ckpt["teacher"].items()
                 if k.startswith("backbone.")}
    model.replace_backbone(gastro_sd)
    if use_lora:
        inject_lora(model, rank=4, lora_alpha=4.0)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


def load_da3_base(device: str) -> EndoDA3:
    return EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=device).eval()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'Checkpoint':<28} {'AbsRel':>8} {'δ<1.25':>8} {'δ<1.05':>8} {'MAE(mm)':>9} {'RMSE(mm)':>10}")
    print("-" * 74)

    model = load_da3_base(device)
    m = evaluate(model, SCARED, img_size=IMG_SIZE, device=device)
    print(f"{'DA3-BASE':<28} {m['AbsRel']:>8.4f} {m['d1']:>8.4f} {m['d3']:>8.4f} {m['MAE']*1000:>9.1f} {m['RMSE']*1000:>10.1f}")
    del model; torch.cuda.empty_cache()
    print("-" * 74)

    for label, ckpt_path, use_lora in CHECKPOINTS:
        if not Path(ckpt_path).exists():
            print(f"{label:<28}  (not found)")
            continue
        model = load_model(ckpt_path, use_lora, device)
        m = evaluate(model, SCARED, img_size=IMG_SIZE, device=device)
        print(f"{label:<28} {m['AbsRel']:>8.4f} {m['d1']:>8.4f} {m['d3']:>8.4f} {m['MAE']*1000:>9.1f} {m['RMSE']*1000:>10.1f}")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
