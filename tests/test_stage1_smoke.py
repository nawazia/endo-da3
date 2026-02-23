"""
Smoke test for Stage 1: dataset loading + one forward/backward pass.

Run:
    python tests/test_stage1_smoke.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

DATA      = Path("/home/in4218/code/data")
GASTRONET = "/home/in4218/code/GastroNet/gastronet/dinov2.pth"
IMG_SIZE  = 336
SEQ_LEN   = 2
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Per-dataset expected depth ranges (metres)
_DEPTH_RANGES = {
    "SimCol3DDataset":           (0.0,  0.21),   # 0–20 cm
    "C3VDDataset":               (0.0,  0.11),   # 0–100 mm
    "EndoSLAMSynthDataset":      (0.0,  0.65),   # 0–63 cm (Colon max)
    "PolypSense3DVirtualDataset":(0.0,  0.26),   # 0–255 mm
}


# ── 1. Datasets ───────────────────────────────────────────────────────────────
def test_datasets():
    from endo_da3.data import (
        SimCol3DDataset, C3VDDataset,
        EndoSLAMSynthDataset, PolypSense3DVirtualDataset,
    )

    # relative_pose: datasets that normalise c2w to first frame (c2w[0] = I)
    configs = [
        (SimCol3DDataset,            DATA / "SimCol3D",     False),
        (C3VDDataset,                DATA / "C3VD",         False),
        (EndoSLAMSynthDataset,       DATA / "EndoSLAM",     True),
        (PolypSense3DVirtualDataset, DATA / "PolypSense3D", True),
    ]

    for cls, root, relative_pose in configs:
        name = cls.__name__
        ds = cls(root, split="train", img_size=IMG_SIZE, seq_len=SEQ_LEN, with_pose=True)
        sample = ds[0]
        images = sample["images"]
        depths = sample["depths"]
        K      = sample["K"]
        c2w    = sample["c2w"]

        d_min, d_max = _DEPTH_RANGES[name]

        print(f"  {name}")
        print(f"    samples={len(ds)}  images={tuple(images.shape)}"
              f"  depths={tuple(depths.shape)}")
        print(f"    depth min={depths[depths>0].min():.4f}"
              f"  max={depths.max():.4f}  (expected ≤{d_max:.2f} m)")
        if relative_pose:
            print(f"    c2w[0]=I? {torch.allclose(c2w[0], torch.eye(4), atol=1e-5)}"
                  f"  |t[1]|={c2w[1,:3,3].norm():.5f} m")
        else:
            det = torch.linalg.det(c2w[0, :3, :3])
            print(f"    c2w absolute  det(R)={det:.4f}  |t[0]|={c2w[0,:3,3].norm():.4f} m")

        assert images.shape == (SEQ_LEN, 3, IMG_SIZE, IMG_SIZE)
        assert depths.shape == (SEQ_LEN, IMG_SIZE, IMG_SIZE)
        assert K.shape      == (3, 3)
        assert c2w.shape    == (SEQ_LEN, 4, 4)
        assert depths.min() >= 0,             f"{name}: negative depth"
        assert depths.max() <= d_max + 1e-3,  \
            f"{name}: depth {depths.max():.4f} > expected max {d_max}"
        if relative_pose:
            assert torch.allclose(c2w[0], torch.eye(4), atol=1e-5), \
                f"{name}: c2w[0] is not identity"
        else:
            det = torch.linalg.det(c2w[0, :3, :3])
            assert abs(det.item() - 1.0) < 1e-3, f"{name}: det(R) = {det:.4f}, expected 1"
        print(f"    [PASS]\n")


# ── 2. Loss (GT ray + point map) ──────────────────────────────────────────────
def test_loss():
    from endo_da3.data import SimCol3DDataset
    from endo_da3.loss import compute_gt_ray_and_pointmap

    ds     = SimCol3DDataset(DATA / "SimCol3D", split="train",
                             img_size=IMG_SIZE, seq_len=SEQ_LEN, with_pose=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch  = next(iter(loader))

    depths = batch["depths"]
    c2w    = batch["c2w"]
    K      = batch["K"]

    Hp, Wp = 192, 192
    ray_gt, point_gt, depth_down, scale = compute_gt_ray_and_pointmap(
        depths, c2w, K, ray_hw=(Hp, Wp)
    )

    print(f"  ray_gt    : {tuple(ray_gt.shape)}  (B,S,Hp,Wp,6)")
    print(f"  point_gt  : {tuple(point_gt.shape)}  (B,S,Hp,Wp,3)")
    print(f"  depth_down: {tuple(depth_down.shape)}")
    print(f"  scale     : {scale:.4f}")

    assert ray_gt.shape   == (2, SEQ_LEN, Hp, Wp, 6)
    assert point_gt.shape == (2, SEQ_LEN, Hp, Wp, 3)

    dirs  = ray_gt[..., :3]
    norms = dirs.norm(dim=-1)
    assert norms.min() >= 0.99, f"ray dir norm < 1: {norms.min():.4f}"
    assert norms.max() <= 3.0,  f"ray dir norm > √3: {norms.max():.4f}"
    print(f"  ray dir norms: min={norms.min():.4f}  max={norms.max():.4f}")
    print("  [PASS]\n")


# ── 3. Forward + backward ─────────────────────────────────────────────────────
def test_train_step():
    from endo_da3 import EndoDA3
    from endo_da3.data import SimCol3DDataset
    from endo_da3.loss import da3_loss

    print("  Loading Endo-DA3 (DA3-BASE weights, no backbone swap)…")
    model = EndoDA3.from_pretrained(img_size=IMG_SIZE, with_camera=False, device=DEVICE)
    model.train()

    ds     = SimCol3DDataset(DATA / "SimCol3D", split="train",
                             img_size=IMG_SIZE, seq_len=SEQ_LEN, with_pose=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch  = next(iter(loader))

    images = batch["images"].to(DEVICE)
    depths = batch["depths"].to(DEVICE)
    c2w    = batch["c2w"].to(DEVICE)
    K      = batch["K"].to(DEVICE)

    out = model(images)
    print(f"  model outputs:")
    for k, v in out.items():
        print(f"    {k:12s}: {tuple(v.shape)}")

    loss, terms = da3_loss(out, depths, c2w, K)
    print(f"\n  loss : {loss.item():.4f}")
    for k, v in terms.items():
        print(f"    {k}: {v:.4f}")

    loss.backward()
    print("  [PASS]\n")


if __name__ == "__main__":
    print("=" * 60)
    print("1. Datasets (all 4)")
    print("=" * 60)
    test_datasets()

    print("=" * 60)
    print("2. Loss (GT ray + point map)")
    print("=" * 60)
    test_loss()

    print("=" * 60)
    print("3. Forward + backward")
    print("=" * 60)
    test_train_step()

    print("All smoke tests passed.")
