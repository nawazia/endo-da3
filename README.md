# Endo-DA3

**Endo-DA3** adapts [Depth Anything 3 (DA3)](https://github.com/DepthAnything/Depth-Anything-3) for endoscopic depth estimation by replacing the DA3 backbone with [GastroNet](https://github.com/...)'s DINOv2, which was pretrained on gastrointestinal imagery.

The result is a full multi-view depth model, backbone, DualDPT head, and camera decoder, ready to fine-tune on endoscopy data.

## Training Pipeline

| Stage | Frozen | Trainable | Data | Loss | Purpose |
|---|---|---|---|---|---|
| 1. Align | GastroNet backbone | DualDPT head, camera_token, q_norm/k_norm | SimCol3D, C3VD, EndoSLAM (synth), PolypSense3D | LD + LM + LP + Lgrad (GT poses) | Teach decoder to read GastroNet features |
| 2a. Real-domain | GastroNet backbone base | LoRA adapters, DualDPT head | Hamlyn daVinci (stereo, RAFT pseudo-GT) | LD + Lgrad | Bridge synthetic→real domain gap |
| 3. Calibrate | GastroNet backbone + LoRA | DualDPT head (low LR) | SCARED (structured-light GT) | LD + Lgrad | Anchor metric scale for clinical use |

### Stage 1 — Align

Trains the DualDPT head on 4 synthetic/simulated endoscopy datasets with ground-truth depth. GastroNet backbone is fully frozen; the decoder learns to decode GastroNet's feature maps into metric depth.

```bash
python -u train/stage1.py \
    --data-path ~/code/data \
    --gastronet /path/to/gastronet/dinov2.pth \
    --out-dir runs/stage1 \
    --batch-size 4 --epochs 30 2>&1 | tee runs/stage1/stage1.log
```

### Stage 2a — Real-domain alignment

Injects LoRA adapters (rank-4) into backbone attention layers and fine-tunes on Hamlyn daVinci stereo video. Pseudo-GT depth is generated offline with RAFT-Stereo (`depth = fx·B / disparity = 2.103 / disparity`).

```bash
# Generate RAFT-Stereo pseudo-GT depth maps (run once)
python tools/generate_hamlyn_pseudogt.py --split train --batch_size 32
python tools/generate_hamlyn_pseudogt.py --split test  --batch_size 32

# Train
python -u train/stage2a.py \
    --stage1-ckpt runs/stage1/best.pt \
    --gastronet /path/to/gastronet/dinov2.pth \
    --hamlyn-root ~/code/data/Hamlyn/daVinci \
    --data-path ~/code/data \
    --out-dir runs/stage2a \
    --batch-size 5 --epochs 20 --lr 5e-5 2>&1 | tee runs/stage2a/stage2a.log
```

### Stage 3 — Calibrate *(planned)*

Fine-tunes the decoder only on SCARED structured-light GT depth to recover metric scale.

## Datasets

| Dataset | Type | Depth | Used in |
|---|---|---|---|
| SimCol3D | Synthetic colonoscopy | Unity depth buffer (0–20cm) | Stage 1 |
| C3VD | Synthetic colon (photorealistic) | Metric tiff (metres) | Stage 1 |
| EndoSLAM | Unity simulation (colon/stomach/SI) | `far_clip − R/100` | Stage 1 |
| PolypSense3D | Synthetic polyp scenes | `(255−R)/1000` m | Stage 1 |
| Hamlyn daVinci | Real robotic surgery video (stereo) | RAFT-Stereo pseudo-GT | Stage 2a |
| SCARED | Real colonoscopy (structured light) | Metric GT | Stage 3 |

## Architecture

```
EndoDA3
├── backbone   DA3-modified DINOv2 ViT-B/14
│              GastroNet weights → patch_embed, pos_embed, blocks.*.qkv/mlp
│              DA3 weights       → camera_token, q_norm, k_norm (blocks 4–11)
├── head       DualDPT — depth + ray (DA3-BASE weights)
├── cam_enc    CameraEnc — known poses → camera token (optional)
└── cam_dec    CameraDec — camera token → extrinsics / intrinsics (optional)
```

**Input**: `(B, S, 3, H, W)` — batch of S-view image sequences

**Output**: `depth`, `depth_conf`, `ray`, `ray_conf`, and optionally `extrinsics`, `intrinsics`

## Setup

```bash
conda activate endo-da3
pip install -e .
```

Dependencies: `torch`, `torchvision`, `einops`, `addict`, `huggingface_hub`, `safetensors`

You will also need the GastroNet checkpoint (`dinov2.pth`) — see [GastroNet access request](#gastronet-access).

## Usage

### Prerequisites

The GastroNet backbone is access-restricted. Request access via **[link TBD]**, then download `dinov2.pth`.

The Endo-DA3 checkpoint distributed with this repo contains **only the trained head and LoRA adapter weights** — it does not include the GastroNet backbone, which must be provided separately.

### Inference

```python
from endo_da3 import EndoDA3
import torch

model = EndoDA3.from_finetuned(
    gastronet_path='path/to/gastronet/dinov2.pth',  # access-restricted, see above
    weights_path='endo_da3_weights.pt',              # download from releases
)

# Single image
x = torch.randn(1, 1, 3, 336, 336).cuda()
with torch.no_grad():
    out = model(x)
# out['depth']:      (1, 1, 336, 336)  metric depth in metres
# out['depth_conf']: (1, 1, 336, 336)  per-pixel confidence

# Multi-view sequence (S=2)
x = torch.randn(1, 2, 3, 336, 336).cuda()
with torch.no_grad():
    out = model(x)
# out['ray']:        (1, 2, H/14, W/14, 6)  world-space ray map
```

## Tests

```bash
# Depth comparison: DA3-BASE vs Endo-DA3
python tests/test_depth.py

# Multi-view alternating attention verification
python tests/test_multiview.py

# Backbone feature comparison (PCA)
python tests/test_da3_dino.py

# Hamlyn stereo reprojection test (verifies camera params)
python tests/test_hamlyn_reprojection.py
```

## PLAN

| Stage | What is Trainable? | Primary Data | Purpose |
|---|---|---|---|
| 1. Align | Decoder Only | Synthetic (SimCol3D, C3VD, EndoSLAM [synth], PolypSense3D [Virtual]) | Map Features → Geometry |
| 2a. Steer (Distillation) | Decoder + LoRA | Unlabeled Real (Hamlyn, StereoMIS) | Fix Texture/Lighting Artifacts |
| 2b. Steer (Distillation) | Decoder + LoRA | Unlabeled Real (EndoMapper, EndoSLAM [Micro/PillCam], PolypSense3D [Clinical]) | Fix Texture/Lighting Artifacts |
| 3. Calibrate | Decoder Only | Labeled Real (SCARED, EndoSLAM [High/LowCam] [Olympus],) | Final Metric Precision |

eval on: SCARED-C, EndoAbS, PolypSense3D (clinical)

Stage 1: The "Translation" Phase (Alignment)
Goal: Teach the Decoder to understand GastroNet's feature maps using perfect geometric data.

Data: Synthetic/Simulated (e.g., SimCol3D).

Backbone: STRICTLY FROZEN. Do not touch GastroNet yet.

Decoder: FULLY TRAINABLE.

Logic: Since synthetic data has "perfect" depth labels, the decoder is forced to find the depth cues hidden in GastroNet's medical embeddings. By the end of this stage, your depth maps should look like a colon, but the "textures" might still be a bit off.

Stage 2: The "Steering" Phase (Refinement)
Goal: Allow the Backbone to "lean" into depth estimation without forgetting its medical knowledge.

Data: Unlabeled Real Endo (e.g., HyperKvasir) + Original DA3 Teacher.

Backbone: LoRA ENABLED. (Only the small LoRA adapters are trainable; the 5M weights stay frozen).

Decoder: FULLY TRAINABLE.

Logic: You use the "Teacher-Student" method here. You feed a real endoscopy image to the original DepthAnything (Teacher) and your Endo-DA3 (Student). You train your Student (LoRA + Decoder) to match the Teacher's output. This fixes the "texture" issues that synthetic data can't capture.

Stage 3: The "Calibration" Phase (Precision)
Goal: Finalize the absolute scale and surgical precision.

Data: Real Ground-Truth Depth (e.g., SCARED, Hamlyn).

Backbone: STRICTLY FROZEN (including the LoRA weights from Stage 2).

Decoder: FULLY TRAINABLE (Low learning rate).

Logic: Now that the model understands the "language" (Stage 1) and the "textures" (Stage 2), you use your most precious, limited data to do a final "calibration" so the depth values (in millimeters) are actually accurate for surgery.

Why this specific order?
Stage 1 builds the geometric foundation (the "shape" of a tube).

Stage 2 handles domain adaptation (the "look" of wet tissue).

Stage 3 provides metric accuracy (the "distance" in mm).

EndoSLAM
  ┌──────────────────────────────────────┬──────────────────────────┬────────────┐
  │                 Data                 │       Supervision        │   Stage    │
  ├──────────────────────────────────────┼──────────────────────────┼────────────┤
  │ UnityCam                             │ Pixelwise GT depth       │ Stage 1  ✓ │
  ├──────────────────────────────────────┼──────────────────────────┼────────────┤
  │ Hamlyn + StereoMIS P1                │ RAFT pseudo-GT (stereo)  │ Stage 2a ✓ │
  ├──────────────────────────────────────┼──────────────────────────┼────────────┤
  │ MiroCam + PillCam + other mono video │ Pose GT → photometric    │ Stage 2b   │
  ├──────────────────────────────────────┼──────────────────────────┼────────────┤
  │ HighCam + LowCam + OlympusCam        │ CT/mesh → rendered depth │ Stage 3  ✓ │
  └──────────────────────────────────────┴──────────────────────────┴────────────┘

1. Investigate the growing artifact problem
2. Implement VLoRA like: DARES: Depth Anything in Robotic Endoscopic Surgery with Self-Supervised Vector-LoRA
3. Investigate unifying Depth Anything V2 (depth) and Reloc3rX (pose) like: Endo-FASt3r

## Acknowledgements

- [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-3) — ByteDance, Apache 2.0
- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI
- [GastroNet](https://github.com/...)  <!-- Update with actual repo link -->
