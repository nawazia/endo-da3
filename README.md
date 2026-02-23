# Endo-DA3

**Endo-DA3** adapts [Depth Anything 3 (DA3)](https://github.com/DepthAnything/Depth-Anything-3) for endoscopic depth estimation by replacing the DA3 backbone with [GastroNet](https://github.com/...)'s DINOv2, which was pretrained on gastrointestinal imagery.

The result is a full multi-view depth model, backbone, DualDPT head, and camera decoder, ready to fine-tune on endoscopy data.

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

You will also need the GastroNet checkpoint (`dinov2.pth`).

## Usage

```python
from endo_da3 import EndoDA3
import torch

# Load DA3-BASE weights then swap backbone with GastroNet
model = EndoDA3.from_pretrained(img_size=336, with_camera=True)

ckpt = torch.load('path/to/gastronet/dinov2.pth', map_location='cpu')
gastro_sd = {k.replace('backbone.', ''): v
             for k, v in ckpt['teacher'].items()
             if k.startswith('backbone.')}
model.replace_backbone(gastro_sd)

# Single-view
x = torch.randn(1, 1, 3, 336, 336).cuda()
out = model(x)
# out['depth']:      (1, 1, 336, 336)
# out['depth_conf']: (1, 1, 336, 336)
# out['ray']:        (1, 1, 192, 192, 6)
# out['extrinsics']: (1, 1, 3, 4)
# out['intrinsics']: (1, 1, 3, 3)

# Multi-view (S=2)
x = torch.randn(1, 2, 3, 336, 336).cuda()
out = model(x)
```

## Tests

```bash
# Depth comparison: DA3-BASE vs Endo-DA3
python tests/test_depth.py

# Multi-view alternating attention verification
python tests/test_multiview.py

# Backbone feature comparison (PCA)
python tests/test_da3_dino.py
```

## PLAN

| Stage | What is Trainable? | Primary Data | Purpose |
|---|---|---|---|
| 1. Align | Decoder Only | Synthetic (SimCol3D, C3VD, EndoSLAM [synth], PolypSense3D [clinical]) | Map Features → Geometry |
| 2. Steer (Distillation) | Decoder + LoRA | Unlabeled Real (HyperKvasir, EndoSLAM [real]) | Fix Texture/Lighting Artifacts |
| 3. Calibrate | Decoder Only | Labeled Real (SCARED, Hamlyn, StereoMIS, SERV-CT) | Final Metric Precision |

eval on: SCARED-C, EndoAbS

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

## Acknowledgements

- [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-3) — ByteDance, Apache 2.0
- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI
- [GastroNet](https://github.com/...)  <!-- Update with actual repo link -->
