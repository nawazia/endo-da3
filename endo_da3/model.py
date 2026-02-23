"""
EndoDA3: DA3 depth model with a swappable DINOv2 backbone for endoscopy fine-tuning.

Architecture
------------
backbone  : DA3-modified DINOv2 ViT-B/14  (build_da3_dino / load_da3_dino)
head      : DualDPT                        (depth + depth_conf + ray + ray_conf)
cam_dec   : CameraDec                      (optional; camera token → extrinsics/intrinsics)
cam_enc   : CameraEnc                      (optional; known poses → camera token for backbone)

Forward outputs (all [B, S, ...])
----------------------------------
Always present:
  depth       : [B, S, H, W]      — metric-free depth (exp-activated)
  depth_conf  : [B, S, H, W]      — depth confidence
  ray         : [B, S, H, W, 6]   — per-pixel ray origin+direction (linear activation)
  ray_conf    : [B, S, H, W]      — ray confidence
When cam_dec is active (with_camera=True):
  extrinsics  : [B, S, 3, 4]      — w2c extrinsic matrices
  intrinsics  : [B, S, 3, 3]      — camera intrinsics
"""

from __future__ import annotations

import torch
import torch.nn as nn

from endo_da3.backbone import (
    DA3_BASE_REPO,
    _BACKBONE_PREFIX,
    _adapt_state_dict,
    build_da3_dino,
    replace_dino_weights,
)
from endo_da3._vendor.dualdpt import DualDPT
from endo_da3._vendor.cam_dec import CameraDec
from endo_da3._vendor.cam_enc import CameraEnc
from endo_da3._vendor.utils.transform import pose_encoding_to_extri_intri
from endo_da3._vendor.geometry import affine_inverse

# DA3-BASE DualDPT config (from da3-base.yaml)
_HEAD_DIM_IN     = 1536          # 2 × 768 because cat_token=True
_HEAD_FEATURES   = 128
_HEAD_OUT_CH     = (96, 192, 384, 768)

_HEAD_PREFIX    = "model.head."
_CAM_DEC_PREFIX = "model.cam_dec."
_CAM_ENC_PREFIX = "model.cam_enc."

# Layers whose features DualDPT consumes (must be exactly 4)
_OUT_LAYERS = [5, 7, 9, 11]


class EndoDA3(nn.Module):
    """
    Full DA3 model with a replaceable DINOv2 backbone.

    Parameters
    ----------
    img_size : int
        Backbone input resolution.  336 = GastroNet native (recommended for
        fine-tuning on endoscopy data).  518 = DA3-BASE native.
    with_camera : bool
        Include CameraEnc / CameraDec.  Set False for depth-only inference.
    """

    def __init__(self, img_size: int = 336, with_camera: bool = True):
        super().__init__()
        self.img_size = img_size

        self.backbone = build_da3_dino(img_size=img_size)

        self.head = DualDPT(
            dim_in=_HEAD_DIM_IN,
            output_dim=2,
            features=_HEAD_FEATURES,
            out_channels=_HEAD_OUT_CH,
        )

        if with_camera:
            self.cam_enc = CameraEnc(dim_out=768)
            self.cam_dec = CameraDec(dim_in=_HEAD_DIM_IN)
        else:
            self.cam_enc = None
            self.cam_dec = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        img_size: int = 336,
        with_camera: bool = True,
        device: str = "cuda",
    ) -> "EndoDA3":
        """
        Build EndoDA3 and load ALL weights from the DA3-BASE checkpoint
        (downloads from HuggingFace on first call, cached afterwards).

        img_size=336 loads with bicubic pos_embed interpolation (518→336).
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        model = cls(img_size=img_size, with_camera=with_camera)

        ckpt_path = hf_hub_download(repo_id=DA3_BASE_REPO, filename="model.safetensors")
        full_sd = load_file(ckpt_path, device="cpu")

        # ---- backbone ----
        backbone_sd = {
            k[len(_BACKBONE_PREFIX):]: v
            for k, v in full_sd.items()
            if k.startswith(_BACKBONE_PREFIX)
        }
        backbone_sd = _adapt_state_dict(backbone_sd, model.backbone)
        m, u = model.backbone.load_state_dict(backbone_sd, strict=False)
        print(f"Backbone loaded from {DA3_BASE_REPO}.")
        if m:
            print(f"  Missing: {m}")
        if u:
            print(f"  Unexpected: {u}")

        # ---- head ----
        # strict=False: the DA3-BASE checkpoint omits LayerNorm weights for aux
        # pyramid levels 1-3 (only level 0 has them).  Those params default to
        # identity (weight=1, bias=0) which matches the no-LN behaviour at init.
        head_sd = {k[len(_HEAD_PREFIX):]: v for k, v in full_sd.items()
                   if k.startswith(_HEAD_PREFIX)}
        m, u = model.head.load_state_dict(head_sd, strict=False)
        if m:
            print(f"  Head — missing (LN init to identity): {m}")
        if u:
            print(f"  Head — unexpected: {u}")

        # ---- camera (optional) ----
        if with_camera:
            cam_enc_sd = {k[len(_CAM_ENC_PREFIX):]: v for k, v in full_sd.items()
                          if k.startswith(_CAM_ENC_PREFIX)}
            model.cam_enc.load_state_dict(cam_enc_sd, strict=True)

            cam_dec_sd = {k[len(_CAM_DEC_PREFIX):]: v for k, v in full_sd.items()
                          if k.startswith(_CAM_DEC_PREFIX)}
            model.cam_dec.load_state_dict(cam_dec_sd, strict=True)

        print("EndoDA3 fully loaded.")
        return model.eval().to(device)

    def replace_backbone(self, dino_state_dict: dict) -> None:
        """
        Swap the DINOv2 weights in-place (e.g. load GastroNet weights),
        keeping DA3-specific parameters (camera_token, q/k-norm, RoPE).

        Example
        -------
        ckpt = torch.load('dinov2.pth', map_location='cpu')
        gastro_sd = {k.replace('backbone.', ''): v
                     for k, v in ckpt['teacher'].items()
                     if k.startswith('backbone.')}
        model.replace_backbone(gastro_sd)
        """
        replace_dino_weights(self.backbone, dino_state_dict)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        x           : (B, S, 3, H, W)   input images, ImageNet-normalised
        extrinsics  : (B, S, 4, 4)       known w2c extrinsics (optional)
        intrinsics  : (B, S, 3, 3)       known camera intrinsics (optional)

        Returns
        -------
        dict with keys: depth, depth_conf, ray, ray_conf
                        + extrinsics, intrinsics  (when cam_dec is present)
        """
        H, W = x.shape[-2], x.shape[-1]

        # Encode known camera poses into a token injected at alt_start
        if extrinsics is not None and self.cam_enc is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, (H, W))
        else:
            cam_token = None

        # Backbone: (B, S, 3, H, W) → list of (patch_tokens, cam_tokens) × 4
        feats, _ = self.backbone.get_intermediate_layers(
            x, n=_OUT_LAYERS, cam_token=cam_token
        )

        # Depth + ray head
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self.head(feats, H, W, patch_start_idx=0)
            # output is an addict.Dict; shapes: [B, S, H, W] / [B, S, H, W, 6]

            # Camera pose estimation from the camera token at the last backbone layer
            if self.cam_dec is not None:
                cam_tokens_last = feats[-1][1]   # (B, S, 2D)
                pose_enc = self.cam_dec(cam_tokens_last)
                c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
                output["extrinsics"] = affine_inverse(c2w)
                output["intrinsics"] = ixt

        return dict(output)
