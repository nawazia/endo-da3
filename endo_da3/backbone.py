"""
DA3-modified DINOv2 ViT-B/14 backbone.

Provides:
  build_da3_dino  — randomly initialised DA3 backbone
  load_da3_dino   — DA3 backbone with pretrained DA3-BASE weights
  replace_dino_weights — swap vanilla DINOv2 weights in-place (e.g. GastroNet)
"""

import torch
import torch.nn.functional as F

from endo_da3._vendor.dinov2.vision_transformer import DinoVisionTransformer, vit_base

# DA3-BASE HuggingFace repo — ViT-B/14, Apache 2.0 licence
DA3_BASE_REPO = "depth-anything/DA3-BASE"

# Prefix separating the DINOv2 backbone from the rest of the DA3 checkpoint
_BACKBONE_PREFIX = "model.backbone.pretrained."


def _adapt_state_dict(src_sd: dict, model: DinoVisionTransformer) -> dict:
    """
    Align a source state dict to the model's parameter shapes, handling the
    one common mismatch: pos_embed trained at a different resolution.
    Bicubically interpolates pos_embed to match the model's size.
    All other shape mismatches are skipped with a warning.
    """
    model_sd = dict(model.state_dict())
    adapted = {}

    for k, v in src_sd.items():
        if k not in model_sd:
            continue
        if v.shape == model_sd[k].shape:
            adapted[k] = v
        elif k == "pos_embed":
            cls, patches = v[:, :1], v[:, 1:]
            src_n = patches.shape[1]
            tgt_n = model_sd[k].shape[1] - 1
            src_hw = int(src_n ** 0.5)
            tgt_hw = int(tgt_n ** 0.5)
            D = patches.shape[-1]
            patches = patches.reshape(1, src_hw, src_hw, D).permute(0, 3, 1, 2).float()
            patches = F.interpolate(patches, size=(tgt_hw, tgt_hw), mode='bicubic', antialias=True)
            patches = patches.permute(0, 2, 3, 1).reshape(1, tgt_n, D).to(v.dtype)
            adapted[k] = torch.cat([cls, patches], dim=1)
            print(f"  pos_embed interpolated {src_hw}×{src_hw} → {tgt_hw}×{tgt_hw}")
        else:
            print(f"  Warning: skipping '{k}' — shape {tuple(v.shape)} != model {tuple(model_sd[k].shape)}")

    return adapted


def build_da3_dino(
    alt_start: int = 4,
    qknorm_start: int = 4,
    rope_start: int = 4,
    rope_freq: int = 100,
    cat_token: bool = True,
    num_register_tokens: int = 0,
    img_size: int = 518,
) -> DinoVisionTransformer:
    """
    Build a DA3-modified DINOv2 ViT-B/14, randomly initialised.

    Architecture vs vanilla DINOv2
    --------------------------------
    Blocks 0..alt_start-1  : standard per-view self-attention
    Blocks alt_start..end  : alternating local (even) / global cross-view (odd)

    Input:  (B, S, C, H, W)
    Output of get_intermediate_layers():
        list of (patch_tokens, camera_token) per requested layer
        patch_tokens  : (B, S, N_patches, D or 2D)
        camera_token  : (B, S, D or 2D)
        Feature dim is 2D when cat_token=True (local ‖ global concatenated).

    img_size note
    -------------
    Use img_size=518 (default) to match DA3-BASE native training resolution.
    Use img_size=336 to match GastroNet native resolution — block weights are
    in-distribution at 576 tokens (24×24 patches); prefer for fine-tuning.
    """
    return vit_base(
        patch_size=14,
        img_size=img_size,
        num_register_tokens=num_register_tokens,
        ffn_layer="mlp",
        alt_start=alt_start,
        qknorm_start=qknorm_start,
        rope_start=rope_start,
        rope_freq=rope_freq,
        cat_token=cat_token,
    )


def load_da3_dino(
    alt_start: int = 4,
    qknorm_start: int = 4,
    rope_start: int = 4,
    rope_freq: int = 100,
    cat_token: bool = True,
    img_size: int = 518,
    device: str = "cuda",
) -> DinoVisionTransformer:
    """
    Build a DA3-modified DINOv2 ViT-B/14 and load ALL weights from the
    pretrained DA3-BASE checkpoint (downloads from HuggingFace if not cached).

    img_size=518  — DA3-BASE native; best for zero-shot with the DA3 depth head.
    img_size=336  — GastroNet native; best when fine-tuning on endoscopy data.
    """
    model = build_da3_dino(
        alt_start=alt_start,
        qknorm_start=qknorm_start,
        rope_start=rope_start,
        rope_freq=rope_freq,
        cat_token=cat_token,
        img_size=img_size,
    )

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ckpt_path = hf_hub_download(repo_id=DA3_BASE_REPO, filename="model.safetensors")
    full_sd = load_file(ckpt_path, device="cpu")

    backbone_sd = {
        k[len(_BACKBONE_PREFIX):]: v
        for k, v in full_sd.items()
        if k.startswith(_BACKBONE_PREFIX)
    }

    backbone_sd = _adapt_state_dict(backbone_sd, model)
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    print(f"Loaded DA3-BASE backbone weights from {DA3_BASE_REPO}.")
    if missing:
        print(f"  Missing (not in DA3-BASE checkpoint): {missing}")
    if unexpected:
        print(f"  Unexpected (not used): {unexpected}")

    return model.eval().to(device)


def replace_dino_weights(
    model: DinoVisionTransformer,
    dino_state_dict: dict,
) -> None:
    """
    Swap vanilla DINOv2 weights inside a DA3-modified backbone in-place,
    leaving DA3-specific parameters (camera_token, q/k-norm, RoPE) untouched.

    Parameters
    ----------
    model:
        A DinoVisionTransformer built by build_da3_dino() or load_da3_dino().
    dino_state_dict:
        State dict from any DINOv2 ViT-B/14 variant (e.g. GastroNet).
        Keys must follow standard DINOv2 naming (patch_embed, blocks.*, norm, …).

    DA3-only keys that are NOT in dino_state_dict and will be kept as-is:
        camera_token
        blocks[i].attn.q_norm.*  (for i >= qknorm_start)
        blocks[i].attn.k_norm.*  (for i >= qknorm_start)
        rope frequency cache (computed, not a stored param)

    Shape-mismatched keys (e.g. pos_embed at a different resolution) are
    bicubically interpolated; unresolvable mismatches are skipped.
    """
    adapted = _adapt_state_dict(dino_state_dict, model)
    missing, unexpected = model.load_state_dict(adapted, strict=False)

    da3_only = {"camera_token"}
    truly_missing = [k for k in missing if k not in da3_only
                     and "q_norm" not in k and "k_norm" not in k]
    if truly_missing:
        print(f"Warning: expected DINOv2 keys not found in supplied state dict:\n  {truly_missing}")
    if unexpected:
        print(f"Warning: keys not used (not in model):\n  {unexpected}")
