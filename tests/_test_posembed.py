"""
Test whether keeping vs interpolating gastronet's pos_embed affects the
grid-like artifacts seen when running GastroNet-DA3 on a natural image.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from endo_da3 import load_da3_dino, replace_dino_weights

IMG_PATH = '/home/in4218/code/Depth-Anything-3/assets/examples/SOH/000.png'
OUT_LAYER = 11
IMG_SIZE  = 518


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def gastro_sd():
    ckpt = torch.load('/home/in4218/code/GastroNet/gastronet/dinov2.pth', map_location='cpu')
    return {k.replace('backbone.', ''): v for k, v in ckpt['teacher'].items()
            if k.startswith('backbone.')}


def interpolate_pos_embed(pos_embed, target_n_patches):
    """Bilinearly interpolate pos_embed from src_n_patches to target_n_patches."""
    # pos_embed: (1, 1 + src_n_patches, D)
    cls_token = pos_embed[:, :1]
    patch_embed = pos_embed[:, 1:]           # (1, src_n, D)
    src_n = patch_embed.shape[1]
    src_h = src_w = int(src_n ** 0.5)
    tgt_h = tgt_w = int(target_n_patches ** 0.5)
    D = patch_embed.shape[-1]

    patch_embed = patch_embed.reshape(1, src_h, src_w, D).permute(0, 3, 1, 2).float()
    patch_embed = F.interpolate(patch_embed, size=(tgt_h, tgt_w), mode='bicubic', antialias=True)
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, tgt_h * tgt_w, D)
    return torch.cat([cls_token, patch_embed], dim=1)


@torch.no_grad()
def run(model, img_tensor):
    x = img_tensor.unsqueeze(1)
    outputs, _ = model.get_intermediate_layers(x, n=[OUT_LAYER])
    return outputs[0][0].squeeze(0).squeeze(0).cpu().float()


def pca_vis(tokens):
    n = tokens.shape[0]
    h = w = int(n ** 0.5)
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(tokens.numpy())
    for i in range(3):
        rgb[:, i] = (rgb[:, i] - rgb[:, i].min()) / (rgb[:, i].max() - rgb[:, i].min() + 1e-8)
    return rgb.reshape(h, w, 3)


def main():
    img = Image.open(IMG_PATH).convert('RGB')
    img_tensor = get_transform()(img).unsqueeze(0).cuda()

    sd = gastro_sd()
    target_n = (IMG_SIZE // 14) ** 2  # 1369 for 518px

    # Model A: keep DA3's pos_embed (current behaviour — shape mismatch skipped)
    print("Building model A: DA3 pos_embed (current behaviour)...")
    model_a = load_da3_dino(device='cuda')
    replace_dino_weights(model_a, sd)

    # Model B: interpolate gastronet's pos_embed to 518px grid
    print("Building model B: interpolated gastronet pos_embed...")
    model_b = load_da3_dino(device='cuda')
    sd_interp = dict(sd)
    sd_interp['pos_embed'] = interpolate_pos_embed(sd['pos_embed'], target_n)
    replace_dino_weights(model_b, sd_interp)

    feats_a = run(model_a, img_tensor)
    feats_b = run(model_b, img_tensor)

    diff = (feats_a - feats_b).abs()
    print(f"\nDifference between DA3 pos_embed vs interpolated gastronet pos_embed:")
    print(f"  Max |diff|:  {diff.max().item():.6f}")
    print(f"  Mean |diff|: {diff.mean().item():.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pca_vis(feats_a))
    axes[0].set_title('DA3 pos_embed (kept)')
    axes[0].axis('off')

    axes[1].imshow(pca_vis(feats_b))
    axes[1].set_title('Gastronet pos_embed (interpolated to 518px)')
    axes[1].axis('off')

    diff_map = diff.mean(-1).numpy().reshape(int(feats_a.shape[0]**0.5), -1)
    im = axes[2].imshow(diff_map, cmap='hot')
    axes[2].set_title('|diff| between the two')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f'pos_embed source effect on GastroNet-DA3 features (layer {OUT_LAYER})', fontsize=12)
    plt.tight_layout()
    out = '/home/in4218/code/endo-da3/posembed_test.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
