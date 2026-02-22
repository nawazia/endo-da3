"""
Compare our load_da3_dino() backbone against the backbone extracted directly
from the full DepthAnything3 model, on a single image.

Both should produce identical outputs since they load the same weights.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

IMG_PATH = '/home/in4218/code/data/kvasir-dataset-v2/normal-z-line/0be91a4e-be3d-4c06-92d7-6e0ee417f55a.jpg'
OUT_LAYERS = [5, 7, 9, 11]   # da3-base.yaml out_layers
IMG_SIZE = 518                 # DA3 backbone img_size


def get_transform(size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

GASTRO_SIZE = 336


def load_da3():
    from endo_da3 import load_da3_dino
    print("  Loading DA3-BASE backbone...")
    return load_da3_dino(device='cuda')


def load_endo_da3():
    import torch
    from endo_da3 import load_da3_dino, replace_dino_weights
    print("  Loading DA3-BASE backbone at 336px (GastroNet native)...")
    model = load_da3_dino(img_size=336, device='cuda')
    print("  Swapping in GastroNet weights...")
    ckpt = torch.load('/home/in4218/code/GastroNet/gastronet/dinov2.pth', map_location='cpu')
    gastro_sd = {
        k.replace('backbone.', ''): v
        for k, v in ckpt['teacher'].items()
        if k.startswith('backbone.')
    }
    replace_dino_weights(model, gastro_sd)
    return model


@torch.no_grad()
def run(model, img_tensor):
    """
    img_tensor: (1, 3, H, W)  →  wrapped to (B=1, S=1, 3, H, W) for DA3
    Returns list of patch-token tensors, one per OUT_LAYERS entry.
    """
    x = img_tensor.unsqueeze(1)   # (1, 1, 3, H, W)
    outputs, _ = model.get_intermediate_layers(x, n=OUT_LAYERS)
    # each output is (patch_tokens, camera_token); patch_tokens: (B, S, N, D or 2D)
    return [patch.squeeze(0).squeeze(0).cpu().float() for patch, _ in outputs]


def pca_vis(tokens):
    """tokens: (N, D) → (H_p, W_p, 3) RGB"""
    n = tokens.shape[0]
    h = w = int(n ** 0.5)
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(tokens.numpy())
    for i in range(3):
        rgb[:, i] = (rgb[:, i] - rgb[:, i].min()) / (rgb[:, i].max() - rgb[:, i].min() + 1e-8)
    return rgb.reshape(h, w, 3)


def main():
    img = Image.open(IMG_PATH).convert('RGB')
    img_tensor_518 = get_transform(IMG_SIZE)(img).unsqueeze(0).cuda()
    img_tensor_336 = get_transform(GASTRO_SIZE)(img).unsqueeze(0).cuda()

    print("=== DA3-BASE backbone (518px) ===")
    da3 = load_da3()
    print("=== DA3 + GastroNet backbone (336px) ===")
    gastro = load_endo_da3()

    print("\nRunning forward passes...")
    da3_feats    = run(da3,    img_tensor_518)
    gastro_feats = run(gastro, img_tensor_336)

    # ── PCA visualisation ──────────────────────────────────────────────────
    # Note: DA3 runs at 518px (37×37 patches), GastroNet-DA3 at 336px (24×24),
    # so a direct diff map is not meaningful — just show the PCA features side by side.
    n_layers = len(OUT_LAYERS)
    fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 7))

    for col, (layer, d, g) in enumerate(zip(OUT_LAYERS, da3_feats, gastro_feats)):
        axes[0, col].imshow(pca_vis(d))
        axes[0, col].set_title(f'DA3 518px — layer {layer}')
        axes[0, col].axis('off')

        axes[1, col].imshow(pca_vis(g))
        axes[1, col].set_title(f'Endo-DA3 336px — layer {layer}')
        axes[1, col].axis('off')

    fig.suptitle('DA3-BASE (518px) vs Endo-DA3 (336px) — PCA features (endoscopy)', fontsize=13)
    plt.tight_layout()
    out_path = '/home/in4218/code/endo-da3/endo_da3_comparison_endo.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
