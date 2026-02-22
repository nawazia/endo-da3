"""
Compare single-view depth from DA3-BASE vs Endo-DA3 on a kvasir endoscopy image.

DA3-BASE  : EndoDA3.from_pretrained(img_size=518)       — original DA3 weights
Endo-DA3  : EndoDA3.from_pretrained(img_size=336) + GastroNet backbone swap

Output: depth map, depth confidence, and log-depth side by side.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

IMG_PATH = '/home/in4218/code/data/kvasir-dataset-v2/normal-z-line/0be91a4e-be3d-4c06-92d7-6e0ee417f55a.jpg'
GASTRONET_CKPT = '/home/in4218/code/GastroNet/gastronet/dinov2.pth'
OUT_PATH = '/home/in4218/code/endo-da3/depth_comparison.png'


def get_transform(size):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_gastronet_sd():
    ckpt = torch.load(GASTRONET_CKPT, map_location='cpu')
    return {k.replace('backbone.', ''): v
            for k, v in ckpt['teacher'].items()
            if k.startswith('backbone.')}


@torch.no_grad()
def run(model, img_tensor):
    """img_tensor: (1, 3, H, W) → wrap to (1, 1, 3, H, W) for single-view."""
    x = img_tensor.unsqueeze(1)
    out = model(x)
    # squeeze B and S dims → (H, W)
    return {k: v.squeeze().cpu().float() for k, v in out.items()
            if v.ndim >= 3}   # skip scalar outputs if any


def depth_to_rgb(depth, vmin=None, vmax=None, cmap='magma_r'):
    """Normalise depth to [0,1] and apply colormap → (H, W, 3) uint8."""
    d = depth.numpy()
    vmin = d.min() if vmin is None else vmin
    vmax = d.max() if vmax is None else vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return (plt.get_cmap(cmap)(norm(d))[..., :3] * 255).astype(np.uint8)


def main():
    from endo_da3 import EndoDA3

    img = Image.open(IMG_PATH).convert('RGB')

    print("Loading DA3-BASE (518px)...")
    da3 = EndoDA3.from_pretrained(img_size=518, with_camera=False, device='cuda')
    t518 = get_transform(518)(img).unsqueeze(0).cuda()
    out_da3 = run(da3, t518)
    del da3

    print("Loading Endo-DA3 (336px) + GastroNet backbone...")
    endo = EndoDA3.from_pretrained(img_size=336, with_camera=False, device='cuda')
    endo.replace_backbone(load_gastronet_sd())
    t336 = get_transform(336)(img).unsqueeze(0).cuda()
    out_endo = run(endo, t336)
    del endo

    depth_da3  = out_da3['depth']
    depth_endo = out_endo['depth']
    conf_da3   = out_da3['depth_conf']
    conf_endo  = out_endo['depth_conf']

    print(f"\nDA3-BASE  depth  — min: {depth_da3.min():.3f}  max: {depth_da3.max():.3f}  mean: {depth_da3.mean():.3f}")
    print(f"Endo-DA3  depth  — min: {depth_endo.min():.3f}  max: {depth_endo.max():.3f}  mean: {depth_endo.mean():.3f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    # Resize the source image to match each model's output for display
    img_518 = img.resize((518, 518), Image.BICUBIC)
    img_336 = img.resize((336, 336), Image.BICUBIC)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    rows = [
        ("DA3-BASE (518px)",  img_518, depth_da3,  conf_da3),
        ("Endo-DA3 (336px)",  img_336, depth_endo, conf_endo),
    ]

    for r, (label, im, depth, conf) in enumerate(rows):
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f'{label}\nInput')

        axes[r, 1].imshow(depth_to_rgb(depth, cmap='magma_r'))
        axes[r, 1].set_title('Depth')

        axes[r, 2].imshow(depth_to_rgb(torch.log1p(depth), cmap='magma_r'))
        axes[r, 2].set_title('Log depth')

        axes[r, 3].imshow(depth_to_rgb(conf, cmap='viridis'))
        axes[r, 3].set_title('Depth confidence')

        for ax in axes[r]:
            ax.axis('off')

    plt.suptitle('Single-view depth: DA3-BASE vs Endo-DA3 (kvasir)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {OUT_PATH}")


if __name__ == '__main__':
    main()
