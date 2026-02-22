"""
Test that DA3's alternating local/global attention actually couples views,
comparing DA3-BASE vs Endo-DA3.

Expected behaviour:
  - Layers < alt_start (4): solo and multi features are IDENTICAL — only local
    (per-view) attention, so other views have no influence yet.
  - Layers >= alt_start (4): solo and multi features DIVERGE — odd blocks run
    global cross-view attention, so each view's features are influenced by the other.
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from endo_da3 import load_da3_dino, replace_dino_weights

IMG_A = '/home/in4218/code/Depth-Anything-3/assets/examples/SOH/000.png'
IMG_B = '/home/in4218/code/Depth-Anything-3/assets/examples/SOH/010.png'
ALT_START = 4
# One layer before alt_start, then several after
LAYERS = [3, 5, 7, 9, 11]


def get_transform(size):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_endo_da3():
    model = load_da3_dino(img_size=336, device='cuda')
    ckpt = torch.load('/home/in4218/code/GastroNet/gastronet/dinov2.pth', map_location='cpu')
    gastro_sd = {k.replace('backbone.', ''): v for k, v in ckpt['teacher'].items()
                 if k.startswith('backbone.')}
    replace_dino_weights(model, gastro_sd)
    return model


@torch.no_grad()
def run(model, *imgs):
    """
    imgs: one or more (1, 3, H, W) tensors → stacked as (1, S, 3, H, W).
    Returns dict: layer → (S, N, D) patch token tensor (on CPU).
    """
    x = torch.cat(imgs, dim=0).unsqueeze(0)  # (1, S, 3, H, W)
    outputs, _ = model.get_intermediate_layers(x, n=LAYERS)
    return {layer: patch.squeeze(0).cpu().float()
            for layer, (patch, _) in zip(LAYERS, outputs)}


def pca_vis(tokens):
    """tokens: (N, D) → (H_p, W_p, 3)"""
    n = tokens.shape[0]
    hw = int(n ** 0.5)
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(tokens.numpy())
    for i in range(3):
        rgb[:, i] = (rgb[:, i] - rgb[:, i].min()) / (rgb[:, i].max() - rgb[:, i].min() + 1e-8)
    return rgb.reshape(hw, hw, 3)


def run_and_report(model, t_a, t_b, label):
    solo_a = run(model, t_a)
    solo_b = run(model, t_b)
    multi  = run(model, t_a, t_b)

    print(f"\n{label}")
    print(f"  {'Layer':>5}  {'mean|diff| A':>14}  {'mean|diff| B':>14}  {'Expected':>10}")
    for layer in LAYERS:
        da = (solo_a[layer][0] - multi[layer][0]).abs().mean().item()
        db = (solo_b[layer][0] - multi[layer][1]).abs().mean().item()
        expected = "coupled" if layer >= ALT_START else "identical"
        print(f"  {layer:>5}  {da:>14.6f}  {db:>14.6f}  {expected:>10}")

    return solo_a, solo_b, multi


def plot_model(axes_row_offset, axes, solo_a, solo_b, multi, model_label):
    """Fill 4 rows of axes for one model: A solo, A multi, B solo, B multi."""
    rows = [
        (f'{model_label}\nA solo',       [solo_a[l][0] for l in LAYERS]),
        (f'{model_label}\nA multi-view', [multi[l][0]  for l in LAYERS]),
        (f'{model_label}\nB solo',       [solo_b[l][0] for l in LAYERS]),
        (f'{model_label}\nB multi-view', [multi[l][1]  for l in LAYERS]),
    ]
    for r, (label, feats) in enumerate(rows):
        for c, (layer, f) in enumerate(zip(LAYERS, feats)):
            ax = axes[axes_row_offset + r, c]
            ax.imshow(pca_vis(f))
            if axes_row_offset + r == 0:
                marker = ' ← global\n    starts' if layer == ALT_START else ''
                ax.set_title(f'layer {layer}{marker}', fontsize=8)
            ax.axis('off')
        axes[axes_row_offset + r, 0].set_ylabel(label, fontsize=8,
                                                 rotation=0, labelpad=90, va='center')


def main():
    print("Loading DA3-BASE (518px)...")
    da3 = load_da3_dino(img_size=518, device='cuda')
    print("Loading Endo-DA3 (336px)...")
    endo = load_endo_da3()

    tf_518 = get_transform(518)
    tf_336 = get_transform(336)
    img_a = Image.open(IMG_A).convert('RGB')
    img_b = Image.open(IMG_B).convert('RGB')

    t_a_518 = tf_518(img_a).unsqueeze(0).cuda()
    t_b_518 = tf_518(img_b).unsqueeze(0).cuda()
    t_a_336 = tf_336(img_a).unsqueeze(0).cuda()
    t_b_336 = tf_336(img_b).unsqueeze(0).cuda()

    solo_a_da3, solo_b_da3, multi_da3   = run_and_report(da3,  t_a_518, t_b_518, "DA3-BASE (518px)")
    solo_a_end, solo_b_end, multi_endo  = run_and_report(endo, t_a_336, t_b_336, "Endo-DA3 (336px)")

    # 8 rows total: 4 per model
    n_layers = len(LAYERS)
    fig, axes = plt.subplots(8, n_layers, figsize=(3.2 * n_layers, 18))

    plot_model(0, axes, solo_a_da3, solo_b_da3, multi_da3,  "DA3-BASE")
    plot_model(4, axes, solo_a_end, solo_b_end, multi_endo, "Endo-DA3")

    # Divider line between the two models
    for ax in axes[3, :]:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

    plt.suptitle(f'Alternating attention: solo (S=1) vs multi-view (S=2)\n'
                 f'Global cross-view attention starts at layer {ALT_START}', fontsize=12)
    plt.tight_layout()
    out = '/home/in4218/code/endo-da3/multiview_attention_test.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
