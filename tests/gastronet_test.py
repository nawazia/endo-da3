import time
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# IMG_PATH = '/home/in4218/code/data/kvasir-dataset-v2/normal-z-line/0be91a4e-be3d-4c06-92d7-6e0ee417f55a.jpg'
IMG_PATH = '/home/in4218/code/data/SOH/000.png'
NUM_PATCHES = 24  # 336 / 14


def get_transform():
    return transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_pca_features(model, img_tensor):
    """Extract patch features and reduce to 3 PCA components."""
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        patch_tokens = features['x_norm_patchtokens'][0].cpu().numpy()

    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_tokens)

    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                              (pca_features[:, i].max() - pca_features[:, i].min())

    return pca_features.reshape(NUM_PATCHES, NUM_PATCHES, 3)


def load_gastronet_model():
    """Load DINOv2 ViT-B/14 with GastroNet weights."""
    model = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitb14_reg',
        pretrained=False,
        img_size=336,
    )
    ckpt = torch.load('/home/in4218/code/GastroNet/gastronet/dinov2.pth', map_location='cpu')
    backbone_sd = {
        k.replace('backbone.', ''): v
        for k, v in ckpt['teacher'].items()
        if k.startswith('backbone.')
    }
    model.load_state_dict(backbone_sd, strict=False)
    model.eval().cuda()
    return model


def load_vanilla_model():
    """Load vanilla pretrained DINOv2 ViT-B/14 with registers."""
    model = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitb14_reg',
        pretrained=True,
    )
    model.eval().cuda()
    return model


def main():
    img = Image.open(IMG_PATH).convert('RGB')
    img_tensor = get_transform()(img).unsqueeze(0).cuda()

    gastronet_model = load_gastronet_model()
    vanilla_model = load_vanilla_model()

    gastronet_pca = extract_pca_features(gastronet_model, img_tensor)
    vanilla_pca = extract_pca_features(vanilla_model, img_tensor)
    time.sleep(10)  # Ensure all GPU operations are complete

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(vanilla_pca)
    axes[1].set_title('Vanilla DINOv2')
    axes[1].axis('off')

    axes[2].imshow(gastronet_pca)
    axes[2].set_title('GastroNet DINOv2')
    axes[2].axis('off')

    plt.suptitle('DINOv2 ViT-B/14 — PCA Feature Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/in4218/code/endo-da3/pca_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved to /home/in4218/code/endo-da3/pca_comparison.png")


if __name__ == "__main__":
    main()
