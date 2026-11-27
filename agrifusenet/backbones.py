"""Frozen ImageNet / self-supervised backbones, input normalisation and train-time augmentation."""
import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
BACKBONES = ("dinov2_s", "resnet50", "convnext_t", "effnet_b0", "mobilenetv3")
_TIMM = {"effnet_b0": "efficientnet_b0.ra_in1k", "convnext_t": "convnext_tiny.fb_in22k_ft_in1k",
         "mobilenetv3": "mobilenetv3_large_100.ra_in1k"}


def build(name, device="cpu"):
    """Returns (module, forward) where forward maps a normalised batch (B, 3, 224, 224) to features (B, D).
    DINOv2 ViT-S/14: class token concatenated with the mean patch token (768-D)."""
    import timm
    import torchvision
    if name == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        m.fc = torch.nn.Identity()
        m = m.eval().to(device)
        return m, lambda x: m(x.to(device)).cpu()
    if name == "dinov2_s":
        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224).eval().to(device)

        def forward(x):
            t = m.forward_features(x.to(device))
            return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1).cpu()
        return m, forward
    m = timm.create_model(_TIMM[name], pretrained=True, num_classes=0).eval().to(device)
    return m, lambda x: m(x.to(device)).cpu()


def to_tensor(a, size=224):
    im = Image.fromarray(np.ascontiguousarray(a)).resize((size, size), Image.BICUBIC)
    return (torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255 - MEAN) / STD


GEOMETRY = T.Compose([T.RandomRotation(15), T.RandomHorizontalFlip(0.5), T.RandomAffine(0, scale=(0.9, 1.1))])
COLOUR = T.ColorJitter(brightness=0.2, saturation=0.1, hue=0.02)


def augment(a, m, gen):
    """One augmented copy of image a and mask m with identical geometry; gen is a numpy Generator."""
    torch.manual_seed(int(gen.integers(1 << 31)))
    im = T.functional.to_image(np.ascontiguousarray(a))
    mk = T.functional.to_image(m.astype(np.uint8)[..., None] * 255)
    out = GEOMETRY(torch.cat([im, mk], 0))
    rgb = COLOUR(out[:3])
    return rgb.permute(1, 2, 0).numpy(), out[3].numpy() > 127
