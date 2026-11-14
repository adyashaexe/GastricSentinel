"""Model definitions.

FusionNet is trained with *modality dropout*: each tabular modality is randomly
hidden during training and the head is told which modalities are present
(`mask`). That makes "fallback when a modality is missing" a trained behaviour
rather than a hand-written confidence threshold.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_image_model(num_classes: int, pretrained: bool = True, trainable: str = "layer4"):
    m = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    for p in m.parameters():
        p.requires_grad = False
    if trainable == "all":
        for p in m.parameters():
            p.requires_grad = True
    elif trainable == "layer4":
        for p in m.layer4.parameters():
            p.requires_grad = True
    elif trainable != "fc":
        raise ValueError("trainable must be one of: fc, layer4, all")
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # new head is always trainable
    return m


def load_image_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    m = build_image_model(len(ck["classes"]), pretrained=False, trainable="fc")
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck


class Backbone(nn.Module):
    """ResNet-50 up to the global-average-pool -> (B, 2048)."""

    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.body = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.body(x).flatten(1)


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FusionNet(nn.Module):
    """Concatenation fusion of image features and (optional) clinical / genomic embeddings."""

    def __init__(self, num_classes: int, clin_dim: int, gen_dim: int,
                 img_dim: int = 2048, emb: int = 32, hidden: int = 128, p: float = 0.3):
        super().__init__()
        self.clin_dim, self.gen_dim = clin_dim, gen_dim
        self.img_proj = nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, hidden), nn.ReLU())
        self.clin_enc = TabularEncoder(clin_dim, emb, 32, p)
        self.gen_enc = TabularEncoder(gen_dim, emb, 64, p) if gen_dim > 0 else None
        self.head = nn.Sequential(
            nn.Dropout(p), nn.Linear(hidden + 2 * emb + 2, hidden), nn.ReLU(),
            nn.Dropout(p), nn.Linear(hidden, num_classes),
        )
        self.emb = emb

    def forward(self, img, clin, gen, mask):
        """mask: (B, 2) float, [clinical_present, genomic_present]."""
        c = self.clin_enc(clin) * mask[:, 0:1]
        if self.gen_enc is not None:
            g = self.gen_enc(gen) * mask[:, 1:2]
        else:
            g = torch.zeros(img.size(0), self.emb, device=img.device)
        return self.head(torch.cat([self.img_proj(img), c, g, mask], dim=1))
