"""Model definitions.

FusionNet is trained with *modality dropout*: each tabular modality is randomly
hidden during training and the head is told which modalities are present
(`mask`). That makes "fallback when a modality is missing" a trained behaviour
rather than a hand-written confidence threshold.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


BACKBONES = ("resnet18", "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny", "mobilenet_v3_large")


class ImageNet(nn.Module):
    """Any supported torchvision backbone behind one interface.

    embed(x)  -> pooled features (B, feat_dim)      forward(x) -> logits
    cam_layer -> the module Grad-CAM hooks          set_trainable(mode) -> fc | last | all
    """

    def __init__(self, arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        if arch not in BACKBONES:
            raise ValueError(f"backbone must be one of: {', '.join(BACKBONES)}")
        self.arch = arch
        w = ("IMAGENET1K_V1" if arch == "resnet50" else "DEFAULT") if pretrained else None
        m = getattr(models, arch)(weights=w)
        self.norm = None
        if arch.startswith("resnet"):
            dim, m.fc = m.fc.in_features, nn.Identity()
        elif arch == "densenet121":
            dim, m.classifier = m.classifier.in_features, nn.Identity()
        elif arch == "efficientnet_b0":
            dim, m.classifier = m.classifier[1].in_features, nn.Identity()
        elif arch == "convnext_tiny":
            dim, self.norm, m.classifier = m.classifier[2].in_features, m.classifier[0], nn.Identity()
        else:  # mobilenet_v3_large
            dim, m.classifier = m.classifier[0].in_features, nn.Identity()
        self.m, self.feat_dim = m, dim
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(dim, num_classes)  # always trainable

    @property
    def cam_layer(self):
        return self.m.layer4 if self.arch.startswith("resnet") else self.m.features

    def feature_map(self, x):
        m = self.m
        if self.arch.startswith("resnet"):
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return m.layer4(m.layer3(m.layer2(m.layer1(x))))
        f = m.features(x)
        return F.relu(f) if self.arch == "densenet121" else f

    def embed(self, x):
        f = self.feature_map(x)
        if self.norm is not None:
            f = self.norm(f)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))

    def _last_stage(self):
        m, a = self.m, self.arch
        if a.startswith("resnet"):
            return [m.layer4]
        if a == "densenet121":
            return [m.features.denseblock4, m.features.norm5]
        if a == "convnext_tiny":
            return list(m.features)[-2:] + [self.norm]
        return list(m.features)[-3:] if a == "mobilenet_v3_large" else list(m.features)[-2:]

    def set_trainable(self, mode: str):
        if mode == "layer4":  # old name
            mode = "last"
        if mode not in ("fc", "last", "all"):
            raise ValueError("trainable must be one of: fc, last, all")
        for p in self.parameters():
            p.requires_grad = mode == "all"
        if mode == "last":
            for mod in self._last_stage():
                for p in mod.parameters():
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True
        return self

    def param_groups(self, lr: float, head_mult: float = 10.0):
        head = list(self.head.parameters())
        ids = {id(p) for p in head}
        body = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        groups = [{"params": head, "lr": lr * head_mult}]
        if body:
            groups.append({"params": body, "lr": lr})
        return groups


def build_image_model(num_classes: int, pretrained: bool = True, trainable: str = "last",
                      arch: str = "resnet50", dropout: float = 0.0) -> ImageNet:
    return ImageNet(arch, num_classes, pretrained, dropout).set_trainable(trainable)


def load_image_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    arch = ck.get("arch", "resnet50")
    m = ImageNet(arch, len(ck["classes"]), pretrained=False)
    state = ck["state_dict"]
    if ck.get("format", 1) < 2:  # checkpoints from before multi-backbone support (plain torchvision ResNet-50)
        state = {("head." + k[3:] if k.startswith("fc.") else "m." + k): v for k, v in state.items()}
    m.load_state_dict(state)
    return m.to(device).eval(), ck


class Backbone(nn.Module):
    """Pooled features of an ImageNet model -> (B, feat_dim)."""

    def __init__(self, net: ImageNet):
        super().__init__()
        self.net, self.dim = net, net.feat_dim

    def forward(self, x):
        return self.net.embed(x)


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
