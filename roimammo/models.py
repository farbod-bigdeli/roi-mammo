"""Backbones and heads.

Backbones (timm, ImageNet-pretrained):
    effb0      efficientnet_b0.ra_in1k
    resnet50   resnet50.a1_in1k
    convnext_t convnext_tiny.fb_in1k
Heads:
    gap        global average pooling + linear (standard)
    attnmil    gated-attention MIL pooling over the feature-map locations (Ilse et al., ICML 2018):
               each spatial location is an instance; attention weights pick the informative ones
Two-view (unit='breast'): one shared backbone for CC and MLO, pooled features concatenated,
then a linear layer (learned fusion instead of averaging predictions).
"""
from __future__ import annotations

import timm
import torch
from torch import nn

BACKBONES = {"effb0": "efficientnet_b0.ra_in1k", "resnet50": "resnet50.a1_in1k",
             "convnext_t": "convnext_tiny.fb_in1k"}


class GatedAttentionPool(nn.Module):
    def __init__(self, c: int, hidden: int = 128):
        super().__init__()
        self.v = nn.Linear(c, hidden)
        self.u = nn.Linear(c, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, f):  # f: (B, C, h, w)
        x = f.flatten(2).transpose(1, 2)  # (B, N, C)
        a = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))  # (B, N, 1)
        a = torch.softmax(a, dim=1)
        return (a * x).sum(1), a.squeeze(-1)


class Classifier(nn.Module):
    def __init__(self, backbone="effb0", head="gap", twoview=False, pretrained=True, drop=0.2):
        super().__init__()
        self.backbone = timm.create_model(BACKBONES.get(backbone, backbone), pretrained=pretrained,
                                          num_classes=0, global_pool="")
        c = self.backbone.num_features
        self.head_type, self.twoview = head, twoview
        self.pool = GatedAttentionPool(c) if head == "attnmil" else None
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * c if twoview else c, 1)

    def embed(self, x):
        f = self.backbone(x)
        if self.pool is not None:
            return self.pool(f)[0]
        return f.mean((2, 3))

    def forward(self, x):
        if self.twoview:  # x: (B, 2, 3, H, W)
            b = x.shape[0]
            z = self.embed(x.flatten(0, 1)).view(b, -1)
        else:
            z = self.embed(x)
        return self.fc(self.drop(z)).squeeze(1)

    def set_grad_checkpointing(self, on=True):
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(on)
