"""Compact CNN baseline for wafer-map defect classification.

Three conv blocks -> global average pool -> linear head. The point of the
baseline is to set the bar the ViT must beat. Keeps the final conv feature map
accessible for Grad-CAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 3,
                 channels: tuple[int, ...] = (32, 64, 128), dropout: float = 0.3):
        super().__init__()
        blocks = []
        cin = in_ch
        for c in channels:
            blocks.append(ConvBlock(cin, c))
            cin = c
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(cin, n_classes)
        self._last_feature_channels = cin
        # Grad-CAM hook storage
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

    def forward(self, x):
        feats = self.features(x)
        if feats.requires_grad:
            feats.register_hook(self._save_grad)
        self._activations = feats
        pooled = self.gap(feats).flatten(1)
        return self.head(self.dropout(pooled))

    def _save_grad(self, grad):
        self._gradients = grad

    @property
    def last_activations(self):
        return self._activations

    @property
    def last_gradients(self):
        return self._gradients
