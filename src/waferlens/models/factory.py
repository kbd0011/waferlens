"""Model factory."""
from __future__ import annotations

import torch.nn as nn

from waferlens.config import ModelCfg
from waferlens.models.cnn import WaferCNN
from waferlens.models.vit import WaferViT


def build_model(name: str, n_classes: int, cfg: ModelCfg, img_size: int = 52) -> nn.Module:
    """Construct a model by name ('cnn' or 'vit')."""
    if name == "cnn":
        return WaferCNN(
            n_classes=n_classes,
            channels=tuple(cfg.cnn.channels),
            dropout=cfg.cnn.dropout,
        )
    if name == "vit":
        return WaferViT(
            n_classes=n_classes,
            img_size=img_size,
            patch_size=cfg.vit.patch_size,
            dim=cfg.vit.dim,
            depth=cfg.vit.depth,
            heads=cfg.vit.heads,
            mlp_dim=cfg.vit.mlp_dim,
            dropout=cfg.vit.dropout,
        )
    raise ValueError(f"Unknown model '{name}'. Options: cnn, vit")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
