"""Vision Transformer from scratch for wafer-map defect classification.

Patch embedding -> [CLS] token + positional embedding -> transformer encoder ->
[CLS] head. Attention weights from the final block are retained for attention-
rollout explainability. Kept intentionally small (~1-3M params) so it trains on
an Apple Silicon MPS device in minutes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                       # (B, dim, H/p, W/p)
        b, d, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, n_patches, dim)
        return x, (gh, gw)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # Only retain attention maps at inference (for explainability). Storing them
        # during training balloons memory across all blocks for no benefit.
        if not self.training:
            self.attn_weights = attn.detach()
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WaferViT(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 3, img_size: int = 52,
                 patch_size: int = 4, dim: int = 128, depth: int = 6,
                 heads: int = 4, mlp_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # pad image up to a multiple of patch_size
        self.pad = (patch_size - img_size % patch_size) % patch_size
        padded = img_size + self.pad
        n_patches = (padded // patch_size) ** 2
        self.patch_embed = PatchEmbed(in_ch, dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.pad:
            x = nn.functional.pad(x, (0, self.pad, 0, self.pad))
        tokens, _ = self.patch_embed(x)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])              # CLS token

    def attention_maps(self) -> list[torch.Tensor]:
        """Per-block attention weights from the most recent forward pass."""
        return [blk.attn.attn_weights for blk in self.blocks if blk.attn.attn_weights is not None]
