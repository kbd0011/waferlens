"""Wafer-map encoding and augmentation."""
from __future__ import annotations

import numpy as np


def to_onehot_chw(maps: np.ndarray) -> np.ndarray:
    """Encode categorical {0,1,2} maps to 3-channel one-hot (N, 3, H, W) float32.

    Channel 0 = blank, channel 1 = good die, channel 2 = defective die.
    Separating the three die states helps the network distinguish the wafer
    boundary from genuine defects.
    """
    maps = np.asarray(maps)
    if maps.ndim == 2:
        maps = maps[None, ...]
    n, h, w = maps.shape
    out = np.zeros((n, 3, h, w), dtype=np.float32)
    for c in range(3):
        out[:, c] = (maps == c).astype(np.float32)
    return out


def pad_or_crop(maps: np.ndarray, size: int) -> np.ndarray:
    """Center pad/crop categorical maps to (N, size, size)."""
    maps = np.asarray(maps)
    if maps.ndim == 2:
        maps = maps[None, ...]
    n, h, w = maps.shape
    out = np.zeros((n, size, size), dtype=maps.dtype)
    for i in range(n):
        m = maps[i]
        # crop if larger
        hs = max(0, (h - size) // 2)
        ws = max(0, (w - size) // 2)
        m = m[hs:hs + min(h, size), ws:ws + min(w, size)]
        ph = (size - m.shape[0]) // 2
        pw = (size - m.shape[1]) // 2
        out[i, ph:ph + m.shape[0], pw:pw + m.shape[1]] = m
    return out


def augment_batch(x: np.ndarray, rng: np.random.Generator,
                  rotate90: bool = True, flip: bool = True) -> np.ndarray:
    """Apply random 90-degree rotations and flips to a (N,3,H,W) batch.

    Wafer-map defect patterns are largely rotation/reflection invariant, so these
    are label-preserving augmentations.
    """
    out = x.copy()
    for i in range(len(out)):
        if rotate90:
            out[i] = np.rot90(out[i], k=int(rng.integers(0, 4)), axes=(1, 2))
        if flip and rng.random() < 0.5:
            out[i] = out[i][:, :, ::-1]
        if flip and rng.random() < 0.5:
            out[i] = out[i][:, ::-1, :]
    return np.ascontiguousarray(out)
