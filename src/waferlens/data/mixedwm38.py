"""MixedWM38 loader.

The dataset ships as Wafer_Map_Datasets.npz with:
  arr_0: (38015, 52, 52) uint8  - 0 blank, 1 good die, 2 defective die
  arr_1: (38015, 8) float        - multi-hot over the 8 base defect types
Source: Junliang Wang et al., Donghua University. Multi-label by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

BASE_CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Scratch", "Random"]


@dataclass
class WaferDataset:
    X: np.ndarray            # (n, H, W) uint8 in {0,1,2}
    Y: np.ndarray            # (n, n_classes) float multi-hot
    classes: list[str]


def load_npz(path: Path) -> WaferDataset:
    """Load a MixedWM38-format .npz (also used for the synthetic sample)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `waferlens make-synthetic` for a smoke sample, "
            "or `waferlens download --dataset mixedwm38` for the real data."
        )
    npz = np.load(path)
    keys = list(npz.keys())
    # Accept either canonical (arr_0/arr_1) or named (images/labels) layouts.
    x_key = "arr_0" if "arr_0" in keys else keys[0]
    y_key = "arr_1" if "arr_1" in keys else keys[1]
    X = npz[x_key].astype(np.uint8)
    Y = npz[y_key].astype(np.float32)
    if Y.ndim == 1:
        # single-label integer -> one-hot
        n_classes = int(Y.max()) + 1
        oh = np.zeros((len(Y), n_classes), dtype=np.float32)
        oh[np.arange(len(Y)), Y.astype(int)] = 1.0
        Y = oh
    logger.info(f"Loaded {len(X)} wafer maps {X.shape[1:]} with {Y.shape[1]} label dims from {path.name}")
    return WaferDataset(X=X, Y=Y, classes=BASE_CLASSES[: Y.shape[1]])
