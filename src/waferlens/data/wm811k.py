"""WM-811K loader (single-label, 9 classes incl. none).

LSWMD.pkl is a pandas DataFrame with a `waferMap` column (variable-size 2D
arrays) and a `failureType` column. Maps are resized to a fixed square edge.
Source: MIR Lab, WM-811K, via Kaggle (qingyi/wm811k-wafer-map).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

# NOTE: This is a SELF-CONTAINED 9-class single-label index space (incl. "none").
# Its ordering (...Random, Scratch, Near-full) intentionally differs from the
# 8-base-type MULTI-LABEL space in synthetic.py / mixedwm38.py BASE_CLASSES
# (...Near-full, Scratch, Random). The two index spaces are independent and must
# NOT be index-mapped onto each other; a class index here does not correspond to
# the same defect type as the same index there.
WM811K_CLASSES = [
    "none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full",
]


def _resize_map(m: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of a categorical wafer map to (size, size)."""
    from PIL import Image
    img = Image.fromarray(m.astype(np.uint8))
    img = img.resize((size, size), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def load_pickle(path: Path, size: int = 52):
    """Load LSWMD.pkl, keep labeled rows, resize maps, return (X, y_int, classes)."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"WM-811K not found at {path}. Download LSWMD.pkl from Kaggle "
            "(qingyi/wm811k-wafer-map) and place it there."
        )
    df = pd.read_pickle(path)
    # failureType is sometimes stored as nested arrays; normalize to str
    def _norm(v):
        if isinstance(v, (list, np.ndarray)):
            return str(v[0][0]) if len(v) and len(np.atleast_1d(v[0])) else ""
        return str(v)

    df = df.copy()
    df["label"] = df["failureType"].apply(_norm).str.strip()
    df = df[df["label"].isin(WM811K_CLASSES)]
    logger.info(f"WM-811K: {len(df)} labeled wafer maps after filtering")

    X = np.stack([_resize_map(m, size) for m in df["waferMap"].values])
    class_to_idx = {c: i for i, c in enumerate(WM811K_CLASSES)}
    y = df["label"].map(class_to_idx).to_numpy(dtype=np.int64)
    return X, y, WM811K_CLASSES
