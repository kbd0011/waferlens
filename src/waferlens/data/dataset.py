"""Torch Dataset wrappers and train/val/test splitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from waferlens.data.transforms import augment_batch, to_onehot_chw


class WaferMapDataset(Dataset):
    """Wraps (maps, labels) as a torch Dataset emitting (3,H,W) float tensors."""

    def __init__(self, maps: np.ndarray, labels: np.ndarray,
                 augment: bool = False, rotate90: bool = True, flip: bool = True,
                 seed: int = 0):
        self.x = to_onehot_chw(maps)              # (N,3,H,W) float32
        self.y = np.asarray(labels, dtype=np.float32)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        self.augment = augment
        self.rotate90 = rotate90
        self.flip = flip
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        xi = self.x[idx]
        if self.augment:
            xi = augment_batch(xi[None], self._rng, self.rotate90, self.flip)[0]
        return torch.from_numpy(np.ascontiguousarray(xi)), torch.from_numpy(self.y[idx])


@dataclass
class Splits:
    train: WaferMapDataset
    val: WaferMapDataset
    test: WaferMapDataset
    classes: list[str]
    pos_weight: np.ndarray | None      # per-class positive weight for BCE


def make_splits(maps: np.ndarray, labels: np.ndarray, classes: list[str],
                val_fraction: float, test_fraction: float, seed: int,
                augment: bool, rotate90: bool, flip: bool) -> Splits:
    """Deterministic random split into train/val/test datasets.

    Group-aware: identical wafer maps (the synthetic generator yields ~21%
    exact duplicates) are kept together so no identical map can straddle the
    train/test boundary (split leakage). All copies of a map are assigned to the
    same split as a unit; every sample is still placed (total is preserved).
    """
    n = len(maps)
    rng = np.random.default_rng(seed)

    # Cluster exact-duplicate maps into groups, then permute and split by group
    # so identical maps never land on both sides of the split.
    maps_arr = np.asarray(maps)
    _, group_of = np.unique(maps_arr.reshape(n, -1), axis=0, return_inverse=True)
    group_of = group_of.reshape(-1)
    n_groups = group_of.max() + 1 if n else 0
    group_perm = rng.permutation(n_groups)
    # Order sample indices by their group's permuted rank; ties (same group)
    # stay contiguous, so a split boundary never cuts through a group.
    sort_key = group_perm[group_of]
    idx = np.argsort(sort_key, kind="stable")

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    # Snap boundaries to group edges so a duplicate group isn't split across sets.
    sorted_keys = sort_key[idx]

    def _snap(boundary: int) -> int:
        if boundary <= 0 or boundary >= n:
            return boundary
        # advance until the group at boundary-1 differs from the group at boundary
        while boundary < n and sorted_keys[boundary] == sorted_keys[boundary - 1]:
            boundary += 1
        return boundary

    n_test = _snap(n_test)
    n_val = _snap(n_test + n_val) - n_test
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    labels = np.asarray(labels, dtype=np.float32)
    multi_label = labels.ndim == 2 and labels.shape[1] > 1

    pos_weight = None
    if multi_label:
        y_train = labels[train_idx]
        pos = y_train.sum(axis=0)
        neg = len(y_train) - pos
        pos_weight = np.where(pos > 0, neg / np.clip(pos, 1, None), 1.0).astype(np.float32)

    return Splits(
        train=WaferMapDataset(maps[train_idx], labels[train_idx], augment, rotate90, flip, seed),
        val=WaferMapDataset(maps[val_idx], labels[val_idx], augment=False, seed=seed),
        test=WaferMapDataset(maps[test_idx], labels[test_idx], augment=False, seed=seed),
        classes=classes,
        pos_weight=pos_weight,
    )
