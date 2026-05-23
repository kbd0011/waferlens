"""Training loop with device auto-selection (MPS > CUDA > CPU)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from waferlens.train.metrics import multilabel_metrics


def select_device(requested: str = "auto") -> torch.device:
    """Pick the best available device. 'auto' prefers MPS (Apple Silicon), then CUDA, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainResult:
    model_name: str
    n_params: int
    device: str
    best_epoch: int
    history: list[dict] = field(default_factory=list)
    test_metrics: dict = field(default_factory=dict)
    threshold: float = 0.5
    classes: list[str] = field(default_factory=list)


@torch.no_grad()
def _evaluate(model, loader, device, threshold, classes) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_proba, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        all_proba.append(torch.sigmoid(logits).cpu().numpy())
        all_true.append(yb.numpy())
    proba = np.concatenate(all_proba)
    true = np.concatenate(all_true)
    metrics = multilabel_metrics(true, proba, threshold, classes)
    return metrics, proba, true


def train_model(model: nn.Module, splits, *, model_name: str, epochs: int, batch_size: int,
                lr: float, weight_decay: float, device_str: str, num_workers: int,
                early_stop_patience: int, threshold: float,
                checkpoint_dir: Path, n_params: int) -> TrainResult:
    """Train a model with BCEWithLogitsLoss (multi-label), early stopping on val macro-F1."""
    device = select_device(device_str)
    logger.info(f"Training {model_name} on {device} ({n_params:,} params)")
    model = model.to(device)

    train_loader = DataLoader(splits.train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(splits.val, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(splits.test, batch_size=batch_size, num_workers=num_workers)

    pos_weight = None
    if splits.pos_weight is not None:
        pos_weight = torch.tensor(splits.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{model_name}_best.pt"

    best_f1 = -1.0
    best_epoch = 0
    patience = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
            running += loss.item() * len(xb)
        sched.step()
        train_loss = running / len(splits.train)
        val_metrics, _, _ = _evaluate(model, val_loader, device, threshold, splits.classes)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_macro_f1": val_metrics["macro_f1"],
                        "val_exact_match": val_metrics["exact_match"]})
        logger.info(f"  epoch {epoch:3d}  loss={train_loss:.4f}  "
                    f"val_macroF1={val_metrics['macro_f1']:.4f}  "
                    f"val_exact={val_metrics['exact_match']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience = 0
            torch.save({"model_state": model.state_dict(),
                        "model_name": model_name,
                        "classes": splits.classes,
                        "threshold": threshold}, ckpt_path)
        else:
            patience += 1
            if patience >= early_stop_patience:
                logger.info(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # Load best and evaluate on test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics, _, _ = _evaluate(model, test_loader, device, threshold, splits.classes)
    logger.info(f"  TEST {model_name}: macroF1={test_metrics['macro_f1']:.4f} "
                f"exact={test_metrics['exact_match']:.4f} mAP={test_metrics['mAP']:.4f}")

    return TrainResult(model_name=model_name, n_params=n_params, device=str(device),
                       best_epoch=best_epoch, history=history, test_metrics=test_metrics,
                       threshold=threshold, classes=splits.classes)
