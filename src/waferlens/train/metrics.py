"""Multi-label and single-label metrics for wafer-map classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def multilabel_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5,
                       class_names: list[str] | None = None) -> dict:
    """Compute the multi-label metric set.

    - exact_match: fraction of samples with ALL labels correct (subset accuracy).
      This is the strict analogue of the '38-pattern accuracy' other papers report.
    - macro/micro F1, Hamming loss, mAP, plus per-class P/R/F1/AP.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    out = {
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    # mAP - guard classes with no positives
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            aps.append(average_precision_score(y_true[:, c], y_proba[:, c]))
    out["mAP"] = float(np.mean(aps)) if aps else 0.0

    if class_names:
        per_class = {}
        for c, name in enumerate(class_names):
            support = int(y_true[:, c].sum())
            per_class[name] = {
                "precision": float(precision_score(y_true[:, c], y_pred[:, c], zero_division=0)),
                "recall": float(recall_score(y_true[:, c], y_pred[:, c], zero_division=0)),
                "f1": float(f1_score(y_true[:, c], y_pred[:, c], zero_division=0)),
                "ap": float(average_precision_score(y_true[:, c], y_proba[:, c])) if support > 0 else 0.0,
                "support": support,
            }
        out["per_class"] = per_class
    return out


def singlelabel_metrics(y_true: np.ndarray, logits: np.ndarray,
                        class_names: list[str] | None = None) -> dict:
    """Metrics for single-label (WM-811K) 9-class classification."""
    y_pred = logits.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return out
