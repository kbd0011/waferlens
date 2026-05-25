"""Inference: load a checkpoint, predict defect patterns, attach triage + saliency."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from waferlens.config import ModelCfg
from waferlens.data.transforms import to_onehot_chw
from waferlens.models.factory import build_model
from waferlens.triage.explain import saliency
from waferlens.triage.rules import triage


@dataclass
class Prediction:
    probabilities: dict[str, float]
    detected: list[str]
    triage: dict
    saliency: np.ndarray | None


def load_checkpoint(path: Path, model_cfg: ModelCfg, img_size: int = 52):
    """Rebuild a model from a checkpoint saved by the training loop."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = build_model(ckpt["model_name"], n_classes=len(classes), cfg=model_cfg, img_size=img_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes, float(ckpt.get("threshold", 0.5))


def predict_map(model, classes: list[str], wafer_map: np.ndarray, threshold: float = 0.5,
                with_saliency: bool = True) -> Prediction:
    """Predict on a single categorical wafer map (H,W in {0,1,2})."""
    x = torch.from_numpy(to_onehot_chw(wafer_map[None]))
    with torch.no_grad():
        proba = torch.sigmoid(model(x))[0].numpy()
    probs = {c: float(p) for c, p in zip(classes, proba)}
    detected = [c for c, p in probs.items() if p >= threshold]
    sal = None
    if with_saliency and detected:
        top_idx = int(np.argmax(proba))
        sal = saliency(model, x, top_idx)
    return Prediction(probabilities=probs, detected=detected,
                      triage=triage(detected), saliency=sal)
