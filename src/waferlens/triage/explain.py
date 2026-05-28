"""Explainability: Grad-CAM for the CNN, attention rollout for the ViT.

Produces a per-die saliency map over the wafer that indicates which regions
drove a given class prediction - the visual evidence behind the triage.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from waferlens.models.cnn import WaferCNN
from waferlens.models.vit import WaferViT


def gradcam(model: WaferCNN, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Grad-CAM saliency for a CNN. x is (1,3,H,W). Returns (H,W) in [0,1]."""
    model.eval()
    x = x.clone().requires_grad_(True)
    logits = model(x)
    model.zero_grad()
    logits[0, class_idx].backward()

    acts = model.last_activations          # (1, C, h, w)
    grads = model.last_gradients           # (1, C, h, w)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


def attention_rollout(model: WaferViT, x: torch.Tensor) -> np.ndarray:
    """Attention rollout for the ViT. x is (1,3,H,W). Returns (H,W) in [0,1].

    Implements Abnar & Zuidema (2020): multiply per-layer attention (averaged over
    heads, with residual) and read the CLS->patch attention, reshaped to the grid.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)
    attns = model.attention_maps()         # list of (1, heads, N, N)
    if not attns:
        return np.zeros(x.shape[2:])
    # Keep the rollout on the model's device (MPS/CUDA/CPU). Building torch.eye on
    # CPU and matmul-ing against on-device attention raises a device-mismatch error.
    device = attns[0].device
    result = torch.eye(attns[0].size(-1), device=device)
    for a in attns:
        a = a.mean(dim=1).squeeze(0)        # (N, N) head-averaged
        a = a + torch.eye(a.size(-1), device=device)   # add residual
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    # CLS row, drop the CLS->CLS entry, take CLS->patches; move to CPU for numpy
    cls_to_patches = result[0, 1:].cpu().numpy()
    g = int(np.sqrt(len(cls_to_patches)))
    grid = cls_to_patches[: g * g].reshape(g, g)
    # upsample to image size
    grid_t = torch.from_numpy(grid)[None, None].float()
    up = F.interpolate(grid_t, size=tuple(x.shape[2:]), mode="bilinear", align_corners=False)
    cam = up.squeeze().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


def saliency(model, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Dispatch to the right explainer based on model type."""
    if isinstance(model, WaferCNN):
        return gradcam(model, x, class_idx)
    if isinstance(model, WaferViT):
        return attention_rollout(model, x)
    raise TypeError(f"No explainer for model type {type(model).__name__}")
