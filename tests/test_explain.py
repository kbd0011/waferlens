import torch

from waferlens.config import load_config
from waferlens.models.factory import build_model
from waferlens.triage.explain import saliency


def _x():
    return torch.randn(1, 3, 52, 52)


def test_gradcam_shape():
    cfg = load_config("configs/config.yaml")
    m = build_model("cnn", 8, cfg.model, 52)
    s = saliency(m, _x(), class_idx=0)
    assert s.shape == (52, 52)
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_attention_rollout_shape():
    cfg = load_config("configs/config.yaml")
    m = build_model("vit", 8, cfg.model, 52)
    m.eval()
    s = saliency(m, _x(), class_idx=0)
    assert s.shape == (52, 52)
