import torch

from waferlens.config import load_config
from waferlens.models.factory import build_model, count_params


def test_cnn_forward():
    cfg = load_config("configs/config.yaml")
    m = build_model("cnn", 8, cfg.model, 52)
    out = m(torch.randn(2, 3, 52, 52))
    assert out.shape == (2, 8)
    assert count_params(m) > 0


def test_vit_forward():
    cfg = load_config("configs/config.yaml")
    m = build_model("vit", 8, cfg.model, 52)
    out = m(torch.randn(2, 3, 52, 52))
    assert out.shape == (2, 8)


def test_vit_handles_non_divisible_size():
    cfg = load_config("configs/config.yaml")
    # 52 is not divisible by patch 4 -> model pads internally; 50 also fine
    m = build_model("vit", 8, cfg.model, 50)
    out = m(torch.randn(1, 3, 50, 50))
    assert out.shape == (1, 8)


def test_unknown_model_raises():
    import pytest
    cfg = load_config("configs/config.yaml")
    with pytest.raises(ValueError):
        build_model("transformer9000", 8, cfg.model, 52)
