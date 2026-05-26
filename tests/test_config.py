from pathlib import Path

import pytest
import yaml

from waferlens.config import WaferLensConfig, load_config


def test_load_config():
    cfg = load_config("configs/config.yaml")
    assert isinstance(cfg, WaferLensConfig)
    assert cfg.data.map_size == 52
    assert len(cfg.data.base_classes) == 8
    assert cfg.model.name in {"cnn", "vit"}


def test_config_rejects_extra(tmp_path):
    raw = yaml.safe_load(Path("configs/config.yaml").read_text())
    raw["nonsense"] = 1
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(Exception):
        load_config(p)
