"""Shared fixtures."""
import pytest

from waferlens.data.synthetic import BASE_CLASSES, generate


@pytest.fixture(scope="session")
def synth():
    """Small synthetic dataset shared across tests."""
    X, Y = generate(n=240, size=52, seed=7)
    return X, Y, BASE_CLASSES


@pytest.fixture
def cfg(tmp_path):
    from waferlens.config import load_config
    return load_config("configs/config.yaml")
