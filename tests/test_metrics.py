import numpy as np

from waferlens.train.metrics import multilabel_metrics


def test_perfect_multilabel():
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    proba = y.astype(float) * 0.9 + 0.05
    m = multilabel_metrics(y, proba, threshold=0.5, class_names=["a", "b", "c"])
    assert m["exact_match"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["hamming_loss"] == 0.0
    assert "per_class" in m


def test_all_wrong_multilabel():
    y = np.array([[1, 0], [0, 1]])
    proba = np.array([[0.1, 0.9], [0.9, 0.1]])
    m = multilabel_metrics(y, proba, threshold=0.5)
    assert m["exact_match"] == 0.0
    assert m["hamming_loss"] == 1.0


def test_metrics_keys_present():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(50, 8))
    proba = rng.random((50, 8))
    m = multilabel_metrics(y, proba, 0.5, class_names=[f"c{i}" for i in range(8)])
    for k in ["exact_match", "hamming_loss", "macro_f1", "micro_f1", "mAP", "per_class"]:
        assert k in m
