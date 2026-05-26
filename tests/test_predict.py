
import numpy as np

from waferlens.config import load_config
from waferlens.models.factory import build_model
from waferlens.predict import predict_map


def test_predict_map_outputs(tmp_path):
    cfg = load_config("configs/config.yaml")
    classes = cfg.data.base_classes
    model = build_model("cnn", 8, cfg.model, 52)
    model.eval()
    wafer = np.random.randint(0, 3, size=(52, 52)).astype(np.uint8)
    pred = predict_map(model, classes, wafer, threshold=0.5, with_saliency=True)
    assert set(pred.probabilities.keys()) == set(classes)
    assert all(0.0 <= v <= 1.0 for v in pred.probabilities.values())
    assert isinstance(pred.detected, list)
    assert "summary" in pred.triage
