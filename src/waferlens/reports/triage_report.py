"""Render an HTML FA triage report for a wafer map."""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger


def _wafer_png(wafer_map: np.ndarray, saliency: np.ndarray | None = None) -> str:
    """Render the wafer map (+optional saliency overlay) to a base64 PNG data URI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if saliency is not None else 1, figsize=(8, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(wafer_map, cmap="viridis", interpolation="nearest")
    axes[0].set_title("Wafer map")
    axes[0].axis("off")
    if saliency is not None:
        axes[1].imshow(wafer_map, cmap="gray", interpolation="nearest")
        axes[1].imshow(saliency, cmap="jet", alpha=0.5, interpolation="bilinear")
        axes[1].set_title("Model saliency")
        axes[1].axis("off")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render(prediction, wafer_map: np.ndarray, template_dir: Path, out_dir: Path,
           sample_id: str = "sample") -> Path:
    """Render the triage report to HTML."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(template_dir),
                      autoescape=select_autoescape(["html", "xml"]))
    tmpl = env.get_template("triage_report.html.j2")
    img = _wafer_png(wafer_map, prediction.saliency)
    ranked = sorted(prediction.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    html = tmpl.render(
        sample_id=sample_id,
        wafer_png=img,
        ranked_probs=ranked,
        triage=prediction.triage,
        detected=prediction.detected,
        generated=datetime.now().isoformat(timespec="seconds"),
    )
    path = out_dir / f"triage_{sample_id}_{datetime.now():%Y%m%d_%H%M%S}.html"
    path.write_text(html)
    logger.info(f"Wrote triage report to {path}")
    return path
