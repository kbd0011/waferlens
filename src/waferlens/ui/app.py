"""Streamlit UI for WaferLens-Sherlock.

Run: streamlit run src/waferlens/ui/app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from waferlens.config import load_config
from waferlens.data.mixedwm38 import load_npz
from waferlens.predict import load_checkpoint, predict_map

st.set_page_config(page_title="WaferLens-Sherlock", layout="wide", page_icon="🔬")
st.title("WaferLens-Sherlock")
st.caption("Wafer-map defect-pattern classification (CNN + ViT) with rule-based FA triage.")
st.info(
    "This demo runs on **synthetic** wafer maps for pipeline verification — the model "
    "and metrics are illustrative, not a benchmark. Real results come from training on "
    "MixedWM38 (see the repo). Triage gives candidate hypotheses, not a diagnosis.",
    icon="🧪",
)

# Quick synthetic training used only as a fallback when no checkpoint is baked into the Space.
QUICK_EPOCHS = 3


@st.cache_resource
def get_cfg():
    return load_config("configs/config.yaml")


def _ensure_synthetic(cfg) -> Path:
    """Make sure a synthetic dataset exists; generate one if the Space lacks it."""
    p = Path(cfg.data.synthetic_path)
    if not p.exists():
        from waferlens.data.synthetic import save_npz
        p.parent.mkdir(parents=True, exist_ok=True)
        save_npz(str(p), n=2000, size=cfg.data.map_size, seed=cfg.data.seed)
    return p


@st.cache_resource(show_spinner=False)
def ensure_checkpoint(model_name: str) -> str:
    """Return the path to a checkpoint, training a quick synthetic one if none is present."""
    cfg = get_cfg()
    ckpt = Path(cfg.train.checkpoint_dir) / f"{model_name}_best.pt"
    if ckpt.exists():
        return str(ckpt)
    with st.spinner(f"No {model_name} checkpoint baked in — training a quick synthetic demo model (one-time)…"):
        from waferlens.data.dataset import make_splits
        from waferlens.data.mixedwm38 import load_npz
        from waferlens.models.factory import build_model, count_params
        from waferlens.train.loop import train_model

        d = load_npz(_ensure_synthetic(cfg))
        splits = make_splits(d.X, d.Y, d.classes, cfg.data.val_fraction, cfg.data.test_fraction,
                             cfg.data.seed, cfg.augment.enabled, cfg.augment.rotate90, cfg.augment.flip)
        n_classes = d.Y.shape[1] if d.Y.ndim == 2 else int(d.Y.max()) + 1
        net = build_model(model_name, n_classes, cfg.model, cfg.data.map_size)
        train_model(net, splits, model_name=model_name, epochs=QUICK_EPOCHS,
                    batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                    weight_decay=cfg.train.weight_decay, device_str="cpu", num_workers=0,
                    early_stop_patience=cfg.train.early_stop_patience, threshold=cfg.train.threshold,
                    checkpoint_dir=cfg.train.checkpoint_dir, n_params=count_params(net))
    return str(ckpt)


cfg = get_cfg()

with st.sidebar:
    st.header("Setup")
    model_name = st.selectbox("Model", ["cnn", "vit"], index=0)
    # Data source: real MixedWM38 if present, else the bundled synthetic sample.
    src = (cfg.data.mixedwm38_path
           if cfg.data.dataset != "synthetic" and Path(cfg.data.mixedwm38_path).exists()
           else _ensure_synthetic(cfg))
    if Path(cfg.train.checkpoint_dir, f"{model_name}_best.pt").exists():
        st.success(f"{model_name} checkpoint loaded")
    else:
        st.warning(f"{model_name}: training a quick synthetic demo model")
    st.caption(f"Data: {src}")


@st.cache_data
def get_data(path):
    d = load_npz(Path(path))
    return d.X, d.Y, d.classes


@st.cache_resource
def get_model(model_name):
    ckpt = ensure_checkpoint(model_name)
    net, classes, threshold = load_checkpoint(Path(ckpt), cfg.model, cfg.data.map_size)
    return net, classes, threshold


X, Y, classes = get_data(str(src))
net, classes, threshold = get_model(model_name)

col_l, col_r = st.columns([1, 1])
with col_l:
    idx = st.number_input("Wafer index", min_value=0, max_value=len(X) - 1, value=0)
    wafer_map = X[int(idx)]
    import plotly.express as px
    fig = px.imshow(wafer_map, color_continuous_scale="viridis",
                    labels={"color": "die state"}, title="Wafer map (0 blank / 1 good / 2 defect)")
    st.plotly_chart(fig, use_container_width=True)

pred = predict_map(net, classes, wafer_map, threshold=threshold, with_saliency=True)

with col_r:
    st.subheader("Predicted defect patterns")
    prob_df = (
        __import__("pandas").DataFrame(
            sorted(pred.probabilities.items(), key=lambda kv: kv[1], reverse=True),
            columns=["defect", "probability"],
        )
    )
    st.dataframe(prob_df, use_container_width=True, hide_index=True)
    if pred.saliency is not None:
        import plotly.graph_objects as go
        fig2 = go.Figure(data=go.Heatmap(z=pred.saliency, colorscale="Jet"))
        fig2.update_layout(title="Model saliency (where the model looked)", height=350)
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("Failure-analysis triage")
if not pred.triage["entries"]:
    st.success(pred.triage["summary"])
else:
    st.write(pred.triage["summary"])
    for e in pred.triage["entries"]:
        with st.expander(f"{e['pattern']}  —  {e['typical_process_area']}", expanded=True):
            st.markdown("**Candidate root causes**")
            for c in e["candidate_causes"]:
                st.markdown(f"- {c}")
            st.markdown("**Recommended checks**")
            for c in e["recommended_checks"]:
                st.markdown(f"- {c}")
    for h in pred.triage["compound_hints"]:
        st.info(f"Compound hint: {h}")

st.caption("Candidate hypotheses for a human FA engineer to confirm — not a final diagnosis.")
