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


@st.cache_resource
def get_cfg():
    return load_config("configs/config.yaml")


cfg = get_cfg()

with st.sidebar:
    st.header("Setup")
    model_name = st.selectbox("Model", ["cnn", "vit"], index=0)
    ckpt_path = Path(cfg.train.checkpoint_dir) / f"{model_name}_best.pt"
    if ckpt_path.exists():
        st.success(f"{model_name} checkpoint found")
    else:
        st.error(f"No {model_name} checkpoint. Run `make demo` or `make train-{model_name}`.")
    # Data source
    src = (cfg.data.synthetic_path if cfg.data.dataset == "synthetic"
           else cfg.data.mixedwm38_path)
    if not Path(src).exists():
        src = cfg.data.synthetic_path
    st.caption(f"Data: {src}")

if not ckpt_path.exists():
    st.stop()

if not Path(src).exists():
    st.warning("No dataset found. Run `make smoke-data` to generate a synthetic sample.")
    st.stop()


@st.cache_data
def get_data(path):
    d = load_npz(Path(path))
    return d.X, d.Y, d.classes


@st.cache_resource
def get_model(model_name):
    net, classes, threshold = load_checkpoint(
        Path(cfg.train.checkpoint_dir) / f"{model_name}_best.pt", cfg.model, cfg.data.map_size)
    return net, classes, threshold


X, Y, classes = get_data(src)
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
