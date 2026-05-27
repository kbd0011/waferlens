# Architecture

```
src/waferlens/
├── config.py                 # Pydantic settings (strict, extra='forbid')
├── cli.py                    # Typer CLI: make-synthetic, download, train, compare, triage
├── predict.py                # checkpoint -> inference -> triage + saliency
├── data/
│   ├── synthetic.py          # procedural wafer-map generator (smoke mode)
│   ├── mixedwm38.py          # MixedWM38 .npz loader (multi-label)
│   ├── wm811k.py             # WM-811K .pkl loader (single-label, resized)
│   ├── download.py           # Kaggle helpers + manual instructions
│   ├── transforms.py         # 3-channel one-hot, pad/crop, augmentation
│   └── dataset.py            # torch Dataset + train/val/test splits + pos_weight
├── models/
│   ├── cnn.py                # CNN baseline (Grad-CAM hooks)
│   ├── vit.py                # ViT from scratch (attention retained at inference)
│   └── factory.py            # build_model(name, ...)
├── train/
│   ├── metrics.py            # multi-label + single-label metrics
│   └── loop.py               # training loop, device auto-select (MPS>CUDA>CPU)
├── triage/
│   ├── rules.py              # defect pattern -> candidate root cause table
│   └── explain.py            # Grad-CAM (CNN) + attention rollout (ViT)
├── reports/
│   ├── triage_report.py      # HTML FA report with embedded saliency PNG
│   └── templates/
└── ui/
    └── app.py                # Streamlit dashboard
```

## Data contract

- Wafer maps are categorical `{0: blank, 1: good die, 2: defective die}`.
- One-hot encoded to 3 channels `(3, H, W)` so the boundary is distinguishable from defects.
- MixedWM38 labels: `(N, 8)` multi-hot. WM-811K labels: `(N,)` int over 9 classes.

## Why these choices

- **3-channel one-hot** beats raw `{0,1,2}` because the network never has to learn that "2 > 1 > 0" is categorical, not ordinal.
- **BCE + per-class pos_weight** handles both imbalance and the multi-label structure in one loss.
- **Rotation/flip augmentation** is label-preserving for wafer maps (defect signatures are largely orientation-invariant), so it's free regularization.
- **Device auto-select** prefers MPS so it "just works" fast on Apple Silicon, with a CPU fallback for CI.
- **Attention stored only at eval** keeps training memory flat; the maps are only needed for the explainability overlay.
