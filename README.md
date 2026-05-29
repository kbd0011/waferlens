# WaferLens-Sherlock

**Wafer-map defect-pattern classification (CNN baseline + Vision Transformer) with a rule-based failure-analysis triage layer.**
Primary dataset: **MixedWM38** (38,015 wafer maps, 8 base defect types, mixed-type by construction). Secondary: WM-811K (9-class).

[![CI](https://github.com/kbd0011/waferlens/actions/workflows/ci.yml/badge.svg)](https://github.com/kbd0011/waferlens/actions)
![python](https://img.shields.io/badge/python-3.12-blue.svg)
![torch](https://img.shields.io/badge/torch-2.x-ee4c2c.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-Hugging%20Face%20Space-ffce1c.svg)](https://huggingface.co/spaces/kbdev0011/waferlens)

▶ **Live demo:** https://huggingface.co/spaces/kbdev0011/waferlens  · synthetic demo (trains on first load); real results train on MixedWM38.

---

## What this is

WaferLens-Sherlock classifies the spatial defect signature on a semiconductor wafer map and then tells a yield engineer *where to look*. The model says **what** the pattern is (Center, Edge-Ring, Scratch, …); the rule-based triage layer maps that to **candidate process root causes** and recommended FA checks. A Grad-CAM / attention-rollout saliency map shows the visual evidence behind each call — the "Sherlock" part.

What sets it apart from a Kaggle notebook:

- **Multi-label done right.** MixedWM38 encodes 8 *base* defect types; the "38 patterns" are just their observed *combinations*. WaferLens models it as 8 independent sigmoid outputs (BCE with per-class positive weighting), so mixed-type wafers are handled natively — no 38-way softmax hack.
- **A baseline before the fancy model.** A compact CNN sets the bar; the ViT has to beat it. `waferlens compare` puts them side by side on macro-F1, exact-match, mAP, Hamming loss, and parameter count. Engineering discipline over leaderboard chasing.
- **An FA triage artifact, not just a prediction.** The defect-to-root-cause rule table is the deliverable a process engineer actually uses.
- **Explainable.** Grad-CAM for the CNN, attention rollout (Abnar & Zuidema 2020) for the ViT.
- **Runs with zero downloads.** A procedural synthetic wafer-map generator lets the entire pipeline — training, triage, UI — run before you ever touch Kaggle. `make demo`.

## Quickstart (macOS, Apple Silicon)

```bash
make install            # uv venv + deps
make demo               # synthetic data -> trains CNN + ViT (3 epochs) -> compares
make ui                 # Streamlit dashboard on http://localhost:8501
```

`make demo` needs no downloads: it generates a synthetic MixedWM38-style dataset, trains both models for a few epochs, and prints the comparison. For real results, get the dataset (below) and run the full training.

## Getting the real data

MixedWM38 ships via Google Drive / Kaggle; WM-811K via Kaggle. Neither supports a plain `curl`, so:

```bash
# Option A — Kaggle CLI (recommended)
uv pip install -e '.[kaggle]'
# put kaggle.json in ~/.kaggle/ (chmod 600), then:
waferlens download --dataset mixedwm38
mv data/Wafer_Map_Datasets.npz data/        # if it landed in a subfolder

# Option B — original repo (Google Drive)
# https://github.com/Junliangwangdhu/WaferMap  ->  Wafer_Map_Datasets.npz  ->  data/

# Then train for real (uses MPS on Apple Silicon automatically):
make train-cnn
make train-vit
make compare
```

## Full training on MixedWM38

```bash
# config.yaml: set data.dataset = mixedwm38, train.epochs = 40
waferlens train --model cnn      # ~minutes on MPS
waferlens train --model vit      # a bit longer; ViT typically wins on mixed types
waferlens compare
waferlens triage --sample 0 --model vit   # writes an HTML FA report
make ui                                     # interactive dashboard
```

Published results on MixedWM38 reach ~98-99% pattern accuracy with a ViT; the CNN baseline is a few points behind at a fraction of the parameters. WaferLens reports **macro-F1, micro-F1, exact-match (subset accuracy), mAP, and Hamming loss** — exact-match is the strict analogue of the "38-pattern accuracy" other papers quote.

## The 8 base defect types → FA triage

| Pattern | Typical process area | Example candidate cause |
|---|---|---|
| Center | CMP / Litho coat / Thermal | CMP dishing, spin-coat center non-uniformity |
| Donut | Litho develop / Etch | develop ring non-uniformity, mid-radius etch band |
| Edge-Loc | Etch / Handling | localized edge clamp damage, azimuthal plasma asymmetry |
| Edge-Ring | Etch / Depo / RTP | focus-ring wear, edge temperature roll-off |
| Loc | Contamination / CMP | particle cluster, CMP micro-scratch |
| Near-full | Process-wide / Test | gross excursion, recipe fault, tester miscalibration |
| Scratch | Handling / CMP | robot end-effector scratch, pad debris drag |
| Random | Cleanroom / Defectivity | random particulate, background defectivity |

Plus compound hints for co-occurring patterns (e.g., Center+Edge-Ring → a single global radial non-uniformity rather than two independent causes).

## Architecture

```
wafer map (52x52, {0,1,2})
   │  one-hot -> (3, 52, 52)
   ▼
┌───────────────┐        ┌───────────────────┐
│ CNN baseline  │   vs   │ ViT (from scratch)│   ← compare on macro-F1 / exact-match / mAP
│ 3 conv blocks │        │ patch=4, depth=6  │
└──────┬────────┘        └─────────┬─────────┘
       │  8 sigmoid (multi-label)  │
       ▼                           ▼
   ┌─────────────────────────────────┐
   │ defect probabilities (8 base)   │
   └──────────────┬──────────────────┘
                  ▼
   ┌──────────────────────────┐   ┌────────────────────────┐
   │ Grad-CAM / attn rollout  │   │ rule-based FA triage    │
   │ (where the model looked) │   │ (where the engineer     │
   └──────────────────────────┘   │  should look)           │
                                   └────────────────────────┘
```

## Honest limitations

- Synthetic mode is for *pipeline verification*, not benchmarking. Real numbers require MixedWM38.
- MixedWM38 is partly GAN-balanced; class frequencies are not raw-fab frequencies.
- The FA triage table encodes documented spatial-signature heuristics — candidate hypotheses for a human to confirm, not a diagnosis.
- This generalizes in principle to advanced-packaging wafer/bump maps (HBM, TSV) but is **not** trained on such data; no HBM/TSV claims are made.

## License & attribution

Code: MIT. MixedWM38: Junliang Wang et al., Donghua University. WM-811K: MIR Lab. See [`docs/references.md`](docs/references.md).
