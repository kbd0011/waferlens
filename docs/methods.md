# Methods

## Problem framing

MixedWM38 has 8 base defect types: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random. A wafer can exhibit several at once (mixed type), so this is a **multi-label** problem: 8 independent binary decisions, not a 38-way single-label one. We train with `BCEWithLogitsLoss` and a per-class `pos_weight = n_neg / n_pos` computed on the training split.

## CNN baseline

Three `Conv-BN-ReLU-MaxPool` blocks (32 → 64 → 128 channels) → global average pool → dropout → linear head of 8 logits. ~95K parameters. The final conv feature map is exposed for Grad-CAM. The baseline exists to make the ViT earn its complexity.

## Vision Transformer (from scratch)

- Patch embedding via a `Conv2d(stride=patch)`; default patch size 4 on a 52 grid (already divisible by 4, so pad=0) → 13×13 = 169 patches.
- Learnable `[CLS]` token + positional embeddings.
- `depth=6` transformer blocks, `dim=128`, `heads=4`, `mlp_dim=256`.
- `[CLS]` representation → linear head of 8 logits. ~0.8M parameters.

Intentionally small so it trains in minutes on an Apple Silicon MPS device.

## Metrics

| Metric | Why |
|---|---|
| Exact-match (subset accuracy) | strict: all 8 labels right — the honest analogue of "38-pattern accuracy" |
| Macro-F1 | treats rare defect types equally |
| Micro-F1 | overall label-level performance |
| mAP | threshold-free ranking quality per class |
| Hamming loss | average per-label error rate |
| Per-class P/R/F1/AP + support | so a weak class can't hide behind a strong macro number |

## Explainability

- **CNN — Grad-CAM** (Selvaraju et al. 2017): gradient-weighted final-conv activations, ReLU, upsampled to the wafer grid.
- **ViT — attention rollout** (Abnar & Zuidema 2020): multiply head-averaged per-layer attention with a residual term, read the `[CLS]→patch` row, reshape to the patch grid, upsample.

Both yield a `(52, 52)` saliency map in `[0, 1]` that overlays the wafer map.

## Augmentation

Random 90° rotations and horizontal/vertical flips. Wafer-map defect classes are (to good approximation) invariant to these, so they're label-preserving and reduce overfitting on the smaller defect classes.

## Splitting

Splitting is **group-aware** on exact-duplicate maps. The synthetic generator emits ~21% identical wafer maps, and a naive random split could place copies of the same map on both sides of the train/test boundary (leakage). `make_splits` clusters identical maps into groups and assigns each group wholesale to a single split, so no identical map straddles the split. Every sample is still placed, so split sizes are approximate (snapped to group boundaries) rather than exact fractions. Synthetic data is for pipeline verification only, not a benchmark.

## Training discipline

- Deterministic seed across split, generator, and init.
- AdamW + cosine LR schedule.
- Early stopping on validation macro-F1.
- Best checkpoint (by val macro-F1) restored before the final test evaluation.
- Optional MLflow tracking (graceful no-op if `mlflow` isn't installed).
