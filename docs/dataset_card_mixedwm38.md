# Dataset Card — MixedWM38

## Provenance

- **Authors**: Junliang Wang et al., Institute of Intelligent Manufacturing, Donghua University (2020).
- **Paper**: "Deformable Convolutional Networks for Efficient Mixed-type Wafer Defect Pattern Recognition," *IEEE TSM*, DOI 10.1109/TSM.2020.3020985.
- **Repo**: https://github.com/Junliangwangdhu/WaferMap
- **Format**: `Wafer_Map_Datasets.npz` with `arr_0` (38015, 52, 52) maps and `arr_1` (38015, 8) multi-hot labels.

## Schema

- Map values: `0` = blank (off-wafer), `1` = good die (passed test), `2` = defective die (failed test).
- Labels: 8-dimensional multi-hot over base types [Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random].
- 38 observed "patterns" = 1 normal + 8 single-type + 29 mixed-type combinations of the 8 bases.

## Notes

- Class balance was partly achieved with GAN-generated maps, so frequencies are **not** raw-fab frequencies.
- A documented dataset issue with a small number of mislabeled maps exists; downstream users sometimes apply a correction. WaferLens does not modify the source file.
- Fixed 52×52 size makes it convenient for both CNN and ViT (patch=4 → pad to 56 → 196 patches).

## Why multi-label

Treating the 38 patterns as independent classes throws away the structure that mixed types are *combinations* of bases. Modeling 8 sigmoid outputs:
- handles unseen base combinations,
- gives per-defect-type probabilities the FA triage layer can act on,
- yields an exact-match metric directly comparable to "38-pattern accuracy."

## WM-811K (secondary)

- Single-label, 9 classes (8 defects + none), variable map sizes, ~172K labeled of 811K total.
- WaferLens resizes maps to a fixed square edge (nearest-neighbour) and trains the same models with a single-label head.
