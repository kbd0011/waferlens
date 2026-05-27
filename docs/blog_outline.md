# Blog post outline — "WaferLens-Sherlock: a wafer-map defect classifier that tells you where to look"

## Hook
Screenshot of the Streamlit triage: a wafer map, the predicted defect types, the saliency overlay, and the FA root-cause card. "A classifier predicts. An engineer needs to know where to look next."

## The dataset everyone gets slightly wrong
MixedWM38's 38 "patterns" are combinations of 8 base defects. Most notebooks do 38-way softmax. That's wrong — it's a multi-label problem. Show the 8-sigmoid framing and why it handles mixed types natively.

## Baseline first (the discipline section)
Build a 95K-param CNN. Get a number. *Then* build the ViT. Show `waferlens compare`. Talk about why beating a baseline matters more than a leaderboard score.

## The ViT from scratch
Patch embedding, CLS token, 6 blocks. Kept small so it trains in minutes on an M-series Mac. Attention rollout for explainability.

## The part that makes it engineering, not Kaggle
The FA triage table: Center → CMP dishing, Edge-Ring → focus-ring wear, Scratch → handling. Compound hints. Saliency overlay as visual evidence. This is the artifact a yield engineer uses.

## Honest limitations
GAN-balanced classes, candidate-not-diagnosis triage, no HBM/TSV claims (generalizes in principle).

## What's next
WM-811K at scale; calibrated probabilities; segmentation (WaferSegClassNet-style) to localize, not just classify; tie into an SPC excursion pipeline (see FabSentinel).

## Close
GitHub link, MIT. Map to the Micron roles this supports (FA Engineer, Yield Enhancement, Process Engineer).
