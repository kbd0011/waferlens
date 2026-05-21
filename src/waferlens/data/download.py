"""Dataset download helpers.

MixedWM38 is distributed via Google Drive from the original Donghua University
repo, and WM-811K via Kaggle - neither supports a plain unauthenticated curl.
This module gives the user a reliable, scripted path on their own machine.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

KAGGLE_DATASETS = {
    "mixedwm38": "co1d7era/mixedtype-wafer-defect-datasets",
    "wm811k": "qingyi/wm811k-wafer-map",
}


def download_kaggle(dataset: str, out_dir: Path) -> None:
    """Download a dataset via the Kaggle API (requires ~/.kaggle/kaggle.json)."""
    if dataset not in KAGGLE_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: {list(KAGGLE_DATASETS)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Run: uv pip install -e '.[kaggle]' "
            "and place your kaggle.json in ~/.kaggle/"
        ) from e
    api = KaggleApi()
    api.authenticate()
    slug = KAGGLE_DATASETS[dataset]
    logger.info(f"Downloading {slug} to {out_dir} (this can take a few minutes)...")
    api.dataset_download_files(slug, path=str(out_dir), unzip=True)
    logger.info("Download complete. Look for the .npz / .pkl in the output directory.")


INSTRUCTIONS = """
Manual download instructions
============================

MixedWM38 (primary, multi-label, 52x52):
  Option A - Kaggle CLI:
      uv pip install -e '.[kaggle]'
      # put kaggle.json in ~/.kaggle/  (chmod 600)
      waferlens download --dataset mixedwm38
      # then move Wafer_Map_Datasets.npz into data/

  Option B - original repo (Google Drive):
      Visit https://github.com/Junliangwangdhu/WaferMap
      Download Wafer_Map_Datasets.npz and place it at data/Wafer_Map_Datasets.npz

WM-811K (secondary, single-label, 9 classes):
      Kaggle: qingyi/wm811k-wafer-map  ->  LSWMD.pkl  ->  data/LSWMD.pkl

No download needed to try the pipeline:
      make smoke-data     # generates data/synthetic_sample.npz
      make demo           # trains CNN + ViT on synthetic data
"""
