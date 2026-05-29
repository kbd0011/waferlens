"""Assemble a Hugging Face Space build tree for WaferLens and optionally deploy it.

The Space gets the package, its config, the bundled synthetic sample, and a small
synthetic-trained checkpoint baked in for a fast cold start. The checkpoint is
trained here at build time (never committed to the GitHub repo — synthetic-trained
weights are meaningless as a benchmark); the app also trains one on startup if a
Space ever lacks it. Upload is via ``huggingface_hub`` so the .pt/.npz binaries are
handled over HTTP/LFS automatically.

Usage::

    python scripts/deploy_space.py --build
    python scripts/deploy_space.py --deploy --repo-id kbd0011/waferlens
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "space_build"
SPACE_SRC = REPO_ROOT / "huggingface" / "spaces"
CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
SYNTH = REPO_ROOT / "data" / "synthetic_sample.npz"

INCLUDE = [
    "src/waferlens",
    "configs/config.yaml",
    "data/synthetic_sample.npz",
    "artifacts/checkpoints/cnn_best.pt",
    "artifacts/checkpoints/vit_best.pt",
]


def ensure_demo_checkpoints(epochs: int = 3) -> None:
    """Train small synthetic CNN + ViT checkpoints if they aren't already on disk."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from waferlens.config import load_config
    from waferlens.data.dataset import make_splits
    from waferlens.data.mixedwm38 import load_npz
    from waferlens.data.synthetic import save_npz
    from waferlens.models.factory import build_model, count_params
    from waferlens.train.loop import train_model

    cfg = load_config(str(REPO_ROOT / "configs" / "config.yaml"))
    if not SYNTH.exists():
        SYNTH.parent.mkdir(parents=True, exist_ok=True)
        save_npz(str(SYNTH), n=2000, size=cfg.data.map_size, seed=cfg.data.seed)
    d = load_npz(SYNTH)
    for model_name in ("cnn", "vit"):
        if (CKPT_DIR / f"{model_name}_best.pt").exists():
            continue
        splits = make_splits(d.X, d.Y, d.classes, cfg.data.val_fraction, cfg.data.test_fraction,
                             cfg.data.seed, cfg.augment.enabled, cfg.augment.rotate90, cfg.augment.flip)
        n_classes = d.Y.shape[1] if d.Y.ndim == 2 else int(d.Y.max()) + 1
        net = build_model(model_name, n_classes, cfg.model, cfg.data.map_size)
        train_model(net, splits, model_name=model_name, epochs=epochs,
                    batch_size=cfg.train.batch_size, lr=cfg.train.lr,
                    weight_decay=cfg.train.weight_decay, device_str="cpu", num_workers=0,
                    early_stop_patience=cfg.train.early_stop_patience, threshold=cfg.train.threshold,
                    checkpoint_dir=str(CKPT_DIR), n_params=count_params(net))
        print(f"trained baked checkpoint: {model_name}")


def build() -> Path:
    ensure_demo_checkpoints()
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(SPACE_SRC / name, BUILD_DIR / name)

    for rel in INCLUDE:
        src = REPO_ROOT / rel
        dst = BUILD_DIR / rel
        if not src.exists():
            raise FileNotFoundError(f"manifest entry missing: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)

    print(f"built {BUILD_DIR} ({sum(1 for _ in BUILD_DIR.rglob('*') if _.is_file())} files)")
    return BUILD_DIR


def deploy(repo_id: str) -> str:
    from huggingface_hub import HfApi, create_repo

    build()
    create_repo(repo_id, repo_type="space", space_sdk="streamlit", exist_ok=True)
    HfApi().upload_folder(
        folder_path=str(BUILD_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy WaferLens demo",
    )
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"deployed -> {url}")
    return url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--repo-id", default="kbdev0011/waferlens")
    args = ap.parse_args()
    if args.deploy:
        deploy(args.repo_id)
    else:
        build()


if __name__ == "__main__":
    main()
