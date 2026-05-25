"""Typer CLI for WaferLens-Sherlock."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from waferlens.config import load_config

app = typer.Typer(help="WaferLens-Sherlock: wafer-map defect classification + FA triage.")
CONFIG_PATH = Path("configs/config.yaml")


def _load_dataset(cfg):
    """Resolve the configured dataset to (maps, labels, classes)."""
    from waferlens.data.mixedwm38 import load_npz

    ds = cfg.data.dataset
    if ds == "synthetic":
        d = load_npz(cfg.data.synthetic_path)
        return d.X, d.Y, d.classes
    if ds == "mixedwm38":
        d = load_npz(cfg.data.mixedwm38_path)
        return d.X, d.Y, d.classes
    if ds == "wm811k":
        from waferlens.data.wm811k import load_pickle
        X, y, classes = load_pickle(cfg.data.wm811k_path, cfg.data.map_size)
        return X, y, classes
    raise ValueError(f"Unknown dataset: {ds}")


@app.command("make-synthetic")
def make_synthetic(n: int = 2000, size: int = 52, out: str = "data/synthetic_sample.npz"):
    """Generate a synthetic MixedWM38-style sample so the pipeline runs with no downloads."""
    from waferlens.data.synthetic import save_npz
    save_npz(out, n=n, size=size, seed=42)
    typer.echo(f"Wrote synthetic sample ({n} maps) -> {out}")


@app.command()
def download(dataset: str = "mixedwm38", out_dir: str = "data"):
    """Download a real dataset via Kaggle (prints manual instructions on failure)."""
    from waferlens.data.download import INSTRUCTIONS, download_kaggle
    try:
        download_kaggle(dataset, Path(out_dir))
    except Exception as e:
        typer.echo(f"Automated download unavailable: {e}")
        typer.echo(INSTRUCTIONS)


@app.command()
def train(model: str = "cnn", epochs: int | None = None, data: str | None = None,
          device: str | None = None):
    """Train a model (cnn or vit) on the configured dataset."""
    from waferlens.data.dataset import make_splits
    from waferlens.models.factory import build_model, count_params
    from waferlens.train.loop import train_model

    cfg = load_config(CONFIG_PATH)
    if data:
        # override: treat as a synthetic/mixedwm38-format npz
        from waferlens.data.mixedwm38 import load_npz
        d = load_npz(Path(data))
        maps, labels, classes = d.X, d.Y, d.classes
    else:
        maps, labels, classes = _load_dataset(cfg)

    splits = make_splits(maps, labels, classes,
                         cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.seed,
                         cfg.augment.enabled, cfg.augment.rotate90, cfg.augment.flip)
    n_classes = labels.shape[1] if labels.ndim == 2 else int(labels.max()) + 1
    net = build_model(model, n_classes, cfg.model, cfg.data.map_size)
    result = train_model(
        net, splits, model_name=model,
        epochs=epochs or cfg.train.epochs, batch_size=cfg.train.batch_size,
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        device_str=device or cfg.train.device, num_workers=cfg.train.num_workers,
        early_stop_patience=cfg.train.early_stop_patience, threshold=cfg.train.threshold,
        checkpoint_dir=cfg.train.checkpoint_dir, n_params=count_params(net),
    )
    # persist metrics
    Path("artifacts").mkdir(exist_ok=True)
    out = Path("artifacts") / f"metrics_{model}.json"
    out.write_text(json.dumps({
        "model": result.model_name, "n_params": result.n_params, "device": result.device,
        "best_epoch": result.best_epoch, "test_metrics": result.test_metrics,
        "history": result.history,
    }, indent=2))
    typer.echo(f"Trained {model}: macro-F1={result.test_metrics['macro_f1']:.3f} "
               f"exact-match={result.test_metrics['exact_match']:.3f} -> {out}")


@app.command()
def compare():
    """Print a side-by-side comparison of trained CNN vs ViT."""
    rows = []
    for model in ["cnn", "vit"]:
        p = Path("artifacts") / f"metrics_{model}.json"
        if p.exists():
            m = json.loads(p.read_text())
            tm = m["test_metrics"]
            rows.append((model, m["n_params"], tm["macro_f1"], tm["micro_f1"],
                         tm["exact_match"], tm["mAP"], tm["hamming_loss"]))
    if not rows:
        typer.echo("No metrics found. Train models first (make train-cnn / make train-vit).")
        raise typer.Exit(1)
    from tabulate import tabulate
    typer.echo(tabulate(
        rows,
        headers=["model", "params", "macro-F1", "micro-F1", "exact-match", "mAP", "hamming"],
        floatfmt=".4f",
    ))


@app.command()
def triage(sample: int = 0, model: str = "cnn", data: str | None = None):
    """Run inference + FA triage on one wafer map and write an HTML report."""
    from waferlens.data.mixedwm38 import load_npz
    from waferlens.predict import load_checkpoint, predict_map
    from waferlens.reports.triage_report import render

    cfg = load_config(CONFIG_PATH)
    src = data or (cfg.data.synthetic_path if cfg.data.dataset == "synthetic"
                   else cfg.data.mixedwm38_path)
    d = load_npz(Path(src))
    wafer_map = d.X[sample]

    ckpt = Path(cfg.train.checkpoint_dir) / f"{model}_best.pt"
    net, classes, threshold = load_checkpoint(ckpt, cfg.model, cfg.data.map_size)
    pred = predict_map(net, classes, wafer_map, threshold=threshold, with_saliency=True)
    path = render(pred, wafer_map, cfg.reports.template_dir, cfg.reports.report_dir,
                  sample_id=f"{model}_{sample}")
    typer.echo(f"Detected: {pred.detected or '[none]'}")
    typer.echo(f"Triage report -> {path}")


if __name__ == "__main__":
    app()
