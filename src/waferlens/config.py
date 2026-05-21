"""Pydantic configuration for WaferLens-Sherlock."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = "mixedwm38"
    mixedwm38_path: Path
    wm811k_path: Path
    synthetic_path: Path
    map_size: int = 52
    val_fraction: float = Field(default=0.15, ge=0, lt=1)
    test_fraction: float = Field(default=0.15, ge=0, lt=1)
    seed: int = 42
    base_classes: list[str]


class AugmentCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    rotate90: bool = True
    flip: bool = True
    jitter_defect_prob: float = Field(default=0.0, ge=0, le=1)


class CNNCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channels: list[int] = [32, 64, 128]
    dropout: float = 0.3


class ViTCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patch_size: int = 4
    dim: int = 128
    depth: int = 6
    heads: int = 4
    mlp_dim: int = 256
    dropout: float = 0.1


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "cnn"
    cnn: CNNCfg
    vit: ViTCfg


class TrainCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    device: str = "auto"
    num_workers: int = 2
    early_stop_patience: int = 8
    threshold: float = 0.5
    amp: bool = False
    checkpoint_dir: Path


class EvalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = 0.5


class TriageCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k_dies: int = 50
    rules_path: Path | None = None


class ReportsCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_dir: Path
    template_dir: Path


class TrackingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    experiment: str = "waferlens"


class WaferLensConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: DataCfg
    augment: AugmentCfg
    model: ModelCfg
    train: TrainCfg
    eval: EvalCfg
    triage: TriageCfg
    reports: ReportsCfg
    tracking: TrackingCfg


def load_config(path: str | Path = "configs/config.yaml") -> WaferLensConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    return WaferLensConfig.model_validate(raw)
