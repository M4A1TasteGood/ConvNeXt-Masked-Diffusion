"""YAML configuration for downstream segmentation-head training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    json: str
    num_workers: int = 0
    random_flip: bool = True


@dataclass
class ModelConfig:
    cmd_checkpoint: str
    pathology_weights: str
    cmd_size: str = "L"
    pathology_model_name: str = "h0-mini"
    disable_pathology: bool = False
    allow_random_cmd: bool = False
    image_size: int = 256
    diffusion_step: int = 50
    patch_size: int = 8
    in_channels: int = 3
    use_all_feature_blocks: bool = False


@dataclass
class TrainConfig:
    output_dir: str = "outputs/downstream"
    batch_size: int = 4
    epochs: int = 150
    lr: float = 2e-4
    weight_decay: float = 1e-4
    val_interval: int = 1
    seed: int = 0
    device: str = "cuda:0"
    grad_clip_norm: float = 1.0
    save_predictions: bool = False


@dataclass
class LossConfig:
    ce_weight: float = 0.5
    dice_weight: float = 0.5
    ignore_index: int = 255


@dataclass
class DownstreamConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def _require_mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"'{name}' must be a mapping in the YAML config.")
    return value


def _build_dataclass(cls, values: dict):
    valid = {item.name for item in cls.__dataclass_fields__.values()}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} option(s): {', '.join(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> DownstreamConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to read downstream YAML configs: pip install pyyaml") from exc

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw = _require_mapping(raw, "config")

    data = _build_dataclass(DataConfig, _require_mapping(raw.get("data"), "data"))
    model = _build_dataclass(ModelConfig, _require_mapping(raw.get("model"), "model"))
    train = _build_dataclass(TrainConfig, _require_mapping(raw.get("train", {}), "train"))
    loss = _build_dataclass(LossConfig, _require_mapping(raw.get("loss", {}), "loss"))
    return DownstreamConfig(data=data, model=model, train=train, loss=loss)
