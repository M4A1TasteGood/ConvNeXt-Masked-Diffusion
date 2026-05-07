"""Checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def require_file(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser()
    if not str(path):
        raise ValueError(f"{label} is required")
    if not value.is_file():
        raise FileNotFoundError(f"{label} not found: {value}")
    return value


def load_torch_file(path: Path, *, weights_only: bool | None = None):
    kwargs = {"map_location": "cpu"}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(str(path), **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(str(path), **kwargs)


def clean_state_dict(state) -> dict:
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return {
        k.removeprefix("module.").removeprefix("model."): v
        for k, v in state.items()
    }


def load_cmd_state_dict(checkpoint: Path) -> dict:
    return clean_state_dict(load_torch_file(checkpoint, weights_only=True))


def load_head_state_dict(checkpoint: Path) -> dict:
    return clean_state_dict(load_torch_file(checkpoint, weights_only=False))
