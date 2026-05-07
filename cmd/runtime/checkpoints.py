"""Checkpoint helpers shared by CMD training and inference."""

from __future__ import annotations

from pathlib import Path

import torch


def require_file(path: str | Path, option_name: str = "path") -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{option_name} not found: {resolved}")
    return resolved


def load_cmd_state_dict(checkpoint: str | Path) -> dict:
    state = torch.load(require_file(checkpoint, "CMD checkpoint"), map_location="cpu")
    if isinstance(state, dict):
        for key in ("ema", "model", "state_dict", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint}")
    return {k.removeprefix("module.").removeprefix("model."): v for k, v in state.items()}


def load_head_state_dict(checkpoint: str | Path) -> dict:
    state = torch.load(require_file(checkpoint, "segmentation head checkpoint"), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported segmentation head checkpoint format: {checkpoint}")
    return {k.removeprefix("module.").removeprefix("model."): v for k, v in state.items()}
