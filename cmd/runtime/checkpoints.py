"""Checkpoint helpers shared by pretraining export, downstream training, and inference."""

from __future__ import annotations

from pathlib import Path

import torch


def _torch_load(path: Path):
    """Load a checkpoint, falling back when ``weights_only=True`` rejects the file."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def require_file(path: str | Path, option_name: str = "path") -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{option_name} not found: {resolved}")
    return resolved


def require_path(path: str | Path, option_name: str = "path") -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{option_name} not found: {resolved}")
    return resolved


def resolve_cmd_checkpoint_path(checkpoint: str | Path) -> Path:
    """Accept a weight file or a directory that contains ``pytorch_model.bin``."""
    resolved = require_path(checkpoint, "CMD checkpoint")
    if resolved.is_dir():
        candidate = resolved / "pytorch_model.bin"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            "CMD checkpoint directory must contain pytorch_model.bin: "
            f"{resolved}"
        )
    return require_file(resolved, "CMD checkpoint")


def _unwrap_state_dict(state: dict, keys: tuple[str, ...]) -> dict:
    for key in keys:
        value = state.get(key)
        if isinstance(value, dict):
            return value
    return state


def load_cmd_state_dict(checkpoint: str | Path) -> dict:
    """Load CMD weights from an EMA file, a training checkpoint, or ``pytorch_model.bin``.

    Training checkpoints store both ``model`` and ``ema``. The EMA weights are
    preferred because downstream feature extraction uses the EMA network.
    """
    state = _torch_load(resolve_cmd_checkpoint_path(checkpoint))
    if isinstance(state, dict):
        state = _unwrap_state_dict(state, ("ema", "model", "state_dict", "module"))
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint}")
    return {k.removeprefix("module.").removeprefix("model."): v for k, v in state.items()}


def load_head_state_dict(checkpoint: str | Path) -> dict:
    state = _torch_load(require_file(checkpoint, "segmentation head checkpoint"))
    if isinstance(state, dict):
        state = _unwrap_state_dict(state, ("model_state_dict", "state_dict", "model", "module"))
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported segmentation head checkpoint format: {checkpoint}")
    return {k.removeprefix("module.").removeprefix("model."): v for k, v in state.items()}
