"""Runtime utilities for released CMD scripts."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device)


def configure_pathology_encoder(weights_path: str | Path) -> None:
    weights_path = Path(weights_path).expanduser()
    hub_dir = weights_path.parent
    if (hub_dir / "config.json").is_file():
        os.environ.setdefault("H0_MINI_HUB_DIR", str(hub_dir))
