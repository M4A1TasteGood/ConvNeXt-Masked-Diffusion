"""Runtime helpers for command-line inference."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def add_model_code_to_path(project_root: Path) -> None:
    project_root = project_root.expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_pathology_encoder(pathology_model_path: Path) -> None:
    os.environ["PATHOLOGY_MODEL_NAME"] = "h0-mini"
    os.environ["PATHOLOGY_MODEL_PATH"] = str(pathology_model_path)
    os.environ.setdefault("H0_MINI_HUB_DIR", str(pathology_model_path.parent))


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use --device cpu.")
        torch.cuda.set_device(device)
    return device
