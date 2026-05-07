"""Frozen H0-mini encoder wrapper for CMD conditioning."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import timm


def _resolve_h0_mini_dir(weights_path: str | Path) -> str:
    override = (os.environ.get("H0_MINI_HUB_DIR") or "").strip()
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    if weights_path:
        candidates.append(Path(weights_path).expanduser().parent)
    for directory in candidates:
        if (directory / "config.json").is_file():
            return str(directory)
    return ""


def _load_pytorch_model_bin(model, weights_path: str | Path) -> None:
    weights_path = Path(weights_path).expanduser()
    if not weights_path.is_file():
        raise FileNotFoundError(f"pathology_model_path not found: {weights_path}")
    try:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if isinstance(state, dict):
        state = {
            k.removeprefix("module.").removeprefix("model."): v
            for k, v in state.items()
        }
    model.load_state_dict(state, strict=True)


def build_h0_mini_encoder(device: torch.device, weights_path: str | Path, dtype=torch.bfloat16):
    local_hub = _resolve_h0_mini_dir(weights_path)
    if local_hub:
        model = timm.create_model(
            f"local-dir:{local_hub}",
            pretrained=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
    else:
        model = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
    _load_pytorch_model_bin(model, weights_path)
    model = model.to(device=device, dtype=dtype).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def extract_h0_mini_features(model, images: torch.Tensor) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    x = (images + 1.0) * 0.5
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    out = model(x.to(dtype=dtype))
    if out.ndim == 3:
        out = out[:, 0]
    return out.to(dtype=torch.float32)
