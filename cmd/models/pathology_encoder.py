"""Frozen pathology foundation model feature extractors.

This module centralises UNI / H0-mini loading so pretraining and downstream
feature extraction use the same preprocessing, feature dimension, and local
checkpoint handling.
"""

import os

import torch
import torch.nn.functional as F
import timm


DEFAULT_UNI_WEIGHTS_PATH = ""

PATHOLOGY_MODEL_DIMS = {
    "uni": 1024,
    "h0-mini": 768,
}


def normalize_pathology_model_name(model_name):
    name = str(model_name or "uni").strip().lower().replace("_", "-")
    aliases = {
        "h0": "h0-mini",
        "h0mini": "h0-mini",
        "bioptimus/h0-mini": "h0-mini",
    }
    name = aliases.get(name, name)
    if name not in PATHOLOGY_MODEL_DIMS:
        raise ValueError(
            f"Unknown pathology_model_name '{model_name}', choose from "
            f"{list(PATHOLOGY_MODEL_DIMS.keys())}"
        )
    return name


def get_pathology_feat_dim(model_name):
    return PATHOLOGY_MODEL_DIMS[normalize_pathology_model_name(model_name)]


def get_default_pathology_weights_path(model_name):
    name = normalize_pathology_model_name(model_name)
    if name == "uni":
        return DEFAULT_UNI_WEIGHTS_PATH
    return ""


def _resolve_h0_mini_local_hub_dir(weights_path):
    """Directory with timm ``config.json`` for offline ``local-dir:`` loading.

    Checks:
      * env ``H0_MINI_HUB_DIR`` (if set and contains config.json)
      * ``dirname(weights_path)/config.json`` (typical HF snapshot layout)
    """
    override = (os.environ.get("H0_MINI_HUB_DIR") or "").strip()
    candidates = []
    if override:
        candidates.append(os.path.abspath(override))
    if weights_path:
        candidates.append(os.path.dirname(os.path.abspath(weights_path)))
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "config.json")):
            return d
    return ""


def _load_pytorch_model_bin(model, weights_path, strict=True):
    if not weights_path:
        return
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"pathology_model_path not found: {weights_path}")

    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")

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
    model.load_state_dict(state, strict=strict)


def build_pathology_encoder(model_name, device, weights_path="", dtype=torch.bfloat16):
    """Build a frozen pathology encoder and return it in eval mode."""
    name = normalize_pathology_model_name(model_name)
    weights_path = weights_path or get_default_pathology_weights_path(name)

    if name == "uni" and not weights_path:
        raise FileNotFoundError("UNI weights are required. Pass --pathology-model-path or set PATHOLOGY_MODEL_PATH.")

    if name == "uni":
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        _load_pytorch_model_bin(model, weights_path, strict=True)
    elif name == "h0-mini":
        # ``hf-hub:`` always hits the network for config.json; use ``local-dir:`` on air-gapped nodes.
        local_hub = _resolve_h0_mini_local_hub_dir(weights_path)
        if local_hub:
            model = timm.create_model(
                f"local-dir:{local_hub}",
                pretrained=False,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        else:
            try:
                model = timm.create_model(
                    "hf-hub:bioptimus/H0-mini",
                    pretrained=False,
                    mlp_layer=timm.layers.SwiGLUPacked,
                    act_layer=torch.nn.SiLU,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to build H0-mini in the current environment.\n"
                    "For offline use, download the H0-mini files and place config.json "
                    "next to pytorch_model.bin, or set H0_MINI_HUB_DIR to that directory.\n"
                    f"Original error: {exc}"
                ) from exc
        _load_pytorch_model_bin(model, weights_path, strict=True)
    else:
        raise AssertionError(f"unhandled pathology model: {name}")

    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_pathology_features(model, images, model_name):
    """Extract CLS pathology features from images in training range [-1, 1]."""
    name = normalize_pathology_model_name(model_name)
    dtype = next(model.parameters()).dtype

    x = (images + 1.0) * 0.5
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    out = model(x.to(dtype=dtype))
    if name == "h0-mini" and out.ndim == 3:
        out = out[:, 0]
    return out.to(dtype=torch.float32)
