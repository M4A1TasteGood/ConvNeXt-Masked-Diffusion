"""Image loading and prediction saving."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .config import TNBCInferenceConfig


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser()
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path not found: {input_path}")
    paths = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"no supported images found under: {input_path}")
    return paths


def load_image(image_path: Path, image_size: int, device: torch.device) -> tuple[Image.Image, torch.Tensor]:
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((image_size, image_size), resample=Image.BICUBIC)
    transform = transforms.Compose([transforms.ToTensor(), lambda x: 2 * x - 1])
    image = transform(resized).unsqueeze(0).to(device)
    return original, image


def make_overlay(original: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    image = np.asarray(original.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(image)
    color[mask == 1] = np.array([255, 0, 0], dtype=np.float32)
    overlay = ((1.0 - alpha) * image + alpha * color).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_prediction(
    image_path: Path,
    original: Image.Image,
    logits: torch.Tensor,
    output_dir: Path,
    *,
    save_prob: bool,
    overlay_alpha: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    mask = probs.argmax(axis=0).astype(np.uint8)

    mask_256 = Image.fromarray(mask)
    mask_256_path = output_dir / f"{stem}_mask.png"
    mask_256.save(mask_256_path)

    mask_original = mask_256.resize(original.size, resample=Image.NEAREST)
    mask_original_path = output_dir / f"{stem}_mask_original_size.png"
    mask_original.save(mask_original_path)

    overlay_path = output_dir / f"{stem}_overlay.png"
    make_overlay(original, np.asarray(mask_original), overlay_alpha).save(overlay_path)

    prob_path = None
    if save_prob:
        prob_path = output_dir / f"{stem}_prob.npy"
        np.save(prob_path, probs)

    return {
        "image": image_path.name,
        "mask": mask_256_path.name,
        "mask_original_size": mask_original_path.name,
        "overlay": overlay_path.name,
        "probability": prob_path.name if prob_path else None,
    }


def write_manifest(output_dir: Path, config: TNBCInferenceConfig, records: list[dict]) -> None:
    manifest = {
        "model": "CMD",
        "task": "TNBC semantic segmentation",
        "image_size": config.image_size,
        "diffusion_step": config.diffusion_step,
        "feature_blocks": list(config.feature_blocks),
        "classes": list(config.class_names),
        "cmd_checkpoint_file": config.cmd_checkpoint.name,
        "tnbc_head_checkpoint_file": config.tnbc_head_checkpoint.name,
        "pathology_encoder": config.pathology_model_name,
        "pathology_checkpoint_file": config.pathology_model_path.name,
        "outputs": records,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
