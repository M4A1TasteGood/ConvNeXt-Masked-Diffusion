"""Image loading and prediction saving for dense-prediction inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

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
    overlay_alpha: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    mask = probs.argmax(axis=0).astype(np.uint8)

    Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / f"{stem}_mask_vis.png")

    mask_full = Image.fromarray(mask).resize(original.size, resample=Image.NEAREST)
    make_overlay(original, np.asarray(mask_full), overlay_alpha).save(output_dir / f"{stem}_overlay.png")
