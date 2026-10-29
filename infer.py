#!/usr/bin/env python3
"""Run the released CMD segmentation head on an image or a folder of images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from inference.checkpoints import require_file, require_path
from inference.config import TNBCInferenceConfig
from inference.io import list_images, load_image, save_prediction
from inference.segmentor import build_segmentor
from inference.utils import configure_pathology_encoder, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMD dense-prediction inference.")
    parser.add_argument("--image", required=True, help="Input image path or a directory of images.")
    parser.add_argument(
        "--cmd",
        required=True,
        help="CMD checkpoint file, or a directory containing pytorch_model.bin.",
    )
    parser.add_argument(
        "--seg-head",
        required=True,
        dest="seg_head",
        help="Segmentation head checkpoint, for example best_model.pth.",
    )
    parser.add_argument("--pathology", required=True, help="H0-mini weights (pytorch_model.bin).")
    parser.add_argument("--output-dir", default="outputs/tnbc", help="Directory for prediction images.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--diffusion-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TNBCInferenceConfig(
        cmd_checkpoint=require_path(args.cmd, "--cmd"),
        seg_head_checkpoint=require_file(args.seg_head, "--seg-head"),
        pathology_weights=require_file(args.pathology, "--pathology"),
        image_size=args.image_size,
        diffusion_step=args.diffusion_step,
    )
    configure_pathology_encoder(config.pathology_weights)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print("[CMD] loading models...")
    segmentor = build_segmentor(config, device)
    print("[CMD] ready.")

    image_paths = list_images(Path(args.image))
    output_dir = Path(args.output_dir).expanduser()
    with torch.inference_mode():
        for image_path in image_paths:
            original, image = load_image(image_path, config.image_size, device)
            logits = segmentor(image)
            save_prediction(
                image_path,
                original,
                logits,
                output_dir,
                overlay_alpha=args.overlay_alpha,
            )
            print(f"[CMD] {image_path.name} -> {output_dir}")

    print(f"[CMD] finished {len(image_paths)} image(s) -> {output_dir}")


if __name__ == "__main__":
    main()
