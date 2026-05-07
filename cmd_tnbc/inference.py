"""Command-line TNBC inference pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .checkpoints import require_file
from .config import TNBCInferenceConfig
from .io import list_images, load_image, save_prediction, write_manifest
from .runtime import configure_pathology_encoder, resolve_device, set_seed
from .segmentor import build_segmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TNBC inference with CMD.")
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output-dir", default="outputs/tnbc_inference", help="Directory for outputs.")
    parser.add_argument(
        "--cmd-checkpoint",
        default=os.environ.get("CMD_CHECKPOINT", ""),
        help="CMD pretrained checkpoint, or set CMD_CHECKPOINT.",
    )
    parser.add_argument(
        "--tnbc-head-checkpoint",
        default=os.environ.get("TNBC_HEAD_CHECKPOINT", ""),
        help="TNBC segmentation head checkpoint, or set TNBC_HEAD_CHECKPOINT.",
    )
    parser.add_argument(
        "--pathology-model-path",
        default=os.environ.get("H0_MINI_WEIGHTS", os.environ.get("PATHOLOGY_MODEL_PATH", "")),
        help="H0-mini pytorch_model.bin path, or set H0_MINI_WEIGHTS.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--diffusion-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-prob", action="store_true", help="Save per-class probability maps.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TNBCInferenceConfig:
    return TNBCInferenceConfig(
        cmd_checkpoint=require_file(args.cmd_checkpoint, "--cmd-checkpoint"),
        tnbc_head_checkpoint=require_file(args.tnbc_head_checkpoint, "--tnbc-head-checkpoint"),
        pathology_model_path=require_file(args.pathology_model_path, "--pathology-model-path"),
        image_size=args.image_size,
        diffusion_step=args.diffusion_step,
    )


def run_inference(args: argparse.Namespace) -> None:
    config = build_config(args)
    configure_pathology_encoder(config.pathology_model_path)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print("[CMD] loading CMD and TNBC head...")
    segmentor = build_segmentor(config, device)
    print("[CMD] model is ready.")
    image_paths = list_images(Path(args.input))
    output_dir = Path(args.output_dir).expanduser()

    records = []
    with torch.inference_mode():
        for image_path in image_paths:
            original, image = load_image(image_path, config.image_size, device)
            logits = segmentor(image)
            records.append(
                save_prediction(
                    image_path,
                    original,
                    logits,
                    output_dir,
                    save_prob=args.save_prob,
                    overlay_alpha=args.overlay_alpha,
                )
            )
            print(f"[CMD] saved TNBC prediction for {image_path.name}")

    write_manifest(output_dir, config, records)
    print(f"[CMD] finished {len(records)} image(s); outputs: {output_dir}")


def main() -> None:
    run_inference(parse_args())
