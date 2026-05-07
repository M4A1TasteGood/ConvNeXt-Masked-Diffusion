"""Pretrain CMD from a JSONL image manifest.

Run from the released repository root:

    python -m pretrain.train_cmd --data-jsonl data/train.jsonl

For multi-GPU training, launch the same script with Accelerate:

    accelerate launch --num_processes 4 pretrain/train_cmd.py --data-jsonl data/train.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmd.data import load_pretrain_data
from cmd.models.factory import create_model_and_diffusion
from cmd.models.pathology_encoder import get_pathology_feat_dim, normalize_pathology_model_name
from cmd.training import TrainLoop, find_latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain CMD from a JSONL image manifest.")
    parser.add_argument("--data-jsonl", "--data_jsonl", dest="data_jsonl", required=True)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default="outputs/pretrain_cmd")
    parser.add_argument("--resume-checkpoint", "--resume_checkpoint", dest="resume_checkpoint", default="")
    parser.add_argument("--cmd-size", dest="cmd_size", default="B", choices=["S", "B", "L", "XL"])
    parser.add_argument("--backbone", default="pixel", choices=["pixel", "vae"])
    parser.add_argument("--pathology-model-name", "--pathology_model_name", dest="pathology_model_name", default="uni")
    parser.add_argument(
        "--pathology-model-path",
        "--pathology_model_path",
        dest="pathology_model_path",
        default=os.environ.get("PATHOLOGY_MODEL_PATH", ""),
        help="Local pathology encoder weights. Required for UNI; optional for online H0-mini setup.",
    )
    parser.add_argument("--image-size", "--image_size", dest="image_size", type=int, default=256)
    parser.add_argument("--diffusion-steps", "--diffusion_steps", dest="diffusion_steps", type=int, default=1000)
    parser.add_argument("--patch-size", "--patch_size", dest="patch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--total-steps", "--total_steps", dest="total_steps", type=int, default=50000)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=16)
    parser.add_argument("--log-interval", "--log_interval", dest="log_interval", type=int, default=100)
    parser.add_argument("--save-interval", "--save_interval", dest="save_interval", type=int, default=5000)
    parser.add_argument("--crop-prob", "--crop_prob", dest="crop_prob", type=float, default=0.8)
    parser.add_argument("--vae-model-name", "--vae_model_name", dest="vae_model_name", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--vae-model-dir", "--vae_model_dir", dest="vae_model_dir", default="")
    parser.add_argument("--vae-scaling-factor", "--vae_scaling_factor", dest="vae_scaling_factor", type=float, default=0.18215)
    parser.add_argument("--vae-latent-channels", "--vae_latent_channels", dest="vae_latent_channels", type=int, default=4)
    parser.add_argument("--no-random-crop", action="store_true")
    parser.add_argument("--no-random-flip", action="store_true")
    return parser.parse_args()


def default_patch_size(backbone: str) -> int:
    return 4 if backbone == "vae" else 8


def main() -> None:
    args = parse_args()
    pathology_model_name = normalize_pathology_model_name(args.pathology_model_name)
    patch_size = args.patch_size or default_patch_size(args.backbone)
    data_jsonl = Path(args.data_jsonl).expanduser()
    pathology_model_path = Path(args.pathology_model_path).expanduser() if args.pathology_model_path else None

    if not data_jsonl.is_file():
        raise FileNotFoundError(f"--data-jsonl not found: {data_jsonl}")
    if pathology_model_name == "uni" and pathology_model_path is None:
        raise FileNotFoundError("--pathology-model-path is required for UNI.")
    if pathology_model_path is not None and not pathology_model_path.is_file():
        raise FileNotFoundError(f"--pathology-model-path not found: {pathology_model_path}")

    if args.backbone == "pixel" and args.image_size % patch_size != 0:
        raise ValueError("--image-size must be divisible by --patch-size for pixel CMD.")
    if args.backbone == "vae" and (args.image_size // 8) % patch_size != 0:
        raise ValueError("--image-size / 8 must be divisible by --patch-size for VAE CMD.")

    resume_checkpoint = args.resume_checkpoint
    if not resume_checkpoint:
        latest = find_latest_checkpoint(args.out_dir)
        if latest:
            resume_checkpoint = latest

    model, diffusion = create_model_and_diffusion(
        image_size=args.image_size,
        in_channels=3,
        pathology_feat_dim=get_pathology_feat_dim(pathology_model_name),
        learn_sigma=False,
        diffusion_steps=args.diffusion_steps,
        use_checkpoint=True,
        patch_size=patch_size,
        cmd_size=args.cmd_size,
        backbone=args.backbone,
        model_arch="cmd",
        pathology_model_name=pathology_model_name,
        pathology_model_path=str(pathology_model_path or ""),
        vae_model_name=args.vae_model_name,
        vae_model_dir=args.vae_model_dir,
        vae_scaling_factor=args.vae_scaling_factor,
        vae_latent_channels=args.vae_latent_channels,
    )

    data = load_pretrain_data(
        data_jsonl=str(data_jsonl),
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=False,
        random_crop=not args.no_random_crop,
        random_flip=not args.no_random_flip,
        crop_prob=args.crop_prob,
        num_workers=args.num_workers,
    )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        lr=args.lr,
        total_steps=args.total_steps,
        out_dir=args.out_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=resume_checkpoint,
        ema_decay=0.9999,
    ).run_loop()


if __name__ == "__main__":
    main()
