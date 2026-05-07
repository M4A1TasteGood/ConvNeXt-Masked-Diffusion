"""CMD model and mask-diffusion factories."""

from __future__ import annotations

import argparse

from .cmd_vae_unet import (
    CMD_VAE_CONFIGS,
    build_cmd_vae_unet,
    get_cmd_vae_feature_info,
    get_cmd_vae_stage_info,
)
from .mask_diffusion import GaussianDiffusion
from .pathology_encoder import get_pathology_feat_dim, normalize_pathology_model_name
from .unet import CMDUNet

BACKBONE_CHOICES = ("pixel", "vae")
MODEL_ARCH_CHOICES = ("cmd",)
DEFAULT_VAE_SCALE_FACTOR = 0.18215
DEFAULT_VAE_DOWNSAMPLE = 8

CMD_CONFIGS = {
    "S": {"hidden_size": 128, "depth": [1, 2, 3, 2, 2, 6, 2, 2, 3, 2, 1], "mult_channels": [1, 1, 1, 2, 2, 4]},
    "B": {"hidden_size": 256, "depth": [1, 2, 3, 2, 2, 6, 2, 2, 3, 2, 1], "mult_channels": [1, 1, 1, 2, 2, 4]},
    "L": {"hidden_size": 512, "depth": [1, 2, 3, 2, 2, 6, 2, 2, 3, 2, 1], "mult_channels": [1, 1, 1, 2, 2, 4]},
    "XL": {"hidden_size": 512, "depth": [2, 3, 4, 3, 3, 9, 3, 3, 4, 3, 2], "mult_channels": [1, 1, 1, 2, 2, 4]},
}


def _resolve_cmd_size(cmd_size: str | None = None) -> str:
    return str(cmd_size or "B").upper()


def get_cmd_feature_info(cmd_size, image_size=512, use_all_blocks=False, backbone="pixel", model_arch="cmd"):
    """Compute output block indices and total feature dimension."""
    cmd_size = _resolve_cmd_size(cmd_size)
    if backbone == "vae":
        return get_cmd_vae_feature_info(cmd_size, use_all_blocks=use_all_blocks)
    if cmd_size not in CMD_CONFIGS:
        raise ValueError(f"Unknown cmd_size '{cmd_size}', choose from {list(CMD_CONFIGS.keys())}")

    cfg = CMD_CONFIGS[cmd_size]
    hidden = cfg["hidden_size"]
    depth = cfg["depth"]
    mult_channels = cfg["mult_channels"]
    levels = (len(depth) + 1) // 2
    stages = levels - 1
    full_mult = list(mult_channels) + list(mult_channels)[:-1][::-1]
    dec_depths = depth[levels:]
    dec_mults = full_mult[levels:]

    offset = 0
    blocks, block_channels = [], []
    for stage_i in range(stages):
        block_count = dec_depths[stage_i]
        channels = int(hidden * dec_mults[stage_i])
        if use_all_blocks:
            for block_i in range(block_count):
                blocks.append(offset + block_i)
                block_channels.append(channels)
        else:
            if block_count >= 3:
                blocks.append(offset + block_count // 2)
                block_channels.append(channels)
            blocks.append(offset + block_count - 1)
            block_channels.append(channels)
        offset += block_count
    return blocks, sum(block_channels)


def get_cmd_stage_last_blocks(cmd_size: str, backbone: str = "pixel", model_arch: str = "cmd"):
    """Return the last output block and channel count for each decoder stage."""
    cmd_size = _resolve_cmd_size(cmd_size)
    if backbone == "vae":
        return get_cmd_vae_stage_info(cmd_size)
    if cmd_size not in CMD_CONFIGS:
        raise ValueError(f"Unknown cmd_size '{cmd_size}', choose from {list(CMD_CONFIGS.keys())}")

    cfg = CMD_CONFIGS[cmd_size]
    hidden = cfg["hidden_size"]
    depth = list(cfg["depth"])
    mult_channels = list(cfg["mult_channels"])
    levels = (len(depth) + 1) // 2
    stages = levels - 1
    full_mult = mult_channels + mult_channels[:-1][::-1]
    dec_depths = depth[levels:]
    dec_mults = full_mult[levels:]

    offset = 0
    blocks, channels = [], []
    for stage_i in range(stages):
        block_count = dec_depths[stage_i]
        ch = int(hidden * dec_mults[stage_i])
        blocks.append(offset + block_count - 1)
        channels.append(ch)
        offset += block_count
    return blocks, channels


def diffusion_defaults():
    return {"learn_sigma": False, "diffusion_steps": 1000}


def model_and_diffusion_defaults():
    defaults = {
        "image_size": 512,
        "in_channels": 3,
        "pathology_feat_dim": 1024,
        "use_checkpoint": True,
        "patch_size": 16,
        "cmd_size": "B",
        "backbone": "pixel",
        "model_arch": "cmd",
        "pathology_model_name": "uni",
        "pathology_model_path": "",
        "vae_model_name": "stabilityai/sd-vae-ft-ema",
        "vae_model_dir": "",
        "vae_scaling_factor": DEFAULT_VAE_SCALE_FACTOR,
        "vae_latent_channels": 4,
    }
    defaults.update(diffusion_defaults())
    return defaults


def _coerce_dir(value):
    if value in (None, "", "None", "null"):
        return None
    return value


def create_model_and_diffusion(
    image_size,
    in_channels,
    pathology_feat_dim,
    learn_sigma,
    diffusion_steps,
    use_checkpoint,
    patch_size,
    cmd_size="",
    backbone="pixel",
    model_arch="cmd",
    pathology_model_name="uni",
    pathology_model_path="",
    vae_model_name="stabilityai/sd-vae-ft-ema",
    vae_model_dir="",
    vae_scaling_factor=DEFAULT_VAE_SCALE_FACTOR,
    vae_latent_channels=4,
    **kwargs,
):
    cmd_size = _resolve_cmd_size(cmd_size)
    pathology_model_name = normalize_pathology_model_name(pathology_model_name)
    if pathology_feat_dim > 0:
        pathology_feat_dim = get_pathology_feat_dim(pathology_model_name)

    model = create_model(
        image_size=image_size,
        in_channels=in_channels,
        learn_sigma=learn_sigma,
        pathology_feat_dim=pathology_feat_dim,
        use_checkpoint=use_checkpoint,
        cmd_size=cmd_size,
        backbone=backbone,
        model_arch=model_arch,
        vae_latent_channels=vae_latent_channels,
    )
    diff_channels = vae_latent_channels if backbone == "vae" else in_channels
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        in_channels=diff_channels,
        patch_size=patch_size,
    )

    model.backbone = backbone
    model.model_arch = "cmd"
    model.pathology_model_name = pathology_model_name
    model.pathology_model_path = pathology_model_path
    model.pathology_feat_dim = pathology_feat_dim
    if backbone == "vae":
        model.vae_model_name = vae_model_name
        model.vae_model_dir = _coerce_dir(vae_model_dir)
        model.vae_scaling_factor = float(vae_scaling_factor)
        model.vae_downsample = DEFAULT_VAE_DOWNSAMPLE

    return model, diffusion


def create_model(
    image_size,
    in_channels=3,
    learn_sigma=False,
    pathology_feat_dim=1024,
    use_checkpoint=True,
    cmd_size="B",
    backbone="pixel",
    model_arch="cmd",
    vae_latent_channels=4,
):
    cmd_size = _resolve_cmd_size(cmd_size)
    if backbone not in BACKBONE_CHOICES:
        raise ValueError(f"Unknown backbone '{backbone}', choose from {BACKBONE_CHOICES}")
    if model_arch not in MODEL_ARCH_CHOICES:
        raise ValueError(f"Unknown model_arch '{model_arch}', choose from {MODEL_ARCH_CHOICES}")

    if backbone == "vae":
        return build_cmd_vae_unet(
            cmd_size=cmd_size,
            in_channels=vae_latent_channels,
            learn_sigma=learn_sigma,
            pathology_feat_dim=pathology_feat_dim,
            use_checkpoint=use_checkpoint,
        )

    if cmd_size not in CMD_CONFIGS:
        raise ValueError(f"Unknown cmd_size '{cmd_size}', choose from {list(CMD_CONFIGS.keys())}")
    cfg = CMD_CONFIGS[cmd_size]
    out_channels = in_channels * 2 if learn_sigma else in_channels
    return CMDUNet(
        image_size=image_size,
        in_channels=in_channels,
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        out_channels=out_channels,
        mult_channels=cfg["mult_channels"],
        mlp_ratio=3,
        pathology_feat_dim=pathology_feat_dim,
        use_checkpoint=use_checkpoint,
    )


def create_gaussian_diffusion(*, steps=1000, in_channels, patch_size):
    return GaussianDiffusion(channels=in_channels, timesteps=steps, loss_type="l1", patch_size=patch_size)


def create_cmd_model(image_size, in_channels, pathology_feat_dim, cmd_size="B", use_checkpoint=True):
    """Build the released pixel-space CMD model used by pretraining and inference."""
    return create_model(
        image_size=image_size,
        in_channels=in_channels,
        learn_sigma=False,
        pathology_feat_dim=pathology_feat_dim,
        use_checkpoint=use_checkpoint,
        cmd_size=cmd_size,
        backbone="pixel",
        model_arch="cmd",
    )


def add_dict_to_argparser(parser, default_dict):
    for key, value in default_dict.items():
        value_type = type(value)
        if value is None:
            value_type = str
        elif isinstance(value, bool):
            value_type = str2bool
        parser.add_argument(f"--{key}", default=value, type=value_type)


def args_to_dict(args, keys):
    return {key: getattr(args, key) for key in keys}


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")
