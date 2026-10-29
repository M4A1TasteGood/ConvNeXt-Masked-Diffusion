"""Single-image CMD feature extraction for the released segmentation head."""

from __future__ import annotations

import torch
import torch.nn as nn

from cmd.models.factory import create_cmd_model, get_cmd_feature_info
from cmd.models.mask_diffusion import CMDMasking
from cmd.models.pathology_encoder import build_pathology_encoder, extract_pathology_features

from .checkpoints import load_cmd_state_dict
from .config import TNBCInferenceConfig


def _save_output_hook(module: nn.Module, _inputs, output):
    module.activations = output.detach().float()
    return output


def collect_features(config: TNBCInferenceConfig, activations: list[torch.Tensor]) -> torch.Tensor:
    size = (config.image_size, config.image_size)
    resized = []
    for features in activations:
        sample = features[0][None]
        sample = nn.functional.interpolate(sample, size=size, mode=config.upsample_mode)
        resized.append(sample[0])
    return torch.cat(resized, dim=0)


class CMDFeatureExtractor(nn.Module):
    def __init__(self, config: TNBCInferenceConfig, device: torch.device):
        super().__init__()
        blocks, feature_dim = get_cmd_feature_info(config.cmd_size)
        if tuple(blocks) != tuple(config.feature_blocks) or feature_dim != config.feature_dim:
            raise ValueError(
                f"CMD-{config.cmd_size} features are blocks {blocks} with dim {feature_dim}, "
                f"but this head expects blocks {config.feature_blocks} and dim {config.feature_dim}."
            )

        self.config = config
        self.steps = [config.diffusion_step]
        self.pathology_model_name = config.pathology_model_name

        self.cmd = create_cmd_model(
            image_size=config.image_size,
            in_channels=config.in_channels,
            pathology_feat_dim=config.pathology_feat_dim,
            cmd_size=config.cmd_size,
        )
        self.diffusion = CMDMasking(
            channels=config.in_channels,
            timesteps=1000,
            patch_size=config.patch_size,
        )
        print("[CMD] loading pretrained CMD checkpoint...")
        missing, unexpected = self.cmd.load_state_dict(
            load_cmd_state_dict(config.cmd_checkpoint),
            strict=False,
        )
        if unexpected:
            raise RuntimeError(f"unexpected CMD checkpoint keys: {len(unexpected)}")
        if missing:
            print(f"[CMD] warning: missing checkpoint keys: {len(missing)}")
        self.cmd.to(device).eval()

        self.feature_blocks = []
        for idx, block in enumerate(self.cmd.output_blocks):
            if idx in config.feature_blocks:
                block.register_forward_hook(_save_output_hook)
                self.feature_blocks.append(block)

        print("[CMD] loading pathology encoder...")
        self.pathology_model = build_pathology_encoder(
            config.pathology_model_name,
            device,
            weights_path=str(config.pathology_weights),
        )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pathology_feat = extract_pathology_features(
            self.pathology_model,
            image,
            self.pathology_model_name,
        )

        activations = []
        for step in self.steps:
            timestep = torch.tensor([int(step)], device=image.device, dtype=torch.long)
            masked_image = self.diffusion.q_sample(image, timestep)
            self.cmd(
                masked_image,
                self.diffusion._scale_timesteps(timestep),
                pathology_feat=pathology_feat,
            )
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        features = collect_features(self.config, activations)
        return features / (torch.norm(features) + 1e-8)
