"""CMD feature extraction used by the TNBC segmentation head."""

from __future__ import annotations

import torch
import torch.nn as nn

from cmd.models.factory import create_cmd_model
from cmd.models.mask_diffusion import CMDMasking
from cmd.models.pathology_encoder import build_pathology_encoder, extract_pathology_features
from cmd.runtime.checkpoints import load_cmd_state_dict

from .config import TNBCInferenceConfig


def _save_activations(module: nn.Module, features, name: str) -> None:
    if isinstance(features, (list, tuple)):
        value = [f.detach().float() if f is not None else None for f in features]
    elif isinstance(features, dict):
        value = {k: f.detach().float() for k, f in features.items()}
    else:
        value = features.detach().float()
    setattr(module, name, value)


def _save_output_hook(module: nn.Module, _inputs, output):
    _save_activations(module, output, "activations")
    return output


def collect_features(config: TNBCInferenceConfig, activations: list[torch.Tensor]) -> torch.Tensor:
    size = (config.image_size, config.image_size)
    resized = []
    for features in activations:
        features = features[0][None]
        features = nn.functional.interpolate(
            features,
            size=size,
            mode=config.upsample_mode,
        )
        resized.append(features[0])
    return torch.cat(resized, dim=0)


class CMDFeatureExtractor(nn.Module):
    """Instantiate CMD and expose the feature maps used by the TNBC head."""

    def __init__(self, config: TNBCInferenceConfig, device: torch.device):
        super().__init__()
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

        print("[CMD] loading H0-mini pathology encoder...")
        self.pathology_model = build_pathology_encoder(
            config.pathology_model_name,
            device,
            weights_path=str(config.pathology_model_path),
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
            t = torch.tensor([int(step)], device=image.device, dtype=torch.long)
            masked_image = self.diffusion.q_sample(image, t)
            self.cmd(
                masked_image,
                self.diffusion._scale_timesteps(t),
                pathology_feat=pathology_feat,
            )
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        features = collect_features(self.config, activations)
        return features / (torch.norm(features) + 1e-8)
