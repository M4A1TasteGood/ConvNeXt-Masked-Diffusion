"""Frozen CMD features for segmentation-head training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmd.models.factory import create_cmd_model, get_cmd_feature_info
from cmd.models.mask_diffusion import CMDMasking
from cmd.models.pathology_encoder import (
    build_pathology_encoder,
    extract_pathology_features,
    get_pathology_feat_dim,
    normalize_pathology_model_name,
)
from inference.checkpoints import load_cmd_state_dict
from inference.utils import configure_pathology_encoder


class FrozenCMDFeatureExtractor(nn.Module):
    """Frozen CMD feature extractor shared by downstream segmentation tasks.

    Each sample is L2-normalized over the concatenated feature map. For a batch
    of one this matches the normalization used at inference time.
    """

    def __init__(
        self,
        *,
        cmd_checkpoint: str,
        cmd_size: str,
        image_size: int,
        feature_blocks: list[int],
        diffusion_step: int,
        pathology_model_name: str,
        pathology_weights: str,
        pathology_feat_dim: int,
        in_channels: int = 3,
        patch_size: int = 8,
        allow_random_cmd: bool = False,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.feature_blocks = tuple(feature_blocks)
        self.diffusion_step = int(diffusion_step)
        self.pathology_model_name = pathology_model_name
        self.use_pathology = pathology_feat_dim > 0
        self.activations: list[torch.Tensor] = []

        self.cmd = create_cmd_model(
            image_size=image_size,
            in_channels=in_channels,
            pathology_feat_dim=pathology_feat_dim,
            cmd_size=cmd_size,
            use_checkpoint=False,
        )
        if cmd_checkpoint:
            missing, unexpected = self.cmd.load_state_dict(load_cmd_state_dict(cmd_checkpoint), strict=False)
            if unexpected:
                print(f"[CMD downstream] warning: skipped unexpected CMD checkpoint keys: {len(unexpected)}")
            if missing:
                print(f"[CMD downstream] warning: missing CMD checkpoint keys: {len(missing)}")
        elif not allow_random_cmd:
            raise FileNotFoundError("model.cmd_checkpoint is required unless model.allow_random_cmd is true.")
        self.cmd.to(device).eval()
        for param in self.cmd.parameters():
            param.requires_grad = False

        self.diffusion = CMDMasking(channels=in_channels, timesteps=1000, patch_size=patch_size)
        self.hooks = []
        for idx, block in enumerate(self.cmd.output_blocks):
            if idx in self.feature_blocks:
                self.hooks.append(block.register_forward_hook(self._save_output))

        self.pathology_model = None
        if self.use_pathology:
            if not pathology_weights:
                raise FileNotFoundError("model.pathology_weights is required when pathology conditioning is enabled.")
            configure_pathology_encoder(pathology_weights)
            self.pathology_model = build_pathology_encoder(
                pathology_model_name,
                device,
                weights_path=pathology_weights,
            )

    def _save_output(self, _module, _inputs, output) -> None:
        self.activations.append(output.detach().float())

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.activations = []
        pathology_feat = None
        if self.pathology_model is not None:
            pathology_feat = extract_pathology_features(
                self.pathology_model,
                image,
                self.pathology_model_name,
            )

        timestep = torch.full((image.shape[0],), self.diffusion_step, device=image.device, dtype=torch.long)
        masked_image = self.diffusion.q_sample(image, timestep)
        self.cmd(masked_image, self.diffusion._scale_timesteps(timestep), pathology_feat=pathology_feat)
        if len(self.activations) != len(self.feature_blocks):
            raise RuntimeError("Failed to collect all requested CMD feature blocks.")
        resized = [
            F.interpolate(feat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            for feat in self.activations
        ]
        features = torch.cat(resized, dim=1)
        norms = torch.linalg.vector_norm(features.flatten(1), dim=1).clamp_min(1e-8)
        return features / norms[:, None, None, None]


def build_feature_extractor(model_cfg, device: torch.device):
    pathology_model_name = normalize_pathology_model_name(model_cfg.pathology_model_name)
    pathology_feat_dim = 0 if model_cfg.disable_pathology else get_pathology_feat_dim(pathology_model_name)
    blocks, feature_dim = get_cmd_feature_info(
        model_cfg.cmd_size,
        use_all_blocks=model_cfg.use_all_feature_blocks,
    )
    extractor = FrozenCMDFeatureExtractor(
        cmd_checkpoint=model_cfg.cmd_checkpoint,
        cmd_size=model_cfg.cmd_size,
        image_size=model_cfg.image_size,
        feature_blocks=blocks,
        diffusion_step=model_cfg.diffusion_step,
        pathology_model_name=pathology_model_name,
        pathology_weights=model_cfg.pathology_weights,
        pathology_feat_dim=pathology_feat_dim,
        in_channels=model_cfg.in_channels,
        patch_size=model_cfg.patch_size,
        allow_random_cmd=model_cfg.allow_random_cmd,
        device=device,
    )
    return extractor.to(device).eval(), feature_dim
