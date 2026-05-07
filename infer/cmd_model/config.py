"""Configuration for released CMD TNBC inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


TNBC_CLASS_NAMES = ("background", "tumor")
TNBC_FEATURE_BLOCKS = (1, 3, 5, 6, 8, 9)


@dataclass(frozen=True)
class TNBCInferenceConfig:
    cmd_checkpoint: Path
    tnbc_head_checkpoint: Path
    pathology_model_path: Path
    image_size: int = 256
    feature_dim: int = 4096
    diffusion_step: int = 50
    feature_blocks: tuple[int, ...] = field(default_factory=lambda: TNBC_FEATURE_BLOCKS)
    class_names: tuple[str, ...] = field(default_factory=lambda: TNBC_CLASS_NAMES)
    pathology_model_name: str = "h0-mini"
    pathology_feat_dim: int = 768
    in_channels: int = 3
    patch_size: int = 8
    cmd_size: str = "L"
    upsample_mode: str = "bilinear"

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return (self.image_size, self.image_size, self.feature_dim)

    def to_feature_args(self) -> dict:
        return {
            "dim": list(self.feature_shape),
            "mask_steps": [self.diffusion_step],
            "blocks": list(self.feature_blocks),
            "upsample_mode": self.upsample_mode,
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "pathology_model_name": self.pathology_model_name,
            "pathology_model_path": str(self.pathology_model_path),
            "pathology_feat_dim": self.pathology_feat_dim,
            "diffusion_steps": 1000,
            "patch_size": self.patch_size,
            "cmd_size": self.cmd_size,
        }
