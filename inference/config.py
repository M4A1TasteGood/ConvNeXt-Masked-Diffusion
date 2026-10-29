"""Inference settings for the released CMD-L TNBC segmentation head."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TNBC_CLASS_NAMES = ("background", "tumor")
# Selective decoder blocks of pixel CMD-L. Channel sum is 4096.
TNBC_FEATURE_BLOCKS = (1, 3, 5, 6, 8, 9)


@dataclass(frozen=True)
class TNBCInferenceConfig:
    cmd_checkpoint: Path
    seg_head_checkpoint: Path
    pathology_weights: Path
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
