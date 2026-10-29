"""End-to-end frozen CMD encoder plus segmentation head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TNBCInferenceConfig
from .features import CMDFeatureExtractor
from .head import build_tnbc_head


class CMDTNBCSegmentor(nn.Module):
    def __init__(self, config: TNBCInferenceConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.feature_extractor = CMDFeatureExtractor(config, device)
        self.head = build_tnbc_head(config, device)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(image).unsqueeze(0)
        return self.head(features)


def build_segmentor(config: TNBCInferenceConfig, device: torch.device) -> CMDTNBCSegmentor:
    return CMDTNBCSegmentor(config, device).to(device).eval()
