"""Frozen-CMD inference for cell-level dense prediction."""

from .config import TNBCInferenceConfig
from .segmentor import CMDTNBCSegmentor, build_segmentor

__all__ = ["TNBCInferenceConfig", "CMDTNBCSegmentor", "build_segmentor"]
