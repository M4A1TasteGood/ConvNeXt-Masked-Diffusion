"""CMD TNBC inference package."""

from .config import TNBCInferenceConfig
from .segmentor import CMDTNBCSegmentor

__all__ = ["TNBCInferenceConfig", "CMDTNBCSegmentor"]
