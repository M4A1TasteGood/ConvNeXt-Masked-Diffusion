"""TNBC segmentation decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from cmd.runtime.checkpoints import load_head_state_dict

from .config import TNBCInferenceConfig


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        super().__init__(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels, 3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class TNBCSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 16),
    ):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(in_channels, head_channels, 3, padding=1)

        in_chs = [head_channels] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList(
            DecoderBlock(in_ch, out_ch)
            for in_ch, out_ch in zip(in_chs, decoder_channels)
        )
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_more(x)
        for block in self.blocks:
            x = block(x)
        return self.seg_head(x)


def build_tnbc_head(config: TNBCInferenceConfig, device: torch.device) -> TNBCSegmentationHead:
    model = TNBCSegmentationHead(
        in_channels=config.feature_dim,
        num_classes=len(config.class_names),
    )
    model.load_state_dict(load_head_state_dict(config.tnbc_head_checkpoint), strict=True)
    return model.to(device).eval()
