"""CMD backbone architecture used for feature extraction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


CMD_CONFIGS = {
    "S": {
        "hidden_size": 128,
        "depth": [1, 2, 3, 2, 2, 6, 2, 2, 3, 2, 1],
        "mult_channels": [1, 1, 1, 2, 2, 4],
    },
    "B": {
        "hidden_size": 256,
        "depth": [1, 2, 3, 2, 2, 6, 2, 2, 3, 2, 1],
        "mult_channels": [1, 1, 1, 2, 2, 4],
    },
    "L": {
        "hidden_size": 512,
        "depth": [1, 2, 3, 2, 2, 6, 2, 2, 3, 2, 1],
        "mult_channels": [1, 1, 1, 2, 2, 4],
    },
    "XL": {
        "hidden_size": 512,
        "depth": [2, 3, 4, 3, 3, 9, 3, 3, 4, 3, 2],
        "mult_channels": [1, 1, 1, 2, 2, 4],
    },
}


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class PathologyProjector(nn.Module):
    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, feat):
        return self.proj(feat)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, affine=False, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.act = nn.GELU()
        self.grn = GRN(int(dim * mlp_ratio))
        self.pwconv2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True),
        )

    def forward(self, x, c):
        h = self.dwconv(x)
        shift, scale, gate = self.adaLN_modulation(c).unsqueeze(2).unsqueeze(3).chunk(3, dim=1)
        h = torch.addcmul(shift, self.norm(h), scale + 1)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        return x + h * gate


class ConvFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.conv(modulate(self.norm(x), shift, scale))


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.conv(x)


class CMDUNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        hidden_size,
        depth,
        out_channels,
        mult_channels=(1, 1, 1, 2, 2, 4),
        mlp_ratio=3,
        pathology_feat_dim=768,
        use_checkpoint=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = hidden_size
        self.pathology_feat_dim = pathology_feat_dim
        self.use_checkpoint = use_checkpoint

        depth = list(depth)
        mult_channels = list(mult_channels)
        if len(depth) % 2 != 1:
            raise ValueError(f"depth must have odd length, got {len(depth)}")
        self.levels = (len(depth) + 1) // 2
        stages = self.levels - 1

        if len(mult_channels) == self.levels:
            full_mc = mult_channels + mult_channels[:-1][::-1]
        else:
            full_mc = mult_channels
        if len(full_mc) != 2 * self.levels - 1:
            raise ValueError("invalid channel multiplier configuration")

        self.depth = depth
        self.mult_channels = full_mc
        hidden = hidden_size
        mc = full_mc

        unique_mults = sorted(set(mc[: self.levels]))
        mult_to_idx = {m: i for i, m in enumerate(unique_mults)}
        self.level_cond_idx = [mult_to_idx[mc[i]] for i in range(self.levels)]

        self.x_embedder = nn.Conv2d(in_channels, int(hidden * mc[0]), 3, padding=1)
        self.t_embedders = nn.ModuleList(
            [TimestepEmbedder(int(hidden * m)) for m in unique_mults]
        )

        self.pathology_projectors = None
        if pathology_feat_dim > 0:
            self.pathology_projectors = nn.ModuleList(
                [PathologyProjector(pathology_feat_dim, int(hidden * m)) for m in unique_mults]
            )

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for enc_i in range(stages):
            ch = int(hidden * mc[enc_i])
            self.enc_blocks.append(
                nn.ModuleList([ConvNeXtBlock(ch, mlp_ratio=mlp_ratio) for _ in range(depth[enc_i])])
            )
            self.downs.append(Downsample(ch, int(hidden * mc[enc_i + 1])))

        lat_ch = int(hidden * mc[stages])
        self.lat_blocks = nn.ModuleList(
            [ConvNeXtBlock(lat_ch, mlp_ratio=mlp_ratio) for _ in range(depth[stages])]
        )

        self.ups = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        self._dec_stage_info = []

        prev_ch = lat_ch
        for dec_i in range(stages):
            enc_level = stages - 1 - dec_i
            dec_ch = int(hidden * mc[enc_level])
            self.ups.append(Upsample(prev_ch, dec_ch))
            self.reduce_chans.append(nn.Conv2d(dec_ch + dec_ch, dec_ch, 1))
            dec_depth = depth[self.levels + dec_i]
            start = len(self.output_blocks)
            for _ in range(dec_depth):
                self.output_blocks.append(ConvNeXtBlock(dec_ch, mlp_ratio=mlp_ratio))
            self._dec_stage_info.append((start, len(self.output_blocks)))
            prev_ch = dec_ch

        final_ch = int(hidden * mc[0])
        self.output_layer = nn.Conv2d(final_ch, final_ch, 3, padding=1)
        self.final_layer = ConvFinalLayer(final_ch, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.xavier_uniform_(self.x_embedder.weight.data.view(self.x_embedder.weight.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)

        if self.pathology_projectors is not None:
            for proj in self.pathology_projectors:
                nn.init.normal_(proj.proj[0].weight, std=0.02)
                nn.init.normal_(proj.proj[2].weight, std=0.02)

        for t_emb in self.t_embedders:
            nn.init.normal_(t_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(t_emb.mlp[2].weight, std=0.02)

        all_blocks = []
        for stage in self.enc_blocks:
            all_blocks.extend(stage)
        all_blocks.extend(self.lat_blocks)
        all_blocks.extend(self.output_blocks)
        for block in all_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    @staticmethod
    def _ckpt_wrapper(module):
        def ckpt_forward(*inputs):
            return module(*inputs)

        return ckpt_forward

    def _run_block(self, block, x, c):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._ckpt_wrapper(block),
                x,
                c,
                use_reentrant=False,
            )
        return block(x, c)

    def forward(self, x, timesteps, pathology_feat=None):
        has_pathology = self.pathology_projectors is not None
        if (pathology_feat is not None) != has_pathology:
            raise ValueError("pathology features must match CMD pathology conditioning")

        h = self.x_embedder(x)
        conds = []
        for i, t_emb_module in enumerate(self.t_embedders):
            t_emb = t_emb_module(timesteps)
            if has_pathology:
                conds.append(t_emb + self.pathology_projectors[i](pathology_feat))
            else:
                conds.append(t_emb)

        stages = self.levels - 1
        skips = []
        for enc_i in range(stages):
            c = conds[self.level_cond_idx[enc_i]]
            for block in self.enc_blocks[enc_i]:
                h = self._run_block(block, h, c)
            skips.append(h)
            h = self.downs[enc_i](h)

        c_lat = conds[self.level_cond_idx[stages]]
        for block in self.lat_blocks:
            h = self._run_block(block, h, c_lat)

        for dec_i in range(stages):
            enc_level = stages - 1 - dec_i
            c = conds[self.level_cond_idx[enc_level]]
            h = self.ups[dec_i](h)
            h = torch.cat([h, skips[enc_level]], dim=1)
            h = self.reduce_chans[dec_i](h)
            start, end = self._dec_stage_info[dec_i]
            for block_idx in range(start, end):
                h = self._run_block(self.output_blocks[block_idx], h, c)

        h = self.output_layer(h)
        return self.final_layer(h, conds[self.level_cond_idx[0]])


def create_cmd_model(
    *,
    image_size: int,
    in_channels: int,
    pathology_feat_dim: int,
    cmd_size: str,
    use_checkpoint: bool = False,
) -> CMDUNet:
    if cmd_size not in CMD_CONFIGS:
        raise ValueError(f"unknown CMD size: {cmd_size}")
    cfg = CMD_CONFIGS[cmd_size]
    return CMDUNet(
        image_size=image_size,
        in_channels=in_channels,
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        out_channels=in_channels,
        mult_channels=cfg["mult_channels"],
        mlp_ratio=3,
        pathology_feat_dim=pathology_feat_dim,
        use_checkpoint=use_checkpoint,
    )
