"""
CMD U-Net for mask diffusion.

Architecture based on CMD with
ConvNeXt blocks and adaLN-Zero conditioning, adapted for full-resolution
inputs with 5 downsampling stages (6 levels).

Key features:
  - ConvNeXtBlock: depthwise conv + adaLN-Zero + pointwise MLP + GRN
  - Per-resolution timestep embedders (shared across levels with the same
    channel width)
  - Pathology foundation model conditioning: 1024-dim UNI features projected
    to each resolution's conditioning dimension, fused as t_emb + p_emb
  - Dense skip connections (concat + 1x1 reduce) at every decoder level
  - PixelShuffle / PixelUnshuffle up / downsampling
  - Flat ``output_blocks`` ModuleList for feature-extraction hooks
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PathologyProjector(nn.Module):
    """Projects pathology foundation model features (e.g. UNI 1024-dim)
    to the conditioning dimension used by adaLN-Zero blocks."""

    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, feat):
        """feat: (B, in_dim) -> (B, hidden_size)"""
        return self.proj(feat)


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block with adaptive LayerNorm-Zero (adaLN-Zero) conditioning.
    Combines depthwise conv and pointwise MLP (as in ConvNeXt) with adaLN modulation on the channel dimension.
    """
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
            nn.Linear(dim, 3 * dim, bias=True)
        )

    def forward(self, x, c):
        """
        x: tensor of shape (B, C, H, W)
        c: conditioning tensor of shape (B, C)
        """
        # Depthwise conv
        h = self.dwconv(x)
        # Compute adaLN parameters
        shift, scale, gate = self.adaLN_modulation(c).unsqueeze(2).unsqueeze(3).chunk(3, dim=1)
        # Apply adaptive LayerNorm-Zero: norm -> scale & shift
        h = self.norm(h)
        h = torch.addcmul(shift, h, scale + 1)
        # Pointwise MLP
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        # Apply gate
        h = h * gate
        # Residual
        return x + h


class ConvFinalLayer(nn.Module):
    """
    Conv-style final layer
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
#  CMDUNet — main model
# ---------------------------------------------------------------------------

class CMDUNet(nn.Module):
    """CMD U-Net adapted for full-resolution mask-diffusion pretraining.

    Architecture mirrors *CMDs* (CMD) but with
    a configurable number of downsampling stages so that it can operate directly
    on raw-pixel inputs (e.g. 512x512 medical images) instead of VAE latents.

    Parameters
    ----------
    image_size : int
        Target spatial resolution (informational, not used in forward).
    in_channels : int
        Number of input image channels.
    hidden_size : int
        Base channel width.
    depth : list[int]
        Block counts per stage.  Must have odd length ``2*levels - 1`` and be
        symmetric (encoder mirrors decoder).  Example for 6 levels (5 down):
        ``[1, 2, 3, 2, 2, 6, 2, 2, 3, 2, 1]``.
    out_channels : int
        Number of output channels.
    mult_channels : tuple[int]
        Per-level channel multiplier (encoder side, length == ``levels``).
        Decoder multipliers are the mirror.  Default ``(1,1,1,2,2,4)`` for
        6 levels.
    mlp_ratio : float
        Hidden-dim expansion in pointwise MLP of ConvNeXtBlock.
    pathology_feat_dim : int
        Dimension of pre-extracted pathology foundation model features
        (e.g. 1024 for UNI).  Set to 0 to disable pathology conditioning.
    use_checkpoint : bool
        Use ``torch.utils.checkpoint`` for memory savings.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        hidden_size,
        depth,
        out_channels,
        mult_channels=(1, 1, 1, 2, 2, 4),
        mlp_ratio=3,
        pathology_feat_dim=1024,
        use_checkpoint=True,
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
        assert len(depth) % 2 == 1, (
            f"depth must have odd length (2*levels-1), got {len(depth)}"
        )
        self.levels = (len(depth) + 1) // 2
        stages = self.levels - 1

        if len(mult_channels) == self.levels:
            full_mc = mult_channels + mult_channels[:-1][::-1]
        else:
            full_mc = mult_channels
        assert len(full_mc) == 2 * self.levels - 1

        self.depth = depth
        self.mult_channels = full_mc
        H = hidden_size
        mc = full_mc  # convenience alias

        # --- unique channel sizes & embedder mapping ---
        unique_mults = sorted(set(mc[: self.levels]))
        mult_to_idx = {m: i for i, m in enumerate(unique_mults)}
        self.level_cond_idx = [mult_to_idx[mc[i]] for i in range(self.levels)]

        # === Input embedding ===
        self.x_embedder = nn.Conv2d(in_channels, int(H * mc[0]), 3, padding=1)

        # === Timestep embedders (one per unique channel size) ===
        self.t_embedders = nn.ModuleList(
            [TimestepEmbedder(int(H * m)) for m in unique_mults]
        )

        # === Pathology feature projectors (one per unique channel size) ===
        self.pathology_projectors = None
        if pathology_feat_dim > 0:
            self.pathology_projectors = nn.ModuleList(
                [
                    PathologyProjector(pathology_feat_dim, int(H * m))
                    for m in unique_mults
                ]
            )

        # === Encoder ===
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for enc_i in range(stages):
            ch = int(H * mc[enc_i])
            stage = nn.ModuleList(
                [ConvNeXtBlock(ch, mlp_ratio=mlp_ratio) for _ in range(depth[enc_i])]
            )
            self.enc_blocks.append(stage)
            next_ch = int(H * mc[enc_i + 1])
            self.downs.append(Downsample(ch, next_ch))

        # === Latent ===
        lat_ch = int(H * mc[stages])
        self.lat_blocks = nn.ModuleList(
            [ConvNeXtBlock(lat_ch, mlp_ratio=mlp_ratio) for _ in range(depth[stages])]
        )

        # === Decoder (flat output_blocks for feature-extraction hooks) ===
        self.ups = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        self._dec_stage_info = []  # (start_idx, end_idx) in output_blocks

        prev_ch = lat_ch
        for dec_i in range(stages):
            enc_level = stages - 1 - dec_i
            dec_ch = int(H * mc[enc_level])
            skip_ch = dec_ch
            self.ups.append(Upsample(prev_ch, dec_ch))
            self.reduce_chans.append(nn.Conv2d(dec_ch + skip_ch, dec_ch, 1))
            dec_depth = depth[self.levels + dec_i]
            start = len(self.output_blocks)
            for _ in range(dec_depth):
                self.output_blocks.append(
                    ConvNeXtBlock(dec_ch, mlp_ratio=mlp_ratio)
                )
            self._dec_stage_info.append((start, len(self.output_blocks)))
            prev_ch = dec_ch

        # === Output ===
        final_ch = int(H * mc[0])
        self.output_layer = nn.Conv2d(final_ch, final_ch, 3, padding=1)
        self.final_layer = ConvFinalLayer(final_ch, out_channels)

        self.initialize_weights()

    # ------------------------------------------------------------------
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
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
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    # ------------------------------------------------------------------
    @staticmethod
    def _ckpt_wrapper(module):
        def ckpt_forward(*inputs):
            return module(*inputs)
        return ckpt_forward

    # ------------------------------------------------------------------
    def _run_block(self, block, x, c):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._ckpt_wrapper(block), x, c, use_reentrant=False
            )
        return block(x, c)

    # ------------------------------------------------------------------
    def forward(self, x, timesteps, pathology_feat=None):
        """
        :param x: (N, C, H, W) input images / masked images.
        :param timesteps: (N,) diffusion timestep indices.
        :param pathology_feat: (N, pathology_feat_dim) pre-extracted pathology
            foundation model features (e.g. UNI). None for unconditional.
        """
        assert (pathology_feat is not None) == (
            self.pathology_projectors is not None
        ), "must specify pathology_feat iff pathology_feat_dim > 0"

        h = x
        h = self.x_embedder(h)

        # --- per-resolution conditioning ---
        conds = []
        for i, t_emb_module in enumerate(self.t_embedders):
            t_emb = t_emb_module(timesteps)
            if self.pathology_projectors is not None:
                p_emb = self.pathology_projectors[i](pathology_feat)
                conds.append(t_emb + p_emb)
            else:
                conds.append(t_emb)

        stages = self.levels - 1

        # --- encoder ---
        skips = []
        for enc_i in range(stages):
            c = conds[self.level_cond_idx[enc_i]]
            for block in self.enc_blocks[enc_i]:
                h = self._run_block(block, h, c)
            skips.append(h)
            h = self.downs[enc_i](h)

        # --- latent ---
        c_lat = conds[self.level_cond_idx[stages]]
        for block in self.lat_blocks:
            h = self._run_block(block, h, c_lat)

        # --- decoder ---
        for dec_i in range(stages):
            enc_level = stages - 1 - dec_i
            c = conds[self.level_cond_idx[enc_level]]
            h = self.ups[dec_i](h)
            h = torch.cat([h, skips[enc_level]], dim=1)
            h = self.reduce_chans[dec_i](h)
            start, end = self._dec_stage_info[dec_i]
            for bi in range(start, end):
                h = self._run_block(self.output_blocks[bi], h, c)

        # --- output ---
        h = self.output_layer(h)
        h = self.final_layer(h, conds[self.level_cond_idx[0]])
        return h

