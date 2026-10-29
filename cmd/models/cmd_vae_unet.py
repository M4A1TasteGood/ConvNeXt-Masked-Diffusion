"""Latent-space CMD: a 3-level ConvNeXt U-Net that runs in VAE latent space.

The topology matches the pixel CMD decoder layout:

    encoder_level_1 (H) -> Down -> encoder_level_2 (2H) -> Down
        -> latent (4H)
        -> Up -> cat(enc2) -> reduce -> decoder_level_2 (2H)
        -> Up -> cat(enc1) -> reduce -> decoder_level_1 (H)
        -> output_layer -> final_layer

Conditioning replaces a class embedding with a pathology-foundation-model
projector. Each channel scale has its own timestep embedder and projector, and
the adaLN-Zero condition is ``c = t_emb + p_emb``. Decoder blocks are also
exposed as a flat ``output_blocks`` list (``decoder_level_2`` then
``decoder_level_1``) so feature hooks can index them the same way as the pixel
model.
"""

import math
import torch
import torch.nn as nn
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
    """Embeds scalar timesteps into vector representations."""

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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class PathologyProjector(nn.Module):
    """Project pathology-encoder features into the adaLN conditioning width."""

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
    """GRN (Global Response Normalization) layer."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with adaLN-Zero conditioning."""

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
        h = self.norm(h)
        h = torch.addcmul(shift, h, scale + 1)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        h = h * gate
        return x + h


class ConvFinalLayer(nn.Module):
    """Conv-style final layer with adaLN."""

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
        x = modulate(self.norm(x), shift, scale)
        x = self.conv(x)
        return x


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


# ---------------------------------------------------------------------------
#  CMD (VAE latent space) - 3-level architecture
# ---------------------------------------------------------------------------

class CMDVAEUNet(nn.Module):
    """Latent-space CMD conditioned on a frozen pathology encoder.

    Parameters
    ----------
    in_channels :
        Latent channels. Stable Diffusion's VAE uses 4.
    hidden_size :
        Base width ``H``. The three scales use ``(H, 2H, 4H)``.
    depth :
        Five integers: ``[enc1, enc2, latent, dec2, dec1]``.
    mlp_ratio :
        Expansion ratio of the pointwise MLP inside each ConvNeXt block.
    learn_sigma :
        When true, the output has ``2 * in_channels`` channels.
    pathology_feat_dim :
        Pathology-encoder width. ``0`` disables pathology conditioning.
    use_checkpoint :
        Enable ``torch.utils.checkpoint`` during training.
    """

    def __init__(
        self,
        in_channels=4,
        hidden_size=512,
        depth=(2, 4, 8, 4, 2),
        mlp_ratio=3,
        learn_sigma=False,
        pathology_feat_dim=1024,
        use_checkpoint=True,
    ):
        super().__init__()
        assert len(depth) == 5, f"CMD requires 5-stage depth, got {depth}"

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.model_channels = hidden_size
        self.pathology_feat_dim = pathology_feat_dim
        self.use_checkpoint = use_checkpoint

        H = hidden_size

        # --- timestep embedders (1 per unique channel size) ---
        self.t_embedder_1 = TimestepEmbedder(H)
        self.t_embedder_2 = TimestepEmbedder(H * 2)
        self.t_embedder_3 = TimestepEmbedder(H * 4)

        # Pathology projectors, one per channel scale.
        self.pathology_projectors = None
        if pathology_feat_dim > 0:
            self.pathology_projectors = nn.ModuleList(
                [
                    PathologyProjector(pathology_feat_dim, H),
                    PathologyProjector(pathology_feat_dim, H * 2),
                    PathologyProjector(pathology_feat_dim, H * 4),
                ]
            )

        # --- Input embedding ---
        self.x_embedder = nn.Conv2d(in_channels, H, kernel_size=3, stride=1, padding=1)

        # --- Encoder 1 ---
        self.encoder_level_1 = nn.ModuleList(
            [ConvNeXtBlock(H, mlp_ratio=mlp_ratio) for _ in range(depth[0])]
        )
        self.down1_2 = Downsample(H, H * 2)

        # --- Encoder 2 ---
        self.encoder_level_2 = nn.ModuleList(
            [ConvNeXtBlock(H * 2, mlp_ratio=mlp_ratio) for _ in range(depth[1])]
        )
        self.down2_3 = Downsample(H * 2, H * 4)

        # --- Latent ---
        self.latent = nn.ModuleList(
            [ConvNeXtBlock(H * 4, mlp_ratio=mlp_ratio) for _ in range(depth[2])]
        )

        # --- Decoder 2 ---
        self.up3_2 = Upsample(H * 4, H * 2)
        self.reduce_chans_2 = nn.Conv2d(H * 4, H * 2, kernel_size=1)
        self.decoder_level_2 = nn.ModuleList(
            [ConvNeXtBlock(H * 2, mlp_ratio=mlp_ratio) for _ in range(depth[3])]
        )

        # --- Decoder 1 ---
        self.up2_1 = Upsample(H * 2, H)
        self.reduce_chans_1 = nn.Conv2d(H * 2, H, kernel_size=1)
        self.decoder_level_1 = nn.ModuleList(
            [ConvNeXtBlock(H, mlp_ratio=mlp_ratio) for _ in range(depth[4])]
        )

        # --- Flat output_blocks for feature extraction hooks ---
        # Order: every decoder_level_2 block, then every decoder_level_1 block.
        self.output_blocks = nn.ModuleList(
            list(self.decoder_level_2) + list(self.decoder_level_1)
        )

        # --- Output ---
        self.output_layer = nn.Conv2d(H, H, kernel_size=3, stride=1, padding=1)
        self.final_layer = ConvFinalLayer(H, self.out_channels)

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

        for t_emb in (self.t_embedder_1, self.t_embedder_2, self.t_embedder_3):
            nn.init.normal_(t_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(t_emb.mlp[2].weight, std=0.02)

        for stage in (
            self.encoder_level_1,
            self.encoder_level_2,
            self.latent,
            self.decoder_level_2,
            self.decoder_level_1,
        ):
            for block in stage:
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
        def _forward(*inputs):
            return module(*inputs)
        return _forward

    def _run_block(self, block, x, c):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._ckpt_wrapper(block), x, c, use_reentrant=False
            )
        return block(x, c)

    # ------------------------------------------------------------------
    def forward(self, x, timesteps, pathology_feat=None):
        """
        Parameters
        ----------
        x :
            ``(N, C_latent, H, W)`` latent tensor produced by the VAE encoder.
        timesteps :
            ``(N,)`` diffusion timesteps.
        pathology_feat :
            ``(N, pathology_feat_dim)`` pathology features, or ``None`` when
            pathology conditioning is disabled.
        """
        assert (pathology_feat is not None) == (
            self.pathology_projectors is not None
        ), "must specify pathology_feat iff pathology_feat_dim > 0"

        t1 = self.t_embedder_1(timesteps)
        t2 = self.t_embedder_2(timesteps)
        t3 = self.t_embedder_3(timesteps)
        if self.pathology_projectors is not None:
            c1 = t1 + self.pathology_projectors[0](pathology_feat)
            c2 = t2 + self.pathology_projectors[1](pathology_feat)
            c3 = t3 + self.pathology_projectors[2](pathology_feat)
        else:
            c1, c2, c3 = t1, t2, t3

        h = self.x_embedder(x)

        # encoder 1
        out_enc_level1 = h
        for block in self.encoder_level_1:
            out_enc_level1 = self._run_block(block, out_enc_level1, c1)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # encoder 2
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            out_enc_level2 = self._run_block(block, out_enc_level2, c2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # latent
        lat = inp_enc_level3
        for block in self.latent:
            lat = self._run_block(block, lat, c3)

        # decoder 2
        inp_dec_level2 = self.up3_2(lat)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        out_dec_level2 = self.reduce_chans_2(inp_dec_level2)
        for block in self.decoder_level_2:
            out_dec_level2 = self._run_block(block, out_dec_level2, c2)

        # decoder 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.reduce_chans_1(inp_dec_level1)
        for block in self.decoder_level_1:
            out_dec_level1 = self._run_block(block, out_dec_level1, c1)

        # output
        out = self.output_layer(out_dec_level1)
        out = self.final_layer(out, c1)
        return out


# ---------------------------------------------------------------------------
#  Size presets for the latent-space CMD.
# ---------------------------------------------------------------------------

CMD_VAE_CONFIGS = {
    "S":  {"hidden_size": 128, "depth": [2, 4, 8, 4, 2]},
    "B":  {"hidden_size": 256, "depth": [2, 4, 8, 4, 2]},
    "L":  {"hidden_size": 512, "depth": [2, 4, 8, 4, 2]},
    "XL": {"hidden_size": 512, "depth": [3, 6, 12, 6, 3]},
}


def build_cmd_vae_unet(
    cmd_size: str,
    in_channels: int = 4,
    learn_sigma: bool = False,
    pathology_feat_dim: int = 1024,
    use_checkpoint: bool = True,
    mlp_ratio: float = 3.0,
):
    if cmd_size not in CMD_VAE_CONFIGS:
        raise ValueError(
            f"Unknown cmd_size '{cmd_size}', "
            f"choose from {list(CMD_VAE_CONFIGS.keys())}"
        )
    cfg = CMD_VAE_CONFIGS[cmd_size]
    return CMDVAEUNet(
        in_channels=in_channels,
        hidden_size=cfg["hidden_size"],
        depth=cfg["depth"],
        mlp_ratio=mlp_ratio,
        learn_sigma=learn_sigma,
        pathology_feat_dim=pathology_feat_dim,
        use_checkpoint=use_checkpoint,
    )


# ---------------------------------------------------------------------------
#  Feature-extraction helpers (mirrors script_util.get_cmd_feature_info)
# ---------------------------------------------------------------------------

def get_cmd_vae_stage_info(cmd_size: str):
    """Last ``output_blocks`` index and channel count of each decoder stage.

    ``output_blocks`` is ``[dec2 blocks (2H), ..., dec1 blocks (H)]``.
    Stages are ordered coarsest to finest.
    """
    if cmd_size not in CMD_VAE_CONFIGS:
        raise ValueError(
            f"Unknown cmd_size '{cmd_size}', "
            f"choose from {list(CMD_VAE_CONFIGS.keys())}"
        )
    cfg = CMD_VAE_CONFIGS[cmd_size]
    H = cfg["hidden_size"]
    d_dec2, d_dec1 = cfg["depth"][3], cfg["depth"][4]

    blocks = [d_dec2 - 1, d_dec2 + d_dec1 - 1]
    channels = [H * 2, H]
    return blocks, channels


def get_cmd_vae_feature_info(cmd_size: str, use_all_blocks: bool = False):
    """Decoder block indices and the concatenated feature dimension.

    Selective mode keeps the last block of each decoder stage, plus the middle
    block when that stage has at least three blocks.
    """
    if cmd_size not in CMD_VAE_CONFIGS:
        raise ValueError(
            f"Unknown cmd_size '{cmd_size}', "
            f"choose from {list(CMD_VAE_CONFIGS.keys())}"
        )
    cfg = CMD_VAE_CONFIGS[cmd_size]
    H = cfg["hidden_size"]
    d_dec2, d_dec1 = cfg["depth"][3], cfg["depth"][4]

    blocks, block_channels = [], []

    offset = 0
    for d, ch in ((d_dec2, H * 2), (d_dec1, H)):
        if use_all_blocks:
            for bi in range(d):
                blocks.append(offset + bi)
                block_channels.append(ch)
        else:
            if d >= 3:
                blocks.append(offset + d // 2)
                block_channels.append(ch)
            blocks.append(offset + d - 1)
            block_channels.append(ch)
        offset += d

    return blocks, sum(block_channels)
