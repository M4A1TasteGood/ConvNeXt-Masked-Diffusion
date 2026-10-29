"""Patch-wise masked diffusion used by CMD pretraining and feature extraction."""

from __future__ import annotations

import torch
from torch import nn


class CMDMasking(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 3,
        timesteps: int = 1000,
        loss_type: str = "l1",
        patch_size: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.patch_size = patch_size

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() * (1000.0 / self.num_timesteps)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Pack ``(N, C, H, W)`` into ``(N, L, patch_size**2 * C)``.

        ``C`` is 3 in pixel space and 4 in VAE latent space.
        """
        p = self.patch_size
        if imgs.shape[2] != imgs.shape[3] or imgs.shape[2] % p != 0:
            raise ValueError(
                f"spatial size {tuple(imgs.shape[2:])} must be square and divisible by patch_size={p}"
            )

        c = imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * c)

    def unpatchify(self, x: torch.Tensor, channels: int | None = None) -> torch.Tensor:
        """Invert :meth:`patchify`."""
        p = self.patch_size
        c = self.channels if channels is None else channels
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"patch sequence length {x.shape[1]} is not a square")

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """Mask a fraction of tokens independently for each sample.

        ``x`` is ``[N, L, D]``. Returned mask uses 0 for keep and 1 for remove.
        """
        n, length, dim = x.shape
        len_keep = int(length * (1 - mask_ratio))

        noise = torch.rand(n, length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        mask = torch.ones([n, length], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def random_masking_img(self, img: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Randomly mask one image or latent map of shape ``[1, C, H, W]``."""
        channels = img.shape[1]
        x = self.patchify(img)
        n, length, dim = x.shape

        x_masked, _mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_tokens = torch.zeros([n, length - x_masked.shape[1], dim], device=x.device)
        x_masked = torch.cat([x_masked, mask_tokens], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dim))
        return self.unpatchify(x_masked, channels=channels)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mask each batch element at the ratio given by its timestep.

        Masking is not differentiated.
        """
        with torch.no_grad():
            x_t = torch.zeros_like(x_start)
            for i in range(t.shape[0]):
                mask_ratio = t[i].item() / self.num_timesteps
                x_t[i] = self.random_masking_img(x_start[i].unsqueeze(0), mask_ratio)
        return x_t

    def training_losses(self, model, x_start, t, pathology_feat=None):
        """Per-sample reconstruction loss, kept unreduced for importance sampling.

        ``pathology_feat`` is ``(N, D)``, or ``None`` for unconditional training.
        """
        x_t = self.q_sample(x_start=x_start, t=t)
        output = model(x_t, t, pathology_feat=pathology_feat)
        batch = x_start.shape[0]

        if self.loss_type == "l1":
            loss = (x_start - output).abs().view(batch, -1).mean(dim=1)
        else:
            raise NotImplementedError(f"unsupported loss_type: {self.loss_type}")
        return {"loss": loss}


GaussianDiffusion = CMDMasking
