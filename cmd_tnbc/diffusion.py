"""Masking process used by CMD feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn


class CMDMasking(nn.Module):
    def __init__(self, *, channels: int = 3, timesteps: int = 1000, patch_size: int = 8):
        super().__init__()
        self.channels = channels
        self.num_timesteps = int(timesteps)
        self.patch_size = patch_size

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        return t.float() * (1000.0 / self.num_timesteps)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        if images.shape[2] != images.shape[3] or images.shape[2] % patch != 0:
            raise ValueError("CMD expects square images divisible by patch_size")
        h = w = images.shape[2] // patch
        x = images.reshape(images.shape[0], self.channels, h, patch, w, patch)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(images.shape[0], h * w, patch**2 * self.channels)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        h = w = int(patches.shape[1] ** 0.5)
        if h * w != patches.shape[1]:
            raise ValueError("number of patches must be a square")
        x = patches.reshape(patches.shape[0], h, w, patch, patch, self.channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(patches.shape[0], self.channels, h * patch, h * patch)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = x.shape
        len_keep = int(length * (1 - mask_ratio))
        noise = torch.rand(batch, length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        return x_masked, ids_restore

    def random_masking_img(self, image: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        x = self.patchify(image)
        x_masked, ids_restore = self.random_masking(x, mask_ratio)
        mask_patch = torch.zeros(
            x.shape[0],
            x.shape[1] - x_masked.shape[1],
            x.shape[2],
            device=x.device,
        )
        x_masked = torch.cat([x_masked, mask_patch], dim=1)
        x_masked = torch.gather(
            x_masked,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        return self.unpatchify(x_masked)

    @torch.no_grad()
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_t = torch.zeros_like(x_start)
        for idx in range(t.shape[0]):
            mask_ratio = t[idx].item() / self.num_timesteps
            x_t[idx] = self.random_masking_img(x_start[idx].unsqueeze(0), mask_ratio)
        return x_t
