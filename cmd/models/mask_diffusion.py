import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
import math


class CMDMasking(nn.Module):
    def __init__(
            self,
            *,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            patch_size=8
    ):
        super().__init__()
        self.channels = channels
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.patch_size = patch_size

    def _scale_timesteps(self, t):
        return t.float() * (1000.0 / self.num_timesteps)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)  —— C 可为 3（像素域）或 4（VAE 潜空间）等。
        x:    (N, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        c = imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, channels=None):
        """
        x:    (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        c = channels if channels is not None else self.channels
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_img(self, img, mask_ratio):
        """
        perform random masking on image / latent map
        :param img: [1, C, H, W]，C 任意（3 = 像素域；4 = VAE 潜空间）
        :param mask_ratio: 0 < ratio < 1
        :return: masked imgs
        """
        channels = img.shape[1]
        x = self.patchify(img)
        N, L, D = x.shape

        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_patch = torch.zeros([N, L - x_masked.shape[1], D], device=x.device)
        x_masked = torch.cat([x_masked, mask_patch], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        img_masked = self.unpatchify(x_masked, channels=channels)
        return img_masked

    def q_sample(self, x_start, t):
        """
        批量前向过程：对 batch 中每个样本按不同 mask_ratio 进行遮蔽。
        由于每个样本的 mask_ratio 不同，需要逐样本处理。
        使用 torch.no_grad() 因为这一步不参与梯度计算。
        """
        with torch.no_grad():
            x_t = torch.zeros_like(x_start)
            for i in range(t.shape[0]):
                mask_ratio = t[i].item() / self.num_timesteps
                x_t[i] = self.random_masking_img(x_start[i].unsqueeze(0), mask_ratio)
        return x_t

    def training_losses(self, model, x_start, t, pathology_feat=None):
        """
        计算训练损失。返回逐样本损失（非标量），以支持 importance sampling。

        :param pathology_feat: (N, D) 病理大模型预提取特征，None 时为无条件生成。
        """
        terms = {}
        x_t = self.q_sample(x_start=x_start, t=t)
        output = model(x_t, t, pathology_feat=pathology_feat)

        B = x_start.shape[0]

        if self.loss_type == 'l1':
            # 逐样本 L1 损失: [B]
            terms["loss"] = (x_start - output).abs().view(B, -1).mean(dim=1)
        else:
            raise NotImplementedError()

        return terms




GaussianDiffusion = CMDMasking
