"""
Training utilities for CMD mask diffusion.

Uses Hugging Face Accelerate for DDP + mixed precision (BF16).
Follows CMD training methodology:
  - AdamW optimizer (lr=1e-4, weight_decay=0)
  - EMA model via deepcopy (decay=0.9999)
  - BF16 mixed precision via Accelerate
  - Gradient checkpointing in the model
  - Pathology foundation model (UNI) conditioning via frozen feature extractor
"""

import os
import re
import logging
from copy import deepcopy
from collections import OrderedDict
from time import time

import torch
import torch.distributed as dist
from accelerate import Accelerator

from cmd.models.pathology_encoder import (
    build_pathology_encoder,
    extract_pathology_features,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir, rank=0):
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(out_dir):
    """Find the latest checkpoint in the output directory."""
    if not os.path.isdir(out_dir):
        return None
    checkpoints = []
    for name in os.listdir(out_dir):
        m = re.match(r"^(\d+)\.pt$", name)
        if m:
            step = int(m.group(1))
            if step > 0:
                checkpoints.append((step, os.path.join(out_dir, name)))
    if not checkpoints:
        return None
    checkpoints.sort()
    return checkpoints[-1][1]


def parse_step_from_filename(filename):
    """Parse step number from checkpoint filename."""
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    try:
        return int(name)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
#  TrainLoop
# ---------------------------------------------------------------------------

def _build_vae(device, model_name="stabilityai/sd-vae-ft-ema", model_dir=None,
               dtype=torch.bfloat16):
    """加载冻结的 SD-VAE（来自 ``diffusers``）。

    默认以 bf16 加载，避免每个 step 在 fp32 下做 256x256 的 VAE encode。
    """
    from diffusers.models import AutoencoderKL

    if model_dir:
        vae = AutoencoderKL.from_pretrained(model_dir, local_files_only=True)
    else:
        vae = AutoencoderKL.from_pretrained(model_name)
    vae = vae.to(device=device, dtype=dtype).eval()
    requires_grad(vae, False)
    return vae

class TrainLoop:
    """Accelerate-based training loop following CMD methodology.

    Parameters
    ----------
    model : nn.Module
        The CMD model with pathology conditioning.
    diffusion : GaussianDiffusion
        The mask-diffusion process.
    data : iterator
        Infinite data iterator yielding (batch, cond) tuples.
    lr : float
        Learning rate.
    total_steps : int
        Total number of training steps.
    out_dir : str
        Output directory for checkpoints and logs.
    log_interval : int
        Steps between log outputs.
    save_interval : int
        Steps between checkpoint saves.
    resume_checkpoint : str
        Path to checkpoint to resume from.
    ema_decay : float
        EMA decay rate.
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        lr=1e-4,
        total_steps=80000,
        out_dir="checkpoints",
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        ema_decay=0.9999,
    ):
        self.diffusion = diffusion
        self.data = data
        self.lr = lr
        self.total_steps = total_steps
        self.out_dir = out_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.ema_decay = ema_decay
        self.use_pathology_cond = (
            hasattr(model, "pathology_projectors")
            and model.pathology_projectors is not None
        )

        # --- backbone / VAE 元信息（由 script_util.create_model_and_diffusion 注入）---
        self.backbone = getattr(model, "backbone", "pixel")
        self.vae_scaling_factor = float(getattr(model, "vae_scaling_factor", 0.18215))
        self.vae_model_name = getattr(model, "vae_model_name", None)
        self.vae_model_dir = getattr(model, "vae_model_dir", None)
        self.pathology_model_name = getattr(model, "pathology_model_name", "uni")
        self.pathology_model_path = getattr(model, "pathology_model_path", "")

        # Disable checkpointing for the small latent-space model, where it is
        # typically slower than the memory it saves.
        if self.backbone == "vae":
            for m in [model, *model.modules()]:
                if getattr(m, "use_checkpoint", False):
                    m.use_checkpoint = False

        self.accelerator = Accelerator(mixed_precision="bf16")
        self.device = self.accelerator.device

        model = model.to(self.device)
        self.ema = deepcopy(model).to(self.device)
        requires_grad(self.ema, False)
        self.ema.eval()

        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

        self.train_steps = 0
        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint, model)

        self.model, self.opt = self.accelerator.prepare(model, self.opt)

        if self.train_steps == 0:
            update_ema(self.ema, self._unwrap_model(), decay=0)

        self.model.train()

        # --- Frozen pathology feature extractor ---
        self.pathology_model = None
        if self.use_pathology_cond:
            if self.accelerator.is_main_process:
                print(
                    f"Loading {self.pathology_model_name} pathology feature "
                    "extractor (bf16)..."
                )
            self.pathology_model = build_pathology_encoder(
                self.pathology_model_name,
                self.device,
                weights_path=self.pathology_model_path,
            )

        # --- Frozen VAE (only for latent-space CMD) ---
        self.vae = None
        if self.backbone == "vae":
            if self.accelerator.is_main_process:
                print("Loading frozen SD-VAE for latent-space diffusion (bf16)...")
            self.vae = _build_vae(
                self.device,
                model_name=self.vae_model_name,
                model_dir=self.vae_model_dir,
            )

        if self.accelerator.is_main_process:
            os.makedirs(out_dir, exist_ok=True)
            self.logger = create_logger(out_dir)
            total_params = sum(
                p.numel() for p in self._unwrap_model().parameters()
            )
            self.logger.info(f"CMD [{self.backbone}] Parameters: {total_params:,}")
            if self.pathology_model is not None:
                pathology_params = sum(
                    p.numel() for p in self.pathology_model.parameters()
                )
                self.logger.info(
                    f"{self.pathology_model_name} (frozen) Parameters: "
                    f"{pathology_params:,}"
                )
            if self.vae is not None:
                vae_params = sum(p.numel() for p in self.vae.parameters())
                self.logger.info(
                    f"VAE (frozen) Parameters: {vae_params:,} "
                    f"scaling={self.vae_scaling_factor}"
                )
        else:
            self.logger = None

    def _unwrap_model(self):
        return self.accelerator.unwrap_model(self.model)

    def _load_checkpoint(self, ckpt_path, model):
        """Load model, EMA, and optimizer from a checkpoint."""
        if not os.path.isfile(ckpt_path):
            return
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            model.load_state_dict(state["model"])
            self.ema.load_state_dict(state["ema"])
            if "opt" in state:
                self.opt.load_state_dict(state["opt"])
            if "step" in state:
                self.train_steps = state["step"]
            if self.accelerator.is_main_process:
                print(f"Resumed from {ckpt_path} at step {self.train_steps}")
        else:
            model.load_state_dict(state, strict=False)
            update_ema(self.ema, model, decay=0)
            if self.accelerator.is_main_process:
                print(f"Loaded weights from {ckpt_path} (no optimizer state)")

    def _save_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        checkpoint = {
            "model": self._unwrap_model().state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.train_steps,
        }
        path = os.path.join(self.out_dir, f"{self.train_steps:07d}.pt")
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

        ema_path = os.path.join(
            self.out_dir, f"ema_{self.ema_decay}_{self.train_steps:07d}.pt"
        )
        torch.save(self.ema.state_dict(), ema_path)
        self.logger.info(f"Saved EMA to {ema_path}")

    @torch.no_grad()
    def _vae_encode(self, images):
        """将像素图像（[-1, 1]）编码到潜空间，并按 SD-VAE scaling 缩放。

        VAE 在 bf16 下 eval 加载，这里把输入 cast 到同一精度，避免每个 step
        都做一次 fp32 → bf16 的隐式转换 / fp32 全量前向。
        """
        vae_dtype = next(self.vae.parameters()).dtype
        posterior = self.vae.encode(images.to(dtype=vae_dtype)).latent_dist
        latents = posterior.sample() * self.vae_scaling_factor
        return latents.to(dtype=torch.float32)

    @torch.no_grad()
    def _extract_pathology_feat(self, batch):
        """Extract pathology features from original images using frozen encoder.

        The encoder expects 224x224 ImageNet-normalised images, but the training
        batch is already normalised to [-1, 1], so preprocessing stays tensor-only.
        """
        return extract_pathology_features(
            self.pathology_model,
            batch,
            self.pathology_model_name,
        )

    def run_loop(self):
        running_loss = 0.0
        log_steps = 0
        start_time = time()

        if self.logger:
            self.logger.info(f"Training for {self.total_steps} steps...")

        while self.train_steps < self.total_steps:
            batch, cond = next(self.data)
            batch = batch.to(self.device, dtype=torch.float32, non_blocking=True)

            pathology_feat = None
            if self.pathology_model is not None:
                pathology_feat = self._extract_pathology_feat(batch)

            if self.vae is not None:
                batch = self._vae_encode(batch)

            t = torch.randint(
                0, self.diffusion.num_timesteps, (batch.shape[0],), device=self.device
            )

            loss_dict = self.diffusion.training_losses(
                self.model, batch, t, pathology_feat=pathology_feat
            )
            loss = loss_dict["loss"].mean()

            self.opt.zero_grad()
            self.accelerator.backward(loss)
            self.opt.step()
            update_ema(self.ema, self._unwrap_model(), decay=self.ema_decay)

            running_loss += loss.item()
            log_steps += 1
            self.train_steps += 1

            if self.train_steps % self.log_interval == 0 and self.logger:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed
                avg_loss = running_loss / log_steps
                self.logger.info(
                    f"(step={self.train_steps:07d}) "
                    f"Loss: {avg_loss:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if self.train_steps % self.save_interval == 0:
                self._save_checkpoint()

        if self.train_steps % self.save_interval != 0:
            self._save_checkpoint()

        if self.logger:
            self.logger.info("Training complete.")
