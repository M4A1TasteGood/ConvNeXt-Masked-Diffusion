"""JSONL image dataset for CMD pretraining."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _get_rank_and_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _multiprocessing_context(num_workers: int):
    if num_workers is None or int(num_workers) <= 0:
        return None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        import multiprocessing as mp

        return mp.get_context("spawn")
    return None


def extract_image_path(record, manifest_path: str, line_no: int) -> str:
    if isinstance(record, str) and record:
        return record
    if isinstance(record, dict):
        for key in ("image_path", "path"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    raise ValueError(
        f"Invalid JSONL record at line {line_no} in {manifest_path}. "
        "Expected a string path or an object containing 'image_path' or 'path'."
    )


def read_jsonl_image_paths(manifest_path: str | Path) -> list[str]:
    manifest_path = str(manifest_path)
    paths: list[str] = []
    with open(manifest_path, "r", encoding="utf-8", buffering=64 * 1024 * 1024) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no} in {manifest_path}: {exc}") from exc
            paths.append(extract_image_path(record, manifest_path, line_no))
    if not paths:
        raise ValueError(f"No image paths found in JSONL file: {manifest_path}")
    return paths


class JsonlPretrainDataset(Dataset):
    """Load RGB images listed in a JSONL manifest for CMD pretraining."""

    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int,
        *,
        random_crop: bool = True,
        random_flip: bool = True,
        crop_prob: float = 0.8,
        shard: int = 0,
        num_shards: int = 1,
    ) -> None:
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.resolution = int(resolution)
        self.random_crop = bool(random_crop)
        self.random_flip = bool(random_flip)
        self.crop_prob = float(crop_prob)
        all_paths = read_jsonl_image_paths(manifest_path)
        self.image_paths = all_paths[shard::num_shards]
        if not self.image_paths:
            raise ValueError("This data shard is empty; reduce the number of processes or add more images.")
        print(f"[CMD data] loaded {len(self.image_paths)} images from JSONL manifest")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            return self._load_and_transform(path), {}
        except Exception:
            return self[random.randint(0, len(self) - 1)]

    def _load_and_transform(self, path: str):
        pil_image = Image.open(path).convert("RGB")
        w, h = pil_image.size
        if h < self.resolution or w < self.resolution:
            scale = max(self.resolution / h, self.resolution / w)
            pil_image = pil_image.resize(
                (round(w * scale), round(h * scale)), resample=Image.BICUBIC
            )

        if self.random_crop and random.random() < self.crop_prob:
            pil_image = T.RandomCrop(self.resolution)(pil_image)
        else:
            pil_image = T.Resize(
                (self.resolution, self.resolution),
                interpolation=T.InterpolationMode.BICUBIC,
            )(pil_image)

        arr = np.array(pil_image)
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1].copy()
        arr = arr.astype(np.float32) / 127.5 - 1.0
        return np.transpose(arr, [2, 0, 1])


def load_pretrain_data(
    *,
    data_jsonl: str | Path,
    batch_size: int,
    image_size: int,
    deterministic: bool = False,
    random_crop: bool = True,
    random_flip: bool = True,
    crop_prob: float = 0.8,
    num_workers: int = 8,
):
    rank, world_size = _get_rank_and_world()
    dataset = JsonlPretrainDataset(
        data_jsonl,
        image_size,
        random_crop=random_crop,
        random_flip=random_flip,
        crop_prob=crop_prob,
        shard=rank,
        num_shards=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context=_multiprocessing_context(num_workers),
        persistent_workers=bool(num_workers and num_workers > 0),
    )
    while True:
        yield from loader
