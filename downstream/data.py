"""JSON-manifest segmentation dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class JsonSegmentationDataset(Dataset):
    """Image and class-index mask pairs listed in a JSON manifest."""

    def __init__(
        self,
        json_path: str | Path,
        split: str,
        image_size: int,
        *,
        mode: str,
        random_flip: bool = True,
    ) -> None:
        self.json_path = Path(json_path).expanduser()
        self.split = split
        self.image_size = int(image_size)
        self.mode = mode
        self.random_flip = random_flip

        with self.json_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.num_classes = int(config.get("num_classes", 2))
        data = config.get("data", {})
        if split not in data:
            raise ValueError(f"Split '{split}' not found in {self.json_path}")
        self.samples = list(data[split])
        if not self.samples:
            raise ValueError(f"Split '{split}' is empty in {self.json_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        image_path = self._resolve_path(item["image_path"])
        mask_path = self._resolve_path(item["mask_path"])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        if self.mode == "train":
            image, mask = self._resize_if_needed(image, mask)
            image, mask = self._paired_random_crop(image, mask)
            if self.random_flip and random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            image = TF.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            mask = TF.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=TF.InterpolationMode.NEAREST,
            )

        image_tensor = TF.to_tensor(image) * 2.0 - 1.0
        mask_array = np.array(mask, dtype=np.uint8)
        invalid = (mask_array != 255) & (mask_array >= self.num_classes)
        if invalid.any():
            raise ValueError(
                f"Mask {mask_path} contains class values outside [0, {self.num_classes - 1}] "
                "and not equal to ignore_index 255."
            )
        mask_tensor = torch.from_numpy(mask_array).long()
        return image_tensor, mask_tensor, image_path.name

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (self.json_path.parent / path).resolve()

    def _resize_if_needed(self, image: Image.Image, mask: Image.Image):
        width, height = image.size
        if height >= self.image_size and width >= self.image_size:
            return image, mask
        scale = max(self.image_size / height, self.image_size / width)
        new_size = (round(width * scale), round(height * scale))
        image = image.resize(new_size, resample=Image.BICUBIC)
        mask = mask.resize(new_size, resample=Image.NEAREST)
        return image, mask

    def _paired_random_crop(self, image: Image.Image, mask: Image.Image):
        top, left, height, width = T.RandomCrop.get_params(image, (self.image_size, self.image_size))
        return TF.crop(image, top, left, height, width), TF.crop(mask, top, left, height, width)


def has_split(json_path: str | Path, split: str) -> bool:
    with Path(json_path).expanduser().open("r", encoding="utf-8") as handle:
        return bool(json.load(handle).get("data", {}).get(split))
