"""Train a segmentation head on frozen CMD features."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from inference.head import SegmentationHead
from inference.utils import resolve_device, set_seed

from .config import DownstreamConfig
from .data import JsonSegmentationDataset, has_split
from .features import build_feature_extractor
from .losses import CombinedLoss
from .metrics import SegmentationMetrics


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def resolve_config_paths(config: DownstreamConfig, config_path: str | Path) -> DownstreamConfig:
    base_dir = Path(config_path).expanduser().resolve().parent
    config.data.json = _resolve_path(config.data.json, base_dir)
    config.model.cmd_checkpoint = _resolve_path(config.model.cmd_checkpoint, base_dir)
    config.model.pathology_weights = _resolve_path(config.model.pathology_weights, base_dir)
    config.train.output_dir = _resolve_path(config.train.output_dir, base_dir)
    return config


def build_loaders(config: DownstreamConfig):
    data_cfg = config.data
    model_cfg = config.model
    train_set = JsonSegmentationDataset(
        data_cfg.json,
        "train",
        model_cfg.image_size,
        mode="train",
        random_flip=data_cfg.random_flip,
    )
    val_split = "val" if has_split(data_cfg.json, "val") else "test"
    val_set = JsonSegmentationDataset(data_cfg.json, val_split, model_cfg.image_size, mode="val")
    test_set = JsonSegmentationDataset(data_cfg.json, "test", model_cfg.image_size, mode="test")

    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=data_cfg.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=data_cfg.num_workers)
    return train_loader, val_loader, test_loader, train_set.num_classes


def evaluate(model, feature_extractor, loader, criterion, device, num_classes: int, ignore_index: int):
    model.eval()
    metrics = SegmentationMetrics(num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, masks, _names in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(feature_extractor(images))
            loss = criterion(logits, masks)
            total_loss += float(loss.item())
            count += 1
            metrics.update(logits.cpu(), masks.cpu())
    return total_loss / max(count, 1), metrics.compute()


def save_predictions(model, feature_extractor, loader, output_dir: Path, device: torch.device) -> None:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for images, _masks, names in loader:
            images = images.to(device)
            logits = model(feature_extractor(images))
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for pred, name in zip(preds, names):
                Image.fromarray(pred).save(pred_dir / f"{Path(name).stem}_pred.png")


def _validate_paths(config: DownstreamConfig) -> None:
    data_json = Path(config.data.json).expanduser()
    if not data_json.is_file():
        raise FileNotFoundError(f"data.json not found: {data_json}")
    cmd_checkpoint = (
        Path(config.model.cmd_checkpoint).expanduser() if config.model.cmd_checkpoint else None
    )
    if cmd_checkpoint is None and not config.model.allow_random_cmd:
        raise FileNotFoundError("model.cmd_checkpoint is required unless model.allow_random_cmd is true.")
    if cmd_checkpoint is not None and not cmd_checkpoint.exists():
        raise FileNotFoundError(f"model.cmd_checkpoint not found: {cmd_checkpoint}")
    pathology_weights = Path(config.model.pathology_weights).expanduser() if config.model.pathology_weights else None
    if (
        not config.model.disable_pathology
        and pathology_weights is not None
        and not pathology_weights.is_file()
    ):
        raise FileNotFoundError(f"model.pathology_weights not found: {pathology_weights}")


def run_training(config: DownstreamConfig, *, config_path: str | Path) -> dict[str, float]:
    config = resolve_config_paths(config, config_path)
    _validate_paths(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    output_dir = Path(config.train.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, num_classes = build_loaders(config)
    feature_extractor, feature_dim = build_feature_extractor(config.model, device)
    model = SegmentationHead(in_channels=feature_dim, num_classes=num_classes).to(device)
    criterion = CombinedLoss(
        ce_weight=config.loss.ce_weight,
        dice_weight=config.loss.dice_weight,
        ignore_index=config.loss.ignore_index,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.train.epochs, 1),
        eta_min=config.train.lr * 1e-3,
    )

    best_mdice = -math.inf
    best_epoch = 0
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for images, masks, _names in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                features = feature_extractor(images)
            logits = model(features)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            if config.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
        scheduler.step()
        train_loss = total_loss / max(count, 1)

        if epoch % config.train.val_interval == 0 or epoch == config.train.epochs:
            val_loss, val_metrics = evaluate(
                model,
                feature_extractor,
                val_loader,
                criterion,
                device,
                num_classes,
                config.loss.ignore_index,
            )
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"mDice={val_metrics['mDice']:.4f} mIoU={val_metrics['mIoU']:.4f}"
            )
            if val_metrics["mDice"] > best_mdice:
                best_mdice = val_metrics["mDice"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "num_classes": num_classes,
                        "feature_dim": feature_dim,
                        "cmd_size": config.model.cmd_size,
                    },
                    output_dir / "best_model.pth",
                )
        else:
            print(f"epoch={epoch:04d} train_loss={train_loss:.4f}")

    best_path = output_dir / "best_model.pth"
    if best_path.is_file():
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = evaluate(
        model,
        feature_extractor,
        test_loader,
        criterion,
        device,
        num_classes,
        config.loss.ignore_index,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_mDice": best_mdice,
        "test_loss": test_loss,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if config.train.save_predictions:
        save_predictions(model, feature_extractor, test_loader, output_dir, device)
    print(json.dumps(summary, indent=2))
    return summary
