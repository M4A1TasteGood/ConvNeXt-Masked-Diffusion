"""Fine-tune a segmentation head on frozen CMD features."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    loaded_cmd = sys.modules.get("cmd")
    if loaded_cmd is not None and not hasattr(loaded_cmd, "__path__"):
        del sys.modules["cmd"]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cmd.models.factory import create_cmd_model, get_cmd_feature_info
from cmd.models.mask_diffusion import CMDMasking
from cmd.models.pathology_encoder import (
    build_pathology_encoder,
    extract_pathology_features,
    get_pathology_feat_dim,
    normalize_pathology_model_name,
)
from cmd.runtime.checkpoints import load_cmd_state_dict
from cmd.runtime.utils import configure_pathology_encoder, resolve_device, set_seed


class JsonSegmentationDataset(Dataset):
    """Segmentation dataset backed by a JSON manifest."""

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
        w, h = image.size
        if h >= self.image_size and w >= self.image_size:
            return image, mask
        scale = max(self.image_size / h, self.image_size / w)
        new_size = (round(w * scale), round(h * scale))
        image = image.resize(new_size, resample=Image.BICUBIC)
        mask = mask.resize(new_size, resample=Image.NEAREST)
        return image, mask

    def _paired_random_crop(self, image: Image.Image, mask: Image.Image):
        i, j, h, w = T.RandomCrop.get_params(image, (self.image_size, self.image_size))
        return TF.crop(image, i, j, h, w), TF.crop(mask, i, j, h, w)


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        super().__init__(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels, 3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Module):
    """Convolutional decoder trained on frozen CMD feature maps."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 16),
    ) -> None:
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(in_channels, head_channels, 3, padding=1)
        in_chs = [head_channels] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList(
            DecoderBlock(in_ch, out_ch) for in_ch, out_ch in zip(in_chs, decoder_channels)
        )
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=3, padding=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_more(x)
        for block in self.blocks:
            x = block(x)
        return self.seg_head(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        valid = target != self.ignore_index
        target_clamped = target.clone()
        target_clamped[~valid] = 0
        target_one_hot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        target_one_hot = target_one_hot * valid
        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).detach().cpu()
        target = target.detach().cpu()
        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        if target.numel() == 0:
            return
        bins = self.num_classes * target + pred
        hist = torch.bincount(bins, minlength=self.num_classes ** 2)
        self.confusion += hist.reshape(self.num_classes, self.num_classes).double()

    def compute(self) -> dict[str, float]:
        diag = torch.diag(self.confusion)
        gt = self.confusion.sum(dim=1)
        pred = self.confusion.sum(dim=0)
        union = gt + pred - diag
        iou = diag / union.clamp_min(1.0)
        dice = (2.0 * diag) / (gt + pred).clamp_min(1.0)
        acc = diag.sum() / self.confusion.sum().clamp_min(1.0)
        return {
            "mDice": float(dice.mean().item()),
            "mIoU": float(iou.mean().item()),
            "Pixel_Acc": float(acc.item()),
        }


class CMDFeatureExtractor(nn.Module):
    """Frozen CMD feature extractor for downstream segmentation."""

    def __init__(
        self,
        *,
        cmd_checkpoint: str,
        cmd_size: str,
        image_size: int,
        feature_blocks: list[int],
        diffusion_step: int,
        pathology_model_name: str,
        pathology_model_path: str,
        pathology_feat_dim: int,
        in_channels: int = 3,
        patch_size: int = 8,
        allow_random_cmd: bool = False,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.feature_blocks = tuple(feature_blocks)
        self.diffusion_step = int(diffusion_step)
        self.pathology_model_name = pathology_model_name
        self.use_pathology = pathology_feat_dim > 0
        self.cmd = create_cmd_model(
            image_size=image_size,
            in_channels=in_channels,
            pathology_feat_dim=pathology_feat_dim,
            cmd_size=cmd_size,
            use_checkpoint=False,
        )
        if cmd_checkpoint:
            missing, unexpected = self.cmd.load_state_dict(load_cmd_state_dict(cmd_checkpoint), strict=False)
            if unexpected:
                print(f"[CMD downstream] warning: skipped unexpected CMD checkpoint keys: {len(unexpected)}")
            if missing:
                print(f"[CMD downstream] warning: missing CMD checkpoint keys: {len(missing)}")
        elif not allow_random_cmd:
            raise FileNotFoundError("--cmd-checkpoint is required unless --allow-random-cmd is set.")
        self.cmd.to(device).eval()
        for param in self.cmd.parameters():
            param.requires_grad = False

        self.diffusion = CMDMasking(channels=in_channels, timesteps=1000, patch_size=patch_size)
        self.hooks = []
        self.activations: list[torch.Tensor] = []
        for idx, block in enumerate(self.cmd.output_blocks):
            if idx in self.feature_blocks:
                self.hooks.append(block.register_forward_hook(self._save_output))

        self.pathology_model = None
        if self.use_pathology:
            if not pathology_model_path:
                raise FileNotFoundError("--pathology-model-path is required when pathology conditioning is enabled.")
            configure_pathology_encoder(pathology_model_path)
            self.pathology_model = build_pathology_encoder(
                pathology_model_name,
                device,
                weights_path=pathology_model_path,
            )

    def _save_output(self, _module, _inputs, output) -> None:
        self.activations.append(output.detach().float())

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.activations = []
        pathology_feat = None
        if self.pathology_model is not None:
            pathology_feat = extract_pathology_features(
                self.pathology_model,
                image,
                self.pathology_model_name,
            )

        t = torch.full((image.shape[0],), self.diffusion_step, device=image.device, dtype=torch.long)
        masked_image = self.diffusion.q_sample(image, t)
        self.cmd(masked_image, self.diffusion._scale_timesteps(t), pathology_feat=pathology_feat)
        if len(self.activations) != len(self.feature_blocks):
            raise RuntimeError("Failed to collect all requested CMD feature blocks.")
        resized = [
            F.interpolate(feat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            for feat in self.activations
        ]
        features = torch.cat(resized, dim=1)
        norms = torch.linalg.vector_norm(features.flatten(1), dim=1).clamp_min(1e-8)
        return features / norms[:, None, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a segmentation head on frozen CMD features.")
    parser.add_argument("--data-json", required=True, help="Segmentation dataset JSON path.")
    parser.add_argument("--output-dir", default="outputs/downstream_seg")
    parser.add_argument("--cmd-checkpoint", default="")
    parser.add_argument("--allow-random-cmd", action="store_true", help="Use random CMD weights for smoke tests only.")
    parser.add_argument("--cmd-size", default="L", choices=["S", "B", "L", "XL"])
    parser.add_argument("--pathology-model-name", default="h0-mini")
    parser.add_argument("--pathology-model-path", default="")
    parser.add_argument("--disable-pathology", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--diffusion-step", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def build_loaders(args: argparse.Namespace):
    train_set = JsonSegmentationDataset(args.data_json, "train", args.image_size, mode="train")
    val_split = "val" if _has_split(args.data_json, "val") else "test"
    val_set = JsonSegmentationDataset(args.data_json, val_split, args.image_size, mode="val")
    test_set = JsonSegmentationDataset(args.data_json, "test", args.image_size, mode="test")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return train_loader, val_loader, test_loader, train_set.num_classes


def _has_split(json_path: str | Path, split: str) -> bool:
    with Path(json_path).expanduser().open("r", encoding="utf-8") as handle:
        return bool(json.load(handle).get("data", {}).get(split))


def build_feature_extractor(args: argparse.Namespace, device: torch.device):
    pathology_model_name = normalize_pathology_model_name(args.pathology_model_name)
    pathology_feat_dim = 0 if args.disable_pathology else get_pathology_feat_dim(pathology_model_name)
    blocks, feature_dim = get_cmd_feature_info(args.cmd_size, args.image_size)
    extractor = CMDFeatureExtractor(
        cmd_checkpoint=args.cmd_checkpoint,
        cmd_size=args.cmd_size,
        image_size=args.image_size,
        feature_blocks=blocks,
        diffusion_step=args.diffusion_step,
        pathology_model_name=pathology_model_name,
        pathology_model_path=args.pathology_model_path,
        pathology_feat_dim=pathology_feat_dim,
        patch_size=args.patch_size,
        allow_random_cmd=args.allow_random_cmd,
        device=device,
    )
    return extractor.to(device).eval(), feature_dim


def evaluate(model, feature_extractor, loader, criterion, device, num_classes: int) -> tuple[float, dict[str, float]]:
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, masks, _names in loader:
            images = images.to(device)
            masks = masks.to(device)
            features = feature_extractor(images)
            logits = model(features)
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_json = Path(args.data_json).expanduser()
    if not data_json.is_file():
        raise FileNotFoundError(f"--data-json not found: {data_json}")
    if args.cmd_checkpoint and not Path(args.cmd_checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"--cmd-checkpoint not found: {args.cmd_checkpoint}")
    if args.pathology_model_path and not Path(args.pathology_model_path).expanduser().is_file():
        raise FileNotFoundError(f"--pathology-model-path not found: {args.pathology_model_path}")

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, num_classes = build_loaders(args)
    feature_extractor, feature_dim = build_feature_extractor(args, device)
    model = SegmentationHead(feature_dim, num_classes).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 1e-3)

    best_mdice = -math.inf
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            count += 1
        scheduler.step()
        train_loss = total_loss / max(count, 1)

        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_loss, val_metrics = evaluate(model, feature_extractor, val_loader, criterion, device, num_classes)
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
                        "cmd_size": args.cmd_size,
                    },
                    output_dir / "best_model.pth",
                )
        else:
            print(f"epoch={epoch:04d} train_loss={train_loss:.4f}")

    best_path = output_dir / "best_model.pth"
    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = evaluate(model, feature_extractor, test_loader, criterion, device, num_classes)
    summary = {
        "best_epoch": best_epoch,
        "best_val_mDice": best_mdice,
        "test_loss": test_loss,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if args.save_predictions:
        save_predictions(model, feature_extractor, test_loader, output_dir, device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
