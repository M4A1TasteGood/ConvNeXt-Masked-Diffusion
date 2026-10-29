"""Confusion-matrix Dice, IoU, and pixel accuracy."""

from __future__ import annotations

import torch


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
        diagonal = torch.diag(self.confusion)
        ground_truth = self.confusion.sum(dim=1)
        predicted = self.confusion.sum(dim=0)
        union = ground_truth + predicted - diagonal
        iou = diagonal / union.clamp_min(1.0)
        dice = (2.0 * diagonal) / (ground_truth + predicted).clamp_min(1.0)
        accuracy = diagonal.sum() / self.confusion.sum().clamp_min(1.0)
        return {
            "mDice": float(dice.mean().item()),
            "mIoU": float(iou.mean().item()),
            "Pixel_Acc": float(accuracy.item()),
        }
