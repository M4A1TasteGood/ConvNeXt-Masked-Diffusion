# CMD Downstream Segmentation

This folder fine-tunes a lightweight convolutional segmentation head on frozen
CMD features. The CMD backbone and the pathology encoder remain frozen; only
`best_model.pth`, the segmentation head checkpoint, is trained.

## Dataset JSON

The training entry point accepts one dataset JSON through `--data-json`. The
file must contain the number of semantic classes and the `train`, `val`, and
`test` splits:

```json
{
  "num_classes": 2,
  "data": {
    "train": [
      {
        "image_path": "images/train_0001.png",
        "mask_path": "masks/train_0001.png"
      }
    ],
    "val": [
      {
        "image_path": "images/val_0001.png",
        "mask_path": "masks/val_0001.png"
      }
    ],
    "test": [
      {
        "image_path": "images/test_0001.png",
        "mask_path": "masks/test_0001.png"
      }
    ]
  }
}
```

Paths may be absolute or relative to the directory containing the JSON file.
If `val` is omitted or empty, the script uses the `test` split for validation;
`test` is still required for the final summary.

## Mask Format

Masks must be single-channel PNG images. The pixel value is the semantic class
index:

- `0`: background.
- `1 ... num_classes - 1`: foreground classes.
- `255`: ignored pixels, excluded from cross-entropy, Dice loss, and metrics.

Use nearest-neighbor interpolation when preparing masks. Do not store RGB color
maps as masks unless the colors have already been converted to class indices.
The script converts non-`L` PNGs to single-channel grayscale, but the resulting
values must still be valid class indices.

## Training

Run from the repository root:

```bash
cd /path/to/cmd_repository
export PYTHONPATH="$PWD"

python downstream/train_seg_head.py \
  --data-json /path/to/dataset.json \
  --output-dir outputs/downstream_glas \
  --cmd-checkpoint /path/to/ema_0.9999_0080000.pt \
  --cmd-size L \
  --pathology-model-name h0-mini \
  --pathology-model-path /path/to/H0-mini/pytorch_model.bin \
  --image-size 256 \
  --diffusion-step 50 \
  --batch-size 4 \
  --epochs 150 \
  --lr 2e-4
```

For UNI conditioning, use `--pathology-model-name uni` and provide the local UNI
`pytorch_model.bin`. To reproduce a no-pathology ablation-style run without a
pathology encoder, pass `--disable-pathology`. By default, pathology
conditioning is enabled.

## Outputs

The output directory contains:

- `best_model.pth`: the best validation segmentation head checkpoint.
- `summary.json`: best validation Dice and final test metrics.
- `predictions/`: optional test prediction PNGs when `--save-predictions` is
  enabled.

`best_model.pth` is compatible with the inference head layout when the same CMD
feature dimension and number of classes are used.

## Smoke Test

For a code-path smoke test without CMD weights, use `--allow-random-cmd` and
`--disable-pathology` on a tiny synthetic dataset. This is only for checking that
the data loader, feature extraction, optimizer, and checkpoint writing work; it
is not a meaningful model evaluation.
