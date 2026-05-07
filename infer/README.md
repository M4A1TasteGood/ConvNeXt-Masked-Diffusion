# CMD TNBC Inference

This folder provides the reviewer-facing TNBC semantic segmentation inference
pipeline. The CMD backbone, mask-diffusion process, checkpoint loading, and
pathology encoder are imported from the shared `cmd` package at the repository
root. This folder contains only the TNBC task configuration, segmentation head,
image I/O, and command-line orchestration.

## Structure

- `tnbc_infer.py`: command-line entry point.
- `cmd_model/config.py`: released TNBC inference configuration.
- `cmd_model/features.py`: CMD feature extraction for TNBC segmentation.
- `cmd_model/head.py`: TNBC segmentation decoder.
- `cmd_model/segmentor.py`: end-to-end CMD feature extractor plus TNBC decoder.
- `cmd_model/io.py`: image loading, prediction saving, overlays, and anonymous
  output metadata.
- `cmd_model/inference.py`: CLI orchestration.

## Checkpoints

Prepare three local files:

- CMD pretrained EMA checkpoint, for example `ema_0.9999_0080000.pt`.
- TNBC segmentation head checkpoint, for example `best_model.pth`.
- H0-mini pathology encoder weights, usually `pytorch_model.bin`.

For offline H0-mini use, keep `config.json` in the same directory as
`pytorch_model.bin`, or set `H0_MINI_HUB_DIR` to that directory.

## Usage

Run from this directory:

```bash
python tnbc_infer.py \
  --input /path/to/tnbc/image_or_folder \
  --output-dir outputs/tnbc_inference \
  --cmd-checkpoint /path/to/ema_0.9999_0080000.pt \
  --tnbc-head-checkpoint /path/to/best_model.pth \
  --pathology-model-path /path/to/H0-mini/pytorch_model.bin
```

If weights are placed in the default release layout, the explicit checkpoint
arguments can be omitted:

- `weight/CMD-L/ema_0.9999_0080000.pt`
- `weight/SegHead/best_model.pth`
- `weight/H0-mini/pytorch_model.bin`

Environment variables are also supported:

```bash
export CMD_CHECKPOINT=/path/to/ema_0.9999_0080000.pt
export TNBC_HEAD_CHECKPOINT=/path/to/best_model.pth
export H0_MINI_WEIGHTS=/path/to/H0-mini/pytorch_model.bin
python tnbc_infer.py --input /path/to/tnbc/image_or_folder
```

## Outputs

For each input image, the script writes:

- `*_mask.png`: 256 x 256 predicted label mask.
- `*_mask_vis.png`: 256 x 256 visualization mask.
- `*_mask_original_size.png`: predicted label mask resized to the original image
  size.
- `*_overlay.png`: red overlay for the TNBC foreground class.
- `*_prob.npy`: optional per-class probabilities when `--save-prob` is enabled.
- `manifest.json`: anonymous run metadata with checkpoint file names, output file
  names, class names, and foreground ratios only.
