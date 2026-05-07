# CMD TNBC Inference

This folder provides a clean TNBC semantic segmentation inference pipeline with CMD features and a trained TNBC segmentation head. It is intended to be run directly from this directory.

## Structure

- `tnbc_infer.py`: command-line entry point.
- `cmd_tnbc/config.py`: released TNBC inference configuration.
- `cmd_tnbc/cmd_model.py`: CMD backbone definition.
- `cmd_tnbc/diffusion.py`: CMD masking process.
- `cmd_tnbc/features.py`: CMD instantiation, checkpoint loading, and feature extraction.
- `cmd_tnbc/head.py`: TNBC segmentation decoder.
- `cmd_tnbc/pathology_encoder.py`: H0-mini feature encoder wrapper.
- `cmd_tnbc/segmentor.py`: end-to-end CMD + TNBC model wrapper.
- `cmd_tnbc/io.py`: image loading, mask saving, overlays, and anonymous metadata.
- `cmd_tnbc/inference.py`: CLI orchestration.

## Requirements

Install Python dependencies compatible with your CUDA/PyTorch setup:

```bash
pip install torch torchvision timm pillow numpy
```

## Checkpoints

Prepare three local files before running:

- CMD pretrained checkpoint: `ema_0.9999_0080000.pt`
- TNBC segmentation head checkpoint: `best_model.pth`
- H0-mini pathology encoder weights: `pytorch_model.bin`

For H0-mini, keep `config.json` in the same directory as `pytorch_model.bin`, or set `H0_MINI_HUB_DIR` to that directory.

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

The checkpoint paths can also be provided with environment variables:

```bash
export CMD_CHECKPOINT=/path/to/ema_0.9999_0080000.pt
export TNBC_HEAD_CHECKPOINT=/path/to/best_model.pth
export H0_MINI_WEIGHTS=/path/to/H0-mini/pytorch_model.bin
python tnbc_infer.py --input /path/to/tnbc/image_or_folder
```

## Outputs

For each input image, the script writes:

- `*_mask.png`: 256 x 256 predicted label mask.
- `*_mask_original_size.png`: predicted label mask resized back to the original image size.
- `*_overlay.png`: red overlay for the TNBC foreground class.
- `*_prob.npy`: optional per-class probabilities when `--save-prob` is enabled.
- `manifest.json`: anonymous run metadata with checkpoint file names only.
# ConvNeXt-Masked-Diffusion
