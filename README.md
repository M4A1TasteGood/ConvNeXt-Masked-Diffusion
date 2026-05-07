# CMD Reviewer Release

This repository contains the implementation of CMD, including
the pretraining code and the TNBC inference pipeline used in the submitted
study. 

## Repository Layout

- `cmd/`: shared CMD implementation, including model definitions, mask-diffusion
  training objective, pathology encoder wrappers, JSONL data loading, checkpoint
  utilities, and the training loop.
- `pretrain/`: command-line entry point for CMD pretraining from a JSONL image
  manifest.
- `downstream/`: segmentation-head fine-tuning on frozen CMD features from a
  JSON dataset with image and single-channel mask paths.
- `infer/`: TNBC semantic segmentation inference pipeline. It reuses the shared
  CMD package and keeps only the TNBC task-specific decoder and output logic.

## Environment

Use the provided `pfm_seg` environment or an equivalent Python environment with
PyTorch, torchvision, timm, accelerate, Pillow, and NumPy installed. VAE-backed
training additionally requires diffusers.

```bash
conda activate pfm_seg
cd /path/to/reviewer_release
export PYTHONPATH="$PWD"
```

## Data Manifest

Pretraining uses JSONL. Each line must be either a string image path or an object
with `image_path`:

```json
"/path/to/image_0001.png"
{"image_path": "/path/to/image_0002.png"}
```

## CMD Pretraining

Single-process smoke run:

```bash
python -m pretrain.train_cmd \
  --data-jsonl /path/to/train.jsonl \
  --out-dir outputs/pretrain_cmd_smoke \
  --cmd-size B \
  --backbone pixel \
  --pathology-model-name uni \
  --pathology-model-path /path/to/UNI/pytorch_model.bin \
  --batch-size 1 \
  --total-steps 1 \
  --num-workers 0 \
  --save-interval 1
```

Multi-GPU training:

```bash
accelerate launch --mixed_precision no --num_processes 4 pretrain/train_cmd.py \
  --data-jsonl /path/to/train.jsonl \
  --out-dir outputs/pretrain_cmd \
  --cmd-size B \
  --backbone pixel \
  --pathology-model-name uni \
  --pathology-model-path /path/to/UNI/pytorch_model.bin \
  --image-size 256 \
  --diffusion-steps 1000 \
  --patch-size 8 \
  --lr 3e-5 \
  --batch-size 32 \
  --total-steps 50000 \
  --num-workers 16 \
  --log-interval 100 \
  --save-interval 5000
```

The training script writes `log.txt`, `{step:07d}.pt`, and
`ema_0.9999_{step:07d}.pt` to the output directory. If `--resume-checkpoint` is
not specified, the latest numbered checkpoint in `--out-dir` is used
automatically.

## TNBC Inference

See `infer/README.md` for the task-specific segmentation pipeline. Required
weights are provided as local paths or environment variables; no private paths
are embedded in the code.

## Downstream Segmentation

See `downstream/README.md` for fine-tuning a segmentation head from a dataset
JSON containing `image_path` and `mask_path` entries. Masks must be
single-channel PNG files whose pixel values are class indices.
