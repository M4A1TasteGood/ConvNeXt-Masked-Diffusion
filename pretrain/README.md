# CMD Pretraining

This folder provides the CMD pretraining entry point. It implements the training
launcher with explicit command-line arguments and JSONL input.

## Input

`train_cmd.py` expects `--data-jsonl`. Each non-empty line must be one of:

```json
"/path/to/image.png"
{"image_path": "/path/to/image.png"}
```

Images are loaded as RGB, resized or randomly cropped to `--image-size`, randomly
flipped unless disabled, and normalized to `[-1, 1]`.

## Default Training Configuration

The default training configuration is:

- `--cmd-size B`
- `--backbone pixel`
- `--image-size 256`
- `--diffusion-steps 1000`
- `--patch-size 8` for pixel CMD, or `4` for VAE CMD
- `--lr 3e-5`
- `--batch-size 32`
- `--total-steps 50000`
- `--save-interval 5000`
- EMA decay `0.9999`

## Commands

Smoke test:

```bash
cd /path/to/cmd_repository
export PYTHONPATH="$PWD"
python -m pretrain.train_cmd \
  --data-jsonl /path/to/train.jsonl \
  --out-dir outputs/pretrain_cmd_smoke \
  --pathology-model-name uni \
  --pathology-model-path /path/to/UNI/pytorch_model.bin \
  --batch-size 1 \
  --total-steps 1 \
  --num-workers 0 \
  --save-interval 1
```

Multi-GPU run:

```bash
accelerate launch --mixed_precision no --num_processes 4 pretrain/train_cmd.py \
  --data-jsonl /path/to/train.jsonl \
  --out-dir outputs/pretrain_cmd \
  --pathology-model-name uni \
  --pathology-model-path /path/to/UNI/pytorch_model.bin
```

For H0-mini, set `--pathology-model-name h0-mini` and pass
`--pathology-model-path /path/to/H0-mini/pytorch_model.bin`. If the H0-mini
configuration is stored offline, keep `config.json` in the same directory or set
`H0_MINI_HUB_DIR`.

## Outputs

The output directory contains:

- `log.txt`: training log.
- `{step:07d}.pt`: model, EMA, optimizer, and step state.
- `ema_0.9999_{step:07d}.pt`: EMA model weights for downstream use.

The script resumes automatically from the latest numbered checkpoint in
`--out-dir` unless `--resume-checkpoint` is specified.
