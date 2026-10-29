# Beyond ViT Tokens: Masked-Diffusion Pretrained Convolutional Pathology Foundation Model for Cell-Level Dense Prediction (NeurIPS 2026)

<p align="left">
  <a href="https://arxiv.org/abs/2605.08276"><img src="https://img.shields.io/badge/arXiv-2605.08276-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/M4A1TasteGood/ConvNeXt_Masked_Diffusion"><img src="https://img.shields.io/badge/Hugging%20Face-Weights-ffd21e.svg?logo=huggingface&logoColor=000" alt="Hugging Face Weights"></a>
  <a href="https://arxiv.org/abs/2605.08276"><img src="https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg" alt="NeurIPS 2026"></a>
</p>

Official code for **CMD** (ConvNeXt Masked-Diffusion), a convolutional pathology foundation model for cell-level dense prediction. CMD pretrains a ConvNeXt U-Net with masked diffusion in pixel space, conditions it on a frozen pathology encoder, and keeps that backbone frozen at downstream time. Only a lightweight segmentation head is trained for each dense-prediction task.

**NeurIPS 2026.** Paper: [arXiv:2605.08276](https://arxiv.org/abs/2605.08276). Released weights: [M4A1TasteGood/ConvNeXt_Masked_Diffusion](https://huggingface.co/M4A1TasteGood/ConvNeXt_Masked_Diffusion).

## What this repository contains

| Path | Role |
| --- | --- |
| `cmd/`, `pretrain/` | Masked-diffusion pretraining |
| `train_downstream.py`, `downstream/`, `configs/` | Fine-tune a segmentation head on frozen CMD features |
| `infer.py`, `inference/` | Inference with a trained head |
| `assets/test.png` | Example 512×512 image used by the inference command below |

Run every command from the repository root. The entry points add that root to `sys.path`. Do not install this tree as a package named `cmd`: the name overlaps Python's standard-library `cmd` module.

## Setup

Python 3.10+, PyTorch, torchvision, timm, NumPy, Pillow, and PyYAML. Pretraining also needs Accelerate. Latent-space pretraining (`--backbone vae`) additionally needs `diffusers`.

```bash
pip install -r requirements.txt
```

## Weights

Download the released CMD-L checkpoint and the TNBC segmentation head from Hugging Face:

[https://huggingface.co/M4A1TasteGood/ConvNeXt_Masked_Diffusion](https://huggingface.co/M4A1TasteGood/ConvNeXt_Masked_Diffusion)

```text
CMD-L/pytorch_model.bin
TNBC_SegHead/best_model.pth
```

`CMD-L` is the pixel-space CMD-L model conditioned on [H0-mini](https://huggingface.co/bioptimus/H0-mini) (768-d). H0-mini is not redistributed here. Download it separately and keep `config.json` in the same directory as `pytorch_model.bin`, or set `H0_MINI_HUB_DIR` to that directory.

The released TNBC head expects CMD-L features at `image_size=256` and `diffusion_step=50`. A training checkpoint such as `0005000.pt` may also be passed to `--cmd`: when both `ema` and `model` are present, the EMA weights are loaded.

## Inference

```bash
python infer.py \
  --image assets/test.png \
  --cmd /path/to/CMD-L \
  --seg-head /path/to/TNBC_SegHead/best_model.pth \
  --pathology /path/to/H0-mini/pytorch_model.bin \
  --output-dir outputs/tnbc
```

`--image` can be a file or a directory. `--cmd` can be `pytorch_model.bin` or the directory that contains it. Each image writes:

- `*_mask_vis.png`: predicted foreground drawn in white
- `*_overlay.png`: red overlay on the original image

## Downstream training

Edit `configs/downstream.yaml`. Paths inside the YAML are resolved relative to the YAML file. Point `data.json` at a manifest like this:

```json
{
  "num_classes": 2,
  "data": {
    "train": [{"image_path": "images/train_0001.png", "mask_path": "masks/train_0001.png"}],
    "val":   [{"image_path": "images/val_0001.png", "mask_path": "masks/val_0001.png"}],
    "test":  [{"image_path": "images/test_0001.png", "mask_path": "masks/test_0001.png"}]
  }
}
```

Image and mask paths are absolute, or relative to the directory that contains the JSON file. Masks are single-channel class indices: `0` is background, `1 ... num_classes - 1` are foreground classes, and `255` is ignored by the loss and the metrics. If `val` is missing or empty, `test` is used for validation. `test` is still required for the final summary.

```bash
python train_downstream.py --config configs/downstream.yaml
```

CMD and the pathology encoder stay frozen. The run writes:

- `best_model.pth`: segmentation head with the best validation mDice
- `summary.json`: best epoch, validation mDice, and test metrics
- `predictions/`: only when `train.save_predictions` is `true`

`best_model.pth` can be passed to `infer.py --seg-head` when the CMD size, feature layout, and class count match the training run. Set `model.cmd_size` to `S`, `B`, `L`, or `XL`. The released weights and TNBC head use `L`.

## Pretraining

Pretraining reads a JSONL manifest. Each non-empty line is an image path or an object with `image_path` (or `path`):

```json
"/path/to/image_0001.png"
{"image_path": "/path/to/image_0002.png"}
```

Images are loaded as RGB, resized or randomly cropped to `--image-size`, flipped unless disabled, and scaled to `[-1, 1]`.

```bash
python -m pretrain.train_cmd \
  --data-jsonl /path/to/train.jsonl \
  --out-dir outputs/pretrain_cmd \
  --cmd-size L \
  --backbone pixel \
  --pathology-model-name h0-mini \
  --pathology-model-path /path/to/H0-mini/pytorch_model.bin \
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

For UNI conditioning, set `--pathology-model-name uni` and pass a local `pytorch_model.bin`. Multi-GPU training uses the same script:

```bash
accelerate launch --mixed_precision bf16 --num_processes 4 pretrain/train_cmd.py \
  --data-jsonl /path/to/train.jsonl \
  --out-dir outputs/pretrain_cmd \
  --cmd-size L \
  --backbone pixel \
  --pathology-model-name h0-mini \
  --pathology-model-path /path/to/H0-mini/pytorch_model.bin
```

The training loop runs in bfloat16 through Accelerate. The output directory contains `log.txt`, `{step:07d}.pt` (model, EMA, optimizer, step), and `ema_0.9999_{step:07d}.pt`. If `--resume-checkpoint` is omitted, training resumes from the latest numbered checkpoint in `--out-dir`.

`--backbone vae` trains in the latent space of `stabilityai/sd-vae-ft-ema` and expects `diffusers`. The default patch size is then 4. The released CMD-L checkpoint is the pixel-space model.

## Citation

```bibtex
@misc{chen2026vittokensmaskeddiffusionpretrained,
      title={Beyond ViT Tokens: Masked-Diffusion Pretrained Convolutional Pathology Foundation Model for Cell-Level Dense Prediction},
      author={Weiming Chen and Xitong Ling and Zhenyang Cai and Xidong Wang and Jiawen Li and Tian Guan and Benyou Wang and Yonghong He},
      year={2026},
      eprint={2605.08276},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.08276},
}
```
