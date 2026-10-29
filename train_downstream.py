#!/usr/bin/env python3
"""Fine-tune a downstream segmentation head from one YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from downstream.config import load_config
from downstream.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a segmentation head on frozen CMD features.")
    parser.add_argument("--config", required=True, help="Path to the downstream YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(config, config_path=args.config)


if __name__ == "__main__":
    main()
