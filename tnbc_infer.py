#!/usr/bin/env python3
"""Entry point for CMD TNBC inference."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmd_model.inference import main


if __name__ == "__main__":
    main()
