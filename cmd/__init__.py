"""ConvNeXt Masked-Diffusion (CMD) pretraining package.

Downstream training and inference import the backbones defined here. The package
name overlaps the Python standard-library ``cmd`` module, so ``Cmd`` is
re-exported for tools that import ``cmd.Cmd`` while this repository is on
``PYTHONPATH``.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

_stdlib_cmd = Path(sysconfig.get_path("stdlib")) / "cmd.py"
_spec = importlib.util.spec_from_file_location("_stdlib_cmd", _stdlib_cmd)
if _spec is not None and _spec.loader is not None:
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    Cmd = _module.Cmd
else:  # pragma: no cover
    Cmd = object

__all__ = ["Cmd"]
