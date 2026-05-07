"""Public CMD package for reviewer reproduction.

The package name is intentionally short for the released artifact. It also
re-exports the standard-library ``cmd.Cmd`` class to avoid surprising tools that
import ``cmd`` while this repository is on ``PYTHONPATH``.
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
