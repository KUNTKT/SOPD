"""SSOPD02: extract capability directions from rollouts."""

from __future__ import annotations

import marshal
from pathlib import Path

if "main" not in globals():
    _pyc = Path(__file__).resolve().parent / "__pycache__" / "ssopd02_direction.cpython-311.pyc"
    if not _pyc.exists():
        raise ImportError(f"missing ssopd02 bytecode: {_pyc}")
    with _pyc.open("rb") as _f:
        _f.read(16)
        _code = marshal.load(_f)
    if _code.co_consts and isinstance(_code.co_consts[0], str) and "Auto-restored" in _code.co_consts[0]:
        raise ImportError("ssopd02_direction.pyc was overwritten by stub loader")
    exec(_code, globals())
