"""Auto-restored: exec bytecode from __pycache__ (original .py sources missing)."""
from __future__ import annotations

import marshal
from pathlib import Path

_stem = Path(__file__).stem
_cache = Path(__file__).resolve().parent / "__pycache__"
_pyc = _cache / f"{_stem}.cpython-311.pyc"
if not _pyc.exists():
    _cands = sorted(_cache.glob(f"{_stem}.cpython-311*.pyc"))
    if not _cands:
        raise ImportError(f"missing pyc for {__file__}")
    _pyc = _cands[0]
with _pyc.open("rb") as _f:
    _f.read(16)
    _code = marshal.load(_f)
exec(_code, globals())
