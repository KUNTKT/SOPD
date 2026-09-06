"""Math answer parsing (loaded from frozen bytecode)."""
from __future__ import annotations

import marshal
from pathlib import Path

_pyc = Path(__file__).resolve().parent / "_math_answer_real.pyc"
with _pyc.open("rb") as _f:
    _f.read(16)
    _code = marshal.load(_f)
exec(_code, globals())
