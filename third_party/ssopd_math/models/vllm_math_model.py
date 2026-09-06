"""vLLM-backed math rollout model."""

from __future__ import annotations

import marshal
from pathlib import Path

if "VLLMMathModel" not in globals():
    _pyc = Path(__file__).resolve().parent / "__pycache__" / "vllm_math_model.cpython-311.pyc"
    if not _pyc.exists():
        raise ImportError(f"missing vllm_math_model bytecode: {_pyc}")
    with _pyc.open("rb") as _f:
        _f.read(16)
        _code = marshal.load(_f)
    if _code.co_consts and isinstance(_code.co_consts[0], str) and "Auto-restored" in _code.co_consts[0]:
        raise ImportError("vllm_math_model.pyc was overwritten by stub loader")
    exec(_code, globals())
