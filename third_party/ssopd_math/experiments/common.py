"""Shared experiment utilities: config load, paths, metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be mapping: {path}")
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("root", str(path.resolve().parents[1]))
    return cfg


def resolve_path(cfg: dict[str, Any], rel: str | Path) -> Path:
    """Resolve a path relative to cfg paths.root.

    If ``rel`` is a string key present in ``cfg['paths']``, use that value
    (ssopd02-style: resolve_path(cfg, 'rollouts')).
    """
    paths = cfg.get("paths") or {}
    if isinstance(rel, str) and rel in paths and rel != "root":
        rel = paths[rel]
    rel = Path(rel)
    if rel.is_absolute():
        return rel
    root = Path(paths["root"])
    return (root / rel).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def write_yaml(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def resolved_layers(
    num_hidden_layers: int | dict[str, Any],
    fractions: list[float] | dict[str, Any] | None = None,
) -> dict[str, Any] | list[int]:
    """Resolve layer indices from depth fractions.

    Supported call styles:
    - resolved_layers(num_hidden_layers, fractions) -> layer_info dict (ssopd02)
    - resolved_layers(cfg, num_hidden_layers) -> list[int] (legacy helper)
    """
    # legacy: resolved_layers(cfg, n_layers) -> list
    if isinstance(num_hidden_layers, dict):
        cfg = num_hidden_layers
        n = int(fractions)  # type: ignore[arg-type]
        layers = cfg.get("layers", {})
        explicit = layers.get("resolved_indices")
        if explicit:
            return [int(x) for x in explicit]
        fracs = layers.get("fractions", [0.25, 0.5, 0.75, 0.9])
        return [min(max(int(float(f) * n), 0), n - 1) for f in fracs]

    n = int(num_hidden_layers)
    fracs = list(fractions) if fractions is not None else [0.25, 0.5, 0.75, 0.9]
    idx0 = [min(max(int(float(f) * n), 0), n - 1) for f in fracs]
    # unique preserve order
    seen = set()
    uniq = []
    for i in idx0:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return {
        "num_hidden_layers": n,
        "embedding_counted": False,
        "index_base": "0-based HuggingFace model.model.layers[k]",
        "fractions": [float(f) for f in fracs],
        "resolved_indices": uniq,
        "hook_indices_1based": [i + 1 for i in uniq],
    }


def _file_fingerprint(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def collect_metadata(cfg: dict[str, Any], caller_file: str | Path | list | None = None) -> dict[str, Any]:
    if isinstance(caller_file, list):
        caller_file = caller_file[0] if caller_file else None
    fp = {}
    if caller_file is not None:
        try:
            fp = {Path(caller_file).name: _file_fingerprint(caller_file)}
        except Exception:
            fp = {}
    meta: dict[str, Any] = {
        "model_name": cfg.get("model", {}).get("name"),
        "model_path": cfg.get("model", {}).get("path"),
        "model_variant": cfg.get("model", {}).get("variant"),
        "model_revision": cfg.get("model", {}).get("revision"),
        "dataset_version": cfg.get("dataset_version"),
        "random_seed": cfg.get("random_seed"),
        "code_version": cfg.get("code_version"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "file_fingerprint": fp,
    }
    try:
        import torch

        meta["pytorch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers

        meta["transformers_version"] = transformers.__version__
    except Exception:
        pass
    rep = cfg.get("representation", {})
    if rep.get("position"):
        meta["representation_position"] = rep["position"]
    if cfg.get("distillation", {}).get("top_k") is not None:
        meta["top_k"] = cfg["distillation"]["top_k"]
    return meta
