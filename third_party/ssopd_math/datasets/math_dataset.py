"""MATH dataset loading and problem sampling."""
from __future__ import annotations

import hashlib
import importlib
import sys
from typing import Any

DEFAULT_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)


def _import_hf_load_dataset():
    """Import HuggingFace datasets.load_dataset, avoiding local ssopd_math/datasets shadow."""
    import importlib.util
    from pathlib import Path

    # Drop a shadowed top-level "datasets" that is actually ssopd_math/datasets.
    mod = sys.modules.get("datasets")
    if mod is not None:
        path = (getattr(mod, "__file__", "") or "").replace("\\", "/")
        if "ssopd_math/datasets" in path or not hasattr(mod, "load_dataset"):
            for key in list(sys.modules):
                if key == "datasets" or key.startswith("datasets."):
                    m2 = sys.modules.get(key)
                    p2 = (getattr(m2, "__file__", "") or "").replace("\\", "/")
                    if "ssopd_math/datasets" in p2 or key == "datasets":
                        del sys.modules[key]

    # Prefer the interpreter's site-packages copy (not cwd-local).
    site_pkgs = [Path(p) for p in sys.path if p.endswith("site-packages")]
    for sp in site_pkgs:
        init_py = sp / "datasets" / "__init__.py"
        if not init_py.is_file():
            continue
        # Ensure "datasets" is not already the local package.
        if "datasets" not in sys.modules:
            # Put site-packages first for this import.
            sys.path.insert(0, str(sp))
            try:
                hf = importlib.import_module("datasets")
            finally:
                if sys.path and sys.path[0] == str(sp):
                    sys.path.pop(0)
        else:
            hf = sys.modules["datasets"]
        if hasattr(hf, "load_dataset") and "ssopd_math/datasets" not in (
            getattr(hf, "__file__", "") or ""
        ).replace("\\", "/"):
            return hf.load_dataset

    raise ImportError(
        "HuggingFace datasets.load_dataset not found (local ssopd_math/datasets shadow?)"
    )


def _extract_gold_from_solution(solution: str) -> str:
    from ssopd_math.verifier.math_answer import (
        last_boxed_only_string,
        remove_boxed,
    )

    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return solution.strip()
    return remove_boxed(boxed)


def load_math_problems(
    source: str = "DigitalLearningGmbH/MATH-lighteval",
    split: str = "train",
    cache_dir: str | None = None,
    n_problems: int | None = None,
    seed: int = 0,
    instruction_suffix: str = DEFAULT_INSTRUCTION,
) -> list[dict[str, Any]]:
    load_dataset = _import_hf_load_dataset()
    # Prefer split= kwarg. Positional 2nd arg is BuilderConfig name (not split);
    # MATH-lighteval configs are algebra/default/... — not "train".
    ds = load_dataset(source, split=split, cache_dir=cache_dir)
    rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        problem = str(ex["problem"]).strip()
        solution = str(ex.get("solution", ""))
        gold = _extract_gold_from_solution(solution)
        problem_id = f"math_{split}_{idx:05d}"
        prompt_user = problem + " " + instruction_suffix
        rows.append(
            {
                "problem_id": problem_id,
                "index": idx,
                "problem": problem,
                "prompt_user": prompt_user,
                "gold_answer": gold,
                "solution_ref": solution,
                "dataset_split": split,
                "source": source,
            }
        )

    if n_problems is not None and n_problems < len(rows):
        import random

        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        selected = sorted(indices[:n_problems])
        rows = [rows[i] for i in selected]
        for i, row in enumerate(rows):
            row["sample_rank"] = i
    return rows


def problem_rollout_seed(base_seed: int, problem_id: str, group_index: int) -> int:
    raw = f"{base_seed}:{problem_id}:{group_index}".encode()
    return int(hashlib.md5(raw).hexdigest()[:8], 16)
