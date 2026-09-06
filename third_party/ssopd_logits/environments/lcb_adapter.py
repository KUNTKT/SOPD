"""LiveCodeBench v6 adapter for Logit-SSOPD / MOPD rollout pipeline."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import random
import zlib
from pathlib import Path
from typing import Any

DATASET_VERSION = "lcb_v6_adapted_v1"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data" / "lcb_v6" / "cache"
LCB_TOOL = "submit_code"

LCB_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the competitive programming "
    "problem in the user message. Put your complete final solution in exactly "
    "one ```python ... ``` fenced code block with no placeholders."
)

# MOPD / R1 eval window for LiveCodeBench v6
DEFAULT_DATE_START = "2024-08-00T00:00:00"
DEFAULT_DATE_END = "2025-01-00T00:00:00"


def _decode_test_cases(raw_public: str, raw_private: str) -> dict[str, Any]:
    public = json.loads(raw_public)
    try:
        private = json.loads(raw_private)
    except Exception:
        private = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(raw_private.encode("utf-8"))))
        )
    full = public + private
    metadata_fn = None
    return {
        "inputs": [t["input"] for t in full],
        "outputs": [t["output"] for t in full],
        "fn_name": metadata_fn,
    }


def build_lcb_prompt(row: dict[str, Any]) -> str:
    """Problem prompt aligned with SEED / LiveCodeBench code_generation template."""
    query = (
        "You will be given a question (problem specification) and will generate a "
        "correct Python program that matches the specification and passes all tests.\n\n"
        f"Question: {row['question_content']}\n\n"
    )
    starter = row.get("starter_code") or ""
    if starter:
        query += (
            "You will use the following starter code to write the solution to the problem "
            "and enclose your code within delimiters.\n"
            f"```python\n{starter}\n```"
        )
    else:
        query += (
            "Read the inputs from stdin solve the problem and write the answer to stdout "
            "(do not directly test on the sample inputs). Enclose your code within delimiters "
            "as follows. Ensure that when the python program runs, it reads the inputs, runs "
            "the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```"
        )
    return query


def row_to_task(row: dict[str, Any], *, split: str = "select") -> dict[str, Any]:
    qid = str(row.get("question_id") or row.get("id") or row.get("task_id"))
    metadata = json.loads(row.get("metadata") or "{}")
    test_blob: dict[str, Any] = {"inputs": [], "outputs": [], "fn_name": metadata.get("func_name")}
    if row.get("public_test_cases"):
        test_blob = _decode_test_cases(str(row["public_test_cases"]), str(row["private_test_cases"]))
        test_blob["fn_name"] = metadata.get("func_name")

    instruction = build_lcb_prompt(row)
    return {
        "task_id": f"lcb_{qid}",
        "instruction": instruction,
        "task_type": "single_step",
        "gold_plan": [
            {
                "step": 0,
                "tool_name": LCB_TOOL,
                "arguments": {"code": ""},
                "argument_options": {},
                "expected_result": "ALL_TESTS_PASS",
                "subgoal": "lcb_code_generation",
            }
        ],
        "valid_tools": [LCB_TOOL],
        "function_docs": [],
        "source": "lcb",
        "split": split,
        "n_steps": 1,
        "lcb": {
            "question_id": qid,
            "contest_date": row.get("contest_date"),
            "starter_code": row.get("starter_code") or "",
            "test_cases": test_blob,
            "metadata": metadata,
        },
    }


def tasks_fingerprint(tasks: list[dict[str, Any]]) -> str:
    ids = sorted(str(t["task_id"]) for t in tasks)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def split_lcb_tasks(
    tasks: list[dict[str, Any]],
    *,
    select_frac: float = 0.5,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    idxs = list(range(len(tasks)))
    rng.shuffle(idxs)
    n_select = int(round(len(tasks) * select_frac))
    select_i = set(idxs[:n_select])
    select: list[dict[str, Any]] = []
    confirm: list[dict[str, Any]] = []
    for i, t in enumerate(tasks):
        t2 = dict(t)
        if i in select_i:
            t2["split"] = "select"
            select.append(t2)
        else:
            t2["split"] = "confirm"
            confirm.append(t2)
    return select, confirm


def _rows_from_hf(
    *,
    date_start: str,
    date_end: str,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    jsonl_files = sorted(
        x.path
        for x in api.list_repo_tree("livecodebench/code_generation_lite", repo_type="dataset")
        if str(x.path).endswith(".jsonl")
    )
    if not jsonl_files:
        raise FileNotFoundError("no jsonl shards in livecodebench/code_generation_lite")

    rows: list[dict[str, Any]] = []
    for fn in jsonl_files:
        path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=fn,
            repo_type="dataset",
        )
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                contest = str(row.get("contest_date", ""))
                if date_start <= contest < date_end:
                    rows.append(row)
    rows.sort(key=lambda r: (str(r.get("contest_date")), str(r.get("question_id"))))
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    return rows


def load_lcb_rows(
    *,
    cache_path: Path | None = None,
    date_start: str = DEFAULT_DATE_START,
    date_end: str = DEFAULT_DATE_END,
    max_rows: int | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    cache = cache_path or (DEFAULT_CACHE / "lcb_v6_rows.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    rows = _rows_from_hf(date_start=date_start, date_end=date_end, max_rows=max_rows)
    cache.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return rows


def load_lcb_category(
    *,
    cache_path: Path | None = None,
    date_start: str = DEFAULT_DATE_START,
    date_end: str = DEFAULT_DATE_END,
    max_rows: int | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    rows = load_lcb_rows(
        cache_path=cache_path,
        date_start=date_start,
        date_end=date_end,
        max_rows=max_rows,
        refresh=refresh,
    )
    return [row_to_task(r, split="select") for r in rows]


def load_lcb_splits(
    *,
    cache_path: Path | None = None,
    select_frac: float = 0.5,
    seed: int = 0,
    max_select: int | None = None,
    max_confirm: int | None = None,
    date_start: str = DEFAULT_DATE_START,
    date_end: str = DEFAULT_DATE_END,
    max_rows: int | None = None,
    refresh: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    tasks = load_lcb_category(
        cache_path=cache_path,
        date_start=date_start,
        date_end=date_end,
        max_rows=max_rows,
        refresh=refresh,
    )
    select, confirm = split_lcb_tasks(tasks, select_frac=select_frac, seed=seed)
    if max_select is not None:
        select = select[: int(max_select)]
    if max_confirm is not None:
        confirm = confirm[: int(max_confirm)]
    return {"select": select, "confirm": confirm, "all": tasks}
