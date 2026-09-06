#!/usr/bin/env python3
"""CPU unit tests for UCE-OPD masks, KL, retrieve-only, and teacher stopgrad."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_uce_opd_lib import (  # noqa: E402
    mean_step_kl,
    retrieve_readonly,
    reverse_kl_token,
    supervised_mask,
)
from uce_library import UceLibrary, make_entry  # noqa: E402
from uce_whole_action import command_content_mask  # noqa: E402


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _Tok:
    eos_token_id = 99

    def decode(self, ids, skip_special_tokens=True):
        table = {10: "ACTION", 11: ":", 12: " look", 13: " go", 14: " to", 99: ""}
        return "".join(table.get(int(i), "") for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [10, 11, 13]


def test_command_mask_and_eos() -> None:
    tok = _Tok()
    ids = [10, 11, 13, 14, 99]
    mask = supervised_mask(tok, ids)
    _assert(mask == [False, False, True, True, True], mask)
    cm = command_content_mask(tok, [10, 11, 12])
    _assert(cm == [False, False, True], cm)
    print("ok mask")


def test_reverse_kl_self_zero() -> None:
    torch.manual_seed(0)
    z = torch.randn(4, 16)
    logp = torch.nn.functional.log_softmax(z, dim=-1)
    kl = reverse_kl_token(logp, logp)
    _assert(float(kl.max()) < 1e-6, f"self KL {kl}")
    print("ok kl_self")


def test_teacher_stopgrad() -> None:
    torch.manual_seed(1)
    s_logits = torch.randn(3, 8, requires_grad=True)
    t_raw = torch.randn(3, 8, requires_grad=True)
    mask = [False, True, True]
    loss = mean_step_kl(s_logits, t_raw, [0, 1, 2], mask)
    loss.backward()
    _assert(s_logits.grad is not None and float(s_logits.grad.abs().sum()) > 0, "student needs grad")
    _assert(t_raw.grad is None, "teacher must be stopgrad")
    print("ok stopgrad")


def test_retrieve_readonly() -> None:
    e = make_entry(
        entry_id="e1",
        task_type="pick_and_place_simple",
        goal="put apple on table",
        actions=["ACTION: go to table", "ACTION: take apple from table"],
        source_task_id="t1",
        usage=3,
    )
    lib = UceLibrary([e])
    before = copy.deepcopy(lib.entries)
    retrieve_readonly(lib, {"task_type": "pick_and_place_simple", "goal": "put apple on table"})
    _assert(lib.entries[0]["usage"] == before[0]["usage"], "usage mutated")
    print("ok retrieve_readonly")


def test_empty_mask_zero_loss() -> None:
    s = torch.randn(2, 5, requires_grad=True)
    t = torch.randn(2, 5)
    loss = mean_step_kl(s, t, [1, 2], [False, False])
    _assert(float(loss) == 0.0, loss)
    loss.backward()
    print("ok empty_mask")


def test_linear_kl_descends() -> None:
    torch.manual_seed(2)
    teacher = nn.Linear(4, 6)
    student = nn.Linear(4, 6)
    with torch.no_grad():
        teacher.weight.copy_(student.weight + 0.5)
    opt = torch.optim.SGD(student.parameters(), lr=0.5)
    x = torch.randn(3, 4)
    losses = []
    for _ in range(8):
        opt.zero_grad()
        loss = mean_step_kl(student(x), teacher(x).detach(), [0, 1, 2], [True, True, True])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    _assert(losses[-1] < losses[0], losses)
    print("ok descend", losses[0], losses[-1])


def main() -> None:
    test_command_mask_and_eos()
    test_reverse_kl_self_zero()
    test_teacher_stopgrad()
    test_retrieve_readonly()
    test_empty_mask_zero_loss()
    test_linear_kl_descends()
    print("ALL_MATH_OK")


if __name__ == "__main__":
    main()
