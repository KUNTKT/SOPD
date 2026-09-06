"""Worker-process pool for environment stepping.

ALFWorld's text engine is Python and CPU-bound, and its per-episode setup parses
a PDDL game file. During a lockstep rollout the GPU sits at zero utilization while
those steps run serially, so the pool exists to overlap them.

Threads would not help: the work holds the GIL. Each worker therefore owns its own
`AgentEnvironment` instances and receives commands over a pipe. Envs are never
sent across the boundary, only observations and verdicts, so nothing here needs to
be picklable except plain dicts.

Determinism is preserved: a slot is pinned to one worker for the whole episode, and
worker assignment is a function of the slot index, so results do not depend on
which worker happens to finish first.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Any, Callable

EnvFactory = Callable[[], Any]


def _worker(
    conn: Any,
    factory: EnvFactory,
    env_kwargs: dict[str, Any],
) -> None:
    envs: dict[int, Any] = {}
    try:
        while True:
            msg = conn.recv()
            op = msg.get("op")
            if op == "stop":
                break
            slot = int(msg["slot"])
            try:
                if op == "reset":
                    env = envs.get(slot)
                    if env is not None:
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()
                    env = factory(**env_kwargs)
                    envs[slot] = env
                    obs = env.reset(msg["task"])
                    conn.send({"ok": True, "slot": slot, "observation": obs})
                elif op == "state":
                    conn.send(
                        {"ok": True, "slot": slot, "state": envs[slot].current_state()}
                    )
                elif op == "step":
                    env = envs[slot]
                    # State, verdict and transition in one round trip: three
                    # separate ones would put the pipe latency on the critical
                    # path of every step of every episode.
                    state = env.current_state()
                    verdict = env.verify_decision(state, msg["action_text"])
                    out = env.step(msg["action_text"])
                    conn.send(
                        {
                            "ok": True,
                            "slot": slot,
                            "state": state,
                            "verdict": verdict,
                            "observation": out.observation,
                            "tool_result": out.tool_result,
                            "success": bool(out.success),
                            "terminal": bool(out.terminal),
                            "failure_reason": out.failure_reason,
                        }
                    )
                elif op == "close":
                    env = envs.pop(slot, None)
                    if env is not None:
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()
                    conn.send({"ok": True, "slot": slot})
                else:
                    conn.send({"ok": False, "slot": slot, "error": f"bad op {op!r}"})
            except Exception:
                conn.send(
                    {"ok": False, "slot": slot, "error": traceback.format_exc()}
                )
    finally:
        for env in envs.values():
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        conn.close()


class EnvPool:
    """Round-robin pool of environment workers, addressed by slot index."""

    def __init__(
        self,
        factory: EnvFactory,
        *,
        n_workers: int,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        # `fork` keeps the ALFWorld imports and ALFWORLD_DATA already resolved in
        # the parent; spawn would re-import textworld in every worker.
        ctx = mp.get_context("fork")
        self.n_workers = int(n_workers)
        self._conns: list[Any] = []
        self._procs: list[Any] = []
        for _ in range(self.n_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child, factory, dict(env_kwargs or {})),
                daemon=True,
            )
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)

    def _conn(self, slot: int) -> Any:
        return self._conns[slot % self.n_workers]

    # -- collective operations ---------------------------------------------

    def _scatter(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send one request per slot, then collect every reply.

        Requests are grouped by worker and sent before any reply is read, which is
        what actually overlaps the work. Replies come back in request order because
        each worker answers its own queue in order.
        """
        by_worker: dict[int, list[dict[str, Any]]] = {}
        for req in requests:
            by_worker.setdefault(int(req["slot"]) % self.n_workers, []).append(req)
        for worker_idx, reqs in by_worker.items():
            conn = self._conns[worker_idx]
            for req in reqs:
                conn.send(req)
        replies: dict[int, dict[str, Any]] = {}
        for worker_idx, reqs in by_worker.items():
            conn = self._conns[worker_idx]
            for _ in reqs:
                reply = conn.recv()
                if not reply.get("ok"):
                    raise RuntimeError(
                        f"env worker {worker_idx} failed: {reply.get('error')}"
                    )
                replies[int(reply["slot"])] = reply
        return [replies[int(r["slot"])] for r in requests]

    def reset(self, slots: list[int], tasks: list[dict[str, Any]]) -> list[str]:
        replies = self._scatter(
            [
                {"op": "reset", "slot": s, "task": t}
                for s, t in zip(slots, tasks)
            ]
        )
        return [r["observation"] for r in replies]

    def step(self, slots: list[int], action_texts: list[str]) -> list[dict[str, Any]]:
        return self._scatter(
            [
                {"op": "step", "slot": s, "action_text": a}
                for s, a in zip(slots, action_texts)
            ]
        )

    def state(self, slots: list[int]) -> list[dict[str, Any]]:
        return [r["state"] for r in self._scatter([{"op": "state", "slot": s} for s in slots])]

    def close_slots(self, slots: list[int]) -> None:
        if slots:
            self._scatter([{"op": "close", "slot": s} for s in slots])

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send({"op": "stop"})
            except (OSError, BrokenPipeError):
                pass
        for proc in self._procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        self._conns = []
        self._procs = []

    def __enter__(self) -> "EnvPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def default_workers(requested: int | None = None) -> int:
    """Leave a core for the generation engine; it is not free either."""
    if requested is not None and int(requested) > 0:
        return int(requested)
    return max(1, (os.cpu_count() or 2) - 2)
