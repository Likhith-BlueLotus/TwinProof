"""Agent loop: parallel tool dispatch, one tool message per call id in order, verifier in the loop."""
from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from agents.dispatch import DISPATCH, Context
from agents.spark_open import SYSTEM, ToolCall

MAX_REJECT = 6
MAX_TURNS = 30
TurnFn = Callable[[list[dict]], tuple[list[ToolCall], str]]  # messages -> (tool calls, text)


def agent_loop(ctx: Context, turn: TurnFn) -> str | None:
    """Final text when the agent stops on its own; None after 6 consecutive rejections or the turn cap."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx.task_brief()}]
    rejects = 0
    for _ in range(MAX_TURNS):
        calls, text = turn(msgs)
        if not calls:
            return text
        msgs.append({"role": "assistant", "content": text or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in calls]})
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            results = list(pool.map(lambda c: _run(c, ctx), calls))
        for c, r in zip(calls, results):
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(r, default=float)})
            if r.get("verdict") in ("REJECTED", "REFUSED"):
                rejects += 1
            elif r.get("verdict") == "ADMITTED":
                rejects = 0
        if rejects >= MAX_REJECT:
            ctx.emit("commentary", "Six consecutive rejections: please film a new experiment.")
            return None
    ctx.emit("commentary", f"Stopped after {MAX_TURNS} turns.")
    return None


def _run(call: ToolCall, ctx: Context) -> dict:
    handler = DISPATCH.get(call["name"])
    if handler is None:
        return {"error": f"unknown tool {call['name']!r}"}
    try:
        args = json.loads(call["args"] or "{}")
        ctx.emit("tool_call", {"name": call["name"], "args": args})
        return handler(args, ctx)
    except Exception as err:  # a malformed call must reach the agent as feedback, not end the run
        return {"error": type(err).__name__, "detail": str(err)[:300]}
