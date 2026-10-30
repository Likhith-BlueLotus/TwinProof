"""Live agent run on a tracked S1 take: resolve a Spark endpoint, let the agent fit, propose and get verified.

Run: uv run --env-file .env python -m scripts.run_agent_s1 logs/runs/s1_s1.npz
Every event (provider, commentary, tool calls, verdicts) goes to logs/runs/agent_<stem>_<time>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np

from agents.dispatch import Context
from agents.loop import agent_loop
from agents.spark_open import resolve, run_turn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("track", type=Path)
    ap.add_argument("--model", type=Path, default=Path("physics/models/s1_pendulum.xml"))
    args = ap.parse_args()
    data = np.load(args.track)
    ok = np.isfinite(data["phi"])
    log_path = Path("logs/runs") / f"agent_{args.track.stem}_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    log = log_path.open("w")

    def emit(kind: str, payload: object) -> None:
        log.write(json.dumps({"t": time.time(), "kind": kind, "data": payload}, default=float) + "\n")
        log.flush()
        if kind != "commentary" or payload.strip():
            print(f"[{kind}] {json.dumps(payload, default=float)[:220]}")

    api, provider, slug = resolve()
    emit("provider", {"provider": provider, "slug": slug})
    ctx = Context(args.model.read_text(), data["t"][ok], data["phi"][ok], float(data["r_meas"]), emit, proposer=f"{provider}:{slug}")
    final = agent_loop(ctx, partial(run_turn, api, slug, emit=emit))
    emit("final", {"text": final, "verdicts": [(r["claim_id"], r["verdict"], r["reason"]) for r in ctx.records]})
    print("log:", log_path)


if __name__ == "__main__":
    main()
