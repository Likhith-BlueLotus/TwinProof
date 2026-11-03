"""Step 6 of the Glimmer SFT data (master doc §3.5): a teacher rationale per training row, as reasoning_content.

Glimmer's chat template trains on reasoning_content (rendered as the model's private turn before the answer), and every
row asks for "Reasoning: medium", so label-only rows would teach the critic to skip the reasoning it is served with.
Teacher: Muse Spark 1.3 (the plan named Spark Open; no key yet). It is told the correct fix and must say whether the
plot supports it; rows it does not support, or whose rationale names another fix, are dropped. Held-out rows get none.

Run: uv run --env-file .env python -m scripts.add_glimmer_rationales  ->  data/glimmer_sft/train_teacher.jsonl
Resumable: finished rows are cached in data/glimmer_sft/rationales.jsonl.
"""
from __future__ import annotations

import base64
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from perception.sam_orchestrator import client

DATA = Path("data/glimmer_sft")
CACHE = DATA / "rationales.jsonl"
OUT = DATA / "train_teacher.jsonl"
MODEL = "muse-spark-1.3"
WORKERS = 8
FIXES = ("add_coulomb_friction", "add_viscous_damping", "adjust_pivot", "switch_to_sliding", "switch_to_rolling", "no_change")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["rationale", "supports"],
          "properties": {"rationale": {"type": "string", "maxLength": 900}, "supports": {"type": "boolean"}}}
PROMPT = ("You are teaching a system-identification critic. The image is the residual of a fitted physics model "
          "(residual phase portrait and residual against time); the statistics are below. The correct fix is {fix} "
          "(parameters: {params}). In two to four sentences, explain what in the plot and statistics points to this "
          "fix, as the critic should reason before answering. Describe only evidence you can see; do not name any "
          "other fix. Set supports=false if the plot does not show evidence for this fix.\n\nStatistics: {stats}")


def contradicts(rationale: str, label: str) -> bool:
    """A rationale that names a different fix teaches the wrong association."""
    words = rationale.lower().replace("-", "_")
    return any(f in words for f in FIXES if f != label)


def with_rationale(row: dict, rationale: str) -> dict:
    """The row with the rationale as the assistant's reasoning_content (a new row; the input is not changed)."""
    *head, answer = row["messages"]
    return {**row, "messages": [*head, {**answer, "reasoning_content": rationale}]}


def teach(api, row: dict) -> dict:
    _, user, answer = row["messages"]
    image, stats = user["content"]
    fix = json.loads(answer["content"])
    url = "data:image/png;base64," + base64.b64encode(Path(image["image"]).read_bytes()).decode()
    r = api.responses.create(
        model=MODEL, reasoning={"effort": "medium"},
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": url},
            {"type": "input_text", "text": PROMPT.format(fix=fix["fix"], params=", ".join(fix["params"]) or "none", stats=stats["text"])}]}],
        text={"format": {"type": "json_schema", "name": "rationale", "schema": SCHEMA, "strict": True}})
    out = json.loads(r.output_text)
    return {"seed": row["seed"], "rationale": re.sub(r"\s+", " ", out["rationale"]).strip(), "supports": bool(out["supports"])}


def main() -> None:
    rows = [json.loads(line) for line in (DATA / "train.jsonl").read_text().splitlines() if line.strip()]
    done = {d["seed"]: d for d in map(json.loads, CACHE.read_text().splitlines())} if CACHE.exists() else {}
    todo = [r for r in rows if r["seed"] not in done]
    api, lock = client(), threading.Lock()
    with CACHE.open("a") as cache, ThreadPoolExecutor(WORKERS) as pool:
        def work(row: dict) -> None:
            d = teach(api, row)
            with lock:
                cache.write(json.dumps(d) + "\n"); cache.flush()
                done[d["seed"]] = d
        for i, _ in enumerate(pool.map(work, todo), 1):
            if i % 100 == 0:
                print(f"{i}/{len(todo)}")
    kept = [with_rationale(r, done[r["seed"]]["rationale"]) for r in rows
            if done[r["seed"]]["supports"] and not contradicts(done[r["seed"]]["rationale"], r["label"])]
    OUT.write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"kept {len(kept)} of {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
