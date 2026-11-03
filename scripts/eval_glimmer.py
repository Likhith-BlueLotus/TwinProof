"""H4: base vs QLoRA Glimmer top-1 fix accuracy on the 200 held-out rows, exact one-sided McNemar (BENCHMARK.md).

Run: uv run python -m scripts.eval_glimmer --base <serve_base url> --tuned <serve_tuned url>  ->  logs/glimmer_eval.json
     uv run --env-file .env python -m scripts.eval_glimmer --spark muse-spark-1.3 --spark muse-spark-1.2
       ->  logs/glimmer_eval_spark.json (Meta Model API models as zero-shot critics on the same rows)
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from verifier.stats import mcnemar_exact_p

HELDOUT = Path("data/glimmer_sft/heldout.jsonl")
OUT = Path("logs/glimmer_eval.json")
SPARK_OUT = Path("logs/glimmer_eval_spark.json")
WORKERS = 8  # one container per endpoint; vLLM batches the concurrent requests
ESCALATE_BELOW = 0.6  # agent escalates to Spark Open under this confidence (master doc §3.5)


def parse_fix(text: str | None) -> dict | None:
    """The critic's answer if it is the schema, else None: vLLM does not enforce JSON with Glimmer's reasoning parser."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)  # tolerates ```json fences and a sentence around the object
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    ok = (isinstance(d, dict) and isinstance(d.get("fix"), str) and isinstance(d.get("params"), list)
          and isinstance(d.get("confidence"), (int, float)) and 0 <= d["confidence"] <= 1)
    return d if ok else None


def ask(client: OpenAI, row: dict) -> tuple[dict | None, float]:
    system, user, _ = row["messages"]
    image, text = user["content"]
    url = "data:image/png;base64," + base64.b64encode(Path(image["image"]).read_bytes()).decode()
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="glimmer-critic", temperature=1.0, top_p=0.95, extra_body={"top_k": 64}, max_tokens=2048,
        messages=[system, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, text]}])
    return parse_fix(r.choices[0].message.content), time.perf_counter() - t0


def _right(p: dict | None, y: str) -> bool:
    return p is not None and p["fix"] == y


def arm(labels: list[str], preds: list[dict | None]) -> dict:
    return {"top1": sum(_right(p, y) for p, y in zip(preds, labels)) / len(labels),
            "json_valid": sum(p is not None for p in preds) / len(labels),
            "escalated": sum(p is None or p["confidence"] < ESCALATE_BELOW for p in preds) / len(labels)}


def summarise(labels: list[str], base: list[dict | None], tuned: list[dict | None]) -> dict:
    right = _right
    b = sum(right(t, y) and not right(s, y) for s, t, y in zip(base, tuned, labels))  # only tuned right
    c = sum(right(s, y) and not right(t, y) for s, t, y in zip(base, tuned, labels))  # only base right
    return {"n": len(labels), "base": arm(labels, base), "tuned": arm(labels, tuned), "discordant": {"tuned_only": b, "base_only": c},
            "mcnemar_p_one_sided": mcnemar_exact_p(b, c)}


# A tuned critic learns the fix vocabulary from training; a zero-shot model must be told it (names and params only).
FIXES = {"add_coulomb_friction": ["frictionloss"], "add_viscous_damping": ["damping"], "adjust_pivot": ["joint_pos"],
         "switch_to_sliding": ["friction"], "switch_to_rolling": ["friction"], "no_change": []}
ZERO_SHOT = (" Choose fix from: " + "; ".join(f"{k} (params {v or 'none'})" for k, v in FIXES.items())
             + ". The image is the residual phase portrait and the residual against time of the fitted structure.")
FIX_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["fix", "params", "confidence"],
              "properties": {"fix": {"enum": list(FIXES)}, "params": {"type": "array", "items": {"type": "string"}},
                             "confidence": {"type": "number"}}}


def ask_spark(api: OpenAI, model: str, row: dict) -> tuple[dict | None, float]:
    """Same image, statistics and instruction through the Meta Model API's Responses endpoint, zero-shot."""
    system, user, _ = row["messages"]
    image, text = user["content"]
    url = "data:image/png;base64," + base64.b64encode(Path(image["image"]).read_bytes()).decode()
    t0 = time.perf_counter()
    r = api.responses.create(
        model=model, instructions=system["content"] + ZERO_SHOT, reasoning={"effort": "medium"},
        text={"format": {"type": "json_schema", "name": "fix", "schema": FIX_SCHEMA, "strict": True}},
        input=[{"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": url},
                                                               {"type": "input_text", "text": text["text"]}]}])
    return parse_fix(r.output_text), time.perf_counter() - t0


def run_spark(models: list[str], rows: list[dict]) -> dict:
    from perception.sam_orchestrator import client  # Meta Model API (MODEL_API_KEY)
    api, labels, out = client(), [r["label"] for r in rows], {}
    for model in models:
        with ThreadPoolExecutor(WORKERS) as pool:
            res = list(pool.map(lambda r: ask_spark(api, model, r), rows))
        secs = sorted(s for _, s in res)
        out[model] = {**arm(labels, [p for p, _ in res]), "median_latency_s": secs[len(secs) // 2],
                      "p95_latency_s": secs[int(0.95 * (len(secs) - 1))]}
        print(model, json.dumps(out[model]))
    return {"n": len(rows), **out}


def run(url: str, rows: list[dict]) -> tuple[list[dict | None], list[float]]:
    client = OpenAI(base_url=url.rstrip("/") + "/v1", api_key="unused", timeout=300, max_retries=3)
    with ThreadPoolExecutor(WORKERS) as pool:
        out = list(pool.map(lambda r: ask(client, r), rows))
    return [p for p, _ in out], [s for _, s in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="serve_base endpoint URL")
    ap.add_argument("--tuned", help="serve_tuned endpoint URL")
    ap.add_argument("--spark", action="append", help="Meta Model API model to score zero-shot (repeatable)")
    args = ap.parse_args()
    rows = [json.loads(line) for line in HELDOUT.read_text().splitlines() if line.strip()]
    if args.spark:
        SPARK_OUT.write_text(json.dumps(run_spark(args.spark, rows), indent=1))
        return
    if not (args.base and args.tuned):
        ap.error("--base and --tuned are both needed for H4 (or pass --spark)")
    base, base_s = run(args.base, rows)
    tuned, tuned_s = run(args.tuned, rows)
    out = summarise([r["label"] for r in rows], base, tuned)
    out["median_latency_s"] = {"base": sorted(base_s)[len(base_s) // 2], "tuned": sorted(tuned_s)[len(tuned_s) // 2]}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
