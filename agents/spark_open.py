"""Muse Spark Open orchestrator: provider resolver, tool schemas, streamed turns (master doc §3.4).

Provider order: Baseten (two slugs) -> Fireworks -> muse-spark-1.3 on api.meta.ai. The last is a labelled
development fallback while partner keys are pending; it is not Spark Open and does not count toward the
open-weight bonus. Every resolution is logged for partner verification.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable

from openai import APIStatusError, OpenAI

log = logging.getLogger("spark_open")
CANDIDATES = [
    ("baseten", "https://inference.baseten.co/v1", "BASETEN_API_KEY", "meta-models/muse-spark-open"),
    ("baseten", "https://inference.baseten.co/v1", "BASETEN_API_KEY", "meta-models/muse-spark-open-1026"),
    ("fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "accounts/fireworks/models/muse-spark-open"),
    ("meta-dev-fallback", "https://api.meta.ai/v1", "MODEL_API_KEY", "muse-spark-1.3"),
]
RAW_CHUNKS_LOGGED = 20  # day-1: confirm reasoning/commentary field names

STRUCTURES = ["pendulum_viscous", "pendulum_viscous_coulomb"]
TOOLS = [
    {"type": "function", "function": {
        "name": "fit_batch",
        "description": "Fit a model structure to the identification take (first 70%) and return the posterior.",
        "parameters": {"type": "object", "required": ["structure"], "properties": {
            "structure": {"enum": STRUCTURES}}}}},
    {"type": "function", "function": {
        "name": "propose_mjcf_patch",
        "description": "Create a structure claim that edits whitelisted MJCF fields. Returns a claim ref.",
        "parameters": {"type": "object", "required": ["patches", "rationale"], "properties": {
            "patches": {"type": "array", "items": {"type": "object", "required": ["xpath", "attr", "value"], "properties": {
                "xpath": {"type": "string"}, "attr": {"type": "string"}, "value": {"type": "string"}}}},
            "rationale": {"type": "string", "maxLength": 400}}}}},
    {"type": "function", "function": {
        "name": "synthesize_controller",
        "description": "Create a controller claim holding the pendulum upright. Give LQR weights or explicit gains. Returns a claim ref.",
        "parameters": {"type": "object", "required": ["kind", "u_max"], "properties": {
            "kind": {"enum": ["lqr"]},
            "Q_diag": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number", "exclusiveMinimum": 0}},
            "R_diag": {"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "number", "exclusiveMinimum": 0}},
            "gains": {"type": "object", "properties": {"k_theta": {"type": "number"}, "k_omega": {"type": "number"}}},
            "u_max": {"type": "number", "exclusiveMinimum": 0}}}}},
    {"type": "function", "function": {
        "name": "submit_claim",
        "description": "Send a claim to the deterministic verifier (G7, G0-G5). Returns the verdict with witness numbers.",
        "parameters": {"type": "object", "required": ["ref"], "properties": {"ref": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "reverify",
        "description": "Operator what-if: re-verify the last controller under changed physics.",
        "parameters": {"type": "object", "properties": {
            "mass_scale": {"type": "number", "minimum": 0.5, "maximum": 2.0},
            "damping_scale": {"type": "number", "minimum": 0.0, "maximum": 3.0},
            "torque_limit": {"type": "number", "exclusiveMinimum": 0}}}}},
]

SYSTEM = (
    "You are TwinProof's physics agent. You never assert a parameter, structure, or controller; you submit claims "
    "and the deterministic verifier decides. Before each tool call, give the operator one line of commentary. "
    "Start by fitting candidate structures. Then synthesize a controller that holds the pendulum upright and submit it. "
    "On a gate failure, read the witness numbers and propose the smallest change that could pass. Edit only "
    "whitelisted MJCF fields: joint pos, damping, frictionloss, armature, friction, solref, solimp, kp, gear, mass. "
    "Never edit goals, options, or limits. Controllers are parametric only; never write code. "
    "Stop once a controller is admitted. After 6 consecutive rejections, ask the operator for a new experiment."
)

Emit = Callable[[str, object], None]
ToolCall = dict  # {"id", "name", "args"}


def resolve() -> tuple[OpenAI, str, str]:
    """First (provider, slug) answering a 1-token call wins."""
    errors = []
    for provider, url, env, slug in CANDIDATES:
        key = os.environ.get(env)
        if not key:
            errors.append(f"{provider}:{slug}: no key")
            continue
        api = OpenAI(base_url=url, api_key=key, timeout=120)
        try:
            api.chat.completions.create(model=slug, max_tokens=1, messages=[{"role": "user", "content": "ok"}])
        except APIStatusError as err:
            errors.append(f"{provider}:{slug}: {err.status_code}")
            continue
        log.info("resolved provider=%s slug=%s", provider, slug)
        return api, provider, slug
    raise RuntimeError("no Spark endpoint; tried " + "; ".join(errors))


def run_turn(api: OpenAI, slug: str, messages: list[dict], *, emit: Emit, effort: str = "high") -> tuple[list[ToolCall], str]:
    """Stream one turn, forwarding reasoning and commentary; return assembled tool calls and text."""
    stream = api.chat.completions.create(
        model=slug, messages=messages, tools=TOOLS, parallel_tool_calls=True, stream=True, reasoning_effort=effort
    )
    calls: dict[int, ToolCall] = {}
    text: list[str] = []
    for n, chunk in enumerate(stream):
        if n < RAW_CHUNKS_LOGGED:
            log.debug("raw chunk %s", chunk.model_dump_json())
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        extra = delta.model_extra or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning")
        if reasoning:
            emit("reasoning", reasoning)
        if delta.content:
            text.append(delta.content)
            emit("commentary", delta.content)
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
            slot["id"] = tc.id or slot["id"]
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["args"] += tc.function.arguments
    return [calls[i] for i in sorted(calls)], "".join(text)
