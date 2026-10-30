"""Muse Spark 1.3 (Standard tier, not -contributor): scene nouns and candidate model classes from a video."""
from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from perception.sam_orchestrator import client, upload_video

MODEL = "muse-spark-1.3"
MODEL_CLASSES = [
    "pendulum_viscous", "pendulum_viscous_coulomb", "pendulum_stiction",
    "incline_rolling", "incline_sliding", "hinge_door", "prismatic",
]
SCENE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["nouns", "model_classes", "marker_visible"],
    "properties": {
        "nouns": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 32}},
        "model_classes": {"type": "array", "items": {"enum": MODEL_CLASSES}},
        "marker_visible": {"type": "boolean"},
    },
}
PROMPT = (
    "This video shows one mechanism filmed for physics identification, with a printed black-and-white square "
    "ArUco marker for scale. List the single moving rigid parts as short noun phrases (one object each, no lists, "
    "at most 3 words), the candidate model classes, and whether the square marker is visible."
)


def scene(video: Path, api: OpenAI | None = None) -> dict:
    """Validated scene description; refuses when the marker is not visible."""
    api = api or client()
    r = api.responses.create(
        model=MODEL,
        reasoning={"effort": "high"},  # capped at high on video (master doc §3.3)
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": PROMPT}, {"type": "input_video", "file_id": upload_video(api, video)}]}],
        text={"format": {"type": "json_schema", "name": "scene", "schema": SCENE_SCHEMA, "strict": True}},
    )
    data = json.loads(r.output_text)
    bad = [c for c in data["model_classes"] if c not in MODEL_CLASSES]
    if bad or not data["nouns"]:
        raise ValueError(f"scene output fails the schema: {data}")
    if not data["marker_visible"]:
        raise ValueError("ArUco marker not visible: film a new take with the marker in frame")
    return data
