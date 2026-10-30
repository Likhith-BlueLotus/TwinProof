"""SAM 3.1 via api.meta.ai: frame-0 concept fan-out -> one box-prompted video request -> full-frame masks.

Raw output_text is cached on disk by request hash, so re-runs and the demo never pay twice and survive an
API outage (master doc §6.1). Parsing and mask decoding use Meta's meta_sam_parser.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import meta_sam_parser as msp
import numpy as np
from openai import APIStatusError, OpenAI

MODEL = "sam-3.1"
CACHE = Path("data/sam_cache")
MAX_OBJECTS = 16
RETRY_STATUS = {429, 500, 502, 503}
RETRIES = 4
FILE_PURPOSE = "user_data"
MULTI_CONCEPT = re.compile(r"[,;+\n]|\b(and|or|not|without)\b", re.I)

Boxes = dict[int, tuple[int, int, int, int]]  # object id -> half-open (left, top, right, bottom)
Masks = dict[int, dict[int, np.ndarray]]  # frame -> object id -> full-frame uint8 mask


def client() -> OpenAI:
    return OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])


def sanitize(noun: str) -> str:
    """SAM takes one concept per request: keep the first fragment, at most 3 words."""
    words = MULTI_CONCEPT.split(noun, maxsplit=1)[0].split()[:3]
    if not words:
        raise ValueError(f"empty concept after sanitize: {noun!r}")
    return " ".join(words)


def _key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def request_text(api: OpenAI, content: list[dict], metadata: dict[str, str], cache_key: str) -> str:
    """Raw output_text ('' = zero matches, a valid result). 400s are never retried."""
    path = CACHE / f"{cache_key}.txt"
    if path.exists():
        return path.read_text()
    for attempt in range(RETRIES):
        try:
            stream = api.responses.create(
                model=MODEL, input=[{"type": "message", "role": "user", "content": content}], metadata=metadata, stream=True
            )
            chunks = []
            for ev in stream:
                if ev.type == "response.output_text.delta":
                    chunks.append(ev.delta)
                elif ev.type in ("response.failed", "error"):
                    raise RuntimeError(f"SAM stream error: {ev}")
            text = "".join(chunks)
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            return text
        except APIStatusError as err:
            if err.status_code in RETRY_STATUS and attempt < RETRIES - 1:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def image_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("could not encode frame")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def detect_frame0(api: OpenAI, frame0: np.ndarray, nouns: list[str]) -> dict[str, Boxes]:
    """One concept request per noun, in parallel. Returns each noun's boxes (SAM-assigned ids)."""
    url = image_data_url(frame0)
    img_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

    def one(noun: str) -> tuple[str, Boxes]:
        text = request_text(api, [{"type": "input_text", "text": noun}, {"type": "input_image", "image_url": url}], {}, _key("img", img_hash, noun))
        return noun, {int(r.object_id): (int(r.left), int(r.top), int(r.right), int(r.bottom)) for r in parse(text, video=False) if r.kind == "box"}

    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(one, [sanitize(n) for n in nouns]))


def box_prompt(boxes: Boxes, width: int, height: int) -> str:
    """Box prompts on frame 0 under our own ids; size is the displayed (post-rotation) media size."""
    if not 1 <= len(boxes) <= MAX_OBJECTS:
        raise ValueError(f"SAM box prompts name 1..{MAX_OBJECTS} objects, got {len(boxes)}")
    objects = [msp.PromptObject(oid, [msp.FramePrompt(0, msp.PromptBox(*b))]) for oid, b in sorted(boxes.items())]
    return msp.build_object_prompt(size=(width, height), objects=objects)


def upload_video(api: OpenAI, path: Path) -> str:
    with open(path, "rb") as f:
        return api.files.create(file=f, purpose=FILE_PURPOSE).id


def track(api: OpenAI, video: Path, prompt: str) -> str:
    """One box-prompted video request; uploads only on a cache miss, so cached runs make no API calls."""
    key = _key("vid", hashlib.sha256(video.read_bytes()).hexdigest()[:16], prompt)
    if (CACHE / f"{key}.txt").exists():
        return (CACHE / f"{key}.txt").read_text()
    content = [{"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": upload_video(api, video)}]
    return request_text(api, content, {"propagation_direction": "forward"}, key)


def parse(text: str, video: bool) -> tuple:
    """Parsed records of a complete response; raises on malformed output instead of guessing."""
    parser = (msp.video_segmentation_format() if video else msp.image_segmentation_format()).create_parser()
    parser.push(text, emit=False)
    result = parser.finish(msp.CompletedOutcome()).result
    if result.diagnostics:
        raise ValueError(f"SAM output diagnostics: {result.diagnostics[:3]}")
    return result.records


def full_frame_masks(text: str, width: int, height: int) -> Masks:
    """Place each box-local mask into a full (height, width) frame."""
    out: Masks = {}
    for r in parse(text, video=True):
        if not isinstance(r, msp.SegmentationMaskRecord):
            continue
        local = np.frombuffer(msp.decode_mask_to_raster(r.mask), np.uint8).reshape(r.mask.height, r.mask.width)
        frame = np.zeros((height, width), np.uint8)
        top, left = int(r.bounds.top), int(r.bounds.left)
        frame[top : top + local.shape[0], left : left + local.shape[1]] = local > 0
        out.setdefault(r.frame.frame_index if r.frame else 0, {})[int(r.object_id)] = frame
    return out
