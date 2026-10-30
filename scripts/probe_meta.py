"""Day-1 probes against api.meta.ai (master doc §8): Spark 1.3 text, SAM concept / zero-match / box / wrong-aspect.

Run: uv run --env-file .env python scripts/probe_meta.py   (raw outputs -> logs/probes/)
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from openai import APIStatusError, OpenAI

OUT = Path("logs/probes")
client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
W, H = 960, 540


def scene() -> np.ndarray:
    """Wooden bar hanging at an angle in front of a board carrying the 80 mm ArUco marker."""
    img = np.full((H, W, 3), (235, 235, 230), np.uint8)
    cv2.rectangle(img, (80, 40), (880, 500), (205, 210, 215), -1)  # backing board
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 120)
    img[60:180, 100:220] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    bar = cv2.boxPoints(((560, 270), (40, 360), 20)).astype(np.int32)
    cv2.fillPoly(img, [bar], (60, 120, 185))  # wood colour (BGR)
    cv2.circle(img, (int(bar[:, 0].mean() - 60), 110), 9, (40, 40, 40), -1)  # pivot bolt
    return img


def data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def sam(name: str, text: str, url: str) -> None:
    rec: dict = {"probe": name, "prompt": text, "t0": time.time()}
    try:
        stream = client.responses.create(
            model="sam-3.1",
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": text}, {"type": "input_image", "image_url": url}]}],
            stream=True,
        )
        events, chunks = [], []
        for ev in stream:
            events.append(ev.type)
            if ev.type == "response.output_text.delta":
                chunks.append(ev.delta)
        rec.update(status="ok", event_types=sorted(set(events)), output_text="".join(chunks))
    except APIStatusError as e:
        rec.update(status=e.status_code, error=str(e.body)[:2000])
    rec["seconds"] = round(time.time() - rec.pop("t0"), 2)
    (OUT / f"sam_{name}.json").write_text(json.dumps(rec, indent=1))
    preview = rec.get("output_text", rec.get("error", ""))
    print(f"SAM {name:12s} status={rec['status']} {rec['seconds']}s len={len(preview)} :: {preview[:160]!r}")


def spark_text() -> None:
    r = client.responses.create(model="muse-spark-1.3", input="Reply with exactly: ok")
    rec = {"probe": "spark_text", "model": r.model, "output_text": r.output_text, "usage": r.usage.model_dump() if r.usage else None}
    (OUT / "spark_text.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"Spark 1.3 text: {r.output_text!r} usage={rec['usage']}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = scene()
    cv2.imwrite(str(OUT / "probe_scene.png"), img)
    url = data_url(img)
    try:
        spark_text()
    except APIStatusError as e:
        print("Spark 1.3 text failed:", e.status_code, str(e.body)[:300])
    sam("concept", "stick", url)
    sam("zero_match", "giraffe", url)
    sam("box", f"<0f>1<|box;x1=480;y1=80;x2=650;y2=460;w={W};h={H}|>", url)
    sam("wrong_aspect", f"<0f>1<|box;x1=480;y1=80;x2=650;y2=460;w={W};h={W}|>", url)


if __name__ == "__main__":
    main()
