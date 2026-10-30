"""Render a synthetic S1 take with known physics and pivot: MuJoCo pendulum drawn on a board with the marker,
seen under mild perspective, 60 fps H.264. For testing Phase D-G before real footage exists.

Run: uv run python -m scripts.make_synthetic_s1  ->  data/videos/synthetic/s1.mp4 + s1_truth.json
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np

from physics.identify_s1 import model_for
from physics.rollout import rollout

OUT = Path("data/videos/synthetic")
FPS, SECONDS, W, H = 60, 15.0, 960, 540  # real takes are 10-15 s (data/FILMING.md)
MM = 2  # plane canvas pixels per mm
BOARD_MM = (600, 450)
MARKER_TL_MM, MARKER_MM = (50, 50), 80
PIVOT_MM, LENGTH_MM, WIDTH_MM = (330, 90), 350, 22
X_TRUE = np.r_[np.log([0.004, 0.0008, 2.0, 0.01]), 0, 0]  # damping, frictionloss, (OU unused here)
THETA0 = 0.6
# plane canvas -> image: mild perspective (camera slightly left of and below centre)
PLANE_TO_IMG = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [BOARD_MM[0] * MM, 0], [BOARD_MM[0] * MM, BOARD_MM[1] * MM], [0, BOARD_MM[1] * MM]]),
    np.float32([[150, 40], [830, 70], [810, 510], [170, 500]]),
)


def board() -> np.ndarray:
    img = np.full((BOARD_MM[1] * MM, BOARD_MM[0] * MM, 3), (200, 205, 210), np.uint8)
    side = MARKER_MM * MM
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, side)
    x, y = MARKER_TL_MM[0] * MM, MARKER_TL_MM[1] * MM
    img[y - 12 : y + side + 12, x - 12 : x + side + 12] = 255  # white quiet zone
    img[y : y + side, x : x + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return img


def frame(base: np.ndarray, theta: float) -> np.ndarray:
    canvas = base.copy()
    p = np.array(PIVOT_MM, float)
    tip = p + LENGTH_MM * np.array([np.sin(theta), np.cos(theta)])
    centre = (p + tip) / 2 * MM
    rect = cv2.boxPoints(((centre[0], centre[1]), (WIDTH_MM * MM, LENGTH_MM * MM), -np.degrees(theta)))
    cv2.fillPoly(canvas, [rect.astype(np.int32)], (40, 110, 190), lineType=cv2.LINE_AA)
    cv2.circle(canvas, (int(p[0] * MM), int(p[1] * MM)), 7 * MM, (30, 30, 30), -1, cv2.LINE_AA)
    return cv2.warpPerspective(canvas, PLANE_TO_IMG, (W, H), borderValue=(235, 235, 230))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_xml = Path("physics/models/s1_pendulum.xml").read_text()
    t = np.arange(round(FPS * SECONDS)) / FPS
    theta = rollout(model_for(base_xml, X_TRUE), np.array([THETA0]), np.array([0.0]), t)[:, 0]
    base = board()
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(OUT / "s1.mp4")],
        stdin=subprocess.PIPE,
    )
    for th in theta:
        enc.stdin.write(frame(base, th).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    truth = {
        "pivot_rel_marker_m": [(PIVOT_MM[0] - MARKER_TL_MM[0]) / 1000, (PIVOT_MM[1] - MARKER_TL_MM[1]) / 1000],
        "marker_side_m": MARKER_MM / 1000, "length_m": LENGTH_MM / 1000, "theta0": THETA0,
        "damping": float(np.exp(X_TRUE[0])), "frictionloss": float(np.exp(X_TRUE[1])), "fps": FPS, "frames": len(t),
    }
    (OUT / "s1_truth.json").write_text(json.dumps(truth, indent=1))
    print(OUT / "s1.mp4", truth)


if __name__ == "__main__":
    main()
