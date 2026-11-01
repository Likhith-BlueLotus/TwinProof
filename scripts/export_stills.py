"""Stills for the landing page: one frame per rig with the identified twin drawn on, as small .webp files.

Run: uv run python -m scripts.export_stills   ->  web/public/stills/<name>.webp
Only stills go into git; the videos stay in data/videos (gitignored).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from api.main import _track, _track_stem, _events, _video_path, get_run

OUT = Path("web/public/stills")
TWIN_BGR = (232, 214, 79)  # --twin #4fd6e8
QUALITY = 82
# name -> (run id, seconds after the twin's first frame): mid-swing or mid-roll, so the ghost is visibly on the object
STILLS = {
    "s1-oct31": ("agent_take3_s1_20261031T194821", 1.6),
    "s1-clamped": ("agent_take3_s1_20261101T160606", 1.2),
    "s2-ramp": ("verify_s2_take1_20261101T170003", 0.55),
    "s2-board": ("verify_nov01_s2_take2_20261101T170008", 0.7),
}


def frame_at(video: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame at {t:.2f} s in {video}")
    return frame


def draw_twin(frame: np.ndarray, overlay: dict, k: int) -> np.ndarray:
    out = frame.copy()
    if overlay.get("kind") == "s2":
        c = np.round(overlay["center_px"][k]).astype(int)
        cv2.circle(out, tuple(c), int(round(overlay["radius_px"])), TWIN_BGR, 3, cv2.LINE_AA)
    else:
        pivot = np.round(overlay["pivot_px"]).astype(int)
        tip = np.round(overlay["tip_px"][k]).astype(int)
        cv2.line(out, tuple(pivot), tuple(tip), TWIN_BGR, 4, cv2.LINE_AA)
        cv2.circle(out, tuple(tip), 6, TWIN_BGR, -1, cv2.LINE_AA)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (run_id, dt) in STILLS.items():
        overlay = get_run(run_id)["overlay"]
        meta, _ = _track(_track_stem(run_id, _events(run_id)))
        video = _video_path(meta)
        if video is None:
            raise SystemExit(f"{run_id}: video not on disk")
        k = int(round(dt * overlay["fps"]))
        img = draw_twin(frame_at(video, overlay["t0"] + dt), overlay, k)
        path = OUT / f"{name}.webp"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_WEBP_QUALITY, QUALITY])
        print(f"{path}  {img.shape[1]}x{img.shape[0]}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
