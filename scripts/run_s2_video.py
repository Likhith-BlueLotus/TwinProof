"""S2 on one incline video: SAM tracks the can; centroid gives distance along the ramp, the dark stripe on its
end face gives rotation; the known can radius corrects the scale (the can is in front of the marker plane).

Run: uv run --env-file .env python -m scripts.run_s2_video data/videos/s2_incline/take1.mp4 --side 0.07 --radius 0.0325 --noun can
Writes logs/runs/<stem>_s2.json and .npz (t, s, psi at 60 Hz over the free roll).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from perception.extract import first_frame, frame_times, homography_video, plane_grid, tracks
from perception.kinematics import prismatic_axis
from perception.sam_orchestrator import box_prompt, client, detect_frame0, full_frame_masks, track

OBJECT_ID = 1
STRIPE_DARKER = 40  # grey levels below the end face's median brightness
MIN_STRIPE_PX = 6
MOVING_MM = 2.0  # displacement that counts as released
AREA_BAND = 0.2  # mask area outside +-20% of resting: blur or occlusion as the can leaves the ramp
DIRECTION_RUN = 0.10  # m of early roll that defines the ramp direction
PERP_MAX = 0.003  # m off that line: the can has left the ramp (it drops off the end)
OUTLIER_RAD = 0.4  # stripe readings this far from the local median trend are misdetections (blur, glare)
INNER_DISC = 0.8  # look for the stripe only inside this fraction of the face radius: the rim and its shadow are dark too


def stripe_angle(frame: np.ndarray, mask: np.ndarray, H: np.ndarray) -> float:
    """Angle in the marker plane from the can's centre to the stripe's centroid (a clock hand: full 2 pi range)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    inside = mask > 0
    ys, xs = np.nonzero(inside)
    cx, cy, radius = xs.mean(), ys.mean(), np.sqrt(inside.sum() / np.pi)
    yy, xx = np.indices(mask.shape)
    inside &= (xx - cx) ** 2 + (yy - cy) ** 2 < (INNER_DISC * radius) ** 2
    dark = inside & (gray < np.median(gray[inside]) - STRIPE_DARKER)
    if dark.sum() < MIN_STRIPE_PX:
        return np.nan
    dys, dxs = np.nonzero(dark)
    pts = cv2.perspectiveTransform(np.float64([[[cx, cy]], [[dxs.mean(), dys.mean()]]]), H).reshape(2, 2)
    d = pts[1] - pts[0]
    return float(np.arctan2(d[1], d[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--side", type=float, required=True, help="measured marker side, metres")
    ap.add_argument("--radius", type=float, required=True, help="measured can radius, metres")
    ap.add_argument("--noun", action="append", required=True)
    ap.add_argument("--box", type=int, nargs=4, help="frame-0 pixel box x0 y0 x1 y1 for the can; skips noun detection")
    args = ap.parse_args()

    t = frame_times(args.video)
    f0 = first_frame(args.video)
    h, w = f0.shape[:2]
    H, marker_frames = homography_video(args.video, args.side)
    api = client()
    if args.box:  # operator box when no concept matches frame 0 (e.g. a release stick over the can)
        noun, box = "operator box", tuple(args.box)
    else:
        found = detect_frame0(api, f0, args.noun)
        noun, boxes = next(((n, b) for n, b in found.items() if b), (None, {}))
        if not boxes:
            raise SystemExit(f"SAM found none of {args.noun} on frame 0: {found}; pass --box x0 y0 x1 y1")
        # the can is the most compact detection (a disc), not the ramp
        box = min(boxes.values(), key=lambda b: abs(np.log((b[2] - b[0]) / (b[3] - b[1]))))
    masks = full_frame_masks(track(api, args.video, box_prompt({OBJECT_ID: box}, w, h)), w, h)
    tr = tracks(masks, t, H, plane_grid(H, w, h), [OBJECT_ID])[OBJECT_ID]

    cap = cv2.VideoCapture(str(args.video))
    psi = np.full(t.size, np.nan)
    for k in range(t.size):
        ok, frame = cap.read()
        m = masks.get(k, {}).get(OBJECT_ID)
        if ok and m is not None and m.any():
            psi[k] = stripe_angle(frame, m, H)
    cap.release()

    seen = np.isfinite(tr.xy[:, 0])
    disp = np.linalg.norm(tr.xy - tr.xy[np.flatnonzero(seen)[0]], axis=1)
    moving = np.flatnonzero(disp > MOVING_MM / 1000)
    if moving.size == 0:
        raise SystemExit("the can never moves")
    rest = slice(0, moving[0])
    area_rest = float(np.nanmedian(tr.area[rest][tr.area[rest] > 0]))
    scale = args.radius / np.sqrt(area_rest / np.pi)  # true radius / apparent radius in the marker plane
    start = max(moving[0] - 6, 0)  # keep a few resting frames: release time is fitted
    d = (tr.xy - tr.xy[moving[0]]) * scale
    run = np.flatnonzero((np.arange(t.size) > moving[0]) & (np.linalg.norm(np.nan_to_num(d), axis=1) > DIRECTION_RUN))
    if run.size == 0:
        raise SystemExit(f"the can never rolls {DIRECTION_RUN} m")
    u = d[run[0]] / np.linalg.norm(d[run[0]])
    perp = np.abs(d @ np.array([-u[1], u[0]]))
    off = ~seen | (np.abs(tr.area / area_rest - 1) > AREA_BAND) | (perp > PERP_MAX)
    after = np.flatnonzero((np.arange(t.size) > moving[0]) & off)
    stop = int(after[0]) if after.size else t.size
    win = slice(start, stop)
    p = tr.xy[win] * scale
    axis, flatness = prismatic_axis(p[np.isfinite(p[:, 0])], np.zeros(np.isfinite(p[:, 0]).sum()))
    s = (p - p[0]) @ axis
    if s[-1] < 0:
        axis, s = -axis, -s
    ok = np.isfinite(psi[win])
    psi_win = np.unwrap(psi[win][ok])
    trend = median_filter(psi_win, size=5, mode="nearest")
    good = np.abs(psi_win - trend) < OUTLIER_RAD
    psi_full = np.interp(t[win], t[win][ok][good], psi_win[good])
    psi_full -= psi_full[0]
    if np.median(np.diff(psi_full)) < 0:  # rolling down the ramp turns the can one way; make it positive
        psi_full = -psi_full

    result = {"video": str(args.video), "noun": noun, "box": box, "marker_frames": marker_frames, "size": [w, h],
              "frames_roll": int(stop - start), "release_frame": int(moving[0]), "stop_frame": stop,
              "scale_correction": float(scale), "apparent_radius_m": float(np.sqrt(area_rest / np.pi)),
              "path_axis": axis.tolist(), "path_flatness": flatness, "path_angle_deg_from_x": float(np.degrees(np.arctan2(axis[1], axis[0]))),
              "distance_m": float(s[-1]), "rotation_rad": float(psi_full[-1]),
              "stripe_frames": int(good.sum()), "H": H.tolist()}
    out = Path("logs/runs") / f"{args.video.parent.name}_{args.video.stem}_s2.json"
    out.write_text(json.dumps(result, indent=1, default=float))
    np.savez(out.with_suffix(".npz"), t=t[win] - t[win][0], s=s, psi=psi_full)
    print(json.dumps({k: v for k, v in result.items() if k not in ("H", "box")}, indent=1, default=float))


if __name__ == "__main__":
    main()
