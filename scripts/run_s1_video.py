"""Live Phase D->G on one S1 video: SAM frame-0 detection -> box-prompted tracking -> metric tracks -> pivot.

Run: uv run --env-file .env python -m scripts.run_s1_video data/videos/synthetic/s1.mp4 --side 0.08 --noun bar
Writes logs/runs/<video stem>_s1.json. Compares with <stem>_truth.json when present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from perception.extract import first_frame, frame_times, homography, plane_grid, tracks
from perception.kinematics import fixed_pivot, kasa_circle, swing_angle
from perception.sam_orchestrator import box_prompt, client, detect_frame0, full_frame_masks, track
from perception.smoother import decimate_uniform, rts_smooth

OBJECT_ID = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--side", type=float, required=True, help="measured marker side, metres")
    ap.add_argument("--noun", action="append", required=True, help="SAM concept for the moving body (repeatable)")
    ap.add_argument("--down", type=float, nargs=2, default=(0.0, 1.0), help="gravity direction in the marker plane (from the static clip)")
    args = ap.parse_args()

    t = frame_times(args.video)
    f0 = first_frame(args.video)
    h, w = f0.shape[:2]
    H = homography(f0, args.side)
    grid = plane_grid(H, w, h)

    api = client()
    found = detect_frame0(api, f0, args.noun)
    noun, boxes = next(((n, b) for n, b in found.items() if b), (None, {}))
    if not boxes:
        raise SystemExit(f"SAM found none of {args.noun} on frame 0: {found}")
    box = max(boxes.values(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    print(f"frame 0: '{noun}' -> box {box}")

    text = track(api, args.video, box_prompt({OBJECT_ID: box}, w, h))
    masks = full_frame_masks(text, w, h)
    tr = tracks(masks, t, H, grid, [OBJECT_ID])[OBJECT_ID]
    seen = np.isfinite(tr.xy[:, 0])
    print(f"masks on {seen.sum()}/{t.size} frames")

    sx = rts_smooth(t, tr.xy[:, 0], sigma=0.001, q_tilde=50.0).states[:, 0]
    sy = rts_smooth(t, tr.xy[:, 1], sigma=0.001, q_tilde=50.0).states[:, 0]
    sth = rts_smooth(t, tr.theta, sigma=0.005, q_tilde=50.0).states[:, 0]
    p = np.c_[sx, sy]
    piv = fixed_pivot(p, sth)
    c_kasa, r_kasa = kasa_circle(p)
    result = {"video": str(args.video), "noun": noun, "box": box, "H": H.tolist(), "down": list(args.down), "size": [w, h], "frames": int(t.size), "frames": int(t.size), "frames_with_mask": int(seen.sum()),
              "pivot_m": piv.center.tolist(), "pivot_valid": piv.valid, "pivot_radius_m": piv.radius, "pivot_rms_rel": piv.rms_rel,
              "kasa_center_m": c_kasa.tolist(), "kasa_radius_m": r_kasa}
    truth_path = args.video.with_name(args.video.stem + "_truth.json")
    if truth_path.exists():
        truth = json.loads(truth_path.read_text())
        result["pivot_error_m"] = float(np.linalg.norm(piv.center - np.array(truth["pivot_rel_marker_m"])))
        result["truth_pivot_m"] = truth["pivot_rel_marker_m"]
    # swing angle for identification at 15 Hz (codec-safe decimation); noise from decimated-vs-smoothed scatter
    td, phi = decimate_uniform(t, swing_angle(tr.xy, piv.center, np.array(args.down)))
    sm = rts_smooth(td, phi, sigma=0.002, q_tilde=50.0)
    sigma_phi = float(np.std(phi - sm.states[:, 0]))
    result["sigma_phi_rad"] = sigma_phi
    out = Path("logs/runs") / f"{args.video.stem}_s1.json"
    np.savez(out.with_suffix(".npz"), t=td, phi=phi, r_meas=sigma_phi**2)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
