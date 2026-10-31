"""Live Phase D->G on one S1 video: SAM frame-0 detection -> box-prompted tracking -> metric tracks -> pivot.

Run: uv run --env-file .env python -m scripts.run_s1_video data/videos/synthetic/s1.mp4 --side 0.08 --noun bar
Writes logs/runs/<video stem>_s1.json. Compares with <stem>_truth.json when present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from perception.extract import camera_shift, first_frame, frame_times, homography_video, plane_grid, shifted_homography, tracks
from perception.kinematics import fixed_pivot, kasa_circle, swing_angle
from perception.sam_orchestrator import box_prompt, client, detect_frame0, full_frame_masks, track
from perception.smoother import decimate_uniform, rts_smooth
from physics.identify_s1 import pendulum_xml

OBJECT_ID = 1
BASE_MODEL = Path("physics/models/s1_pendulum.xml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--side", type=float, required=True, help="measured marker side, metres")
    ap.add_argument("--noun", action="append", required=True, help="SAM concept for the moving body (repeatable)")
    ap.add_argument("--down", type=float, nargs=2, help="gravity direction in the marker plane; default: estimated from the swing")
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip before the free swing (hand release)")
    ap.add_argument("--mass", type=float, help="bar mass, kg (from a scale); writes the twin model when given")
    ap.add_argument("--mass-assumed", action="store_true", help="mark --mass as a placeholder, not a measurement")
    ap.add_argument("--pivot-measured", type=float, nargs=2, help="tape-measured pivot from the marker's top-left corner, metres")
    ap.add_argument("--length", type=float, help="tape-measured bar length, metres (default: estimated from the masks)")
    ap.add_argument("--marker-from", type=Path, help="take with a cleanly visible marker; used, shift-corrected, when this one has none")
    ap.add_argument("--rig-from", type=Path, help="run JSON of a well-conditioned take of the same rig: reuse its pivot and down "
                    "(a small swing barely constrains the pivot)")
    args = ap.parse_args()

    t_all = frame_times(args.video)
    f0 = first_frame(args.video)
    h, w = f0.shape[:2]
    registration = None
    try:
        H, marker_frames = homography_video(args.video, args.side)
    except ValueError:
        if not args.marker_from:
            raise
        H_ref, marker_frames = homography_video(args.marker_from, args.side)
        shift, n_features = camera_shift(args.marker_from, args.video)
        H = shifted_homography(H_ref, shift)
        registration = {"from": str(args.marker_from), "shift_px": shift.tolist(), "features": n_features}
        print(f"marker hidden in this take: using {args.marker_from.name}, camera shift {shift.round(2)} px ({n_features} features)")
    grid = plane_grid(H, w, h)

    api = client()
    found = detect_frame0(api, f0, args.noun)
    noun, boxes = next(((n, b) for n, b in found.items() if b), (None, {}))
    if not boxes:
        raise SystemExit(f"SAM found none of {args.noun} on frame 0: {found}")
    box = max(boxes.values(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    print(f"frame 0: '{noun}' -> box {box}; marker in {marker_frames} sampled frames")

    text = track(api, args.video, box_prompt({OBJECT_ID: box}, w, h))
    tr_all = tracks(full_frame_masks(text, w, h), t_all, H, grid, [OBJECT_ID])[OBJECT_ID]
    keep = t_all >= args.start  # the free swing only: before release a hand moves the bar
    t, xy, theta, lam1 = t_all[keep], tr_all.xy[keep], tr_all.theta[keep], tr_all.lam1[keep]
    seen = np.isfinite(xy[:, 0])
    print(f"masks on {seen.sum()}/{t.size} free-swing frames")

    sx = rts_smooth(t, xy[:, 0], sigma=0.001, q_tilde=50.0).states[:, 0]
    sy = rts_smooth(t, xy[:, 1], sigma=0.001, q_tilde=50.0).states[:, 0]
    sth = rts_smooth(t, theta, sigma=0.005, q_tilde=50.0).states[:, 0]
    p = np.c_[sx, sy]
    piv = fixed_pivot(p, sth)
    c_kasa, r_kasa = kasa_circle(p)
    rig = json.loads(args.rig_from.read_text()) if args.rig_from else None
    if rig:  # same pin, same registered plane: the rig's pivot, not this take's poorly conditioned one
        center = np.asarray(rig["pivot_m"])
        piv = piv._replace(center=center, radius=float(np.nanmedian(np.linalg.norm(p - center, axis=1))))
    if rig:
        down = np.asarray(rig["down"], float)
    elif args.down:
        down = np.asarray(args.down, float)
    else:  # a symmetric swing averages to vertical
        u = p - piv.center
        down = (u / np.linalg.norm(u, axis=1, keepdims=True)).mean(0)
    down = down / np.linalg.norm(down)
    length_video = float(np.sqrt(12 * np.nanmedian(lam1)))  # uniform bar: lam1 = L^2 / 12
    length = args.length or length_video
    result = {"video": str(args.video), "noun": noun, "box": box, "H": H.tolist(), "marker_frames": marker_frames,
              "down": down.tolist(), "down_estimated": not args.down, "size": [w, h], "start_s": args.start,
              "t0_video": float(t[0]), "frames": int(t.size), "frames_with_mask": int(seen.sum()),
              "pivot_m": piv.center.tolist(), "pivot_valid": piv.valid, "pivot_radius_m": piv.radius,
              "pivot_rms_rel": piv.rms_rel, "pivot_sd_m": float(np.sqrt(np.trace(piv.cov))),
              "kasa_center_m": c_kasa.tolist(), "kasa_radius_m": r_kasa, "bar_length_m": length,
              "bar_length_video_m": length_video, "registration": registration,
              "rig_from": str(args.rig_from) if args.rig_from else None}
    if args.pivot_measured:
        result["pivot_measured_m"] = list(args.pivot_measured)
        result["pivot_error_m"] = float(np.linalg.norm(piv.center - np.array(args.pivot_measured)))
    truth_path = args.video.with_name(args.video.stem + "_truth.json")
    if truth_path.exists():
        truth = json.loads(truth_path.read_text())
        result["pivot_error_m"] = float(np.linalg.norm(piv.center - np.array(truth["pivot_rel_marker_m"])))
        result["truth_pivot_m"] = truth["pivot_rel_marker_m"]
    out = Path("logs/runs") / f"{args.video.stem}_s1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mass:
        model_path = out.with_name(f"{args.video.stem}_model.xml")
        model_path.write_text(pendulum_xml(BASE_MODEL.read_text(), length, piv.radius, args.mass))
        result.update(model=str(model_path), mass_kg=args.mass, mass_assumed=args.mass_assumed)
    # swing angle for identification at 15 Hz (codec-safe decimation); noise from decimated-vs-smoothed scatter
    td, phi = decimate_uniform(t, swing_angle(xy, piv.center, down))
    sm = rts_smooth(td, phi, sigma=0.002, q_tilde=50.0)
    sigma_phi = float(np.std(phi - sm.states[:, 0]))
    result["sigma_phi_rad"] = sigma_phi
    np.savez(out.with_suffix(".npz"), t=td, phi=phi, r_meas=sigma_phi**2)
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: v for k, v in result.items() if k not in ("H", "box")}, indent=1, default=float))


if __name__ == "__main__":
    main()
