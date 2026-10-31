"""Video -> metric tracks: PTS timestamps, ArUco homography, SAM masks rectified to the marker plane, moments."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from perception.masks import PlaneGrid, detect_aruco, elongation, homography_from_corners, moments, rectify, unwrap_theta
from perception.sam_orchestrator import Masks

GRID_DELTA_M = 0.0005
GRID_MAX_M = 2.0  # clip the rectified window: far-off image corners map to huge plane extents


class Track(NamedTuple):
    t: np.ndarray  # s, from container PTS
    xy: np.ndarray  # (K, 2) m, marker-plane centroid; NaN where the mask is missing
    theta: np.ndarray  # rad, unwrapped; NaN where missing or elongation too low
    area: np.ndarray  # m^2
    lam1: np.ndarray  # m^2, major-axis second moment (uniform bar: length = sqrt(12 lam1)); NaN where missing


def frame_times(video: Path) -> np.ndarray:
    """Presentation timestamps in seconds (phones record at a variable frame rate)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    t = np.array([float(f["best_effort_timestamp_time"]) for f in json.loads(out.stdout)["frames"]])
    return t - t[0]


def first_frame(video: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))  # applies the container's rotation by default
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read {video}")
    return img


def plane_grid(H: np.ndarray, width: int, height: int) -> PlaneGrid:
    corners = cv2.perspectiveTransform(np.float64([[[0, 0]], [[width, 0]], [[width, height]], [[0, height]]]), H).reshape(-1, 2)
    lo = np.clip(corners.min(0), -GRID_MAX_M, GRID_MAX_M)
    hi = np.clip(corners.max(0), -GRID_MAX_M, GRID_MAX_M)
    return PlaneGrid(float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]), GRID_DELTA_M)


def tracks(masks: Masks, t: np.ndarray, H: np.ndarray, grid: PlaneGrid, object_ids: list[int]) -> dict[int, Track]:
    """Rectify-then-moments per frame and object. shortcut: H from frame 0 only (tripod); re-detect per frame if the camera moves."""
    out = {}
    for oid in object_ids:
        xy, th, area, lam1 = np.full((t.size, 2), np.nan), np.full(t.size, np.nan), np.zeros(t.size), np.full(t.size, np.nan)
        for k in range(t.size):
            mask = masks.get(k, {}).get(oid)
            if mask is None or not mask.any():
                continue
            m = moments(rectify(mask, H, grid), grid)
            xy[k], area[k], lam1[k] = m.centroid, m.area, m.lam1
            if elongation(m) >= 0.15:
                th[k] = m.theta
        ok = np.isfinite(th)
        th[ok] = unwrap_theta(th[ok])
        out[oid] = Track(t, xy, th, area, lam1)
    return out


def homography(frame0: np.ndarray, marker_side_m: float, marker_id: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    return homography_from_corners(detect_aruco(gray, marker_id), marker_side_m)


MIN_MARKER_FRAMES = 5


def homography_video(video: Path, marker_side_m: float, marker_id: int = 0, step: int = 5) -> tuple[np.ndarray, int]:
    """Homography from the median marker corners over every `step`-th frame where the marker is fully visible.

    The moving body can pass in front of the marker, so frame 0 alone may fail; the median over many
    frames also averages down corner noise (tripod assumed: the camera must not move).
    """
    cap = cv2.VideoCapture(str(video))
    corners, k = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if k % step == 0:
            try:
                corners.append(detect_aruco(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), marker_id))
            except LookupError:
                pass
        k += 1
    cap.release()
    corners = consistent_corners(corners)
    if len(corners) < MIN_MARKER_FRAMES:
        raise ValueError(f"marker cleanly visible in only {len(corners)} sampled frames; need {MIN_MARKER_FRAMES}")
    return homography_from_corners(np.median(corners, axis=0), marker_side_m), len(corners)


CONSENSUS_PX = 0.75


def consistent_corners(corners: list[np.ndarray]) -> np.ndarray:
    """Drop decodes whose side length or centre disagrees with the median by > CONSENSUS_PX.

    A body passing over the marker can leave the code readable while corrupting an edge: the
    side shrinks and the centre jumps by pixels, which would bias the homography.
    """
    if not corners:
        return np.zeros((0, 4, 2))
    c = np.asarray(corners)
    side = np.linalg.norm(np.roll(c, -1, axis=1) - c, axis=2).mean(1)
    centre = c.mean(1)
    ok = (np.abs(side - np.median(side)) < CONSENSUS_PX) & (np.linalg.norm(centre - np.median(centre, 0), axis=1) < CONSENSUS_PX)
    return c[ok]


STATIC_STD = 3.0  # grey levels: pixels steadier than this across the take are background
PATCH_HALF, SEARCH_PX, MIN_MATCH = 12, 8, 0.9


def _gray_frames(video: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            frames.append(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32))
    cap.release()
    return frames


def camera_shift(ref_video: Path, video: Path, samples: int = 12, points: int = 60) -> tuple[np.ndarray, int]:
    """Median image translation (dx, dy) of `video` relative to `ref_video`, from static background texture.

    Static pixels are those whose brightness barely changes across each take, so the moving body
    and any hands drop out by themselves. Each feature is matched with sub-pixel peak refinement.
    """
    ref_frames, frames = _gray_frames(ref_video, samples), _gray_frames(video, samples)
    static = (np.std(ref_frames, 0) < STATIC_STD) & (np.std(frames, 0) < STATIC_STD)
    margin = PATCH_HALF + SEARCH_PX
    static[:margin], static[-margin:], static[:, :margin], static[:, -margin:] = False, False, False, False
    ref, img = np.median(ref_frames, 0).astype(np.float32), np.median(frames, 0).astype(np.float32)
    pts = cv2.goodFeaturesToTrack(ref, points, 0.01, 2 * PATCH_HALF, mask=static.astype(np.uint8))
    if pts is None:
        raise ValueError("no static background texture to register the takes")
    shifts = []
    for x, y in pts.reshape(-1, 2).astype(int):
        tpl = ref[y - PATCH_HALF:y + PATCH_HALF, x - PATCH_HALF:x + PATCH_HALF]
        win = img[y - margin:y + margin, x - margin:x + margin]
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (px, py) = cv2.minMaxLoc(r)
        if score < MIN_MATCH or not (0 < px < r.shape[1] - 1 and 0 < py < r.shape[0] - 1):
            continue
        dx = 0.5 * (r[py, px - 1] - r[py, px + 1]) / (r[py, px - 1] - 2 * r[py, px] + r[py, px + 1])
        dy = 0.5 * (r[py - 1, px] - r[py + 1, px]) / (r[py - 1, px] - 2 * r[py, px] + r[py + 1, px])
        shifts.append((px + dx - SEARCH_PX, py + dy - SEARCH_PX))
    if len(shifts) < 5:
        raise ValueError(f"only {len(shifts)} background features matched; cannot register the takes")
    return np.median(np.array(shifts), 0), len(shifts)


def shifted_homography(H_ref: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Image -> plane homography for a camera translated by `shift` pixels relative to the reference take.

    shortcut: pure translation; a tripod knocked into rotation needs a full image registration instead.
    """
    return H_ref @ np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]])
