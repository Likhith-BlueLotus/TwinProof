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
        xy, th, area = np.full((t.size, 2), np.nan), np.full(t.size, np.nan), np.zeros(t.size)
        for k in range(t.size):
            mask = masks.get(k, {}).get(oid)
            if mask is None or not mask.any():
                continue
            m = moments(rectify(mask, H, grid), grid)
            xy[k], area[k] = m.centroid, m.area
            if elongation(m) >= 0.15:
                th[k] = m.theta
        ok = np.isfinite(th)
        th[ok] = unwrap_theta(th[ok])
        out[oid] = Track(t, xy, th, area)
    return out


def homography(frame0: np.ndarray, marker_side_m: float, marker_id: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    return homography_from_corners(detect_aruco(gray, marker_id), marker_side_m)
