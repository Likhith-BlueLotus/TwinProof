"""ArUco homography, rectify-then-moments, orientation unwrap and quality flags (master doc §2.1–2.2)."""
from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

ELONGATION_MIN = 0.15
OCCLUSION_AREA_TOL = 0.2


class PlaneGrid(NamedTuple):
    """Metric raster on the marker plane; (x0, y0) is the plane point at grid pixel (0, 0)."""

    x0: float
    y0: float
    width_m: float
    height_m: float
    delta: float  # metres per grid pixel


class Moments(NamedTuple):
    area: float  # m^2
    centroid: tuple[float, float]  # m, plane frame
    theta: float  # rad, modulo pi
    lam1: float
    lam2: float


def detect_aruco(gray: np.ndarray, marker_id: int, dictionary: int = cv2.aruco.DICT_4X4_50) -> np.ndarray:
    """Corners (TL, TR, BR, BL) in image pixels of the given marker id."""
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary))
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or marker_id not in ids.ravel():
        raise LookupError(f"ArUco id {marker_id} not found")
    return corners[list(ids.ravel()).index(marker_id)].reshape(4, 2).astype(np.float64)


def homography_from_corners(corners_px: np.ndarray, side_m: float) -> np.ndarray:
    """Image -> plane homography mapping the marker corners to (0,0),(s,0),(s,s),(0,s)."""
    plane = np.array([[0, 0], [side_m, 0], [side_m, side_m], [0, side_m]], np.float64)
    H, _ = cv2.findHomography(np.asarray(corners_px, np.float64), plane)
    if H is None:
        raise ValueError("degenerate marker corners")
    return H


def rectify(mask: np.ndarray, H: np.ndarray, grid: PlaneGrid) -> np.ndarray:
    """Warp an image-space mask onto the metric grid (nearest neighbour, so it stays binary)."""
    to_grid = np.array([[1 / grid.delta, 0, -grid.x0 / grid.delta], [0, 1 / grid.delta, -grid.y0 / grid.delta], [0, 0, 1]])
    size = (round(grid.width_m / grid.delta), round(grid.height_m / grid.delta))
    return cv2.warpPerspective(mask.astype(np.uint8), to_grid @ H, size, flags=cv2.INTER_NEAREST)


def moments(rect_mask: np.ndarray, grid: PlaneGrid) -> Moments:
    """Area, centroid, orientation and covariance eigenvalues of a rectified mask, in metres."""
    rows, cols = np.nonzero(rect_mask)
    if rows.size < 3:
        raise ValueError("mask is empty or too small")
    x = grid.x0 + (cols + 0.5) * grid.delta
    y = grid.y0 + (rows + 0.5) * grid.delta
    xb, yb = x.mean(), y.mean()
    mu20, mu02, mu11 = np.mean((x - xb) ** 2), np.mean((y - yb) ** 2), np.mean((x - xb) * (y - yb))
    half_gap = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11**2)
    return Moments(
        area=rows.size * grid.delta**2,
        centroid=(float(xb), float(yb)),
        theta=float(0.5 * np.arctan2(2 * mu11, mu20 - mu02)),
        lam1=float((mu20 + mu02) / 2 + half_gap),
        lam2=float((mu20 + mu02) / 2 - half_gap),
    )


def elongation(m: Moments) -> float:
    """1 - lam2/lam1; orientation is only trusted above ELONGATION_MIN."""
    return 1.0 - m.lam2 / m.lam1 if m.lam1 > 0 else 0.0


def unwrap_theta(theta: np.ndarray) -> np.ndarray:
    """Remove the modulo-pi ambiguity of moment orientation."""
    out = np.asarray(theta, dtype=float).copy()
    for k in range(1, out.size):
        out[k] += np.pi * np.round((out[k - 1] - out[k]) / np.pi)
    return out


def occlusion_flags(area: np.ndarray) -> np.ndarray:
    """Frames whose mask area deviates more than 20% from the median."""
    area = np.asarray(area, dtype=float)
    med = np.median(area)
    return np.abs(area - med) / med > OCCLUSION_AREA_TOL
