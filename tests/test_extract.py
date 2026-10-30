"""Offline Phase D-G check: drawn masks (standing in for SAM) under perspective -> metric tracks -> pivot."""
import cv2
import numpy as np

from perception.extract import plane_grid, tracks
from perception.kinematics import fixed_pivot
from perception.masks import homography_from_corners

MM = 2
PLANE_TO_IMG = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [1200, 0], [1200, 900], [0, 900]]), np.float32([[150, 40], [830, 70], [810, 510], [170, 500]])
)
MARKER_TL, SIDE, PIVOT, LENGTH = np.array([50.0, 50.0]), 80.0, np.array([330.0, 90.0]), 350.0


def _img(points_mm):
    return cv2.perspectiveTransform((np.asarray(points_mm, float) * MM).reshape(-1, 1, 2), PLANE_TO_IMG).reshape(-1, 2)


def test_pivot_from_rectified_masks_matches_truth():
    corners = _img([MARKER_TL, MARKER_TL + [SIDE, 0], MARKER_TL + [SIDE, SIDE], MARKER_TL + [0, SIDE]])
    H = homography_from_corners(corners, SIDE / 1000)
    t = np.arange(120) / 60
    theta = 0.5 * np.cos(2 * np.pi * 0.8 * t)
    masks = {}
    for k, th in enumerate(theta):
        tip = PIVOT + LENGTH * np.array([np.sin(th), np.cos(th)])
        normal = 11 * np.array([np.cos(th), -np.sin(th)])
        quad = _img([PIVOT + normal, tip + normal, tip - normal, PIVOT - normal])
        m = np.zeros((540, 960), np.uint8)
        cv2.fillPoly(m, [np.round(quad).astype(np.int32)], 1)
        masks[k] = {1: m}
    tr = tracks(masks, t, H, plane_grid(H, 960, 540), [1])[1]
    fit = fixed_pivot(tr.xy, tr.theta)
    assert np.linalg.norm(fit.center - (PIVOT - MARKER_TL) / 1000) < 0.003
    assert abs(fit.radius - LENGTH / 2000) < 0.003


def test_missing_masks_become_nan():
    H = np.eye(3)
    tr = tracks({}, np.arange(3) / 60, H, plane_grid(H, 10, 10), [1])[1]
    assert np.all(np.isnan(tr.xy)) and np.all(np.isnan(tr.theta))
