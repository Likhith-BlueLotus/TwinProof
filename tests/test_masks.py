import cv2
import numpy as np
import pytest

from perception.masks import (
    PlaneGrid, detect_aruco, elongation, homography_from_corners, moments, occlusion_flags,
    rectify, unwrap_theta,
)

SIDE = 0.08
# oblique camera: plane metres -> image pixels
PLANE_TO_IMG = np.array([[2600.0, 700.0, 300.0], [-150.0, 2400.0, 250.0], [0.35, 0.9, 1.0]])


def _to_img(xy):
    return cv2.perspectiveTransform(np.asarray(xy, np.float64).reshape(-1, 1, 2), PLANE_TO_IMG).reshape(-1, 2)


def _bar_mask(center, length, width, angle, shape=(900, 1100)):
    c, s = np.cos(angle), np.sin(angle)
    half = np.array([[-length, -width], [length, -width], [length, width], [-length, width]]) / 2
    corners = half @ np.array([[c, s], [-s, c]]) + center
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(_to_img(corners)).astype(np.int32)], 1)
    return mask


def test_homography_maps_marker_corners_to_metres():
    marker = np.array([[0, 0], [SIDE, 0], [SIDE, SIDE], [0, SIDE]])
    H = homography_from_corners(_to_img(marker), SIDE)
    assert cv2.perspectiveTransform(_to_img(marker).reshape(-1, 1, 2), H).reshape(-1, 2) == pytest.approx(marker, abs=1e-6)


def test_rectify_then_moments_recovers_plane_centroid_and_angle():
    marker = np.array([[0, 0], [SIDE, 0], [SIDE, SIDE], [0, SIDE]])
    H = homography_from_corners(_to_img(marker), SIDE)
    grid = PlaneGrid(x0=-0.05, y0=-0.05, width_m=0.4, height_m=0.4, delta=0.0005)
    mom = moments(rectify(_bar_mask((0.15, 0.12), 0.20, 0.02, 0.4), H, grid), grid)
    assert mom.centroid == pytest.approx((0.15, 0.12), abs=1e-3)
    assert mom.theta == pytest.approx(0.4, abs=0.01)
    assert elongation(mom) > 0.9


def test_image_centroid_mapped_afterwards_is_biased():
    # why the doc demands rectify-before-moments
    marker = np.array([[0, 0], [SIDE, 0], [SIDE, SIDE], [0, SIDE]])
    H = homography_from_corners(_to_img(marker), SIDE)
    ys, xs = np.nonzero(_bar_mask((0.15, 0.12), 0.20, 0.02, 0.4))
    naive = cv2.perspectiveTransform(np.array([[[xs.mean(), ys.mean()]]]), H).ravel()
    assert np.linalg.norm(naive - (0.15, 0.12)) > 2e-3


def test_unwrap_theta_removes_pi_jumps():
    truth = np.linspace(0, 3 * np.pi, 200)
    wrapped = np.mod(truth + np.pi / 2, np.pi) - np.pi / 2
    assert unwrap_theta(wrapped) == pytest.approx(truth, abs=1e-9)


def test_occlusion_flags_area_outliers():
    area = np.full(50, 1.0)
    area[10] = 0.7
    assert list(np.flatnonzero(occlusion_flags(area))) == [10]


def test_detect_aruco_finds_rendered_marker():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = np.full((400, 400), 255, np.uint8)
    img[100:300, 100:300] = cv2.aruco.generateImageMarker(d, 7, 200)
    corners = detect_aruco(img, marker_id=7)
    assert corners == pytest.approx(np.array([[100, 100], [299, 100], [299, 299], [100, 299]]), abs=1.5)
