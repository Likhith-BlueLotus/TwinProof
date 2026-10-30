import numpy as np
import pytest

from perception.kinematics import fixed_pivot, icr, kasa_circle, prismatic_axis, slip

PIVOT = np.array([0.21, -0.04])
R = 0.35


def _swing(n=400, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4, n)
    th = 0.5 * np.sin(2 * np.pi * 0.8 * t)
    om = 0.5 * 2 * np.pi * 0.8 * np.cos(2 * np.pi * 0.8 * t)
    # body centroid hangs below the pivot; body orientation tracks the swing angle
    p = PIVOT + R * np.c_[np.sin(th), np.cos(th)] + noise * rng.standard_normal((n, 2))
    v = R * om[:, None] * np.c_[np.cos(th), -np.sin(th)]
    return p, v, -th, -om  # rotation sense in an image-like (y-down) frame


def test_fixed_pivot_within_1cm_under_noise():
    p, _, th, _ = _swing(noise=0.002)
    fit = fixed_pivot(p, th + 0.003 * np.random.default_rng(1).standard_normal(th.size))
    assert np.linalg.norm(fit.center - PIVOT) < 0.01
    assert fit.radius == pytest.approx(R, abs=0.01)
    assert fit.valid


def test_fixed_pivot_rejects_non_rotating_motion():
    p = np.c_[np.linspace(0, 0.3, 100), np.zeros(100)]
    assert not fixed_pivot(p, np.zeros(100)).valid


def test_icr_points_at_pivot():
    p, v, _, om = _swing()
    c = icr(p, v, om, omega_min=0.5)
    assert np.max(np.linalg.norm(c - PIVOT, axis=1)) < 1e-9


def test_kasa_matches_pivot():
    p, *_ = _swing(noise=0.001)
    c, r = kasa_circle(p)
    assert np.linalg.norm(c - PIVOT) < 0.01 and r == pytest.approx(R, abs=0.01)


def test_prismatic_axis_for_sliding_block():
    s = np.linspace(0, 0.5, 80)
    axis = np.array([np.cos(0.3), np.sin(0.3)])
    p = s[:, None] * axis + 0.0005 * np.random.default_rng(0).standard_normal((80, 2))
    found, ratio = prismatic_axis(p, np.zeros(80))
    assert abs(found @ axis) == pytest.approx(1, abs=1e-4) and ratio < 0.01
    assert prismatic_axis(p, np.linspace(0, 1, 80)) is None  # rotates -> not prismatic


def test_slip_detects_sliding():
    v_t = np.array([1.0, 1.0])
    omega = np.array([1.0 / 0.02, 0.5 / 0.02])
    _, _, sliding = slip(v_t, omega, rho=0.02)
    assert list(sliding) == [False, True]
