"""ICR, multi-frame fixed pivot, Kåsa cross-check, prismatic fallback and slip (master doc §2.4–2.5)."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

J = np.array([[0.0, -1.0], [1.0, 0.0]])
PIVOT_COV_REL_MAX = 0.1  # sqrt(tr Cov(c)) < 0.1 r
PIVOT_RMS_REL_MAX = 0.03  # RMS(|p - c| - r) / r < 0.03
SLIDING_EPS = 0.1


class PivotFit(NamedTuple):
    center: np.ndarray  # (2,) m
    cov: np.ndarray  # (2, 2)
    radius: float
    rms_rel: float
    valid: bool


def icr(p: np.ndarray, v: np.ndarray, omega: np.ndarray, omega_min: float) -> np.ndarray:
    """Instantaneous centre c = p + J v / omega for frames with |omega| > omega_min."""
    keep = np.abs(omega) > omega_min
    return p[keep] + (v[keep] @ J.T) / omega[keep, None]


def fixed_pivot(p: np.ndarray, theta: np.ndarray) -> PivotFit:
    """Weighted LS over all frame pairs of (I - R_kj) c = p_j - R_kj p_k."""
    p, theta = np.asarray(p, float), np.asarray(theta, float)
    k, j = np.triu_indices(len(p), 1)
    d = theta[j] - theta[k]
    cos, sin = np.cos(d), np.sin(d)
    # R_kj p_k and (I - R_kj)^T t, written out to avoid (pairs, 2, 2) arrays
    rp = np.c_[cos * p[k, 0] - sin * p[k, 1], sin * p[k, 0] + cos * p[k, 1]]
    t = p[j] - rp
    w = np.sin(d / 2) ** 2
    n = np.sum(w * 4 * np.sin(d / 2) ** 2)
    if n < 1e-9:
        return PivotFit(np.full(2, np.nan), np.full((2, 2), np.inf), np.nan, np.inf, False)
    imr_t = np.c_[(1 - cos) * t[:, 0] - sin * t[:, 1], sin * t[:, 0] + (1 - cos) * t[:, 1]]
    c = (w[:, None] * imr_t).sum(0) / n
    res = np.c_[(1 - cos) * c[0] + sin * c[1], -sin * c[0] + (1 - cos) * c[1]] - t
    sigma2 = np.sum(w * np.sum(res**2, 1)) / np.sum(w)
    cov = sigma2 / n * np.eye(2)
    dist = np.linalg.norm(p - c, axis=1)
    r = float(np.median(dist))
    rms_rel = float(np.sqrt(np.mean((dist - r) ** 2)) / r)
    valid = bool(np.sqrt(np.trace(cov)) < PIVOT_COV_REL_MAX * r and rms_rel < PIVOT_RMS_REL_MAX)
    return PivotFit(c, cov, r, rms_rel, valid)


def kasa_circle(p: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic circle fit, used only as a cross-check of the fixed pivot."""
    A = np.c_[p, np.ones(len(p))]
    D, E, F = np.linalg.lstsq(A, -np.sum(p**2, 1), rcond=None)[0]
    c = np.array([-D / 2, -E / 2])
    return c, float(np.sqrt(D**2 / 4 + E**2 / 4 - F))


def prismatic_axis(
    p: np.ndarray, theta: np.ndarray, eps_theta: float = 0.05, eps_p: float = 0.01
) -> tuple[np.ndarray, float] | None:
    """Sliding axis and singular-value ratio, or None if the body rotates or barely moves."""
    if np.max(np.abs(theta - theta[0])) > eps_theta or np.max(np.linalg.norm(p - p[0], axis=1)) <= eps_p:
        return None
    _, s, vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
    return vt[0], float(s[1] / s[0])


def slip(v_t: np.ndarray, omega: np.ndarray, rho: float, v0: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slip s = v_t - omega*rho, relative slip, and sliding flag (relative slip > 0.1)."""
    s = np.asarray(v_t) - np.asarray(omega) * rho
    eps = np.abs(s) / np.maximum(np.abs(v_t), v0)
    return s, eps, eps > SLIDING_EPS
