"""Ornstein-Uhlenbeck residual: exact discretization, Kalman NLL, moment initializer (master doc §2.7)."""
from __future__ import annotations

import numpy as np


def discretize(kappa: float, sigma: float, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-step decay exp(-kappa dt) and process variance sigma^2/(2 kappa) (1 - exp(-2 kappa dt))."""
    a = np.exp(-kappa * dt)
    return a, sigma**2 / (2 * kappa) * (1 - a**2)


def ou_innovations(
    residual: np.ndarray, t: np.ndarray, kappa: np.ndarray, sigma: np.ndarray, r_meas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman innovations nu and their variances S for y = r + meas noise, r ~ OU per channel. residual: (K, m)."""
    res = np.atleast_2d(np.asarray(residual, float).T).T
    r_meas = np.broadcast_to(np.asarray(r_meas, float), res.shape)
    dt = np.diff(np.asarray(t, float))
    nu, var = np.empty_like(res), np.empty_like(res)
    for j in range(res.shape[1]):
        a, q = discretize(kappa[j], sigma[j], dt)
        x, p = 0.0, sigma[j] ** 2 / (2 * kappa[j])  # stationary prior
        for k in range(res.shape[0]):
            if k > 0:
                x, p = a[k - 1] * x, a[k - 1] ** 2 * p + q[k - 1]
            var[k, j] = p + r_meas[k, j]
            nu[k, j] = res[k, j] - x
            g = p / var[k, j]
            x, p = x + g * nu[k, j], (1 - g) * p
    return nu, var


def ou_nll(residual: np.ndarray, t: np.ndarray, kappa: np.ndarray, sigma: np.ndarray, r_meas: np.ndarray) -> float:
    """0.5 * sum(log 2 pi S + nu^2 / S) over all channels and frames."""
    nu, var = ou_innovations(residual, t, kappa, sigma, r_meas)
    return float(0.5 * np.sum(np.log(2 * np.pi * var) + nu**2 / var))


def init_ou(residual: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """kappa = -ln(rho_1)/mean_dt, sigma^2 = 2 kappa Var(r); rho_1 clipped into (0, 1)."""
    res = np.atleast_2d(np.asarray(residual, float).T).T
    dt = float(np.mean(np.diff(t)))
    r = res - res.mean(0)
    rho = np.sum(r[1:] * r[:-1], 0) / np.sum(r**2, 0)
    rho = np.clip(rho, 1e-3, 1 - 1e-6)
    kappa = -np.log(rho) / dt
    return kappa, np.sqrt(2 * kappa * r.var(0))
