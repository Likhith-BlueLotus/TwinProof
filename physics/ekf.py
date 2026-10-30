"""One-step-ahead innovations of the twin (prediction-error method) for G1's whiteness test.

State [q, q_dot, r]: the MuJoCo twin propagates (q, q_dot) between samples, r is the OU residual, and the
measurement is observe(q) + r + white noise. Process noise on acceleration (white, PSD `accel_psd`) is the
only extra parameter; it is tuned by maximum likelihood on the fit window and then frozen. Open-loop
simulation residuals are not white over many swings (a 0.1% period error becomes phase drift), which is why
whiteness is tested on one-step innovations, as in standard system identification.
"""
from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np
from scipy.optimize import minimize_scalar

from physics.identify_s1 import OBS_SCALE, N_CAL, observe
from physics.reset import set_state

FD_EPS = 1e-5
PSD_BOUNDS = (np.log(1e-10), np.log(1e2))
P0 = np.diag([1e-6, 1e-4, 0.0])  # start-state uncertainty (rad^2, (rad/s)^2); r starts at its stationary variance


class Innovations(NamedTuple):
    nu: np.ndarray  # innovations
    var: np.ndarray  # their predicted variances
    nll: float


def _step(model: mujoco.MjModel, data: mujoco.MjData, q: float, v: float, dt: float) -> np.ndarray:
    set_state(model, data, np.array([q]), np.array([v]))
    for _ in range(max(1, round(dt / model.opt.timestep))):
        mujoco.mj_step(model, data)
    return np.array([data.qpos[0], data.qvel[0]])


def _slope(q: float, x: np.ndarray) -> float:
    c = OBS_SCALE * np.asarray(x[6 : 6 + N_CAL]) if len(x) >= 6 + N_CAL else np.zeros(N_CAL)
    return 1 + c[1] + 2 * c[2] * q + 3 * c[3] * q**2


def innovations(model: mujoco.MjModel, x: np.ndarray, t: np.ndarray, y: np.ndarray, start: tuple[float, float],
                r_meas: float, accel_psd: float) -> Innovations:
    """EKF over the whole take from the true-space start state at t[0]; x supplies OU and calibration."""
    kappa, sigma = np.exp(x[2]), np.exp(x[3])
    data = mujoco.MjData(model)
    m = np.array([start[0], start[1], 0.0])
    P = P0.copy()
    P[2, 2] = sigma**2 / (2 * kappa)
    nu, var, nll = np.empty(t.size), np.empty(t.size), 0.0
    for k in range(t.size):
        if k > 0:
            dt = t[k] - t[k - 1]
            f0 = _step(model, data, m[0], m[1], dt)
            F = np.zeros((3, 3))
            for j in range(2):
                d = np.zeros(2)
                d[j] = FD_EPS
                F[:2, j] = (_step(model, data, *(m[:2] + d), dt) - _step(model, data, *(m[:2] - d), dt)) / (2 * FD_EPS)
            a = np.exp(-kappa * dt)
            F[2, 2] = a
            Q = np.zeros((3, 3))
            Q[:2, :2] = accel_psd * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
            Q[2, 2] = sigma**2 / (2 * kappa) * (1 - a**2)
            m = np.r_[f0, a * m[2]]
            P = F @ P @ F.T + Q
        H = np.array([_slope(m[0], x), 0.0, 1.0])
        var[k] = H @ P @ H + r_meas
        nu[k] = y[k] - (observe(np.array([m[0]]), x)[0] + m[2])
        nll += 0.5 * (np.log(2 * np.pi * var[k]) + nu[k] ** 2 / var[k])
        g = P @ H / var[k]
        m = m + g * nu[k]
        P = P - np.outer(g, H @ P)
    return Innovations(nu, var, float(nll))


def tune_accel_psd(model: mujoco.MjModel, x: np.ndarray, t: np.ndarray, y: np.ndarray, start: tuple[float, float], r_meas: float) -> float:
    """Maximum-likelihood process-noise PSD on the fit window only."""
    res = minimize_scalar(lambda lw: innovations(model, x, t, y, start, r_meas, np.exp(lw)).nll, bounds=PSD_BOUNDS, method="bounded")
    return float(np.exp(res.x))
