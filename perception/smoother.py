"""Constant-acceleration Kalman filter + RTS smoother per channel (master doc §2.3)."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.signal import resample_poly, savgol_filter

H = np.array([[1.0, 0.0, 0.0]])
P0_DIAG_SCALE = 1e4  # vague prior on rate and acceleration
KAISER_BETA = 8.0  # anti-alias window for decimation; pre-registered in BENCHMARK.md
EDGE_TRIM = 3  # decimated samples dropped at each end
DECIMATE = 4  # 60 fps -> 15 Hz


class Smoothed(NamedTuple):
    states: np.ndarray  # (K, 3): q, q_dot, q_ddot
    covs: np.ndarray  # (K, 3, 3)
    nll: float  # innovation negative log-likelihood, for tuning q_tilde


def _transition(dt: float, q_tilde: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.array([[1.0, dt, dt**2 / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    Q = q_tilde * np.array(
        [
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ]
    )
    return F, Q


def rts_smooth(
    t: np.ndarray,
    z: np.ndarray,
    sigma: float,
    q_tilde: float,
    r_scale: np.ndarray | None = None,
) -> Smoothed:
    """Smooth one channel. NaN in z skips the update (R=inf); r_scale inflates R (occlusion gamma)."""
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    K = t.size
    if z.shape != (K,) or K < 2:
        raise ValueError("t and z must be 1-D, same length, >= 2 samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    first = np.flatnonzero(np.isfinite(z))
    if first.size == 0:
        raise ValueError("no finite measurements")
    r = sigma**2 * (np.ones(K) if r_scale is None else np.asarray(r_scale, dtype=float))

    xf = np.zeros((K, 3))
    Pf = np.zeros((K, 3, 3))
    xp = np.zeros((K, 3))
    Pp = np.zeros((K, 3, 3))
    x = np.array([z[first[0]], 0.0, 0.0])
    P = np.diag([sigma**2, P0_DIAG_SCALE, P0_DIAG_SCALE])
    nll = 0.0
    for k in range(K):
        if k > 0:
            F, Q = _transition(t[k] - t[k - 1], q_tilde)
            x, P = F @ x, F @ P @ F.T + Q
        xp[k], Pp[k] = x, P
        if np.isfinite(z[k]):
            s = P[0, 0] + r[k]
            nu = z[k] - x[0]
            g = P[:, 0] / s
            x = x + g * nu
            P = P - np.outer(g, P[0, :])
            nll += 0.5 * (np.log(2 * np.pi * s) + nu**2 / s)
        xf[k], Pf[k] = x, P

    xs, Ps = xf.copy(), Pf.copy()
    for k in range(K - 2, -1, -1):
        F, _ = _transition(t[k + 1] - t[k], q_tilde)
        C = Pf[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return Smoothed(xs, Ps, float(nll))


def savgol_rate(t: np.ndarray, z: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
    """Diagnostic-only derivative: resample uniformly, Savitzky-Golay, map back. Never used downstream."""
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(z)
    tu = np.linspace(t[0], t[-1], t.size)
    zu = np.interp(tu, t[ok], np.asarray(z)[ok])
    du = savgol_filter(zu, window, poly, deriv=1, delta=tu[1] - tu[0])
    return np.interp(t, tu, du)


def decimate_uniform(t: np.ndarray, z: np.ndarray, factor: int = DECIMATE) -> tuple[np.ndarray, np.ndarray]:
    """Resample onto a uniform grid, then polyphase Kaiser-FIR downsample with cutoff at the new Nyquist.

    Codecs cycle frame types (I/P/B) every few frames, putting periodic error on mask edges at fs/2..fs/5.
    At 60 -> 15 Hz with beta 8 those cycles are suppressed by >= 99 dB, white noise stays white (lag acf
    ~0.04: G1's measurement model needs this; a 5 Hz Butterworth gave 0.48) and the 0.8 Hz gain is
    1.00001. Linear edge padding keeps start-up error < 4e-4 rad; EDGE_TRIM samples are dropped per end.
    Missing samples are interpolated first.
    """
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(z)
    tu = np.linspace(t[0], t[-1], t.size)
    zu = np.interp(tu, t[ok], np.asarray(z, dtype=float)[ok])
    zd = resample_poly(zu, 1, factor, window=("kaiser", KAISER_BETA), padtype="line")
    keep = slice(EDGE_TRIM, -EDGE_TRIM)
    return tu[::factor][: zd.size][keep], zd[keep]
