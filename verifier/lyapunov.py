"""Lyapunov matrix, Lipschitz-sound annulus check and c* bisection (master doc §2.9 G2)."""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import chi2

from verifier.bounds import state_bound

MAX_GRID_POINTS = 400_000
BISECT_ITERS = 10
PAD = 1.2  # grid cells may extend to sqrt(V) <= PAD sqrt(c)
INNER_RATIO = 2.0  # c0 search step and thin-annulus width

StepFn = Callable[[np.ndarray], np.ndarray]  # batch of states (n, nx) -> next states
LipFn = Callable[[float], tuple[float, float]]  # level c -> (L_dV w.r.t. P-norm, friction eps) valid on {V <= c}


class RoaResult(NamedTuple):
    c_star: float  # 0.0 if no level set passed
    c0: float
    max_dv: float  # at the last level set tested
    lipschitz: float
    friction_eps: float
    grid_points: int
    reason: str


def lyapunov_matrix(A_cl: np.ndarray, Q_L: np.ndarray) -> np.ndarray:
    """P with A_cl^T P A_cl - P = -Q_L."""
    return solve_discrete_lyapunov(A_cl.T, Q_L)


def noise_ball_c0(P: np.ndarray, A_cl: np.ndarray, W: np.ndarray) -> float:
    """chi2_{n,0.99} * lambda_max(P Sigma_inf), Sigma_inf the stationary closed-loop covariance under W."""
    sigma_inf = solve_discrete_lyapunov(A_cl, W)
    return float(chi2.ppf(0.99, P.shape[0]) * np.max(np.real(np.linalg.eigvals(P @ sigma_inf))))


def pendulum_lipschitz(P, K, inertia, mgl, damping, frictionloss, dt, c) -> tuple[float, float]:
    """Sound bounds on {V <= c}, distances in the P-norm ||x||_P = sqrt(x^T P x), for a single hinge under ZOH.

    Open-loop flow: ||dPhi||_P <= exp(mu dt), mu = max log-norm of P^1/2 J P^-1/2 over J's vertices
    (d sin q / dq = cos q in [-1, 1]; J is affine in it, the log-norm is convex). Held input: u = clip(-K x0),
    1-Lipschitz clip. So ||F(x) - F(y)||_P <= ell ||x - y||_P and |dV(x) - dV(y)| <= 2 sqrt(c)(ell^2 + 1)||x - y||_P.
    Coulomb friction |tau| <= frictionloss is a bounded disturbance whose effect on dV is at most eps.
    shortcut: bounds the continuous-time model; MuJoCo integrator error at its timestep is not included.
    """
    w, v = np.linalg.eigh(P)
    p_half, p_inv_half = v @ np.diag(np.sqrt(w)) @ v.T, v @ np.diag(1 / np.sqrt(w)) @ v.T
    mu = max(
        np.max(np.linalg.eigvalsh((M + M.T) / 2))
        for cos_q in (-1.0, 1.0)
        for M in [p_half @ np.array([[0.0, 1.0], [-mgl * cos_q / inertia, -damping / inertia]]) @ p_inv_half]
    )
    flow = np.exp(mu * dt)
    integral = (flow - 1) / mu if abs(mu) > 1e-12 else dt  # int_0^dt exp(mu s) ds
    b_p = np.linalg.norm(p_half @ np.array([0.0, 1.0 / inertia]))
    ell = flow + b_p * integral * np.linalg.norm(K @ p_inv_half, 2)
    l_dv = 2 * np.sqrt(c) * (ell**2 + 1)
    e = b_p * frictionloss * integral
    eps = 2 * ell * np.sqrt(c) * e + e**2
    return float(l_dv), float(eps)


def shell_check(step: StepFn, P: np.ndarray, a: float, b: float, lip_fn: LipFn) -> tuple[bool, float, int, str]:
    """Sound check that dV < 0 on the shell {a <= V <= b}.

    Grids z = P^1/2 x, where V = |z|^2 and the P-norm cell radius is exactly h sqrt(n)/2. Passes iff
    max dV(grid) + L * cell_radius + 2 eps < 0, with L, eps valid on {V <= PAD^2 b} (cells may stick out).
    """
    n = P.shape[0]
    w, v = np.linalg.eigh(P)
    p_inv_half = v @ np.diag(1 / np.sqrt(w)) @ v.T
    lipschitz, eps = lip_fn(PAD**2 * b)
    coarse = _grid(np.sqrt(b), 41)
    r = np.linalg.norm(coarse, axis=1)
    coarse = coarse[(r >= np.sqrt(a)) & (r <= np.sqrt(b))]
    dv_coarse = _dv(step, P, p_inv_half, coarse)
    if dv_coarse.max(initial=-np.inf) + 2 * eps >= 0:
        return False, float(dv_coarse.max()), len(coarse), f"max dV + 2 eps = {dv_coarse.max() + 2 * eps:.3g} >= 0 on the coarse grid"
    delta = -(dv_coarse.max() + 2 * eps) / 2
    # 0.9: per_axis rounding makes the real spacing up to ~11% larger than h
    h = 0.9 * min(2 * delta / (lipschitz * np.sqrt(n)), 2 * (PAD - 1) * np.sqrt(b) / np.sqrt(n))
    per_axis = int(np.ceil(2 * np.sqrt(b) / h)) + 2
    if per_axis**n > MAX_GRID_POINTS:
        return False, float(dv_coarse.max()), per_axis**n, f"grid over budget ({per_axis**n} points)"
    Z = _grid(np.sqrt(b) + h, per_axis)
    spacing = 2 * (np.sqrt(b) + h) / (per_axis - 1)  # actual step, slightly above h
    pad = spacing * np.sqrt(n) / 2  # exact P-norm cell radius: sqrt(V) moves at most this much within a cell
    if np.sqrt(b) + pad > PAD * np.sqrt(b):
        return False, float(dv_coarse.max()), per_axis**n, "cells exceed the set where L is valid"
    r = np.linalg.norm(Z, axis=1)
    Z = Z[(r >= np.sqrt(a) - pad) & (r <= np.sqrt(b) + pad)]
    dv = _dv(step, P, p_inv_half, Z)
    worst = float(dv.max()) + lipschitz * pad + 2 * eps
    return worst < 0, float(dv.max()), len(Z), "" if worst < 0 else f"bound {worst:.3g} >= 0"


def _dv(step: StepFn, P: np.ndarray, p_inv_half: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = Z @ p_inv_half
    Xn = step(X)
    return np.einsum("ij,jk,ik->i", Xn, P, Xn) - np.sum(Z**2, 1)


def _grid(half: float, per_axis: int, n: int = 2) -> np.ndarray:
    axis = np.linspace(-half, half, per_axis)
    return np.stack(np.meshgrid(*[axis] * n, indexing="ij"), -1).reshape(-1, n)


def certify_roa(step: StepFn, P: np.ndarray, c0_noise: float, c_hi: float, lip_fn: LipFn) -> RoaResult:
    """Inner level c0 >= c0_noise (grown past the friction dead band) and the largest c* <= c_hi such that
    every shell [c_i, 2 c_i] between them passes shell_check. c_star = 0.0 means no certificate."""
    if not c0_noise > 0 or INNER_RATIO * c0_noise > c_hi:  # also stops a 0 or NaN c0 looping forever
        return RoaResult(0.0, c0_noise, np.nan, *lip_fn(c_hi), 0, f"inner level {c0_noise:.3g} leaves no shell below c_hi {c_hi:.3g}")
    c0, pts_total = c0_noise, 0
    while True:
        ok, max_dv, pts, why = shell_check(step, P, c0, INNER_RATIO * c0, lip_fn)
        pts_total += pts
        if ok:
            break
        c0 *= INNER_RATIO
        if INNER_RATIO * c0 > c_hi:
            return RoaResult(0.0, c0_noise, max_dv, *lip_fn(c_hi), pts_total, f"no shell in (c0, c_hi] certifies: {why}")
    lo = INNER_RATIO * c0
    while lo < c_hi:
        hi = min(INNER_RATIO * lo, c_hi)
        res = shell_check(step, P, lo, hi, lip_fn)
        pts_total += res[2]
        if res[0]:
            lo, max_dv = hi, res[1]
            continue
        for _ in range(BISECT_ITERS):  # c* lies in this shell; each pass extends the certified [c0, lo]
            mid = np.sqrt(lo * hi)
            res = shell_check(step, P, lo, mid, lip_fn)
            pts_total += res[2]
            lo, hi = (mid, hi) if res[0] else (lo, mid)
        break
    # V can rise inside {V <= c0} (no decrease is proven there): with ||F x||_P <= ell sqrt(c0) + e,
    # sup V(F x) = ell^2 c0 + eps (pendulum_lipschitz's L = 2 sqrt(c)(ell^2 + 1)). It must stay within c*.
    l_dv, eps = lip_fn(c0)
    jump = (l_dv / (2 * np.sqrt(c0)) - 1) * c0 + eps
    if not jump <= lo:
        return RoaResult(0.0, float(c0), float(max_dv), l_dv, eps, pts_total, f"inner ball can jump to V={jump:.3g} > c*={lo:.3g}")
    return RoaResult(float(lo), float(c0), float(max_dv), *lip_fn(PAD**2 * lo), pts_total, "")
