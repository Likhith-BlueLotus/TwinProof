"""Run the controller gates for an S1 LQR claim across the identified posterior."""
from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np

from physics.controllers import CONTROL_PERIOD, LQRSpec, closed_loop_step, linearize, lqr_gain, pendulum_constants
from verifier.gates import SPECTRAL_MAX, GateResult, g2_stability, g3_bounds
from verifier.lyapunov import RoaResult, certify_roa, lyapunov_matrix, noise_ball_c0, pendulum_lipschitz

ROA_WORST_SAMPLES = 2  # ROA on the nominal twin plus the posterior samples with the largest spectral radius
THETA_REGION_MAX = 1.0  # rad: c_hi is the level whose theta bound is this


class ControllerVerdict(NamedTuple):
    gates: dict[str, GateResult]
    K: np.ndarray
    P: np.ndarray
    roa: RoaResult

    @property
    def admitted(self) -> bool:
        return all(g.passed for g in self.gates.values())


def verify_s1_lqr(nominal: mujoco.MjModel, samples: list[mujoco.MjModel], spec: LQRSpec, ou_kappa: float, ou_sigma: float) -> ControllerVerdict:
    """G2 then G3 (stops at the first failure, like the full verifier)."""
    A, B = linearize(nominal, spec.q_eq)
    K = lqr_gain(A, B, spec)
    Q_L = np.diag(spec.q_diag) + K.T @ np.diag(spec.r_diag) @ K
    P = lyapunov_matrix(A - B @ K, Q_L)
    lin = [linearize(m, spec.q_eq) for m in samples]
    A_s, B_s = np.array([a for a, _ in lin]), np.array([b for _, b in lin])

    a = np.exp(-ou_kappa * CONTROL_PERIOD)
    W = np.diag([ou_sigma**2 / (2 * ou_kappa) * (1 - a**2), 0.0])  # OU disturbance on theta per control period
    c0 = noise_ball_c0(P, A - B @ K, W) if np.max(np.abs(np.linalg.eigvals(A - B @ K))) < 1 else 0.0
    c_hi = THETA_REGION_MAX**2 / np.linalg.inv(P)[0, 0]
    rho = np.max(np.abs(np.linalg.eigvals(A_s - B_s @ K)), axis=1)
    if np.linalg.eigvalsh(P).min() <= 0 or rho.max() > SPECTRAL_MAX:
        roa = RoaResult(0.0, 0.0, np.nan, np.nan, np.nan, 0, "skipped: linear stability checks failed")
    else:
        worst = [samples[i] for i in np.argsort(rho)[::-1][:ROA_WORST_SAMPLES]]
        roa = min((_roa(m, K, P, spec, c0, c_hi) for m in [nominal, *worst]), key=lambda r: r.c_star)

    gates = {"G2": g2_stability(A_s, B_s, K, P, roa)}
    if gates["G2"].passed:
        gates["G3"] = g3_bounds(K, P, roa.c_star, spec.u_max, float(nominal.actuator_ctrlrange[0, 1]))
    return ControllerVerdict(gates, K, P, roa)


def _roa(model: mujoco.MjModel, K: np.ndarray, P: np.ndarray, spec: LQRSpec, c0: float, c_hi: float) -> RoaResult:
    inertia, mgl, damping, fl = pendulum_constants(model, spec.q_eq)
    lip = lambda c: pendulum_lipschitz(P, K, inertia, mgl, damping, fl, CONTROL_PERIOD, c)  # noqa: E731
    return certify_roa(lambda X: closed_loop_step(model, X, K, spec), P, c0, c_hi, lip)
