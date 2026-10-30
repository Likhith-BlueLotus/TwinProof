"""Parametric controllers (master doc §3.7). No LLM-written code is ever executed: agents only pick parameters."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout
from scipy.linalg import solve_discrete_are

from physics.reset import set_state

CONTROL_PERIOD = 0.02  # s, zero-order hold; G2's spectral bound 0.98 is per control period


@dataclass(frozen=True)
class LQRSpec:
    """u = clip(-K (x - x_eq), +-u_max), K from discrete LQR on the nominal twin."""

    q_eq: float
    q_diag: tuple[float, ...]
    r_diag: tuple[float, ...]
    u_max: float
    gains: tuple[float, ...] | None = None  # explicit K row overrides LQR synthesis


def substeps(model: mujoco.MjModel) -> int:
    return max(1, round(CONTROL_PERIOD / model.opt.timestep))


def linearize(model: mujoco.MjModel, q_eq: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete (A, B) over one control period, composed from MuJoCo's per-timestep FD Jacobians.

    Coulomb frictionloss is zeroed first: at zero velocity FD reads dry friction as huge damping. G2 treats
    friction as a bounded disturbance instead (pendulum_lipschitz eps).
    """
    model = copy.deepcopy(model)
    model.dof_frictionloss[:] = 0
    data = mujoco.MjData(model)
    set_state(model, data, np.array([q_eq]), np.zeros(model.nv))
    nx = 2 * model.nv + model.na
    A, B = np.zeros((nx, nx)), np.zeros((nx, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    Ac, Bc = np.eye(nx), np.zeros_like(B)
    for _ in range(substeps(model)):
        Ac, Bc = A @ Ac, A @ Bc + B
    return Ac, Bc


def lqr_gain(A: np.ndarray, B: np.ndarray, spec: LQRSpec) -> np.ndarray:
    if spec.gains is not None:
        return np.atleast_2d(np.asarray(spec.gains, float))
    Q, R = np.diag(spec.q_diag), np.diag(spec.r_diag)
    S = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)


def policy(X: np.ndarray, K: np.ndarray, spec: LQRSpec) -> np.ndarray:
    return np.clip(-X @ K.T, -spec.u_max, spec.u_max)


def closed_loop_step(model: mujoco.MjModel | list[mujoco.MjModel], X: np.ndarray, K: np.ndarray, spec: LQRSpec) -> np.ndarray:
    """Nonlinear one-control-period map F_cl for a batch of deviation states X (n, 2).

    A list of models (same structure, e.g. posterior samples) simulates row i on model i.
    """
    first = model[0] if isinstance(model, list) else model
    u = policy(X, K, spec)
    state0 = np.c_[np.zeros(len(X)), X[:, 0] + spec.q_eq, X[:, 1]]  # [time, qpos, qvel]
    n = substeps(first)
    states, _ = mj_rollout.rollout(model, mujoco.MjData(first), state0, np.repeat(u[:, None, :], n, 1))
    return np.c_[states[:, -1, 1] - spec.q_eq, states[:, -1, 2]]


def pendulum_constants(model: mujoco.MjModel, q_eq: float) -> tuple[float, float, float, float]:
    """(inertia incl. armature, m g l_c, viscous damping, Coulomb frictionloss) for a single-hinge twin."""
    data = mujoco.MjData(model)
    set_state(model, data, np.array([q_eq]), np.zeros(1))
    M = np.zeros((1, 1))
    mujoco.mj_fullM(model, data, M)
    lever = np.linalg.norm(data.xipos[1] - data.xanchor[0])
    mgl = model.body_mass[1] * np.linalg.norm(model.opt.gravity) * lever
    return float(M[0, 0]), float(mgl), float(model.dof_damping[0]), float(model.dof_frictionloss[0])
