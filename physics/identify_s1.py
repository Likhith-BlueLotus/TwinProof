"""S1 pendulum identification.

x = [log damping, log frictionloss, log kappa, log sigma, d_theta0, d_omega0]. The initial state is a
nuisance parameter (theta0 + THETA0_SCALE*d_theta0, ...): the smoother is least certain at the ends of
a take, and a wrong initial state otherwise leaks into damping and the OU sigma.
"""
from __future__ import annotations

import mujoco
import numpy as np

from physics.loss import composite_loss
from physics.mjcf_patch import apply_patch
from physics.ou import ou_innovations
from physics.rollout import rollout

NAMES = ["damping", "frictionloss", "ou_kappa", "ou_sigma", "d_theta0", "d_omega0"]
PHYSICAL = [0, 1, 2, 3]  # log-space entries whose relative std the identifiability gate checks
JOINT = ".//joint[@name='hinge']"
THETA0_SCALE, OMEGA0_SCALE = 0.02, 0.2  # rad, rad/s per unit offset, so CMA-ES sees O(1) coordinates
LOWER = np.r_[np.log([1e-6, 1e-7, 1e-2, 1e-5]), -5, -5]
UPPER = np.r_[np.log([1.0, 0.5, 1e2, 1.0]), 5, 5]
LAM_PRIOR = 1e-3  # weak pull to x0; frozen in BENCHMARK.md before held-out eval


def xml_for(base_xml: str, x: np.ndarray) -> str:
    b, fl = np.exp(x[:2])
    return apply_patch(base_xml, {f"{JOINT}@damping": f"{b:.9g}", f"{JOINT}@frictionloss": f"{fl:.9g}"})


def model_for(base_xml: str, x: np.ndarray) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml_for(base_xml, x))


def residual(base_xml: str, x: np.ndarray, t: np.ndarray, theta: np.ndarray, state0: tuple[float, float]) -> np.ndarray:
    """theta_obs - theta_sim, simulating from the smoother's state0 plus the fitted offsets."""
    q0 = np.array([state0[0] + THETA0_SCALE * x[4]])
    v0 = np.array([state0[1] + OMEGA0_SCALE * x[5]])
    return theta - rollout(model_for(base_xml, x), q0, v0, t)[:, 0]


def make_objective(base_xml: str, t: np.ndarray, theta: np.ndarray, state0: tuple[float, float], r_meas: float, x0: np.ndarray):
    def objective(x: np.ndarray) -> float:
        x = np.asarray(x)
        res = residual(base_xml, x, t, theta, state0)
        return composite_loss(
            res, t, np.exp(x[2:3]), np.exp(x[3:4]), r_meas, x, x0, np.ones(x.size),
            margins=np.zeros(0), lam_prior=LAM_PRIOR, lam_margin=0.0,
        )
    return objective


def innovations(base_xml: str, x: np.ndarray, t: np.ndarray, theta: np.ndarray, state0: tuple[float, float], r_meas: float) -> np.ndarray:
    """Normalised OU innovations nu / sqrt(S); white if the twin + OU explains the data (G1)."""
    nu, var = ou_innovations(residual(base_xml, x, t, theta, state0), t, np.exp(x[2:3]), np.exp(x[3:4]), r_meas)
    return (nu / np.sqrt(var))[:, 0]
