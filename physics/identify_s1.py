"""S1 pendulum identification.

x = [log damping, log frictionloss, log kappa, log sigma, d_theta0, d_omega0, c0..c3]. Nuisance terms:
- initial state (theta0 + THETA0_SCALE*d_theta0, ...): the smoother is least certain at the ends of a take, and a
  wrong initial state otherwise leaks into damping and the OU sigma;
- static output calibration phi_obs = q + c0 + c1 q + c2 q^2 + c3 q^3: homography anisotropy, residual pivot
  offset and lens distortion map each true angle to a slightly wrong measured one, a swing-synchronous residual
  no noise model whitens. A time-invariant map of q cannot absorb wrong dynamics (those make the residual
  differ between up- and down-swing at the same q), so G1 still tests the physics. Omitted (x of length 6)
  means identity.
"""
from __future__ import annotations

import mujoco
import numpy as np

from physics.loss import composite_loss
from physics.mjcf_patch import apply_patch
from physics.rollout import rollout

NAMES = ["damping", "frictionloss", "ou_kappa", "ou_sigma", "d_theta0", "d_omega0", "c0", "c1", "c2", "c3"]
PHYSICAL = [0, 1]  # MJCF parameters phi (master doc §2.0): what the identifiability gate judges
REPORTED = [0, 1, 2, 3]  # phi plus the OU noise-model hyperparameters (kappa can sit at its bound when the residual is white)
JOINT = ".//joint[@name='hinge']"
THETA0_SCALE, OMEGA0_SCALE = 0.02, 0.2  # rad, rad/s per unit offset, so CMA-ES sees O(1) coordinates
OBS_SCALE = 0.02  # calibration coefficient per unit offset
N_CAL = 4
UNOBSERVE_ITERS = 20
LOWER = np.r_[np.log([1e-6, 1e-7, 1e-2, 1e-5]), -5, -5, [-5] * N_CAL]
UPPER = np.r_[np.log([1.0, 0.5, 1e2, 1.0]), 5, 5, [5] * N_CAL]
LAM_PRIOR = 1e-3  # weak pull to x0; frozen in BENCHMARK.md before held-out eval
CAL_PRIOR_SD = 1.0  # calibration coefficients ~ N(0, (OBS_SCALE * 1)^2) = +-2%: a measured marker bounds homography error


MOTOR_GRAVITY_RATIO = 2.0  # twin motor limit = 2 m g r: can hold the bar upright with margin, so G3 stays meaningful


def pendulum_xml(base_xml: str, length: float, centroid_radius: float, mass: float) -> str:
    """S1 twin for a uniform bar of `length` whose centroid hangs `centroid_radius` below the pivot.

    The bar extends (L - 2r)/2 above the pivot and (L + 2r)/2 below it; geometry comes from the video,
    mass from a scale.
    """
    import re

    if not 0 < 2 * centroid_radius <= length:
        raise ValueError(f"centroid radius {centroid_radius:.3f} m is inconsistent with bar length {length:.3f} m")
    top, bottom = (length - 2 * centroid_radius) / 2, (length + 2 * centroid_radius) / 2
    limit = MOTOR_GRAVITY_RATIO * mass * 9.81 * centroid_radius
    xml = re.sub(r'fromto="[^"]*"', f'fromto="0 0 {top:.5f} 0 0 {-bottom:.5f}"', base_xml, count=1)
    xml = re.sub(r'(<geom name="rod"[^>]*mass=")[^"]*"', rf'\g<1>{mass:.5g}"', xml)
    return re.sub(r'ctrlrange="[^"]*"', f'ctrlrange="{-limit:.5g} {limit:.5g}"', xml)


def xml_for(base_xml: str, x: np.ndarray) -> str:
    b, fl = np.exp(x[:2])
    return apply_patch(base_xml, {f"{JOINT}@damping": f"{b:.9g}", f"{JOINT}@frictionloss": f"{fl:.9g}"})


def model_for(base_xml: str, x: np.ndarray) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml_for(base_xml, x))


def _cal(x: np.ndarray) -> np.ndarray:
    return OBS_SCALE * np.asarray(x[6 : 6 + N_CAL]) if len(x) >= 6 + N_CAL else np.zeros(N_CAL)


def observe(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Measured swing angle for true angle q under the fitted static calibration."""
    c = _cal(x)
    return q + c[0] + c[1] * q + c[2] * q**2 + c[3] * q**3


def unobserve(phi: float, phi_dot: float, x: np.ndarray) -> tuple[float, float]:
    """Inverse calibration for a measured (angle, rate) by Newton's method; rate by the chain rule."""
    c = _cal(x)
    slope = lambda q: 1 + c[1] + 2 * c[2] * q + 3 * c[3] * q**2  # noqa: E731
    q = float(phi)
    for _ in range(UNOBSERVE_ITERS):
        q -= (float(observe(np.array([q]), x)[0]) - phi) / slope(q)
    if not slope(q) > 0:
        raise ValueError("calibration is not monotone at this angle")
    return q, float(phi_dot / slope(q))


def start_state(x: np.ndarray, state0: tuple[float, float]) -> tuple[float, float]:
    """True-space (angle, rate) at the start of a take: the smoother's measured state, un-calibrated, plus offsets."""
    q, v = unobserve(state0[0], state0[1], x)
    return q + THETA0_SCALE * x[4], v + OMEGA0_SCALE * x[5]


def predict(base_xml: str, x: np.ndarray, t: np.ndarray, start: tuple[float, float]) -> np.ndarray:
    """Measured-angle prediction over t, simulating from the true-space start state at t[0]."""
    return observe(rollout(model_for(base_xml, x), np.array([start[0]]), np.array([start[1]]), t)[:, 0], x)


def residual(base_xml: str, x: np.ndarray, t: np.ndarray, theta: np.ndarray, state0: tuple[float, float]) -> np.ndarray:
    return theta - predict(base_xml, x, t, start_state(x, state0))


def prior_precision(size: int) -> np.ndarray:
    """Per-coordinate precision for composite_loss's lam_prior term: weak everywhere, informative on calibration."""
    prec = np.ones(size)
    prec[6 : 6 + N_CAL] = 0.5 / (LAM_PRIOR * CAL_PRIOR_SD**2)  # lam_prior * prec = 1 / (2 sd^2): a Gaussian log-prior
    return prec


def make_objective(base_xml: str, t: np.ndarray, theta: np.ndarray, state0: tuple[float, float], r_meas: float, x0: np.ndarray):
    def objective(x: np.ndarray) -> float:
        x = np.asarray(x)
        res = residual(base_xml, x, t, theta, state0)
        return composite_loss(
            res, t, np.exp(x[2:3]), np.exp(x[3:4]), r_meas, x, x0, prior_precision(x.size),
            margins=np.zeros(0), lam_prior=LAM_PRIOR, lam_margin=0.0,
        )
    return objective
